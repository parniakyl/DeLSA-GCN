"""End-to-end smoke test: data -> tokenizer -> model -> loss -> metrics.

Verifies that the encoder downloads, the TSV parses, one forward pass runs, and
the loss and metrics compute — before committing to a multi-hour training run.

Usage:
    python tools/sanity_check.py --config configs/aapd_phase1.yaml
    python tools/sanity_check.py --config configs/rmsc_phase1.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import yaml
from transformers import AutoTokenizer

from src.console import enable_utf8_output
from src.data import MultiLabelSample, load_tsv, make_collate_fn
from src.losses import build_loss_from_config
from src.metrics import evaluate_all, format_metrics
from src.models.encoder_classifier import EncoderClassifier


def main():
    enable_utf8_output()
    parser = argparse.ArgumentParser(description="Pipeline smoke test")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg["dataset"]
    num_labels = data_cfg["num_labels"]

    print("=== Sanity check ===")
    print(f"Config:  {args.config}")
    print(f"Dataset: {data_cfg.get('name', '?')} ({num_labels} labels)")
    print(f"Encoder: {cfg['model']['encoder_name']}")

    # 1) Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["encoder_name"])
    print(f"Tokenizer vocab size: {len(tokenizer)}")

    # 2) Data (fall back to a synthetic batch if the corpus isn't there yet)
    train_path = Path(data_cfg["train_path"])
    if not train_path.is_absolute():
        train_path = PROJECT_ROOT / train_path
    if train_path.exists():
        print(f"\nLoading 10 training samples from {train_path}...")
        samples = load_tsv(str(train_path), num_labels=num_labels)[:10]
        print(f"  First sample text (truncated): {samples[0].text[:120]}...")
        print(f"  # positive labels in it: {int(samples[0].labels.sum())}")
    else:
        print(f"\n⚠ Train file not found at {train_path}")
        print("  Using a synthetic batch to verify the forward pass instead.")
        print(f"  (or run: python tools/make_synthetic_data.py --config {args.config})")
        fake_labels = np.zeros(num_labels, dtype=np.float32)
        fake_labels[: min(2, num_labels)] = 1.0
        samples = [
            MultiLabelSample(
                text="We propose a novel graph convolutional network for "
                     "multi-label text classification.",
                labels=fake_labels.copy(),
            )
            for _ in range(4)
        ]

    # 3) Collate
    collate = make_collate_fn(tokenizer, max_seq_len=data_cfg["max_seq_len"])
    batch = collate(samples[:4])
    print("\nBatch shapes:")
    print(f"  input_ids:      {tuple(batch['input_ids'].shape)}")
    print(f"  attention_mask: {tuple(batch['attention_mask'].shape)}")
    print(f"  labels:         {tuple(batch['labels'].shape)}")

    # 4) Model forward
    print("\nBuilding model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EncoderClassifier(
        encoder_name=cfg["model"]["encoder_name"],
        num_labels=num_labels,
        dropout=cfg["model"]["dropout"],
    ).to(device)
    print(f"  Trainable params: {model.num_parameters(True)/1e6:.2f}M")

    batch = {k: v.to(device) for k, v in batch.items()}
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    print(f"  Logits shape: {tuple(out['logits'].shape)}")
    print(f"  Pooled shape: {tuple(out['pooled'].shape)}")

    # 5) Loss
    loss_fn = build_loss_from_config(
        cfg["loss"], num_labels=num_labels,
        train_label_matrix=batch["labels"].cpu().numpy(),
    ).to(device)
    loss = loss_fn(out["logits"], batch["labels"])
    print(f"  Loss value: {loss.item():.4f}")

    # 6) Metrics (detach — these logits still carry grad from the forward pass)
    scores = torch.sigmoid(out["logits"].float()).detach().cpu().numpy()
    metrics = evaluate_all(scores, batch["labels"].cpu().numpy())
    print(f"  Batch metrics: {format_metrics(metrics)}")

    print("\n✓ Sanity check passed.")
    print(f"\nNext: python scripts/01_train_phase1.py --config {args.config}")


if __name__ == "__main__":
    main()
