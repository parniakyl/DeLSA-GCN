"""Run a handful of training steps with verbose diagnostics.

Use this before launching a long run to confirm the loss is finite, gradients
are sane, and the sigmoid outputs aren't already saturated. Nothing is saved.

Usage:
    python tools/debug_first_steps.py --config configs/aapd_phase1.yaml
    python tools/debug_first_steps.py --config configs/rmsc_phase1.yaml --n_steps 10

Healthy output: finite loss, gradient norm below ~10, and a visible gap between
the mean sigmoid at positive vs negative labels after a few steps.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.console import enable_utf8_output
from src.data import MultiLabelDataset, label_matrix, load_split_file, make_collate_fn
from src.losses import build_loss_from_config
from src.models.encoder_classifier import EncoderClassifier
from src.train_utils import build_optimizer, set_seed


def main():
    enable_utf8_output()
    parser = argparse.ArgumentParser(description="Verbose first-N-steps debugger")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--n_steps", type=int, default=5)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["train"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_cfg = cfg["dataset"]
    num_labels = data_cfg["num_labels"]
    print("Loading data...")
    train_samples = load_split_file(data_cfg["train_path"], num_labels)
    print(f"  Train samples: {len(train_samples)}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["encoder_name"])
    collate = make_collate_fn(tokenizer, max_seq_len=data_cfg["max_seq_len"])
    train_loader = DataLoader(
        MultiLabelDataset(train_samples), batch_size=cfg["train"]["batch_size"],
        shuffle=True, collate_fn=collate, num_workers=0, pin_memory=True,
    )

    train_label_matrix = label_matrix(train_samples)
    counts = train_label_matrix.sum(0)
    print(f"  Per-label positive counts (first 10): {counts[:10].tolist()}")
    print(f"  Min/Max/Mean per-label count: "
          f"{counts.min():.0f} / {counts.max():.0f} / {counts.mean():.1f}")

    print("\nBuilding model...")
    model = EncoderClassifier(
        encoder_name=cfg["model"]["encoder_name"],
        num_labels=num_labels,
        dropout=cfg["model"]["dropout"],
        gradient_checkpointing=cfg["train"].get("gradient_checkpointing", False),
        freeze_layers=cfg["model"].get("freeze_layers", 0),
    ).to(device)
    print(f"  Trainable params: {model.num_parameters(True)/1e6:.2f}M")

    loss_fn = build_loss_from_config(
        cfg["loss"], num_labels=num_labels, train_label_matrix=train_label_matrix,
    ).to(device)
    print(f"  Loss: {cfg['loss']['name']} | "
          f"class_balanced={cfg['loss'].get('class_balanced', False)}")

    optimizer, _ = build_optimizer(model, cfg["train"])
    scaler = torch.cuda.amp.GradScaler(
        enabled=cfg["train"].get("fp16", False) and device.type == "cuda")
    print(f"  FP16 enabled: {scaler.is_enabled()}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        print(f"  GPU memory BEFORE training: used={(total-free)/1e9:.2f} GB / "
              f"total={total/1e9:.2f} GB")

    print(f"\n=== Running {args.n_steps} debug steps ===")
    model.train()
    optimizer.zero_grad()
    train_iter = iter(train_loader)

    for step in range(args.n_steps):
        batch = next(train_iter)
        input_ids      = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels         = batch["labels"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            out = model(input_ids=input_ids, attention_mask=attention_mask)

        logits_fp32 = out["logits"].float()   # loss always in fp32
        loss = loss_fn(logits_fp32, labels)
        scaler.scale(loss).backward()

        with torch.no_grad():
            sig = torch.sigmoid(logits_fp32)
            print(f"\n--- Step {step+1} ---")
            print(f"  loss = {loss.item():.6f}  | finite = {torch.isfinite(loss).item()}")
            print(f"  logits  range: [{logits_fp32.min().item():.4f}, "
                  f"{logits_fp32.max().item():.4f}], mean={logits_fp32.mean().item():.4f}")
            print(f"  sigmoid range: [{sig.min().item():.6f}, {sig.max().item():.6f}], "
                  f"mean={sig.mean().item():.6f}")
            print(f"  positives in batch: {labels.sum().item():.0f} "
                  f"| shape={tuple(labels.shape)} | pos rate={labels.mean().item():.4f}")
            if (labels > 0.5).any():
                pos = sig[labels > 0.5]
                print(f"  sigmoid @ positive labels: mean={pos.mean().item():.6f}, "
                      f"std={pos.std().item():.6f}")
            neg = sig[labels < 0.5]
            print(f"  sigmoid @ negative labels: mean={neg.mean().item():.6f}, "
                  f"std={neg.std().item():.6f}")

        if scaler.is_enabled():
            scaler.unscale_(optimizer)   # unscale so the norms below are real
        total_grad_norm, max_grad, n_nan_grads = 0.0, 0.0, 0
        for p in model.parameters():
            if p.grad is None:
                continue
            g = p.grad
            if not torch.isfinite(g).all():
                n_nan_grads += 1
            total_grad_norm += float((g.float() ** 2).sum())
            max_grad = max(max_grad, float(g.abs().max()))
        print(f"  grad norm = {math.sqrt(total_grad_norm):.6f} | "
              f"max abs grad = {max_grad:.6f} | #non-finite-grad params = {n_nan_grads}")

        with torch.no_grad():
            w_norm = math.sqrt(sum(float((p.data.float() ** 2).sum())
                                   for p in model.parameters()))
            print(f"  weight norm = {w_norm:.4f}")

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            print(f"  GPU mem: allocated={torch.cuda.memory_allocated()/1e9:.2f} GB | "
                  f"reserved={torch.cuda.memory_reserved()/1e9:.2f} GB | "
                  f"free={free/1e9:.2f} GB / {total/1e9:.2f} GB")

        nn.utils.clip_grad_norm_(model.parameters(), cfg["train"].get("max_grad_norm", 1.0))
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        del out, logits_fp32, loss, sig
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n✓ All {args.n_steps} steps completed.")
    print("  If the loss stayed finite and the grad norm stayed under ~10, run:")
    print(f"  python scripts/01_train_phase1.py --config {args.config}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
