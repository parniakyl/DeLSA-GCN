"""Check whether a trained model has collapsed to all-zeros or all-ones.

Loads a checkpoint (or a randomly initialised model, as a floor) and reports the
distribution of sigmoid outputs on validation, split by true label.

Healthy:
  sigmoid @ positive labels  mean ~0.7-0.9, std ~0.1-0.2
  sigmoid @ negative labels  mean ~0.05-0.2, std ~0.05-0.15
  clear separation between the two

Collapsed to all-ones:   both means ~0.95+ (no separation)
Collapsed to all-zeros:  both means <0.05  (no separation)

Usage:
    python tools/diagnose_collapse.py --config configs/aapd_phase1.yaml \
        --checkpoint outputs/aapd/phase1/best_model.pt
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
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.console import enable_utf8_output
from src.data import MultiLabelDataset, load_split_file, make_collate_fn
from src.models.encoder_classifier import EncoderClassifier


@torch.no_grad()
def diagnose(model, loader, device, threshold: float = 0.5):
    model.eval()
    pos_sigmoid, neg_sigmoid, all_targets, pred_counts = [], [], [], []

    for batch in loader:
        labels = batch["labels"].to(device, non_blocking=True)
        out = model(
            input_ids=batch["input_ids"].to(device, non_blocking=True),
            attention_mask=batch["attention_mask"].to(device, non_blocking=True),
        )
        scores = torch.sigmoid(out["logits"].float())

        pos_mask = labels > 0.5
        if pos_mask.any():
            pos_sigmoid.append(scores[pos_mask].cpu().numpy())
        if (~pos_mask).any():
            neg_sigmoid.append(scores[~pos_mask].cpu().numpy())

        all_targets.append(labels.cpu().numpy())
        pred_counts.append((scores > threshold).int().sum(dim=1).cpu().numpy())

    all_pos = np.concatenate(pos_sigmoid)
    all_neg = np.concatenate(neg_sigmoid)
    all_targets = np.concatenate(all_targets)
    pred_counts = np.concatenate(pred_counts)

    print(f"\n{'='*60}")
    print("Sigmoid Distribution Analysis")
    print(f"{'='*60}")
    for name, arr in (("Positive labels (target=1)", all_pos),
                      ("Negative labels (target=0)", all_neg)):
        print(f"  {name}:")
        print(f"    count: {len(arr)}")
        print(f"    mean:  {arr.mean():.4f}")
        print(f"    std:   {arr.std():.4f}")
        print(f"    min:   {arr.min():.4f}")
        print(f"    max:   {arr.max():.4f}")
        print(f"    % above 0.5: {(arr > 0.5).mean()*100:.1f}%")
        print(f"    % above 0.3: {(arr > 0.3).mean()*100:.1f}%")
        print()

    pos_mean, neg_mean = all_pos.mean(), all_neg.mean()
    separation = pos_mean - neg_mean
    print(f"  Separation: pos_mean - neg_mean = {separation:.4f}")
    print(f"  Predictions per sample: mean={pred_counts.mean():.1f}, "
          f"std={pred_counts.std():.1f}")
    print(f"  Ground-truth labels per sample: mean={all_targets.sum(1).mean():.1f}")

    print(f"\n{'='*60}")
    print("Diagnosis")
    print(f"{'='*60}")
    if pos_mean > 0.9 and neg_mean > 0.9:
        print("  ❌ COLLAPSED TO ALL-ONES: the model predicts ~1 for everything")
        print("     → switch loss.name from asl to bce")
        print("     → or set ASL clip=0 and gamma_pos=1")
    elif pos_mean < 0.1 and neg_mean < 0.1:
        print("  ❌ COLLAPSED TO ALL-ZEROS: the model predicts ~0 for everything")
        print("     → check the learning rate (likely too high)")
        print("     → check that the labels parsed correctly (see tools/sanity_check.py)")
    elif separation < 0.1:
        print("  ⚠ WEAK SEPARATION: pos and neg are indistinguishable")
        print("     → train longer, or raise the learning rate")
    elif separation < 0.3:
        print("  ⚠ MODERATE SEPARATION: learning, but not confident")
        print("     → train longer, or try eval.threshold 0.3 instead of 0.5")
    else:
        print("  ✓ HEALTHY: the model discriminates positives from negatives")
        if neg_mean > 0.3:
            print(f"     ...though neg_mean={neg_mean:.3f} is high — consider "
                  f"eval.threshold 0.4")
        if pos_mean < 0.6:
            print(f"     ...though pos_mean={pos_mean:.3f} is low — consider "
                  f"eval.threshold 0.4")


def main():
    enable_utf8_output()
    parser = argparse.ArgumentParser(description="Diagnose output collapse")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Phase-1 .pt checkpoint. Omitted = untrained model.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg = cfg["dataset"]
    num_labels = data_cfg["num_labels"]
    print("Loading val data...")
    val_samples = load_split_file(data_cfg["val_path"], num_labels)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["encoder_name"])
    collate = make_collate_fn(tokenizer, max_seq_len=data_cfg["max_seq_len"])
    val_loader = DataLoader(
        MultiLabelDataset(val_samples), batch_size=cfg["train"]["eval_batch_size"],
        shuffle=False, collate_fn=collate, num_workers=0, pin_memory=True,
    )

    model = EncoderClassifier(
        encoder_name=cfg["model"]["encoder_name"],
        num_labels=num_labels,
        dropout=cfg["model"]["dropout"],
        gradient_checkpointing=cfg["train"].get("gradient_checkpointing", False),
        freeze_layers=cfg["model"].get("freeze_layers", 0),
    ).to(device)

    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded epoch {ckpt.get('epoch', '?')}")
    else:
        print("Using an UNTRAINED model (random weights) as a floor for comparison")

    diagnose(model, val_loader, device, threshold=cfg["eval"]["threshold"])


if __name__ == "__main__":
    main()
