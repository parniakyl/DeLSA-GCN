"""Phase 1 — train the encoder baseline (encoder + mean-pool + linear head).

Works for every dataset; the encoder comes from `model.encoder_name` in the
config, so English/Chinese/Persian differ only by config.

Usage:
    python scripts/01_train_phase1.py --config configs/aapd_phase1.yaml
    python scripts/01_train_phase1.py --config configs/rmsc_phase1.yaml
    python scripts/01_train_phase1.py --config configs/phate_phase1.yaml

Outputs (under `train.save_dir`):
    best_model.pt   checkpoint with the best val micro-F1
    results.json    per-epoch history + final test metrics
    config.yaml     the resolved config actually used
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import yaml
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from src.console import enable_utf8_output
from src.data import label_matrix, load_split_file
from src.losses import build_loss_from_config
from src.metrics import evaluate_all, format_metrics
from src.models.encoder_classifier import EncoderClassifier
from src.reference_scores import print_baseline_comparison
from src.train_utils import (
    build_loaders,
    build_optimizer,
    evaluate_with_scores,
    find_best_threshold,
    load_config,
    report_vram_estimate,
    resolve_test_metrics,
    set_seed,
    train_one_epoch,
)


def main():
    enable_utf8_output()
    parser = argparse.ArgumentParser(description="Phase 1: encoder baseline")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--override", type=str, default=None,
                        help="Optional YAML override file (e.g. for an ablation).")
    args = parser.parse_args()

    cfg = load_config(args.config, args.override)
    train_cfg = cfg["train"]
    data_cfg = cfg["dataset"]
    model_cfg = cfg["model"]
    num_labels = data_cfg["num_labels"]
    dataset_name = data_cfg.get("name", "unknown")

    set_seed(train_cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Dataset: {dataset_name} ({num_labels} labels)")
    print(f"Encoder: {model_cfg['encoder_name']}")

    # Sparse label sets can leave every score below 0.5, making micro-F1 exactly
    # 0. When that happens we re-fit the threshold on validation each epoch.
    use_adaptive_threshold = cfg.get("eval", {}).get("adaptive_threshold", True)
    if use_adaptive_threshold:
        print("  Adaptive threshold: ENABLED (searches 0.05-0.6 if micro_f1≈0)")

    save_dir = Path(train_cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, indent=2)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    print("\nLoading data...")
    train_samples = load_split_file(data_cfg["train_path"], num_labels)
    val_samples   = load_split_file(data_cfg["val_path"],   num_labels)
    test_samples  = load_split_file(data_cfg["test_path"],  num_labels)
    print(f"  Train: {len(train_samples)} | Val: {len(val_samples)} | Test: {len(test_samples)}")

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["encoder_name"])
    train_loader, val_loader, test_loader = build_loaders(
        train_samples, val_samples, test_samples, tokenizer, cfg
    )
    train_label_matrix = label_matrix(train_samples)

    # ------------------------------------------------------------------
    # Model, loss, optimizer, scheduler
    # ------------------------------------------------------------------
    print("\nBuilding model...")
    model = EncoderClassifier(
        encoder_name=model_cfg["encoder_name"],
        num_labels=num_labels,
        dropout=model_cfg["dropout"],
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", False),
        freeze_layers=model_cfg.get("freeze_layers", 0),
    ).to(device)
    print(f"  Trainable params: {model.num_parameters(True)/1e6:.2f}M")
    print(f"  Total params:     {model.num_parameters(False)/1e6:.2f}M")

    loss_fn = build_loss_from_config(
        cfg["loss"], num_labels=num_labels, train_label_matrix=train_label_matrix,
    ).to(device)

    optimizer, optim_name = build_optimizer(model, train_cfg)

    total_steps = len(train_loader) * train_cfg["epochs"] // train_cfg["grad_accum_steps"]
    warmup_steps = int(total_steps * train_cfg["warmup_ratio"])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    use_fp16 = train_cfg.get("fp16", False) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    print(f"  FP16: {use_fp16} | Grad accum: {train_cfg['grad_accum_steps']} "
          f"| Total steps: {total_steps}")
    report_vram_estimate(model, optim_name, use_fp16)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_metric = -1.0
    best_metric_name = train_cfg.get("save_best_metric", "val_micro_f1")
    best_epoch = -1
    history = []
    eval_threshold = cfg["eval"]["threshold"]

    print("\n=== Training (Phase 1) ===")
    print(f"Save best on: {best_metric_name}")

    for epoch in range(train_cfg["epochs"]):
        t0 = time.time()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        train_loss, ok = train_one_epoch(
            model, train_loader, loss_fn, optimizer, scheduler, scaler,
            device=device,
            grad_accum_steps=train_cfg["grad_accum_steps"],
            max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
            log_every=train_cfg.get("log_every", 50),
            epoch=epoch,
        )
        if not ok:
            print(f"Epoch {epoch+1} aborted due to non-finite loss. Stopping training.")
            break

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        val_metrics, val_scores, val_targets = evaluate_with_scores(
            model, val_loader, loss_fn, device, threshold=eval_threshold
        )
        if use_adaptive_threshold and val_metrics.get("micro_f1", 0) < 0.01:
            epoch_thr, _ = find_best_threshold(
                val_scores, val_targets, threshold_range=(0.05, 0.6), step=0.05
            )
            val_loss = val_metrics["loss"]
            val_metrics = evaluate_all(val_scores, val_targets, threshold=epoch_thr)
            val_metrics["loss"] = val_loss
            val_metrics["threshold"] = epoch_thr
            print(f"  (adaptive threshold={epoch_thr:.2f} → "
                  f"val micro_f1={val_metrics['micro_f1']*100:.2f}%)")

        val_metrics["train_loss"] = train_loss
        val_metrics["epoch"] = epoch + 1
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            **{k: v for k, v in val_metrics.items() if k != "loss"},
        })

        print(f"Epoch {epoch+1:02d}/{train_cfg['epochs']} | "
              f"train_loss={train_loss:.4f} | val_loss={val_metrics['loss']:.4f} | "
              f"{format_metrics(val_metrics)} | time={time.time() - t0:.1f}s")

        cur_metric = val_metrics.get(best_metric_name.replace("val_", ""), -1.0)
        if cur_metric > best_metric:
            best_metric = cur_metric
            best_epoch = epoch + 1
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
            }, save_dir / "best_model.pt")
            print(f"  ✓ New best {best_metric_name}={best_metric:.4f}, saved.")

    if best_epoch < 0:
        print("\nNo epoch completed successfully — nothing to evaluate.")
        return

    # ------------------------------------------------------------------
    # Final test evaluation (threshold decided on validation only)
    # ------------------------------------------------------------------
    print(f"\n=== Final test evaluation (best epoch {best_epoch}) ===")
    ckpt = torch.load(save_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    val_metrics_default, val_scores, val_targets = evaluate_with_scores(
        model, val_loader, loss_fn, device, threshold=eval_threshold
    )
    test_metrics_default, test_scores, test_targets = evaluate_with_scores(
        model, test_loader, loss_fn, device, threshold=eval_threshold
    )

    test_metrics, used_threshold, per_label_thrs = resolve_test_metrics(
        val_metrics_default, val_scores, val_targets,
        test_metrics_default, test_scores, test_targets,
        default_threshold=eval_threshold,
    )

    print(format_metrics(test_metrics))
    print(f"\nTest metrics (as percentages) [threshold={used_threshold}]:")
    for k, v in test_metrics.items():
        if k == "loss":
            continue
        print(f"  {k}: {v*100:.2f}")

    results = {
        "best_epoch": best_epoch,
        "best_val_metric": best_metric,
        "best_threshold": used_threshold if isinstance(used_threshold, (int, float)) else None,
        "per_label_thresholds": per_label_thrs.tolist() if per_label_thrs is not None else None,
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
        "history": history,
    }
    with open(save_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {save_dir / 'results.json'}")

    print_baseline_comparison(dataset_name, test_metrics)


if __name__ == "__main__":
    main()
