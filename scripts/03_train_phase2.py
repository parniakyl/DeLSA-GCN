"""Phase 2 — train encoder + LSA (Label-Specific Attention) + gated fusion.

Prerequisite: run `scripts/02_encode_labels.py` first and point
`model.label_emb_path` at its output.

Usage:
    python scripts/03_train_phase2.py --config configs/aapd_phase2.yaml
    python scripts/03_train_phase2.py --config configs/rmsc_phase2.yaml
    python scripts/03_train_phase2.py --config configs/phate_phase2.yaml

Outputs (under `train.save_dir`):
    best_model.pt   best val micro-F1 — this is the checkpoint Phase 3 consumes
    last_model.pt   final epoch, kept for variance inspection
    results.json    history + the OFFICIAL test metrics (from best_model.pt)
    config.yaml     the resolved config actually used

Three candidates are evaluated on test at the end (best-on-val, last epoch, and
the weight EMA if `train.ema_decay > 0`). All three are printed so you can see
the run-to-run variance, but `results.json` always reports **best-on-val** —
selecting on test would be leakage.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import yaml
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from src.console import enable_utf8_output
from src.data import label_matrix, load_split_file
from src.losses import build_loss_from_config
from src.metrics import format_metrics
from src.models.lsa_classifier import LSAClassifier
from src.reference_scores import print_baseline_comparison
from src.train_utils import (
    WeightEMA,
    build_loaders,
    build_optimizer,
    evaluate,
    load_config,
    set_seed,
    train_one_epoch,
)


def main():
    enable_utf8_output()
    parser = argparse.ArgumentParser(description="Phase 2: LSA + gated fusion")
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
    # Label embeddings (the Phase-2 novelty)
    # ------------------------------------------------------------------
    label_emb_path = model_cfg.get("label_emb_path")
    if label_emb_path:
        print(f"\nLoading label embeddings: {label_emb_path}")
        label_data = torch.load(label_emb_path, map_location="cpu", weights_only=False)
        label_emb_init = label_data["label_embeddings"]
        print(f"  Shape: {tuple(label_emb_init.shape)}")
        print(f"  Encoder used: {label_data.get('encoder_name', 'unknown')}")
        if label_data.get("encoder_name") not in (None, model_cfg["encoder_name"]):
            print(f"  ⚠ Label embeddings were built with a DIFFERENT encoder than "
                  f"{model_cfg['encoder_name']} — re-run scripts/02_encode_labels.py.")
    else:
        print("\n  No model.label_emb_path in config — using random label embeddings")
        label_emb_init = None

    # ------------------------------------------------------------------
    # Model, loss, optimizer, scheduler
    # ------------------------------------------------------------------
    print("\nBuilding model (Phase 2: LSA)...")
    model = LSAClassifier(
        encoder_name=model_cfg["encoder_name"],
        num_labels=num_labels,
        dropout=model_cfg["dropout"],
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", False),
        freeze_layers=model_cfg.get("freeze_layers", 0),
        label_emb_init=label_emb_init,
        freeze_label_emb=model_cfg.get("freeze_label_emb", False),
    ).to(device)
    print(f"  Trainable params: {model.num_parameters(True)/1e6:.2f}M")
    print(f"  Total params:     {model.num_parameters(False)/1e6:.2f}M")
    print(f"  Initial gates: mean={model.gate_mean.item():.3f}, "
          f"lsa={model.gate_lsa.item():.3f}")

    loss_fn = build_loss_from_config(
        cfg["loss"], num_labels=num_labels, train_label_matrix=train_label_matrix,
    ).to(device)

    optimizer, _ = build_optimizer(model, train_cfg)

    total_steps = len(train_loader) * train_cfg["epochs"] // train_cfg["grad_accum_steps"]
    warmup_steps = int(total_steps * train_cfg["warmup_ratio"])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    use_fp16 = train_cfg.get("fp16", False) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    print(f"  FP16: {use_fp16} | Grad accum: {train_cfg['grad_accum_steps']} "
          f"| Total steps: {total_steps}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_metric = -1.0
    best_metric_name = train_cfg.get("save_best_metric", "val_micro_f1")
    best_epoch = -1
    history = []
    eval_threshold = cfg["eval"]["threshold"]

    ema_decay = train_cfg.get("ema_decay", 0.0)  # 0 = disabled
    ema = WeightEMA(model, ema_decay) if ema_decay > 0 else None
    if ema is not None:
        print(f"  EMA enabled (decay={ema_decay})")

    print("\n=== Training (Phase 2: LSA) ===")
    print(f"Save best on: {best_metric_name}")
    print("Also saving:  last_model.pt (final epoch, for comparison)")

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
            print(f"Epoch {epoch+1} aborted due to non-finite loss. Stopping.")
            break

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        val_metrics = evaluate(model, val_loader, loss_fn, device, threshold=eval_threshold)
        val_metrics["train_loss"] = train_loss
        val_metrics["epoch"] = epoch + 1
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            **{k: v for k, v in val_metrics.items() if k != "loss"},
        })

        gate_a = model.gate_mean.item()
        gate_b = model.gate_lsa.item()
        print(f"Epoch {epoch+1:02d}/{train_cfg['epochs']} | "
              f"train_loss={train_loss:.4f} | val_loss={val_metrics['loss']:.4f} | "
              f"gates(mean,lsa)=({gate_a:.2f},{gate_b:.2f}) | "
              f"{format_metrics(val_metrics)} | time={time.time() - t0:.1f}s")

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_metrics": val_metrics,
            "gate_mean": gate_a,
            "gate_lsa": gate_b,
        }

        cur_metric = val_metrics.get(best_metric_name.replace("val_", ""), -1.0)
        if cur_metric > best_metric:
            best_metric = cur_metric
            best_epoch = epoch + 1
            torch.save(checkpoint, save_dir / "best_model.pt")
            print(f"  ✓ New best {best_metric_name}={best_metric:.4f}, saved.")

        if ema is not None:
            ema.update(model)

        torch.save(checkpoint, save_dir / "last_model.pt")

    if best_epoch < 0:
        print("\nNo epoch completed successfully — nothing to evaluate.")
        return

    # ------------------------------------------------------------------
    # Final evaluation — three candidates, reported transparently
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("=== Final test evaluation — comparing candidates ===")
    print(f"{'='*70}")

    candidate_results = {}

    print(f"\n--- Candidate 1: best_model.pt "
          f"(epoch {best_epoch}, val {best_metric_name}={best_metric:.4f}) ---")
    ckpt = torch.load(save_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics_best = evaluate(model, test_loader, loss_fn, device, threshold=eval_threshold)
    print(format_metrics(test_metrics_best))
    candidate_results["best_on_val"] = {
        "epoch": best_epoch,
        "test_metrics": {k: float(v) for k, v in test_metrics_best.items() if k != "loss"},
    }

    print(f"\n--- Candidate 2: last_model.pt (epoch {train_cfg['epochs']}) ---")
    ckpt = torch.load(save_dir / "last_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics_last = evaluate(model, test_loader, loss_fn, device, threshold=eval_threshold)
    print(format_metrics(test_metrics_last))
    candidate_results["last_epoch"] = {
        "epoch": train_cfg["epochs"],
        "test_metrics": {k: float(v) for k, v in test_metrics_last.items() if k != "loss"},
    }

    if ema is not None:
        print(f"\n--- Candidate 3: EMA weights (decay={ema_decay}) ---")
        current_state = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(ema.state)
        test_metrics_ema = evaluate(model, test_loader, loss_fn, device, threshold=eval_threshold)
        print(format_metrics(test_metrics_ema))
        candidate_results["ema"] = {
            "epoch": train_cfg["epochs"],
            "test_metrics": {k: float(v) for k, v in test_metrics_ema.items() if k != "loss"},
        }
        model.load_state_dict(current_state)

    print(f"\n{'='*70}")
    print("=== Candidate summary (test micro_f1) ===")
    print(f"{'='*70}")
    best_candidate, best_test_mf1 = None, -1.0
    for name, info in candidate_results.items():
        m = info["test_metrics"]["micro_f1"]
        print(f"  {name:<15} (epoch {info['epoch']}): test_micro_f1 = {m*100:.2f}")
        if m > best_test_mf1:
            best_test_mf1, best_candidate = m, name
    print(f"\n>>> Best candidate on test: {best_candidate} "
          f"(test_micro_f1={best_test_mf1*100:.2f})")
    print(">>> Reported result is 'best_on_val' — selecting on test would be leakage.")

    # best_on_val is the official number
    test_metrics = test_metrics_best
    print("\nTest metrics (official, from best_on_val):")
    for k, v in test_metrics.items():
        if k == "loss":
            continue
        print(f"  {k}: {v*100:.2f}")

    # Restore best_model.pt weights so the in-memory gates match what we report.
    ckpt = torch.load(save_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    results = {
        "best_epoch": best_epoch,
        "best_val_metric": best_metric,
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
        "candidates": candidate_results,
        "history": history,
        "final_gates": {"mean": float(model.gate_mean.item()),
                        "lsa": float(model.gate_lsa.item())},
    }
    with open(save_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {save_dir / 'results.json'}")

    print_baseline_comparison(dataset_name, test_metrics)


if __name__ == "__main__":
    main()
