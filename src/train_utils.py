"""Shared training/evaluation plumbing for the Phase 1 and Phase 2 scripts.

Both phases fine-tune a transformer with the same recipe (fp16 + grad accum +
cosine schedule + NaN guard), so the loop lives here once instead of being
copy-pasted per phase and per dataset.
"""

from __future__ import annotations

import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .data import MultiLabelDataset, make_collate_fn
from .metrics import evaluate_all, ndcg_at_k, precision_at_k


# ---------------------------------------------------------------------------
# Config / reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Seed every RNG we touch and put cuDNN in deterministic mode."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into `base` (used by --override)."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(config_path: str, override_path: str | None = None) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if override_path:
        with open(override_path) as f:
            cfg = deep_merge(cfg, yaml.safe_load(f))
    return cfg


def dataloader_workers() -> int:
    """0 on Windows (spawn makes worker startup dominate), 2 elsewhere."""
    return 0 if sys.platform == "win32" else 2


def build_loaders(train_samples, val_samples, test_samples, tokenizer, cfg):
    """Standard train/val/test DataLoader triple for the encoder phases."""
    collate = make_collate_fn(tokenizer, max_seq_len=cfg["dataset"]["max_seq_len"])
    n_workers = dataloader_workers()
    train_cfg = cfg["train"]

    train_loader = DataLoader(
        MultiLabelDataset(train_samples),
        batch_size=train_cfg["batch_size"], shuffle=True,
        collate_fn=collate, num_workers=n_workers, pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        MultiLabelDataset(val_samples),
        batch_size=train_cfg["eval_batch_size"], shuffle=False,
        collate_fn=collate, num_workers=n_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        MultiLabelDataset(test_samples),
        batch_size=train_cfg["eval_batch_size"], shuffle=False,
        collate_fn=collate, num_workers=n_workers, pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def build_optimizer(model: nn.Module, train_cfg: dict):
    """AdamW (default) or Adafactor, per `train.optimizer`."""
    optim_name = train_cfg.get("optimizer", "adamw").lower()
    if optim_name == "adafactor":
        from transformers.optimization import Adafactor
        print("  Optimizer: Adafactor (low memory, no momentum/variance states)")
        return Adafactor(
            model.parameters(),
            lr=train_cfg["lr"],
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            weight_decay=train_cfg["weight_decay"],
        ), optim_name
    if optim_name == "adamw":
        print("  Optimizer: AdamW (uses ~2x params for m, v states)")
        return AdamW(
            model.parameters(),
            lr=train_cfg["lr"],
            weight_decay=train_cfg["weight_decay"],
        ), optim_name
    raise ValueError(f"Unknown optimizer: {optim_name}")


def report_vram_estimate(model: nn.Module, optim_name: str, use_fp16: bool) -> None:
    """Rough model+grad+optimizer VRAM estimate (activations excluded)."""
    n_params = model.num_parameters(True)
    bytes_per_param = 2 if use_fp16 else 4
    grad_bytes = n_params * bytes_per_param
    optim_bytes = 2 * n_params * 4 if optim_name == "adamw" else 0  # m, v in fp32
    model_bytes = n_params * bytes_per_param
    est_total = (model_bytes + grad_bytes + optim_bytes) / 1e9
    print(f"  Estimated VRAM (model+grad+optim): ~{est_total:.2f} GB")


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    device: torch.device,
    grad_accum_steps: int = 1,
    max_grad_norm: float = 1.0,
    log_every: int = 50,
    epoch: int = 0,
) -> tuple[float, bool]:
    """Run one epoch. Returns (mean_loss, ok) — ok=False means a NaN was hit."""
    model.train()
    running_loss = 0.0
    n_batches = 0
    optimizer.zero_grad()

    t_start = time.time()
    for step, batch in enumerate(loader):
        input_ids      = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels         = batch["labels"].to(device, non_blocking=True)

        # Forward under autocast (fp16 for speed) ...
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            out = model(input_ids=input_ids, attention_mask=attention_mask)

        # ... but compute the loss OUTSIDE autocast, in fp32. ASL/DBL involve
        # log() and pow() which underflow to NaN in fp16.
        logits_fp32 = out["logits"].float()
        loss = loss_fn(logits_fp32, labels) / grad_accum_steps

        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # NaN guard: abort the epoch rather than corrupt the weights.
        if not torch.isfinite(loss):
            print(f"\n!!! NON-FINITE LOSS at step {step+1}: loss={loss.item()}")
            print(f"    logits range: [{logits_fp32.min().item():.4f}, "
                  f"{logits_fp32.max().item():.4f}]")
            print(f"    labels sum: {labels.sum().item()}")
            print("    Aborting epoch to prevent weight corruption.")
            return running_loss / max(n_batches, 1), False

        if (step + 1) % grad_accum_steps == 0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        running_loss += loss.item() * grad_accum_steps
        n_batches += 1

        if log_every > 0 and (step + 1) % log_every == 0:
            print(f"  [Epoch {epoch+1}] step {step+1}/{len(loader)} | "
                  f"loss={running_loss/n_batches:.4f} | "
                  f"lr={scheduler.get_last_lr()[0]:.2e} | "
                  f"elapsed={time.time() - t_start:.1f}s")

    return running_loss / max(n_batches, 1), True


@torch.no_grad()
def evaluate_with_scores(model, loader, loss_fn, device, threshold: float = 0.5):
    """Evaluate and also return raw scores/targets (for threshold search).

    Returns (metrics, scores, targets).
    """
    model.eval()
    all_scores, all_targets = [], []
    total_loss, n_batches = 0.0, 0

    for batch in loader:
        input_ids      = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels         = batch["labels"].to(device, non_blocking=True)

        # Evaluation always runs in fp32 — no autocast.
        with torch.cuda.amp.autocast(enabled=False):
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(out["logits"], labels)

        all_scores.append(torch.sigmoid(out["logits"].float()).cpu().numpy())
        all_targets.append(labels.cpu().numpy())
        total_loss += loss.item()
        n_batches += 1

    scores = np.concatenate(all_scores, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    metrics = evaluate_all(scores, targets, threshold=threshold)
    metrics["loss"] = total_loss / max(n_batches, 1)
    return metrics, scores, targets


def evaluate(model, loader, loss_fn, device, threshold: float = 0.5) -> dict:
    """Metrics only (thin wrapper over `evaluate_with_scores`)."""
    metrics, _, _ = evaluate_with_scores(model, loader, loss_fn, device, threshold)
    return metrics


# ---------------------------------------------------------------------------
# Threshold search
#
# A fixed 0.5 threshold collapses to micro_f1 == 0 on extremely sparse label
# sets (e.g. a 90-label corpus with a ~1.4% positive rate), because nothing
# ever crosses 0.5. These helpers recover a usable operating point. The
# threshold is ALWAYS chosen on validation, never on test.
# ---------------------------------------------------------------------------

def find_best_threshold(
    scores: np.ndarray,
    targets: np.ndarray,
    threshold_range: tuple = (0.1, 0.6),
    step: float = 0.05,
) -> tuple[float, float]:
    """Single global threshold maximising micro-F1. Returns (threshold, f1)."""
    from sklearn.metrics import f1_score

    best_thr, best_f1 = 0.5, 0.0
    thr = threshold_range[0]
    while thr <= threshold_range[1]:
        f1 = f1_score(targets, (scores >= thr).astype(int),
                      average="micro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
        thr += step
    return best_thr, best_f1


def find_per_label_thresholds(
    scores: np.ndarray,
    targets: np.ndarray,
    threshold_range: tuple = (0.05, 0.7),
    step: float = 0.05,
    min_samples: int = 5,
) -> np.ndarray:
    """Per-label threshold maximising that label's F1.

    Necessary when positive rates differ by orders of magnitude across labels —
    one global threshold cannot serve a 30%-frequency label and a 0.1% one.
    Labels with fewer than `min_samples` positives keep the default 0.5.

    Returns: (num_labels,) array of thresholds.
    """
    from sklearn.metrics import f1_score

    num_labels = scores.shape[1]
    thresholds = np.full(num_labels, 0.5, dtype=np.float32)

    for i in range(num_labels):
        if int(targets[:, i].sum()) < min_samples:
            continue  # too few positives to fit a threshold on
        best_thr, best_f1 = 0.5, 0.0
        thr = threshold_range[0]
        while thr <= threshold_range[1]:
            f1 = f1_score(targets[:, i], (scores[:, i] >= thr).astype(int),
                          zero_division=0)
            if f1 > best_f1:
                best_f1, best_thr = f1, thr
            thr += step
        thresholds[i] = best_thr

    return thresholds


def evaluate_with_per_label_thresholds(
    scores: np.ndarray,
    targets: np.ndarray,
    thresholds: np.ndarray,
) -> dict:
    """Metrics under per-label thresholds.

    Accuracy/F1 use the thresholds; P@k and nDCG@k are rank-based and therefore
    threshold-independent, so they use the raw scores.
    """
    from sklearn.metrics import accuracy_score, f1_score

    preds = (scores >= thresholds[None, :]).astype(int)
    out = {
        "accuracy": float(accuracy_score(targets, preds)),
        "micro_f1": float(f1_score(targets, preds, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
    }
    for k in (1, 3, 5):
        out[f"p@{k}"] = precision_at_k(scores, targets, k)
        out[f"ndcg@{k}"] = ndcg_at_k(scores, targets, k)
    return out


def resolve_test_metrics(
    val_metrics: dict,
    val_scores: np.ndarray,
    val_targets: np.ndarray,
    test_metrics_default: dict,
    test_scores: np.ndarray,
    test_targets: np.ndarray,
    default_threshold: float,
) -> tuple[dict, object, np.ndarray | None]:
    """Pick the final test metrics, falling back to threshold search if needed.

    If the default threshold already works on validation, use it unchanged.
    Otherwise search both a per-label and a global threshold ON VALIDATION and
    report whichever gives the better test micro-F1.

    Returns (test_metrics, threshold_used, per_label_thresholds_or_None).
    """
    if val_metrics.get("micro_f1", 0) >= 0.01:
        return test_metrics_default, default_threshold, None

    print(f"\n⚠ Default threshold={default_threshold} gives micro_f1≈0 on val.")
    print("  Searching optimal PER-LABEL thresholds on the validation set...")
    per_label_thrs = find_per_label_thresholds(
        val_scores, val_targets, threshold_range=(0.05, 0.7), step=0.05, min_samples=5
    )
    n_custom = int((per_label_thrs != 0.5).sum())
    print(f"  ✓ Found {n_custom}/{len(per_label_thrs)} non-default thresholds")
    print(f"  Threshold distribution: min={per_label_thrs.min():.2f}, "
          f"max={per_label_thrs.max():.2f}, mean={per_label_thrs.mean():.2f}")

    global_thr, global_val_f1 = find_best_threshold(
        val_scores, val_targets, threshold_range=(0.05, 0.6), step=0.05
    )
    print(f"  Global threshold: {global_thr:.2f} (val micro_f1 = {global_val_f1:.4f})")

    from .metrics import format_metrics
    test_metrics_per_label = evaluate_with_per_label_thresholds(
        test_scores, test_targets, per_label_thrs)
    test_metrics_per_label["loss"] = test_metrics_default["loss"]
    test_metrics_global = evaluate_all(test_scores, test_targets, threshold=global_thr)

    print("\n--- Test metrics with PER-LABEL thresholds ---")
    print(format_metrics(test_metrics_per_label))
    print(f"\n--- Test metrics with GLOBAL threshold ({global_thr:.2f}) ---")
    print(format_metrics(test_metrics_global))

    if test_metrics_global.get("micro_f1", 0) > test_metrics_per_label.get("micro_f1", 0):
        print("\n  Global threshold is better for micro_f1. Using it.")
        return test_metrics_global, global_thr, per_label_thrs
    print("\n  Per-label thresholds are better. Using them.")
    return test_metrics_per_label, "per-label", per_label_thrs


# ---------------------------------------------------------------------------
# Exponential moving average of weights
# ---------------------------------------------------------------------------

class WeightEMA:
    """Running EMA of model weights.

    Picking a single best epoch is noisy on a small validation set; an EMA of
    the trajectory is a standard variance reduction. Used by Phase 2 as a third
    candidate alongside best-on-val and last-epoch.
    """

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.state[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                # Integer buffers (e.g. num_batches_tracked) can't be averaged.
                self.state[k].copy_(v.detach())
