"""Evaluation metrics for multi-label classification.

We compute:
- Accuracy         (exact match ratio across all labels)
- Micro-F1         (averaged over all (sample, label) decisions)
- Macro-F1         (averaged per-label F1, then averaged)
- nDCG@k  for k in {1, 3, 5}
- P@k     for k in {1, 3, 5}   (Precision@k on ranked scores)

All metrics operate on:
- scores:  (N, L)  float array of model scores (after sigmoid, in [0,1])
- targets: (N, L)  int   array of ground-truth labels in {0, 1}

Threshold for accuracy/F1 is 0.5 (configurable).
For P@k / nDCG@k we rank by score descending, so they are threshold-free.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def accuracy_multilabel(scores: np.ndarray, targets: np.ndarray, threshold: float = 0.5) -> float:
    """Exact-match accuracy (most strict)."""
    preds = (scores >= threshold).astype(np.int32)
    return float(accuracy_score(targets, preds))


def micro_f1(scores: np.ndarray, targets: np.ndarray, threshold: float = 0.5) -> float:
    preds = (scores >= threshold).astype(np.int32)
    return float(f1_score(targets, preds, average="micro", zero_division=0))


def macro_f1(scores: np.ndarray, targets: np.ndarray, threshold: float = 0.5) -> float:
    preds = (scores >= threshold).astype(np.int32)
    return float(f1_score(targets, preds, average="macro", zero_division=0))


def precision_at_k(scores: np.ndarray, targets: np.ndarray, k: int) -> float:
    """Mean Precision@k.

    For each sample take the top-k labels by score; P@k = |top-k ∩ relevant| / k.
    """
    n = scores.shape[0]
    k = min(k, scores.shape[1])
    topk_idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    hits = 0.0
    for i in range(n):
        hits += targets[i, topk_idx[i]].sum()
    return float(hits / (n * k))


def dcg_at_k(scores: np.ndarray, targets: np.ndarray, k: int) -> np.ndarray:
    """Discounted Cumulative Gain @ k, per sample."""
    n = scores.shape[0]
    k = min(k, scores.shape[1])
    topk_idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))  # (k,)
    dcg = np.zeros(n, dtype=np.float64)
    for i in range(n):
        gains = targets[i, topk_idx[i]].astype(np.float64)  # binary gain
        dcg[i] = float((gains * discounts).sum())
    return dcg


def ndcg_at_k(scores: np.ndarray, targets: np.ndarray, k: int) -> float:
    """Mean nDCG@k. IDCG assumes the min(|relevant|, k) top ranks are all hit."""
    n = scores.shape[0]
    k = min(k, scores.shape[1])
    dcg = dcg_at_k(scores, targets, k)

    ideal = np.zeros(n, dtype=np.float64)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    for i in range(n):
        m = min(int(targets[i].sum()), k)
        if m > 0:
            ideal[i] = float(discounts[:m].sum())

    # Samples with no relevant label contribute 0 (and must not divide by 0).
    ndcg = np.zeros(n, dtype=np.float64)
    np.divide(dcg, ideal, out=ndcg, where=(ideal > 0))
    return float(ndcg.mean())


def evaluate_all(
    scores: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
    top_ks: Iterable[int] = (1, 3, 5),
) -> dict[str, float]:
    """Compute the full metric suite reported in the paper."""
    scores = _to_numpy(scores).astype(np.float64)
    targets = _to_numpy(targets).astype(np.int64)

    out: dict[str, float] = {}
    out["accuracy"] = accuracy_multilabel(scores, targets, threshold)
    out["micro_f1"] = micro_f1(scores, targets, threshold)
    out["macro_f1"] = macro_f1(scores, targets, threshold)
    for k in top_ks:
        out[f"p@{k}"] = precision_at_k(scores, targets, k)
        out[f"ndcg@{k}"] = ndcg_at_k(scores, targets, k)
    return out


def format_metrics(metrics: dict[str, float]) -> str:
    """One-line metric summary in a stable key order."""
    keys = ["accuracy", "micro_f1", "macro_f1",
            "ndcg@1", "ndcg@3", "ndcg@5", "p@1", "p@3", "p@5"]
    parts = [f"{k}={metrics[k]*100:.2f}" for k in keys if k in metrics]
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
#
# Naive bootstrapping (recompute every metric from scratch on each of 1000
# resamples) is far too slow: nDCG loops over documents in Python. The trick is
# that every quantity we need is either per-(document, label) — TP/FP/FN — or
# per-document — P@k, nDCG@k, exact match. None of them depend on which OTHER
# documents are in the sample. So we compute them ONCE, then each bootstrap
# replicate is just a sum/mean over resampled row indices.
# ---------------------------------------------------------------------------

def per_sample_stats(
    scores: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
    top_ks: Iterable[int] = (1, 3, 5),
) -> dict:
    """Precompute the per-document quantities every metric aggregates over."""
    scores = _to_numpy(scores).astype(np.float64)
    targets = _to_numpy(targets).astype(np.int64)
    preds = (scores >= threshold).astype(np.int64)

    stats = {
        "n": scores.shape[0],
        "tp": (preds & targets),                    # (N, L)
        "fp": (preds & (1 - targets)),              # (N, L)
        "fn": ((1 - preds) & targets),              # (N, L)
        "exact": (preds == targets).all(axis=1).astype(np.float64),   # (N,)
        "top_ks": tuple(top_ks),
    }
    for k in top_ks:
        kk = min(k, scores.shape[1])
        topk_idx = np.argpartition(-scores, kth=kk - 1, axis=1)[:, :kk]
        gains = np.take_along_axis(targets, topk_idx, axis=1).astype(np.float64)
        stats[f"p@{k}"] = gains.sum(axis=1) / kk                       # (N,)

        discounts = 1.0 / np.log2(np.arange(2, kk + 2))
        dcg = (gains * discounts).sum(axis=1)
        n_rel = np.minimum(targets.sum(axis=1), kk)
        ideal = np.array([discounts[:m].sum() if m > 0 else 0.0 for m in n_rel])
        ndcg = np.zeros_like(dcg)
        np.divide(dcg, ideal, out=ndcg, where=(ideal > 0))
        stats[f"ndcg@{k}"] = ndcg                                      # (N,)
    return stats


def metrics_from_stats(stats: dict, idx: np.ndarray | None = None) -> dict[str, float]:
    """Aggregate precomputed per-sample stats over the rows in `idx`."""
    sl = slice(None) if idx is None else idx
    tp = stats["tp"][sl].sum(axis=0).astype(np.float64)   # (L,)
    fp = stats["fp"][sl].sum(axis=0).astype(np.float64)
    fn = stats["fn"][sl].sum(axis=0).astype(np.float64)

    tp_s, fp_s, fn_s = tp.sum(), fp.sum(), fn.sum()
    denom = 2 * tp_s + fp_s + fn_s
    micro = (2 * tp_s / denom) if denom > 0 else 0.0

    per_label_denom = 2 * tp + fp + fn
    per_label_f1 = np.zeros_like(tp)
    np.divide(2 * tp, per_label_denom, out=per_label_f1, where=(per_label_denom > 0))

    out = {
        "accuracy": float(stats["exact"][sl].mean()),
        "micro_f1": float(micro),
        "macro_f1": float(per_label_f1.mean()),
    }
    for k in stats["top_ks"]:
        out[f"p@{k}"] = float(stats[f"p@{k}"][sl].mean())
        out[f"ndcg@{k}"] = float(stats[f"ndcg@{k}"][sl].mean())
    return out


def bootstrap_metrics(
    scores: np.ndarray,
    targets: np.ndarray,
    n_boot: int = 1000,
    ci: float = 95.0,
    threshold: float = 0.5,
    top_ks: Iterable[int] = (1, 3, 5),
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Percentile bootstrap CIs over test documents.

    Resamples documents (not labels) with replacement, which is the right unit:
    documents are the independent observations.

    Returns {metric: {"point", "mean", "lo", "hi", "std"}} as fractions in [0,1].
    `point` is the metric on the real sample; `mean`/`lo`/`hi` come from the
    bootstrap distribution.
    """
    stats = per_sample_stats(scores, targets, threshold=threshold, top_ks=top_ks)
    n = stats["n"]
    point = metrics_from_stats(stats)

    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {k: [] for k in point}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        for k, v in metrics_from_stats(stats, idx).items():
            draws[k].append(v)

    lo_q, hi_q = (100 - ci) / 2, 100 - (100 - ci) / 2
    return {
        k: {
            "point": point[k],
            "mean": float(np.mean(v)),
            "std": float(np.std(v, ddof=1)),
            "lo": float(np.percentile(v, lo_q)),
            "hi": float(np.percentile(v, hi_q)),
        }
        for k, v in draws.items()
    }
