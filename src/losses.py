"""Loss functions for multi-label classification.

Implements:
- BCEWithLogitsLoss                             (`loss.name: bce`)  — used by all configs
- Asymmetric Loss, Ben-Baruch et al. 2021       (`loss.name: asl`)
- Focal Loss, Lin et al. 2017                   (`loss.name: focal`)
- Distribution-Balanced Loss, Wu et al. 2020    (`loss.name: dbl`)
- Class-Balanced reweighting, Cui et al. 2019   (`loss.class_balanced: true`)

`build_loss_from_config` is the only entry point the scripts use.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Asymmetric Loss (ASL)
# Paper: "Asymmetric Loss for Multi-Label Classification" (ICCV 2021)
# ---------------------------------------------------------------------------

class AsymmetricLoss(nn.Module):
    """Asymmetric Loss.

    For each label logit z and target y:
        x_pos = sigmoid(z)
        x_neg = 1 - x_pos

        if y == 1:  loss = -((1 - x_pos) ** gamma_pos) * log(x_pos + eps)
        if y == 0:  x_neg = (x_neg - clip).clamp_min(0)
                    loss = -(x_neg ** gamma_neg) * log(1 - x_neg + eps)

    gamma_neg > gamma_pos makes negative gradients softer, reducing the
    "negative flooding" problem in multi-label long-tail scenarios.
    """

    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 2.0,        # lowered from 4.0 -> more stable
        clip: float = 0.05,
        eps: float = 1e-6,             # raised from 1e-8 -> safe under fp16
        reduction: str = "mean",
        force_fp32: bool = True,       # always compute in fp32 to avoid NaN
    ):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps
        self.reduction = reduction
        self.force_fp32 = force_fp32

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits / targets: (B, L), targets in {0, 1}

        # CRITICAL: cast to fp32 to avoid fp16 underflow in log() and pow()
        if self.force_fp32:
            logits = logits.float()
            targets = targets.float()

        xs_pos = torch.sigmoid(logits)
        # Clamp to [eps, 1-eps] BEFORE any log/pow to avoid log(0) / 0**gamma
        xs_pos = xs_pos.clamp(min=self.eps, max=1.0 - self.eps)
        xs_neg = (1.0 - xs_pos).clamp(min=0.0)

        # Asymmetric clipping on negatives
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg - self.clip).clamp(min=0.0)

        # After clipping xs_neg can be 0 -> 0**gamma == 0, so neg_loss is 0
        # there, which is the intended ASL behaviour.
        pos_loss = (1.0 - xs_pos) ** self.gamma_pos * torch.log(xs_pos)
        one_minus_xs_neg = (1.0 - xs_neg).clamp(min=self.eps)
        neg_loss = xs_neg ** self.gamma_neg * torch.log(one_minus_xs_neg)

        loss = -(targets * pos_loss + (1 - targets) * neg_loss)

        # Defensive: replace any residual non-finite values with 0.
        loss = torch.where(torch.isfinite(loss), loss, torch.zeros_like(loss))

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


# ---------------------------------------------------------------------------
# Focal Loss (ablation comparison)
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, reduction: str = "mean", force_fp32: bool = True):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.force_fp32 = force_fp32

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.force_fp32:
            logits = logits.float()
            targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ((1 - p_t) ** self.gamma) * bce
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ---------------------------------------------------------------------------
# Distribution-Balanced Loss (Wu et al. 2020)
# "Distribution-Balanced Loss for Multi-Label Classification in Long-Tailed Data"
# ---------------------------------------------------------------------------

class DistributionBalancedLoss(nn.Module):
    """Distribution-Balanced Loss (DBL).

    Three components that specifically address long-tail multi-label:
      1. Re-sampling weighting: rebalance positive/negative per class
      2. Negative tolerance: reduce over-suppression of negatives
      3. Focal-style re-weighting on hard examples

    Unlike the other losses here, DBL consumes `class_weights` inside its own
    forward pass, so `WeightedMultiLabelLoss` must not re-apply them.
    """

    def __init__(
        self,
        focal_gamma: float = 2.0,            # focal loss gamma
        beta: float = 0.999,                 # CB reweighting beta
        neg_tolerance: float = 0.5,          # negative suppression factor
        reduction: str = "none",
        force_fp32: bool = True,
    ):
        super().__init__()
        self.focal_gamma = focal_gamma
        self.beta = beta
        self.neg_tolerance = neg_tolerance
        self.reduction = reduction
        self.force_fp32 = force_fp32

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                class_weights: torch.Tensor = None) -> torch.Tensor:
        if self.force_fp32:
            logits = logits.float()
            targets = targets.float()

        probs = torch.sigmoid(logits)

        # Focal-style reweighting:
        #   positives weighted by (1 - p)^gamma, negatives by p^gamma
        pos_weight = (1 - probs).pow(self.focal_gamma)
        neg_weight = probs.pow(self.focal_gamma)

        bce = -(targets * torch.log(probs.clamp(min=1e-7)) +
                (1 - targets) * torch.log((1 - probs).clamp(min=1e-7)))

        loss = targets * pos_weight * bce + (1 - targets) * neg_weight * bce

        if class_weights is not None:
            # Positives: up-weight rare classes.
            # Negatives: down-weight easy negatives via the tolerance factor.
            pos_class_w = class_weights.unsqueeze(0).to(logits.device)
            neg_class_w = (1.0 / class_weights.clamp(min=0.1)).unsqueeze(0).to(logits.device)
            neg_class_w = neg_class_w * self.neg_tolerance
            loss = loss * (targets * pos_class_w + (1 - targets) * neg_class_w)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ---------------------------------------------------------------------------
# Class-Balanced reweighting (Cui et al. 2019)
# ---------------------------------------------------------------------------

def class_balanced_weights(
    class_counts: np.ndarray,
    beta: float = 0.999,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Per-class weight from the effective number of samples.

    Effective number E_n = (1 - beta^n) / (1 - beta)
    weight_c = (1 - beta) / (1 - beta^n_c),  normalised so mean(weight) == 1.
    """
    class_counts = np.asarray(class_counts, dtype=np.float64)
    # Avoid log(0): labels with zero positive samples get a tiny count.
    class_counts = np.where(class_counts == 0, 0.5, class_counts)
    effective_num = 1.0 - np.power(beta, class_counts)
    weights = (1.0 - beta) / (effective_num + eps)
    weights = weights / np.mean(weights)
    return torch.from_numpy(weights).float()


# ---------------------------------------------------------------------------
# Wrapper combining a base loss + per-class weights
# ---------------------------------------------------------------------------

class WeightedMultiLabelLoss(nn.Module):
    """Apply per-class weights to a multi-label loss.

    loss = reduction(loss_fn(logits, targets) * class_weights)

    DBL is special-cased: it takes `class_weights` as a forward argument and
    applies them internally, so we must not weight twice.
    """

    def __init__(
        self,
        loss_fn: nn.Module,
        class_weights: torch.Tensor | None = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.loss_fn = loss_fn
        self.register_buffer(
            "class_weights",
            class_weights if class_weights is not None else torch.ones(1),
        )
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if isinstance(self.loss_fn, DistributionBalancedLoss):
            per_sample_per_class = self.loss_fn(
                logits, targets, class_weights=self.class_weights.to(logits.device)
            )
            if per_sample_per_class.dim() == 1:
                return per_sample_per_class
            if self.reduction == "mean":
                return per_sample_per_class.mean()
            elif self.reduction == "sum":
                return per_sample_per_class.sum()
            return per_sample_per_class

        # Standard path (BCE / ASL / Focal)
        per_sample_per_class = self.loss_fn(logits, targets)  # (B, L) if reduction='none'
        if per_sample_per_class.dim() == 1:
            # reduction already happened inside loss_fn
            return per_sample_per_class

        w = self.class_weights.to(logits.device)
        if w.shape[0] != per_sample_per_class.shape[-1]:
            # fallback if weights weren't initialised properly
            w = torch.ones_like(per_sample_per_class)
        weighted = per_sample_per_class * w.unsqueeze(0)

        if self.reduction == "mean":
            return weighted.mean()
        elif self.reduction == "sum":
            return weighted.sum()
        return weighted


def build_loss_from_config(
    cfg: dict,
    num_labels: int,
    train_label_matrix: np.ndarray | None = None,
) -> WeightedMultiLabelLoss:
    """Build the loss described by the `loss:` block of a config file."""
    name = cfg.get("name", "asl").lower()

    if name == "bce":
        base = nn.BCEWithLogitsLoss(reduction="none")
    elif name == "asl":
        base = AsymmetricLoss(
            gamma_pos=cfg.get("gamma_pos", 0.0),
            gamma_neg=cfg.get("gamma_neg", 4.0),
            clip=cfg.get("clip", 0.05),
            reduction="none",
        )
    elif name == "focal":
        base = FocalLoss(gamma=cfg.get("gamma", 2.0), reduction="none")
    elif name in ("dbl", "distribution_balanced"):
        base = DistributionBalancedLoss(
            focal_gamma=cfg.get("focal_gamma", 2.0),
            beta=cfg.get("cb_beta", 0.999),
            neg_tolerance=cfg.get("neg_tolerance", 0.5),
            reduction="none",
        )
    else:
        raise ValueError(f"Unknown loss: {name}")

    if cfg.get("class_balanced", False):
        if train_label_matrix is None:
            raise ValueError("class_balanced=True requires train_label_matrix")
        counts = train_label_matrix.sum(axis=0)
        weights = class_balanced_weights(counts, beta=cfg.get("cb_beta", 0.999))
    else:
        weights = torch.ones(num_labels)

    return WeightedMultiLabelLoss(base, class_weights=weights, reduction="mean")
