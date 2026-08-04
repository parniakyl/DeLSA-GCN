"""Phase 1: transformer encoder + mean-pool + linear head.

Encoder-agnostic: the same class serves all three datasets, only
`encoder_name` changes.

    AAPD   (English)  microsoft/deberta-v3-base
    RMSC-V2 (Chinese) hfl/chinese-roberta-wwm-ext
    PHATE  (Persian)  HooshvareLab/roberta-fa-zwnj-base

Architecture:
    encoder (optionally with frozen bottom layers)
        -> mean-pool over non-pad tokens
        -> dropout
        -> Linear(hidden_size, num_labels)

Supports gradient checkpointing, bottom-layer freezing, and fp16 (via the
caller's GradScaler/autocast).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


def freeze_bottom_layers(encoder: nn.Module, n_layers: int, tag: str = "") -> None:
    """Freeze the embedding table plus the first `n_layers` encoder blocks.

    Layer index is parsed from parameter names (`...layer.<i>....`), which
    covers DeBERTa, BERT and RoBERTa naming alike.
    """
    frozen = 0
    for name, param in encoder.named_parameters():
        if "embeddings" in name.lower():
            param.requires_grad = False
            frozen += 1
    for name, param in encoder.named_parameters():
        parts = name.split(".")
        layer_idx = None
        for i, p in enumerate(parts):
            if p == "layer" and i + 1 < len(parts):
                try:
                    layer_idx = int(parts[i + 1])
                except ValueError:
                    pass
                break
        if layer_idx is not None and layer_idx < n_layers:
            param.requires_grad = False
            frozen += 1

    total = sum(1 for _ in encoder.parameters())
    trainable = sum(1 for p in encoder.parameters() if p.requires_grad)
    print(f"  [{tag or 'Layer freeze'}] Froze {frozen}/{total} encoder params "
          f"({trainable} trainable remain)")


def enable_gradient_checkpointing(encoder: nn.Module) -> None:
    """Turn on gradient checkpointing (~40% VRAM saved, ~30% slower)."""
    if hasattr(encoder, "gradient_checkpointing_enable"):
        try:
            encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            # Older transformers versions don't accept the kwargs dict.
            encoder.gradient_checkpointing_enable()
    else:
        encoder.config.gradient_checkpointing = True


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean over non-pad tokens. (B, L, H) x (B, L) -> (B, H)."""
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden)   # (B, L, 1)
    summed = (last_hidden * mask).sum(dim=1)                   # (B, H)
    denom = mask.sum(dim=1).clamp(min=1.0)                     # (B, 1)
    return summed / denom


class EncoderClassifier(nn.Module):
    """Phase 1 baseline: encoder + mean-pool + linear head."""

    def __init__(
        self,
        encoder_name: str = "microsoft/deberta-v3-base",
        num_labels: int = 54,
        dropout: float = 0.1,
        gradient_checkpointing: bool = False,
        freeze_layers: int = 0,
    ):
        """
        Args:
            encoder_name:           HuggingFace model id
            num_labels:             number of output labels
            dropout:                dropout before the classifier
            gradient_checkpointing: trade compute for VRAM
            freeze_layers:          freeze embeddings + the bottom N encoder
                                    blocks (0 = train everything). Valid range
                                    is 0..num_layers-1 (12 for a -base model).
        """
        super().__init__()
        self.num_labels = num_labels
        self.freeze_layers = freeze_layers

        self.config = AutoConfig.from_pretrained(encoder_name)
        self.encoder = AutoModel.from_pretrained(encoder_name)

        if gradient_checkpointing:
            enable_gradient_checkpointing(self.encoder)
        if freeze_layers > 0:
            freeze_bottom_layers(self.encoder, freeze_layers, tag="EncoderClassifier")

        hidden_size = self.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        return_dict: bool = True,
    ):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        pooled = mean_pool(outputs.last_hidden_state, attention_mask)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        if return_dict:
            return {"logits": logits, "pooled": pooled}
        return logits, pooled

    def num_parameters(self, only_trainable: bool = True) -> int:
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
