"""Phase 2: encoder + LSA (Label-Specific Attention) + gated fusion.

This is the first real novelty over S-GCN. For each label we compute attention
between that label's description embedding and the document's token-level
encoder outputs, producing a label-aware document representation that is then
scored by a shared per-label head.

Key additions over Phase 1:
1. Label description embeddings (encoded label text, see scripts/02_encode_labels.py)
2. Label-specific attention over document tokens
3. Gated fusion of (mean-pool logits) + (LSA logits)

No data leakage: label information enters ONLY through
  - the encoder's reading of the label *description text* ("Artificial Intelligence")
  - trainable attention parameters
The ground-truth label vector is never an input.

Architecture:
    text -> encoder -> token embeddings (B, L, H)
                          |
          +---------------+---------------+
          v                               v
    Mean-pool head                    LSA head
    (Phase 1 path)          label embeddings (num_labels, H)
    Linear(H, num_labels)   x token embeddings^T
    -> logits_mean          -> scores (B, num_labels, L)
                            -> softmax over tokens -> weighted sum
                            -> label-aware doc reps (B, num_labels, H)
                            -> shared Linear(H, 1)
                            -> logits_lsa (B, num_labels)
          |                               |
          +--------> gated fusion <-------+
                a*mean + b*lsa   (a, b trainable scalars)
                        |
                  final logits (B, num_labels)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

from .encoder_classifier import (
    enable_gradient_checkpointing,
    freeze_bottom_layers,
    mean_pool,
)


class LabelSpecificAttention(nn.Module):
    """Label-Specific Attention.

    For each label l with embedding e_l in R^H and a document with token
    embeddings T in R^{LxH}:

        score_{l,t} = (e_l . T_t) / temperature
        alpha_{l,t} = softmax_t(score_{l,t})
        d_l         = sum_t alpha_{l,t} * T_t        (label-aware doc rep)
        logit_l     = w^T d_l + b                    (shared head)

    Conceptually close to LSAN (Wang et al. 2019) and LSAN-BERT.
    """

    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels

        # One shared Linear(H, 1) applied to every label-aware doc rep.
        self.label_classifier = nn.Linear(hidden_size, 1, bias=True)
        self.dropout = nn.Dropout(dropout)

        # Learnable attention temperature, initialised to sqrt(H).
        self.temperature = nn.Parameter(torch.tensor(float(hidden_size) ** 0.5))

        nn.init.normal_(self.label_classifier.weight, std=0.02)
        nn.init.zeros_(self.label_classifier.bias)

    def forward(
        self,
        token_embeddings: torch.Tensor,   # (B, L, H)
        attention_mask: torch.Tensor,     # (B, L)
        label_embeddings: torch.Tensor,   # (num_labels, H)
    ) -> torch.Tensor:
        """Returns logits of shape (B, num_labels)."""
        H = token_embeddings.shape[-1]

        if label_embeddings.shape[-1] != H:
            raise ValueError(
                f"Label embedding dim {label_embeddings.shape[-1]} != hidden_size {H}. "
                f"Project label embeddings to hidden_size before calling LSA."
            )

        # (B, L, H) x (N, H) -> (B, N, L)
        scores = torch.einsum("blh, nh -> bnl", token_embeddings, label_embeddings)
        scores = scores / self.temperature.clamp(min=1.0)

        # Mask padded positions before the softmax.
        scores = scores.masked_fill(attention_mask.unsqueeze(1) == 0, -1e4)
        attn_weights = F.softmax(scores, dim=-1)  # (B, N, L)

        # (B, N, L) x (B, L, H) -> (B, N, H)
        label_aware_reps = torch.einsum("bnl, blh -> bnh", attn_weights, token_embeddings)
        label_aware_reps = self.dropout(label_aware_reps)

        return self.label_classifier(label_aware_reps).squeeze(-1)  # (B, N)

    def num_parameters(self, only_trainable: bool = True) -> int:
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


class LSAClassifier(nn.Module):
    """Phase 2 model: encoder + LSA + gated fusion.

    Path A (mean-pool): identical to Phase 1.
    Path B (LSA):       label-specific attention over document tokens.
    Fusion:             logits = gate_mean * logits_mean + gate_lsa * logits_lsa,
                        with both gates trainable scalars initialised to 1.0.

    Encoder-agnostic — the same class is used for the English, Chinese and
    Persian encoders.
    """

    def __init__(
        self,
        encoder_name: str = "microsoft/deberta-v3-base",
        num_labels: int = 54,
        dropout: float = 0.1,
        gradient_checkpointing: bool = False,
        freeze_layers: int = 0,
        label_emb_init: torch.Tensor | None = None,
        freeze_label_emb: bool = False,
    ):
        """
        Args:
            label_emb_init:   optional (num_labels, hidden_size) tensor used to
                              initialise the label embeddings; random if None.
            freeze_label_emb: keep label embeddings fixed. Reasonable when
                              label_emb_init comes from a strong encoder.
        """
        super().__init__()
        self.num_labels = num_labels

        self.config = AutoConfig.from_pretrained(encoder_name)
        self.encoder = AutoModel.from_pretrained(encoder_name)

        if gradient_checkpointing:
            enable_gradient_checkpointing(self.encoder)

        hidden_size = self.config.hidden_size
        self.hidden_size = hidden_size

        if freeze_layers > 0:
            freeze_bottom_layers(self.encoder, freeze_layers, tag="LSAClassifier")

        # Path A: mean-pool head
        self.dropout = nn.Dropout(dropout)
        self.mean_classifier = nn.Linear(hidden_size, num_labels)
        nn.init.normal_(self.mean_classifier.weight, std=0.02)
        nn.init.zeros_(self.mean_classifier.bias)

        # Path B input: label embeddings (num_labels, hidden_size)
        if label_emb_init is not None:
            if label_emb_init.shape != (num_labels, hidden_size):
                raise ValueError(
                    f"label_emb_init shape {label_emb_init.shape} != "
                    f"({num_labels}, {hidden_size})"
                )
            self.label_embeddings = nn.Parameter(label_emb_init.clone().float())
        else:
            self.label_embeddings = nn.Parameter(
                torch.randn(num_labels, hidden_size) * 0.02
            )
        if freeze_label_emb:
            self.label_embeddings.requires_grad = False

        self.lsa = LabelSpecificAttention(
            hidden_size=hidden_size,
            num_labels=num_labels,
            dropout=dropout,
        )

        # Gated fusion scalars — unconstrained, initialised to 1.0 each.
        self.gate_mean = nn.Parameter(torch.tensor(1.0))
        self.gate_lsa = nn.Parameter(torch.tensor(1.0))

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
        last_hidden = outputs.last_hidden_state  # (B, L, H)

        # Path A: mean-pool
        pooled = self.dropout(mean_pool(last_hidden, attention_mask))
        logits_mean = self.mean_classifier(pooled)

        # Path B: LSA
        logits_lsa = self.lsa(
            token_embeddings=last_hidden,
            attention_mask=attention_mask,
            label_embeddings=self.label_embeddings,
        )

        logits = self.gate_mean * logits_mean + self.gate_lsa * logits_lsa

        if return_dict:
            return {
                "logits": logits,
                "logits_mean": logits_mean,
                "logits_lsa": logits_lsa,
                "pooled": pooled,
            }
        return logits, pooled, logits_mean, logits_lsa

    def num_parameters(self, only_trainable: bool = True) -> int:
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


@torch.no_grad()
def encode_label_descriptions(
    encoder_name: str,
    label_descriptions: list[str],
    device: torch.device | None = None,
    max_seq_len: int = 32,
) -> torch.Tensor:
    """Mean-pooled encoding of each label description.

    Loads a fresh copy of the encoder, encodes, then frees it — this runs once
    offline, so the extra load is irrelevant.

    Returns: (num_labels, hidden_size) CPU tensor.
    """
    from transformers import AutoTokenizer

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    enc = tokenizer(
        label_descriptions,
        padding="max_length",
        truncation=True,
        max_length=max_seq_len,
        return_tensors="pt",
    )

    encoder = AutoModel.from_pretrained(encoder_name).to(device)
    encoder.eval()

    outputs = encoder(
        input_ids=enc["input_ids"].to(device),
        attention_mask=enc["attention_mask"].to(device),
        return_dict=True,
    )
    label_embs = mean_pool(outputs.last_hidden_state, enc["attention_mask"].to(device))

    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return label_embs.cpu()
