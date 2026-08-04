"""Phase 3: Hetero-GCN + Label Correlation Matrix, trained standalone.

Two deliberate design choices:

  1. NO fusion with Phase 2 *during training*. The graph model has to learn
     useful representations on its own; giving it the Phase-2 logits as a
     shortcut let it converge to "copy Phase 2" and learn nothing.
     Fusion happens only at test time, as an ensemble whose weight is chosen
     on the validation set (see scripts/06_train_phase3.py).

  2. An explicit, learnable label correlation matrix (num_labels x num_labels)
     initialised from training-set label co-occurrence. S-GCN only captures
     label dependencies implicitly, via graph propagation.

Flow:
    node features -> Hetero-GCN (2 layers) -> raw per-node logits
                                                  |
                                          doc nodes only
                                                  |
                                    Label Correlation Matrix
                                                  |
                            refined = raw + raw @ LCM
                                                  |
                             (test time) ensemble with Phase 2
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .hetero_gcn import HeteroGCN

# Checkpoints written before the GAT->GCN rename store the backbone under
# `hetero_gat.*`. GraphClassifier.load_state_dict remaps them transparently so
# old checkpoints keep loading.
_LEGACY_PREFIX = "hetero_gat."
_CURRENT_PREFIX = "hetero_gcn."


class LabelCorrelationMatrix(nn.Module):
    """Learnable label correlation matrix.

    Initialised from training label co-occurrence:
        C[i,j] = # docs carrying both label i and j
        L[i,j] = C[i,j] / sqrt(C[i,i] * C[j,j])     (cosine-like, diagonal zeroed)

    then scaled by `scale_init` and made learnable, so it starts as a gentle
    nudge rather than dominating the raw logits.

    Forward is residual: refined = logits + logits @ L, i.e. each label's logit
    keeps its own value plus a contribution from correlated labels.
    """

    def __init__(
        self,
        num_labels: int,
        train_labels: np.ndarray,
        scale_init: float = 0.3,
    ):
        super().__init__()
        self.num_labels = num_labels

        # train_labels: (N, L) binary
        cooccur = train_labels.T @ train_labels          # (L, L)
        diag = np.diag(cooccur).copy()
        diag[diag == 0] = 1                              # avoid div-by-zero
        corr = cooccur / np.sqrt(np.outer(diag, diag))   # in [0, 1]
        np.fill_diagonal(corr, 0)                        # no self-influence

        self.label_corr = nn.Parameter(torch.from_numpy(corr).float() * scale_init)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """(N, num_labels) -> (N, num_labels), residual + correlation influence."""
        return logits + torch.matmul(logits, self.label_corr)

    def get_matrix(self) -> torch.Tensor:
        """Current matrix on CPU (for analysis / figures)."""
        return self.label_corr.detach().cpu()


class GraphClassifier(nn.Module):
    """Phase 3 model: Hetero-GCN backbone + Label Correlation Matrix head."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_labels: int,
        train_labels: np.ndarray,
        num_relations: int = 3,
        dropout: float = 0.3,
        residual: bool = True,
        lcm_scale: float = 0.3,
    ):
        super().__init__()
        self.num_labels = num_labels

        self.hetero_gcn = HeteroGCN(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=num_labels,
            num_relations=num_relations,
            dropout=dropout,
            residual=residual,
        )
        print("  [Model] Hetero-GCN (per-relation conv, learned relation weights)")

        self.label_corr = LabelCorrelationMatrix(
            num_labels=num_labels,
            train_labels=train_labels,
            scale_init=lcm_scale,
        )

    def forward(
        self,
        H: torch.Tensor,
        adj_list: list,
        doc_node_indices: torch.Tensor = None,
    ) -> dict:
        """
        Args:
            H:                (N_total, in_dim) features for every node
            adj_list:         one sparse adjacency per relation
            doc_node_indices: (N_docs,) indices of the document nodes. If None,
                              `H` is assumed to already contain only doc nodes.

        Returns:
            {"logits": refined, "raw_logits": before the LCM}
        """
        raw_logits = self.hetero_gcn(H, adj_list)  # (N_total, num_labels)

        if doc_node_indices is not None:
            doc_raw_logits = raw_logits[doc_node_indices]
        else:
            doc_raw_logits = raw_logits

        return {
            "logits": self.label_corr(doc_raw_logits),
            "raw_logits": doc_raw_logits,
        }

    def load_state_dict(self, state_dict, strict: bool = True, **kwargs):
        """Load weights, accepting pre-rename checkpoints (`hetero_gat.*`)."""
        if any(k.startswith(_LEGACY_PREFIX) for k in state_dict):
            print(f"  [Model] Remapping legacy '{_LEGACY_PREFIX}*' keys "
                  f"to '{_CURRENT_PREFIX}*'")
            state_dict = {
                (_CURRENT_PREFIX + k[len(_LEGACY_PREFIX):]
                 if k.startswith(_LEGACY_PREFIX) else k): v
                for k, v in state_dict.items()
            }
        return super().load_state_dict(state_dict, strict=strict, **kwargs)

    def num_parameters(self, only_trainable: bool = True) -> int:
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
