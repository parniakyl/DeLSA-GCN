"""Heterogeneous Graph Convolutional Network (Hetero-GCN).

Per layer, per relation r:
    h_r = A_r @ (H @ W_r)              relation-specific graph convolution
    h   = sum_r(alpha_r * h_r)         learned relation combination
    h   = ReLU(BatchNorm(h + H))       residual + normalisation
    h   = Dropout(h)

Two layers; the second projects to `num_labels`, giving per-node logits.

This is a **GCN**, not a GAT: aggregation is the fixed, pre-normalised
adjacency A_r (IDF / PMI edge weights), with no per-edge attention
coefficients computed from node features. An attention-based variant was tried
and dropped (it went OOM on RMSC-V2 at this graph size). The learned `alpha_r`
weights are a global, relation-level gate, not edge attention — do not describe
them as attention heads.

What makes it different from S-GCN's plain GCN:
  1. Relation-specific transformations W_r — one projection per edge type
  2. Learned relation weights alpha_r — the model decides which relation matters
  3. Residual connections
  4. Semantic node initialisation from a transformer encoder, rather than
     TextGCN-style random/one-hot node features
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationGraphConv(nn.Module):
    """Graph convolution for a single relation type: H' = A_norm @ (H @ W) + b.

    `A_norm` is pre-normalised at graph-build time (see src/graph_builder.py),
    so there is nothing to normalise here.
    """

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, H: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H:   (N, in_dim) dense node features
            adj: (N, N) sparse (or dense) normalised adjacency
        Returns:
            (N, out_dim)
        """
        Z = self.linear(H)
        if adj.is_sparse:
            return torch.sparse.mm(adj, Z)
        return adj @ Z


class HeteroGCN(nn.Module):
    """Two-layer heterogeneous GCN over a fixed set of relations.

      Layer 1: in_dim     -> hidden_dim  (residual + BN + ReLU + dropout)
      Layer 2: hidden_dim -> out_dim     (BN)
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_relations: int = 3,
        dropout: float = 0.5,
        residual: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_relations = num_relations
        self.dropout = dropout
        self.residual = residual

        self.conv1 = nn.ModuleList([
            RelationGraphConv(in_dim, hidden_dim) for _ in range(num_relations)
        ])
        self.rel_weights1 = nn.Parameter(torch.ones(num_relations))

        self.conv2 = nn.ModuleList([
            RelationGraphConv(hidden_dim, out_dim) for _ in range(num_relations)
        ])
        self.rel_weights2 = nn.Parameter(torch.ones(num_relations))

        # Residual projection only needed when the dims differ.
        if residual and in_dim != hidden_dim:
            self.residual_proj = nn.Linear(in_dim, hidden_dim, bias=False)
        else:
            self.residual_proj = None

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, H: torch.Tensor, adj_list: list) -> torch.Tensor:
        """
        Args:
            H:        (N, in_dim) node features
            adj_list: one sparse adjacency per relation, in a fixed order
        Returns:
            (N, out_dim) node logits
        """
        # ---------------- Layer 1 ----------------
        h_rel = [self.conv1[r](H, adj_list[r]) for r in range(self.num_relations)]

        # Softmax over relation weights -> a normalised relation gate.
        alpha1 = F.softmax(self.rel_weights1, dim=0)
        h = sum(a * hr for a, hr in zip(alpha1, h_rel))

        if self.residual:
            h = h + (self.residual_proj(H) if self.residual_proj is not None else H)

        h = self.bn1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # ---------------- Layer 2 ----------------
        h_rel2 = [self.conv2[r](h, adj_list[r]) for r in range(self.num_relations)]
        alpha2 = F.softmax(self.rel_weights2, dim=0)
        out = sum(a * hr for a, hr in zip(alpha2, h_rel2))

        return self.bn2(out)

    def get_relation_weights(self) -> dict:
        """Current relation gates per layer (for analysis / ablation tables)."""
        return {
            "layer1": F.softmax(self.rel_weights1, dim=0).detach().cpu().numpy(),
            "layer2": F.softmax(self.rel_weights2, dim=0).detach().cpu().numpy(),
        }
