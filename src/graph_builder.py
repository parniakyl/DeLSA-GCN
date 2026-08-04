"""Heterogeneous graph builder for multi-label text classification.

Builds a graph with 3 node types and 3 edge types:
  Nodes: document, word, label
  Edges:
    - doc-word   (weight = IDF)        — document contains word
    - word-word  (weight = PMI > 0)    — word co-occurrence in a sliding window
    - word-label (weight = PMI > 0)    — label treated as a word for the PMI calc

There are NO doc-label edges. That is what keeps the transductive setting
leakage-free (same choice as S-GCN): a test document is connected only to the
words it contains.

The graph spans ALL documents (train + val + test), which is standard for
graph-based text classification (TextGCN, S-GCN). Test *labels* are never used:
PMI/IDF statistics come from the training split only, and only test document
*topology* (which words appear) is visible.

Adapted from S-GCN's build_graph.py, refactored to emit one adjacency matrix
per relation.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import scipy.sparse as sp
import torch


def build_hetero_graph(
    tokenized_docs: List[List[str]],
    labels: np.ndarray,
    word_id_map: dict,
    word_list: list,
    train_size: int,
    test_size: int,
    val_size: int = 0,
    window_size: int = 20,
    verbose: bool = True,
) -> Tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix, dict]:
    """Build the heterogeneous graph.

    Node ordering assumed by every downstream consumer:
      [0, train_size)                                        train doc nodes
      [train_size, train_size+val_size)                      val doc nodes
      [train_size+val_size, total_docs)                      test doc nodes
      [total_docs, total_docs+vocab_size)                    word nodes
      [total_docs+vocab_size, ...+num_labels)                label nodes

    Args:
        tokenized_docs: tokenized documents (train+val+test, in that order)
        labels:         (N_docs, num_labels) multi-hot matrix (only train rows used)
        word_id_map:    {word_string: word_index}
        word_list:      list of word strings (index = position in list)
        train_size, val_size, test_size: split sizes
        window_size:    sliding window size for the word-word PMI
        verbose:        print progress

    Returns:
        adj_doc_word:   (total_docs, vocab_size)  sparse — IDF weighted
        adj_word_word:  (vocab_size, vocab_size)  sparse — positive PMI
        adj_word_label: (vocab_size, num_labels)  sparse — positive PMI
        metadata:       dict of graph stats
    """
    if verbose:
        print("Building heterogeneous graph...")
    t0 = time.time()

    total_docs = train_size + val_size + test_size
    vocab_size = len(word_list)
    num_labels = labels.shape[1]

    # ===================================================================
    # 1. doc-word edges (IDF weight)
    # ===================================================================
    if verbose:
        print("  [1/3] Building doc-word edges (IDF)...")

    # Document frequency: how many docs contain each word
    word_doc_freq = np.zeros(vocab_size, dtype=np.int64)
    for doc_idx in range(total_docs):
        for word in set(tokenized_docs[doc_idx]):
            if word in word_id_map:
                word_doc_freq[word_id_map[word]] += 1

    dw_row, dw_col, dw_val = [], [], []
    for doc_idx in range(total_docs):
        seen = set()
        for word in tokenized_docs[doc_idx]:
            if word not in word_id_map:
                continue
            wid = word_id_map[word]
            if wid in seen:
                continue
            seen.add(wid)
            # IDF weight (same formula as S-GCN)
            idf = np.log(1.0 * train_size / (word_doc_freq[wid] + 1))
            dw_row.append(doc_idx)
            dw_col.append(wid)
            dw_val.append(idf)

    adj_doc_word = sp.csr_matrix(
        (dw_val, (dw_row, dw_col)),
        shape=(total_docs, vocab_size),
        dtype=np.float32,
    )

    # ===================================================================
    # 2. word-word edges (positive PMI over sliding windows)
    #    Only TRAIN documents contribute, so no test co-occurrence statistics
    #    leak into the graph weights.
    # ===================================================================
    if verbose:
        print("  [2/3] Building word-word edges (PMI)...")

    word_occurrence = defaultdict(int)       # word_id -> #windows containing it
    word_pair_occurrence = defaultdict(int)  # (id1, id2) -> #windows containing both
    total_windows = 0

    for doc_idx in range(train_size):
        doc_word_ids = [word_id_map[w] for w in tokenized_docs[doc_idx] if w in word_id_map]
        for start in range(0, len(doc_word_ids), 1):
            window = doc_word_ids[start:start + window_size]
            if len(window) == 0:
                continue
            total_windows += 1
            unique_in_window = list(set(window))
            for wid in unique_in_window:
                word_occurrence[wid] += 1
            for i in range(len(unique_in_window)):
                for j in range(i + 1, len(unique_in_window)):
                    w1, w2 = sorted([unique_in_window[i], unique_in_window[j]])
                    word_pair_occurrence[(w1, w2)] += 1

    ww_row, ww_col, ww_val = [], [], []
    for (w1, w2), count in word_pair_occurrence.items():
        if count <= 0:
            continue
        pmi = np.log((count * total_windows) /
                     (word_occurrence[w1] * word_occurrence[w2] + 1e-8))
        if pmi <= 0:
            continue  # keep positive PMI only
        ww_row.extend([w1, w2])
        ww_col.extend([w2, w1])
        ww_val.extend([pmi, pmi])

    adj_word_word = sp.csr_matrix(
        (ww_val, (ww_row, ww_col)),
        shape=(vocab_size, vocab_size),
        dtype=np.float32,
    )

    # ===================================================================
    # 3. word-label edges (positive PMI, labels treated as pseudo-words)
    # ===================================================================
    if verbose:
        print("  [3/3] Building word-label edges (PMI)...")

    # Label "word" ids live just past the real vocabulary.
    ext_word_occurrence = dict(word_occurrence)
    for label_idx in range(num_labels):
        ext_word_occurrence[vocab_size + label_idx] = int(labels[:train_size, label_idx].sum())

    word_label_pair_occurrence = defaultdict(int)  # (word_id, label_idx) -> count
    for doc_idx in range(train_size):
        doc_labels = np.where(labels[doc_idx] > 0.5)[0]
        doc_words = set(word_id_map[w] for w in tokenized_docs[doc_idx] if w in word_id_map)
        for label_idx in doc_labels:
            for wid in doc_words:
                word_label_pair_occurrence[(wid, label_idx)] += 1

    wl_row, wl_col, wl_val = [], [], []
    for (wid, label_idx), count in word_label_pair_occurrence.items():
        if count <= 0:
            continue
        pmi = np.log(
            (count * total_windows) /
            (ext_word_occurrence[wid] * ext_word_occurrence[vocab_size + label_idx] + 1e-8)
        )
        if pmi <= 0:
            continue
        wl_row.append(wid)
        wl_col.append(label_idx)
        wl_val.append(pmi)

    adj_word_label = sp.csr_matrix(
        (wl_val, (wl_row, wl_col)),
        shape=(vocab_size, num_labels),
        dtype=np.float32,
    )

    # ===================================================================
    # Normalisation
    # ===================================================================
    if verbose:
        print("  Normalizing adjacency matrices...")

    adj_doc_word = normalize_adj_bipartite(adj_doc_word)
    adj_word_word = normalize_adj_symmetric(adj_word_word + sp.eye(vocab_size))
    adj_word_label = normalize_adj_bipartite(adj_word_label)

    metadata = {
        "total_docs": total_docs,
        "vocab_size": vocab_size,
        "num_labels": num_labels,
        "total_nodes": total_docs + vocab_size + num_labels,
        "doc_word_edges": adj_doc_word.nnz,
        "word_word_edges": adj_word_word.nnz,
        "word_label_edges": adj_word_label.nnz,
        "window_size": window_size,
    }

    if verbose:
        print(f"  Graph built in {time.time() - t0:.1f}s")
        print(f"  Total nodes: {metadata['total_nodes']}")
        print(f"  doc-word edges: {metadata['doc_word_edges']}")
        print(f"  word-word edges: {metadata['word_word_edges']}")
        print(f"  word-label edges: {metadata['word_label_edges']}")

    return adj_doc_word, adj_word_word, adj_word_label, metadata


def normalize_adj_symmetric(adj: sp.csr_matrix) -> sp.csr_matrix:
    """Symmetric normalization: D^{-1/2} A D^{-1/2}."""
    rowsum = np.array(adj.sum(axis=1)).flatten()
    # NOTE: np.power(..., where=cond) does NOT zero-fill the positions where
    # cond is False — it leaves them as uninitialised memory. Zero-degree nodes
    # (rowsum == 0) were picking up garbage scaling factors, which is what made
    # logits/loss explode. Fix: pre-allocate `out` as zeros so the False
    # positions stay exactly 0.0.
    d_inv_sqrt = np.zeros_like(rowsum, dtype=np.float64)
    np.power(rowsum, -0.5, where=rowsum > 0, out=d_inv_sqrt)
    d_inv_sqrt[np.isinf(d_inv_sqrt) | np.isnan(d_inv_sqrt)] = 0.0
    d_mat = sp.diags(d_inv_sqrt)
    return d_mat.dot(adj).dot(d_mat).tocsr()


def normalize_adj_bipartite(adj: sp.csr_matrix) -> sp.csr_matrix:
    """Bipartite normalization D_row^{-1/2} A D_col^{-1/2}.

    Rows and columns index different node types with different degree scales,
    so they need separate degree matrices.
    """
    rowsum = np.array(adj.sum(axis=1)).flatten()
    colsum = np.array(adj.sum(axis=0)).flatten()

    # Same zero-fill caveat as normalize_adj_symmetric.
    d_row_inv_sqrt = np.zeros_like(rowsum, dtype=np.float64)
    np.power(rowsum, -0.5, where=rowsum > 0, out=d_row_inv_sqrt)
    d_row_inv_sqrt[np.isinf(d_row_inv_sqrt) | np.isnan(d_row_inv_sqrt)] = 0.0

    d_col_inv_sqrt = np.zeros_like(colsum, dtype=np.float64)
    np.power(colsum, -0.5, where=colsum > 0, out=d_col_inv_sqrt)
    d_col_inv_sqrt[np.isinf(d_col_inv_sqrt) | np.isnan(d_col_inv_sqrt)] = 0.0

    return sp.diags(d_row_inv_sqrt).dot(adj).dot(sp.diags(d_col_inv_sqrt)).tocsr()


def sparse_to_torch(adj: sp.csr_matrix, device: str = "cpu") -> torch.Tensor:
    """Convert a scipy sparse matrix to a coalesced torch sparse COO tensor."""
    adj = adj.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((adj.row, adj.col)).astype(np.int64))
    values = torch.from_numpy(adj.data)
    shape = torch.Size(adj.shape)
    return torch.sparse_coo_tensor(indices, values, shape, device=device).coalesce()


def load_relation_adjacencies(graph_dir, device) -> tuple[list, tuple[int, int, int, int]]:
    """Load the 3 saved blocks and assemble one full-graph adjacency per relation.

    Every node lives in a single [docs | words | labels] index space, so each
    relation becomes an (N_total, N_total) sparse matrix with its block(s)
    placed at the right offset (and transposed for the reverse direction).

    Returns (adj_list, (n_docs, n_words, n_labels, n_total)).
    """
    from pathlib import Path

    graph_dir = Path(graph_dir)
    adj_dw = sp.load_npz(graph_dir / "adj_doc_word.npz")
    adj_ww = sp.load_npz(graph_dir / "adj_word_word.npz")
    adj_wl = sp.load_npz(graph_dir / "adj_word_label.npz")

    n_docs, n_words = adj_dw.shape
    n_labels = adj_wl.shape[1]
    n_total = n_docs + n_words + n_labels

    def build_full_adj(blocks):
        rows, cols, vals = [], [], []
        for r_start, c_start, mat in blocks:
            if mat is None or mat.nnz == 0:
                continue
            coo = mat.tocoo()
            rows.extend((coo.row + r_start).tolist())
            cols.extend((coo.col + c_start).tolist())
            vals.extend(coo.data.tolist())
        return sp.csr_matrix((vals, (rows, cols)), shape=(n_total, n_total), dtype=np.float32)

    relations = [
        ("doc-word",   [(0, n_docs, adj_dw), (n_docs, 0, adj_dw.T)]),
        ("word-word",  [(n_docs, n_docs, adj_ww)]),
        ("word-label", [(n_docs, n_docs + n_words, adj_wl),
                        (n_docs + n_words, n_docs, adj_wl.T)]),
    ]

    print(f"  Graph dims: docs={n_docs}, words={n_words}, labels={n_labels}, total={n_total}")
    adj_list = []
    for i, (name, blocks) in enumerate(relations, start=1):
        full = build_full_adj(blocks)
        print(f"  Relation {i} ({name}): nnz={full.nnz}")
        adj_list.append(sparse_to_torch(full, device=device))

    return adj_list, (n_docs, n_words, n_labels, n_total)
