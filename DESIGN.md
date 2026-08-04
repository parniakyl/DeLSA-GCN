# DeLSA-GCN — Design Document

> **DeBERTa-fused Label-Semantic Attention Graph Convolutional Network
> for Multi-Label Text Classification**

Architecture, training procedure, evaluation protocol and ablation plan.
The primary baseline is **S-GCN (KBS 2024)**; the primary dataset is **AAPD**,
with **RMSC-V2** (Chinese) and **PHATE** (Persian) as cross-lingual evidence
that the method is not English- or domain-specific.

For run commands see `README.md`.

---

## 1. Baselines to beat

### AAPD (54 labels, English)

| Metric   | S-GCN | Minimum target | Stretch target |
|----------|-------|----------------|----------------|
| P@1      | 84.40 | 86.0           | 88.0           |
| P@3      | 62.00 | 63.5           | 66.0           |
| P@5      | 41.78 | 43.0           | 45.0           |
| nDCG@3   | 80.66 | 82.0           | 84.0           |
| nDCG@5   | 84.47 | 86.0           | 88.0           |
| Micro-F1 | 73.28 | 75.5           | 78.0           |
| Macro-F1 | (n/r) | 40.0           | 45.0           |

### RMSC-V2 (22 labels, Chinese)

| Metric   | TextGCN | InductGCN | S-GCN |
|----------|---------|-----------|-------|
| P@1      | 70.25   | 81.47     | 88.91 |
| P@3      | 51.29   | 58.25     | 62.21 |
| P@5      | 36.14   | 40.19     | 41.80 |
| nDCG@3   | 69.08   | 79.14     | 85.34 |
| nDCG@5   | 75.43   | 84.90     | 90.20 |
| Micro-F1 | 61.23   | 69.80     | 74.99 |

### PHATE (4 labels, Persian)

> ⚠ **The PHATE baseline numbers in `src/reference_scores.py` are placeholders**
> — they were copied from the AAPD micro-F1 column and are not real Persian
> hate-speech results. Replace them with properly cited values (or delete the
> entries) before any figure or table built from them goes into the paper.

All published numbers live in one place: `src/reference_scores.py`. Training
scripts and figures read from there, so a correction propagates everywhere.

---

## 2. Critical analysis of S-GCN (why we can beat it)

1. **Weak encoder.** S-GCN uses a 300-d BiLSTM on AAPD. BERT-LiGCN with plain
   BERT already reaches micro-F1 75.89; DeBERTa-v3 should push higher.
2. **No semantic word initialisation.** S-GCN's word nodes are randomly
   initialised trainable parameters. We initialise them from averaged
   contextual encoder embeddings.
3. **MSE + λ loss is a poor fit for multi-label.** We use BCE with
   class-balanced reweighting (Asymmetric Loss and Distribution-Balanced Loss
   are also implemented, for ablations).
4. **Plain GCN with a single edge type.** We use a heterogeneous GCN with
   relation-specific projections and a learned gate over relations.
5. **Label semantics unused**, despite arXiv/genre/hate-speech labels carrying
   real meaning. We add a Label-Specific Attention module driven by encoded
   label descriptions.
6. **Label dependencies only implicit.** We add an explicit, learnable label
   correlation matrix.

---

## 3. Architecture

### 3.1 Phase 1 — encoder baseline

`src/models/encoder_classifier.py` · `EncoderClassifier`

```
text → transformer encoder → token embeddings (B, L, H)
     → mean-pool over non-pad tokens → dropout → Linear(H, num_labels)
```

Encoder-agnostic. The three datasets differ only by `model.encoder_name`:
DeBERTa-v3-base (English), Chinese RoBERTa-wwm-ext, Persian RoBERTa-fa-zwnj.
Supports gradient checkpointing, bottom-layer freezing and fp16.

### 3.2 Phase 2 — Label-Specific Attention + gated fusion

`src/models/lsa_classifier.py` · `LabelSpecificAttention`, `LSAClassifier`

For each label ℓ with embedding `e_ℓ ∈ R^H` and document token embeddings
`T ∈ R^{L×H}`:

```
score_{ℓ,t} = (e_ℓ · T_t) / temperature       temperature learnable, init √H
α_{ℓ,t}     = softmax_t(score_{ℓ,t})          padded positions masked to −1e4
d_ℓ         = Σ_t α_{ℓ,t} · T_t               label-aware document representation
logit_ℓ     = w^T d_ℓ + b                     one shared head across labels
```

Fused with the Phase-1 path by two trainable scalars:

```
logits = gate_mean · logits_mean + gate_lsa · logits_lsa
```

Label embeddings are initialised from encoded label *descriptions*
(`scripts/02_encode_labels.py`, text from `src/labels.py`) and fine-tuned unless
`model.freeze_label_emb: true`. Conceptually related to LSAN (Wang et al. 2019).

Phase 2 also keeps an EMA of the weights (`train.ema_decay`) and evaluates three
candidates on test — best-on-val, last epoch, EMA — printing all three so
run-to-run variance is visible. **The reported number is always best-on-val**;
picking by test score would be leakage.

### 3.3 Phase 3 — heterogeneous graph

**Graph construction** (`src/graph_builder.py`, driven by `scripts/04_build_graph.py`)

| Node type | Count | Initial feature                                     |
|-----------|-------|-----------------------------------------------------|
| document  | N     | Phase-2 pooled embedding                             |
| word      | V     | mean of the embeddings of documents containing it    |
| label     | L     | encoded label description                            |

| Edge type   | Weight                        | Computed from      |
|-------------|-------------------------------|--------------------|
| doc–word    | IDF                           | all documents      |
| word–word   | positive PMI, sliding window  | **train split only** |
| word–label  | positive PMI                  | **train split only** |

There is **no doc–label edge**. Node ordering is fixed everywhere as
`[train docs | val docs | test docs | words | labels]`. Adjacencies are
symmetrically normalised (`D^{-1/2} A D^{-1/2}`, bipartite variant for
doc–word and word–label).

Word segmentation follows `model.language` (`src/tokenization.py`): whitespace
for English, the encoder's WordPiece tokenizer for Chinese and Persian.
Whitespace splitting is meaningless for Chinese (no spaces) and lossy for
Persian (ZWNJ-joined compounds).

**Hetero-GCN** (`src/models/hetero_gcn.py` · `HeteroGCN`)

Per layer, per relation r:

```
h_r = A_r (H W_r)                 relation-specific convolution
h   = Σ_r softmax(α)_r · h_r      learned gate over relations
h   = ReLU(BatchNorm(h + H))      residual
h   = Dropout(h)
```

Two layers; the second projects to `num_labels`.

> **Naming note.** This is a **GCN**, not a GAT. Aggregation uses the fixed,
> pre-normalised adjacency (IDF / PMI weights); there are no per-edge attention
> coefficients computed from node features. `α_r` is a global, relation-level
> gate — do not describe it as attention heads. An attention-based variant was
> prototyped and dropped (OOM on RMSC-V2 at this graph size).
>
> Checkpoints written before this rename store the backbone under
> `hetero_gat.*`; `GraphClassifier.load_state_dict` remaps those keys
> transparently, so old checkpoints still load.

**Label Correlation Matrix** (`src/models/graph_classifier.py`)

```
C      = Yᵀ Y                                  train-split co-occurrence
L[i,j] = C[i,j] / √(C[i,i]·C[j,j]),  diag = 0  cosine-like
init   = L · lcm_scale                         learnable thereafter
refined_logits = logits + logits · L           residual
```

**Standalone training + test-time ensemble.** The graph model never sees the
Phase-2 logits during training. At test time:

```
scores = w · σ(graph_logits) + (1 − w) · σ(phase2_logits)
```

with `w` grid-searched on validation (step 0.05). If
`phase3.phase1.{config,checkpoint}` are set, a graph+Phase-1 ensemble is also
reported — but that comparison is decided on test (no Phase-1 validation scores
are cached), so it is exploratory only and is labelled as such in the output.

### 3.4 Loss

`src/losses.py`. All configs currently use `bce` with `class_balanced: true`
(Cui et al. 2019 effective-number reweighting, `cb_beta = 0.999`). Also
available for ablation: Asymmetric Loss (Ben-Baruch et al. 2021), Focal Loss,
and Distribution-Balanced Loss (Wu et al. 2020).

The forward pass runs under fp16 autocast, but **the loss is always computed in
fp32** — ASL and DBL involve `log()` and `pow()`, which underflow to NaN in
fp16. A non-finite loss aborts the epoch rather than corrupting the weights.

---

## 4. Evaluation protocol

Metrics (`src/metrics.py`): exact-match accuracy, micro-F1, macro-F1,
P@{1,3,5}, nDCG@{1,3,5}. P@k and nDCG@k are rank-based and therefore
threshold-independent.

Default decision threshold 0.5. On very sparse label sets no score may ever
cross 0.5, giving micro-F1 = 0; when `eval.adaptive_threshold` is on, both a
global and a per-label threshold are fitted **on validation** and the better one
is applied to test. AAPD uses the adaptive path; RMSC-V2 and PHATE disable it
(`adaptive_threshold: false`) because their labels are dense enough.

Nothing is ever tuned on test: threshold, ensemble weight, and checkpoint
selection all use validation.

---

## 5. Ablation plan

Re-run Phase 3 with one component removed at a time:

| # | Ablation | How |
|---|----------|-----|
| A1 | − LSA module | Phase 1 encoder in place of Phase 2 in the ensemble |
| A2 | − Hetero-GCN | report Phase 2 alone |
| A3 | − semantic word init | random word node features in `05_compute_node_features.py` |
| A4 | − label description init | `--override` with `model.label_emb_path: null` |
| A5 | − class-balanced weighting | `loss.class_balanced: false` |
| A6 | − Label Correlation Matrix | `phase3.lcm_scale: 0.0` |
| A7 | ASL instead of BCE | `loss.name: asl` |
| A8 | − relation-specific gate | fix `rel_weights` uniform in `HeteroGCN` |

Phases 1 and 2 accept `--override <file.yaml>`, which is deep-merged over the
base config — use it for ablations rather than editing the main configs.

Report the final model over 5 seeds (`scripts/07_multiseed_run.py`), which also
runs a one-sample t-test against the baseline and emits a LaTeX table.

---

## 6. Hardware budget

| Resource       | Available | Projected peak                           |
|----------------|-----------|------------------------------------------|
| GPU memory     | 8 GB      | ~4.5 GB (Phase 1/2) / ~7 GB (Phase 3)    |
| System RAM     | 32 GB     | ~6 GB (graph + feature cache)            |
| Disk           | —         | ~2 GB per dataset (checkpoints + cache)  |
| Time per epoch | —         | ~6 min Phase 1, ~7 min Phase 2 (AAPD)    |
| Phase 3        | —         | full-batch, seconds per epoch after setup |

Phase 3 is full-batch over the whole graph, so `train.batch_size: 1` and
`epochs` counts full graph passes (2000, with early stopping).

---

## 7. Reproducibility

- `train.seed` seeds Python, NumPy, PyTorch and CUDA, and pins cuDNN to
  deterministic mode.
- Every run writes its resolved config next to the checkpoint as `config.yaml`.
- Per-epoch validation metrics and final test metrics are saved to
  `results.json`.
- The checkpoint saved is the one with the best validation micro-F1.

---

## 8. Paper outline

1. Introduction
2. Related work — MLTC, GNN-based text classification, label-aware methods
3. Method (DeLSA-GCN)
   1. Preliminaries
   2. Transformer encoder
   3. Label-Specific Attention module
   4. Hetero-GCN with semantic node initialisation
   5. Label correlation matrix
   6. Gated fusion and test-time ensemble
   7. Loss function
4. Experiments
   1. Datasets — AAPD, RMSC-V2, PHATE
   2. Baselines — LSTM, TextGCN, SGC, S-GCN, BERT-LiGCN, InductGCN
   3. Main results
   4. Ablation
   5. Analysis — label-attention visualisation, long-tail behaviour,
      cross-lingual transfer
5. Conclusion
