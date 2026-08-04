# DeLSA-GCN
Label-Specific Attention and Relation-Aware Heterogeneous Graph for Multilingual Multi-label Text Classification

Three datasets, one codebase:

| Dataset     | Language | Labels | Encoder                              |
|-------------|----------|--------|--------------------------------------|
| **AAPD**    | English  | 54     | `microsoft/deberta-v3-base`          |
| **RMSC-V2** | Chinese  | 22     | `hfl/chinese-roberta-wwm-ext`        |
| **PHATE**   | Persian  | 4      | `HooshvareLab/roberta-fa-zwnj-base`  |

The scripts are dataset-agnostic. Everything that differs between datasets —
encoder, sequence length, label table, graph parameters — lives in the config
file, so the commands below have the same shape for all three.

---

## Architecture

Three phases, trained in sequence. Each phase reuses the previous one's
checkpoint; the final prediction is a **test-time ensemble** of Phase 2 and
Phase 3, with the mixing weight fitted on validation.

```
                          ┌──────────────────────────────┐
  text ───► Transformer ──┤ token embeddings   (B, L, H) │
            encoder       └──────┬───────────────┬───────┘
                                 │               │
        ┌────────────────────────┘               └────────────────┐
        ▼                                                         ▼
┌──────────────────┐                              ┌───────────────────────────┐
│ PHASE 1          │                              │ PHASE 2 · LSA head        │
│ mean-pool        │                              │ label embeddings (L, H)   │
│ + Linear(H, L)   │                              │  · token embeddings ᵀ     │
│ → logits_mean    │                              │ → attention over tokens   │
└────────┬─────────┘                              │ → label-aware doc reps    │
         │                                        │ → shared Linear(H, 1)     │
         │                                        │ → logits_lsa              │
         │                                        └───────────┬───────────────┘
         │                                                    │
         └──────────► gated fusion: α·logits_mean + β·logits_lsa
                       (α, β trainable scalars)  =  PHASE 2 logits
                                    │
                                    │  (frozen; cached to disk)
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ PHASE 3 · Heterogeneous graph over  documents | words | labels           │
│                                                                          │
│   node features ◄─ doc:   Phase-2 pooled embedding                       │
│                    word:  mean embedding of the docs containing it       │
│                    label: encoded label description                      │
│                                                                          │
│   3 edge types  ── doc–word    (IDF)                                     │
│                    word–word   (positive PMI, sliding window)            │
│                    word–label  (positive PMI)          NO doc–label edge │
│                                                                          │
│   Hetero-GCN (2 layers): per-relation conv + learned relation weights    │
│                          + residual + BatchNorm                          │
│                     ↓                                                    │
│   Label Correlation Matrix (L×L, learnable, init from co-occurrence)     │
│                     ↓                                                    │
│   graph logits  (trained STANDALONE — no Phase-2 shortcut)               │
└───────────────────────────────┬──────────────────────────────────────────┘
                                ▼
              TEST TIME:  w · σ(graph) + (1−w) · σ(Phase 2)
                          w grid-searched on the VALIDATION set
```

### What makes it different from a plain Hetero-GCN pipeline

1. **Semantic node initialisation** — word and label nodes start from
   transformer embeddings, not TextGCN-style random/one-hot vectors.
2. **Relation-specific convolutions** — each edge type gets its own projection
   `W_r`, combined by a learned softmax gate `α_r` over relations.
3. **Explicit label correlation** — a learnable `L×L` matrix initialised from
   training co-occurrence, instead of relying on graph propagation alone.
4. **Standalone graph training** — fusing with Phase 2 *during* training let the
   graph converge to "copy Phase 2". Training it alone and ensembling afterwards
   forces it to learn something of its own.

> The graph convolution is a **GCN, not a GAT**: aggregation uses the fixed,
> pre-normalised adjacency (IDF / PMI edge weights). No per-edge attention
> coefficient is computed from node features. The learned `α_r` is a global,
> relation-level gate — do not describe it as attention heads.

### No label leakage

| Channel                     | Source                              | Leak? |
|-----------------------------|-------------------------------------|-------|
| word–label PMI edges        | **train-split** co-occurrence only  | No    |
| LSA module                  | encoded label *description text*    | No    |
| gated fusion / ensemble     | trainable scalars / val-fitted `w`  | No    |
| label correlation matrix    | **train-split** co-occurrence only  | No    |

The graph is transductive (it spans train+val+test documents — standard for
TextGCN/S-GCN), but **there is no doc–label edge**: a test document connects
only to the words it contains. Test labels are never read, and every threshold
and ensemble weight is chosen on validation.

---

## Layout

```
├── README.md                      ← you are here
├── DESIGN.md                      ← architecture spec, ablation plan, targets
├── requirements.txt
├── configs/
│   ├── aapd_phase1.yaml   aapd_phase2.yaml   aapd_phase3.yaml
│   ├── rmsc_phase1.yaml   rmsc_phase2.yaml   rmsc_phase3.yaml
│   └── phate_phase1.yaml  phate_phase2.yaml  phate_phase3.yaml
├── src/
│   ├── data.py                    TSV loading, Dataset, collate
│   ├── labels.py                  label names + descriptions per dataset
│   ├── tokenization.py            word-node tokenizers (whitespace / WordPiece)
│   ├── graph_builder.py           IDF / PMI graph construction + normalisation
│   ├── node_features.py           doc / word / label node features
│   ├── losses.py                  BCE, ASL, Focal, DBL, class-balanced weights
│   ├── metrics.py                 accuracy, micro/macro-F1, P@k, nDCG@k
│   ├── train_utils.py             shared train/eval loop, threshold search, EMA
│   ├── console.py                 UTF-8 stdout (Windows code pages)
│   └── models/
│       ├── encoder_classifier.py  Phase 1  EncoderClassifier
│       ├── lsa_classifier.py      Phase 2  LabelSpecificAttention, LSAClassifier
│       ├── hetero_gcn.py          Phase 3  RelationGraphConv, HeteroGCN
│       └── graph_classifier.py    Phase 3  LabelCorrelationMatrix, GraphClassifier
├── scripts/                       the pipeline, in run order
│   ├── 01_train_phase1.py
│   ├── 02_encode_labels.py
│   ├── 03_train_phase2.py
│   ├── 04_build_graph.py
│   ├── 05_compute_node_features.py
│   ├── 06_train_phase3.py
│   └──  07_multiseed_run.py
├── tools/                         optional helpers, not part of the pipeline
│   ├── check_gpu.py               is the GPU free? do the data + tokenizer load?
│   ├── sanity_check.py            one forward pass end to end
│   ├── debug_first_steps.py       verbose first N training steps
│   ├── diagnose_collapse.py       has the model collapsed to all-0 / all-1?
│   └── make_synthetic_data.py     fake corpus for smoke tests
├── data/                          (gitignored) your TSV files + generated graphs
└── outputs/                       (gitignored) checkpoints, results.json, figures
```

---

## Setup

Everything below assumes the **Anaconda `base`** environment (that is what these
runs were developed against — Python 3.12, torch 2.6.0+cu118).

```bash
conda activate base

# 1. PyTorch for your CUDA version — see https://pytorch.org/get-started/locally/
#    (skip if torch is already installed: python -c "import torch; print(torch.__version__)")
pip install torch --index-url https://download.pytorch.org/whl/cu118

# 2. Everything else
pip install -r requirements.txt

# 3. Verify the GPU is free and the encoder downloads
python tools/check_gpu.py --config configs/aapd_phase1.yaml
```

No `torch-geometric` needed — the Hetero-GCN runs on plain `torch.sparse`.

**Fonts for figures.** Chinese and Persian figure text needs a font covering
that script; `src/plotting.py` picks one automatically (Microsoft YaHei /
Tahoma on Windows) and warns if none is installed. Persian additionally needs
`arabic-reshaper` + `python-bidi` (both in `requirements.txt`) for correct
letter joining and right-to-left order.

### Data format

One sample per line, **tab**-separated:

```
<multi-hot binary string>\t<raw text>
```

The multi-hot string length must equal `dataset.num_labels`, and bit *i* must
correspond to index *i* of that dataset's list in `src/labels.py`.

Place the files where the configs expect them:

```
data/aapd/aapd_train.tsv   data/aapd/aapd_validation.tsv   data/aapd/aapd_test.tsv
data/rmsc/train.tsv        data/rmsc/val.tsv               data/rmsc/test.tsv
data/phate/train.tsv       data/phate/val.tsv              data/phate/test.tsv
```

---

## Running the pipeline

Run the phases **in order** — each one consumes the previous one's checkpoint.

### AAPD (English)

```bash
# ── Phase 1: encoder baseline ───────────────────────────────────────────────
python scripts/01_train_phase1.py --config configs/aapd_phase1.yaml

# ── Phase 2: + LSA + gated fusion ───────────────────────────────────────────
python scripts/02_encode_labels.py \
    --dataset aapd \
    --encoder microsoft/deberta-v3-base \
    --output data/aapd/label_embeddings.pt
python scripts/03_train_phase2.py --config configs/aapd_phase2.yaml

# ── Phase 3: + Hetero-GCN + LCM, then ensemble ──────────────────────────────
python scripts/04_build_graph.py           --config configs/aapd_phase3.yaml
python scripts/05_compute_node_features.py --config configs/aapd_phase3.yaml
python scripts/06_train_phase3.py          --config configs/aapd_phase3.yaml
```

### RMSC-V2 (Chinese)

```bash
python scripts/01_train_phase1.py --config configs/rmsc_phase1.yaml

python scripts/02_encode_labels.py \
    --dataset rmsc \
    --encoder hfl/chinese-roberta-wwm-ext \
    --output data/rmsc/rmsc_cn_label_embeddings.pt
python scripts/03_train_phase2.py --config configs/rmsc_phase2.yaml

python scripts/04_build_graph.py           --config configs/rmsc_phase3.yaml
python scripts/05_compute_node_features.py --config configs/rmsc_phase3.yaml
python scripts/06_train_phase3.py          --config configs/rmsc_phase3.yaml
```

### PHATE (Persian)

```bash
python scripts/01_train_phase1.py --config configs/phate_phase1.yaml

python scripts/02_encode_labels.py \
    --dataset phate \
    --encoder HooshvareLab/roberta-fa-zwnj-base \
    --output data/phate/phate_label_embeddings.pt
python scripts/03_train_phase2.py --config configs/phate_phase2.yaml

python scripts/04_build_graph.py           --config configs/phate_phase3.yaml
python scripts/05_compute_node_features.py --config configs/phate_phase3.yaml
python scripts/06_train_phase3.py          --config configs/phate_phase3.yaml
```

> `04_build_graph.py` reads `phase3.min_freq` and `phase3.window_size` from the
> config. Both can be overridden per run: `--min_freq 5 --window_size 30`.
>
> **The word tokenizer follows `model.language`.** `english` → whitespace +
> punctuation stripping; `chinese` / `persian` → the encoder's own WordPiece
> tokenizer. `04_build_graph.py` and `05_compute_node_features.py` must be run
> with the same config, or the word ids won't line up.

---

## Multi-seed runs (mean ± std for the paper)

```bash
python scripts/07_multiseed_run.py \
    --config configs/aapd_phase3.yaml \
    --seeds 42,123,456,789,2024 \
    --tag crossvalid_
```

The phase is inferred from the config (`phase3:` block → Phase 3,
`head: lsa_gated` → Phase 2, otherwise Phase 1); pass `--phase {1,2,3}` to force
it. Per-seed runs land in `outputs/<tag>_seed<seed>/`, and the aggregate goes to
`outputs/<tag>_aggregated.json` plus a ready-to-paste
`outputs/<tag>_table.tex`. Add `--skip-training` to re-aggregate existing runs
without retraining.

Same command for the other datasets and phases:

```bash
python scripts/07_multiseed_run.py --config configs/rmsc_phase3.yaml  --seeds 42,123,456,789,2024 --tag rmsc_crossvalid
python scripts/07_multiseed_run.py --config configs/phate_phase3.yaml --seeds 42,123,456,789,2024 --tag phate_crossvalid
python scripts/07_multiseed_run.py --config configs/aapd_phase2.yaml  --seeds 42,123,456,789,2024 --tag aapd_crossvalid
```

---

## Outputs

Every training run writes to `train.save_dir`:

| File            | Contents                                                       |
|-----------------|----------------------------------------------------------------|
| `best_model.pt` | checkpoint at the best validation micro-F1                     |                                  |
| `results.json`  | per-epoch history + final test metrics                         |
| `config.yaml`   | the exact config that produced this run                        |

Reported metrics: accuracy (exact match), micro-F1, macro-F1, P@{1,3,5},
nDCG@{1,3,5}. `results.json → test_metrics` is always the number to quote — it
comes from the best-on-validation checkpoint.

---

## Troubleshooting

| Symptom | Try |
|---|---|
| CUDA out of memory | lower `train.batch_size` (raise `grad_accum_steps` to keep the effective batch), lower `dataset.max_seq_len`, or set `train.gradient_checkpointing: true` |
| Phase 3 OOM | lower `phase3.hidden_dim` (512 → 256), or raise `phase3.min_freq` to shrink the vocabulary |
| micro-F1 stuck at 0 | expected on very sparse label sets — `eval.adaptive_threshold: true` refits the threshold on validation. Confirm with `tools/diagnose_collapse.py` |
| Loss goes NaN | training aborts the epoch by design. Inspect with `tools/debug_first_steps.py` |
| `label_emb_init shape mismatch` | `02_encode_labels.py` was run with a different encoder or dataset than the phase-2 config |
| Persian text is disconnected / reversed in figures | `pip install arabic-reshaper python-bidi` |
| Chinese/Persian text shows as boxes | no font covering that script is installed — see `_FONT_CANDIDATES` in `src/plotting.py` |
| Old Phase-3 checkpoint won't load | it will — `GraphClassifier.load_state_dict` remaps pre-rename `hetero_gat.*` keys automatically |
| Garbled / crashing console output | `src/console.py` forces UTF-8; if it still breaks, run `chcp 65001` first on Windows |
| Word count differs between graph and node features | both steps must use the same `model.language`; rerun `04_build_graph.py` and `05_compute_node_features.py` with the same config |

---

## Reproducibility

* `train.seed` seeds Python, NumPy and PyTorch, and pins cuDNN to deterministic mode.
* Each run copies its resolved config next to the checkpoint (`config.yaml`).
* Thresholds and ensemble weights are always fitted on validation, never test.
* Reference hardware: RTX 3060 Ti (8 GB) + 32 GB RAM.

See `DESIGN.md` for the full architecture spec, target metrics, and the
ablation plan.
