"""Phase 3, step 2 — compute and cache the graph's initial node features.

Loads the frozen Phase-2 checkpoint and writes four arrays:

    doc_embeddings.npy    (N_docs,  H)  pooled encoder embedding per document
    word_embeddings.npy   (V,       H)  mean embedding of docs containing the word
    label_embeddings.npy  (L,       H)  encoded label description
    phase2_logits.npy     (N_docs,  L)  Phase-2 logits, for the test-time ensemble

Run `scripts/04_build_graph.py` first — the vocabulary saved there defines the
word rows, and this script reuses the same tokenizer so the ids match.

Usage:
    python scripts/05_compute_node_features.py --config configs/aapd_phase3.yaml
    python scripts/05_compute_node_features.py --config configs/rmsc_phase3.yaml
    python scripts/05_compute_node_features.py --config configs/phate_phase3.yaml
"""

from __future__ import annotations

import argparse
import pickle as pkl
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import yaml
from transformers import AutoTokenizer

from src.console import enable_utf8_output
from src.data import load_split_file
from src.labels import get_labels
from src.models.lsa_classifier import LSAClassifier
from src.node_features import (
    compute_doc_embeddings,
    compute_doc_logits,
    compute_label_embeddings,
    compute_word_embeddings,
)
from src.tokenization import make_word_tokenizer


def main():
    enable_utf8_output()
    parser = argparse.ArgumentParser(description="Cache graph node features")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--phase2_checkpoint", type=str, default=None,
                        help="Path to Phase 2 best_model.pt (default: from config)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Inference batch size for encoding documents")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["dataset"]
    model_cfg = cfg["model"]
    phase3_cfg = cfg.get("phase3", {})
    num_labels = data_cfg["num_labels"]
    dataset_name = data_cfg.get("name", "unknown")
    encoder_name = model_cfg["encoder_name"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset: {dataset_name} ({num_labels} labels)")
    print(f"Encoder: {encoder_name}")

    ckpt_path = args.phase2_checkpoint or phase3_cfg.get("phase2", {}).get("checkpoint")
    if not ckpt_path:
        raise SystemExit("No Phase 2 checkpoint: set phase3.phase2.checkpoint in the "
                         "config or pass --phase2_checkpoint")
    print(f"Phase 2 checkpoint: {ckpt_path}")

    # ------------------------------------------------------------------
    # Frozen Phase 2 model
    # ------------------------------------------------------------------
    print("\nLoading Phase 2 model...")
    model = LSAClassifier(
        encoder_name=encoder_name,
        num_labels=num_labels,
        dropout=model_cfg["dropout"],
        gradient_checkpointing=False,   # inference only — checkpointing just slows it
        freeze_layers=0,
        label_emb_init=None,            # overwritten by the checkpoint
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Loaded epoch {ckpt.get('epoch', '?')}")

    # ------------------------------------------------------------------
    # Data + graph vocabulary
    # ------------------------------------------------------------------
    print(f"\nLoading {dataset_name} data...")
    all_samples = (
        load_split_file(data_cfg["train_path"], num_labels)
        + load_split_file(data_cfg["val_path"], num_labels)
        + load_split_file(data_cfg["test_path"], num_labels)
    )
    print(f"  Total docs: {len(all_samples)}")

    graph_dir = Path(phase3_cfg.get("graph_dir", f"data/{dataset_name}/graph"))
    with open(graph_dir / "word_id_map.pkl", "rb") as f:
        word_id_map = pkl.load(f)
    with open(graph_dir / "word_list.pkl", "rb") as f:
        word_list = pkl.load(f)
    print(f"  Vocabulary: {len(word_list)} words (from {graph_dir})")

    tokenizer = AutoTokenizer.from_pretrained(encoder_name)

    # ------------------------------------------------------------------
    # 1. Document nodes
    # ------------------------------------------------------------------
    print("\n[1/4] Computing document embeddings...")
    doc_embeddings = compute_doc_embeddings(
        model=model,
        samples=all_samples,
        tokenizer=tokenizer,
        max_seq_len=data_cfg["max_seq_len"],
        batch_size=args.batch_size,
        device=device,
    )
    print(f"  Doc embeddings: {doc_embeddings.shape}")

    # ------------------------------------------------------------------
    # 2. Word nodes — must use the SAME tokenizer as 04_build_graph.py
    # ------------------------------------------------------------------
    print("\n[2/4] Computing word embeddings...")
    word_tokenize = make_word_tokenizer(model_cfg.get("language", "english"), encoder_name)
    tokenized_docs = [word_tokenize(s.text) for s in all_samples]
    word_embeddings = compute_word_embeddings(
        doc_embeddings=doc_embeddings,
        tokenized_docs=tokenized_docs,
        word_id_map=word_id_map,
        vocab_size=len(word_list),
    )
    print(f"  Word embeddings: {word_embeddings.shape}")

    # ------------------------------------------------------------------
    # 3. Label nodes
    # ------------------------------------------------------------------
    print("\n[3/4] Computing label embeddings...")
    _, label_descriptions = get_labels(dataset_name)
    label_embeddings = compute_label_embeddings(
        encoder_name=encoder_name,
        label_descriptions=label_descriptions,
        device=device,
        max_seq_len=32,
    )
    print(f"  Label embeddings: {label_embeddings.shape}")

    # ------------------------------------------------------------------
    # 4. Phase 2 logits (cached for the test-time ensemble)
    # ------------------------------------------------------------------
    print("\n[4/4] Computing Phase 2 logits (for the ensemble)...")
    phase2_logits = compute_doc_logits(
        model=model,
        samples=all_samples,
        tokenizer=tokenizer,
        max_seq_len=data_cfg["max_seq_len"],
        batch_size=args.batch_size,
        device=device,
    )
    print(f"  Phase 2 logits: {phase2_logits.shape}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    save_dir = Path(phase3_cfg.get("features_dir", f"data/{dataset_name}/node_features"))
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving node features to {save_dir}/")
    np.save(save_dir / "doc_embeddings.npy", doc_embeddings)
    np.save(save_dir / "word_embeddings.npy", word_embeddings)
    np.save(save_dir / "label_embeddings.npy", label_embeddings)
    np.save(save_dir / "phase2_logits.npy", phase2_logits)

    print("\n✓ Node features saved!")
    print(f"  Doc embeddings:   {doc_embeddings.shape}")
    print(f"  Word embeddings:  {word_embeddings.shape}")
    print(f"  Label embeddings: {label_embeddings.shape}")
    print(f"  Phase 2 logits:   {phase2_logits.shape}")


if __name__ == "__main__":
    main()
