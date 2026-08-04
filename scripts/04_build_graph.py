"""Phase 3, step 1 — build the heterogeneous graph and save it to disk.

Word segmentation follows `model.language` in the config:
  english (default)  whitespace + punctuation stripping
  chinese / persian  the encoder's own WordPiece tokenizer

`scripts/05_compute_node_features.py` reuses the same tokenizer, so word ids
always line up with the graph's word nodes.

Usage:
    python scripts/04_build_graph.py --config configs/aapd_phase3.yaml
    python scripts/04_build_graph.py --config configs/rmsc_phase3.yaml
    python scripts/04_build_graph.py --config configs/phate_phase3.yaml

Graph parameters come from the `phase3:` block (`min_freq`, `window_size`) and
can be overridden on the command line.

Outputs (under `phase3.graph_dir`):
    adj_doc_word.npz  adj_word_word.npz  adj_word_label.npz
    word_list.pkl     word_id_map.pkl    metadata.pkl
"""

from __future__ import annotations

import argparse
import pickle as pkl
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import scipy.sparse as sp
import yaml

from src.console import enable_utf8_output
from src.data import label_matrix, load_split_file
from src.graph_builder import build_hetero_graph
from src.tokenization import build_vocab, make_word_tokenizer


def main():
    enable_utf8_output()
    parser = argparse.ArgumentParser(description="Build the heterogeneous graph")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--min_freq", type=int, default=None,
                        help="Override phase3.min_freq (minimum word frequency)")
    parser.add_argument("--window_size", type=int, default=None,
                        help="Override phase3.window_size (PMI sliding window)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["dataset"]
    phase3_cfg = cfg.get("phase3", {})
    num_labels = data_cfg["num_labels"]
    dataset_name = data_cfg.get("name", "unknown")
    encoder_name = cfg["model"]["encoder_name"]
    language = cfg["model"].get("language", "english")

    min_freq = args.min_freq if args.min_freq is not None else phase3_cfg.get("min_freq", 10)
    window_size = args.window_size if args.window_size is not None \
        else phase3_cfg.get("window_size", 20)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    print(f"Loading {dataset_name} data...")
    train_samples = load_split_file(data_cfg["train_path"], num_labels)
    val_samples = load_split_file(data_cfg["val_path"], num_labels)
    test_samples = load_split_file(data_cfg["test_path"], num_labels)
    print(f"  Train: {len(train_samples)} | Val: {len(val_samples)} | Test: {len(test_samples)}")

    train_size, val_size, test_size = len(train_samples), len(val_samples), len(test_samples)
    total_docs = train_size + val_size + test_size
    all_samples = train_samples + val_samples + test_samples

    # ------------------------------------------------------------------
    # Tokenize (documents order = train, then val, then test)
    # ------------------------------------------------------------------
    print("\nTokenizing documents...")
    tokenize = make_word_tokenizer(language, encoder_name)
    tokenized_docs = [tokenize(s.text) for s in all_samples]
    total_tokens = sum(len(d) for d in tokenized_docs)
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Avg tokens/doc: {total_tokens / max(total_docs, 1):.1f}")
    print(f"  Sample (first 20 tokens of first doc): {tokenized_docs[0][:20]}")

    print(f"\nBuilding vocabulary (min_freq={min_freq})...")
    word_list, word_id_map = build_vocab(tokenized_docs, min_freq=min_freq)

    labels = label_matrix(all_samples)

    # ------------------------------------------------------------------
    # Graph
    # ------------------------------------------------------------------
    print(f"\nBuilding heterogeneous graph (window_size={window_size})...")
    adj_doc_word, adj_word_word, adj_word_label, metadata = build_hetero_graph(
        tokenized_docs=tokenized_docs,
        labels=labels,
        word_id_map=word_id_map,
        word_list=word_list,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        window_size=window_size,
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    save_dir = Path(phase3_cfg.get("graph_dir", f"data/{dataset_name}/graph"))
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving graph to {save_dir}/")
    sp.save_npz(save_dir / "adj_doc_word.npz", adj_doc_word)
    sp.save_npz(save_dir / "adj_word_word.npz", adj_word_word)
    sp.save_npz(save_dir / "adj_word_label.npz", adj_word_label)

    with open(save_dir / "word_list.pkl", "wb") as f:
        pkl.dump(word_list, f)
    with open(save_dir / "word_id_map.pkl", "wb") as f:
        pkl.dump(word_id_map, f)
    with open(save_dir / "metadata.pkl", "wb") as f:
        pkl.dump({
            **metadata,
            "train_size": train_size,
            "val_size": val_size,
            "test_size": test_size,
            "total_docs": total_docs,
            "dataset": dataset_name,
            "encoder_name": encoder_name,
            "language": language,
            "min_freq": min_freq,
        }, f)

    print("\n✓ Graph saved successfully!")
    print(f"  Location: {save_dir}/")
    print("  Files: adj_doc_word.npz, adj_word_word.npz, adj_word_label.npz,")
    print("         word_list.pkl, word_id_map.pkl, metadata.pkl")
    print("\nGraph stats:")
    for k, v in metadata.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
