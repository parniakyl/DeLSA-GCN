"""Phase 2, step 1 — encode label descriptions into initial label embeddings.

Produces a (num_labels, hidden_size) tensor used to initialise the LSA label
embeddings in Phase 2. Use the SAME encoder here as in the phase-2 config,
otherwise the embeddings live in a different space than the token embeddings
they are dot-producted against.

Usage:
    # AAPD (54 labels, English)
    python scripts/02_encode_labels.py --dataset aapd \
        --encoder microsoft/deberta-v3-base \
        --output data/aapd/label_embeddings.pt

    # RMSC-V2 (22 labels, Chinese)
    python scripts/02_encode_labels.py --dataset rmsc \
        --encoder hfl/chinese-roberta-wwm-ext \
        --output data/rmsc/rmsc_cn_label_embeddings.pt

    # PHATE (4 labels, Persian)
    python scripts/02_encode_labels.py --dataset phate \
        --encoder HooshvareLab/roberta-fa-zwnj-base \
        --output data/phate/phate_label_embeddings.pt

The label text itself comes from `src/labels.py` — edit the description tables
there, not this script.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.console import enable_utf8_output
from src.labels import DATASETS, get_labels
from src.models.lsa_classifier import encode_label_descriptions


def main():
    enable_utf8_output()
    parser = argparse.ArgumentParser(description="Encode label descriptions")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=list(DATASETS.keys()),
                        help="Dataset name (selects the label description table)")
    parser.add_argument("--encoder", type=str, required=True,
                        help="HuggingFace encoder id — must match the phase-2 config")
    parser.add_argument("--output", type=str, default=None,
                        help="Output .pt path (default: data/<dataset>/label_embeddings.pt)")
    parser.add_argument("--max_seq_len", type=int, default=32,
                        help="Max tokens per label description. 32 is plenty.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dataset: {args.dataset}")
    print(f"Device:  {device}")
    print(f"Encoder: {args.encoder}")

    names, descriptions = get_labels(args.dataset)
    print(f"\nLabel descriptions ({len(descriptions)}):")
    for i, (name, desc) in enumerate(zip(names, descriptions)):
        print(f"  {i:2d} | {name:<25} -> {desc}")

    print(f"\nEncoding {len(descriptions)} label descriptions with {args.encoder}...")
    label_embs = encode_label_descriptions(
        encoder_name=args.encoder,
        label_descriptions=descriptions,
        device=device,
        max_seq_len=args.max_seq_len,
    )

    print(f"\nLabel embeddings shape: {tuple(label_embs.shape)}")
    print(f"  Mean: {label_embs.mean():.4f}")
    print(f"  Std:  {label_embs.std():.4f}")

    output_path = Path(args.output) if args.output else \
        Path(f"data/{args.dataset}/label_embeddings.pt")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "label_embeddings": label_embs,
        "label_names": names,
        "label_descriptions": descriptions,
        "encoder_name": args.encoder,
        "dataset": args.dataset,
    }, output_path)
    print(f"\n✓ Saved to {output_path}")
    print("\nNext step (make sure model.label_emb_path in the config points here):")
    print(f"  python scripts/03_train_phase2.py --config configs/{args.dataset}_phase2.yaml")


if __name__ == "__main__":
    main()
