"""Generate a tiny synthetic dataset in the project's TSV format.

For smoke-testing the pipeline end to end before the real corpus is in place.
The text is random word salad and the labels are random multi-hot vectors —
NEVER report numbers produced from this.

Usage:
    python tools/make_synthetic_data.py --config configs/aapd_phase1.yaml

Writes the three files named by `dataset.{train,val,test}_path` in the config.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.console import enable_utf8_output

import yaml

WORDS = (
    "we propose a novel method for multi-label text classification based on "
    "graph neural networks with attention mechanism the model leverages "
    "pretrained transformer encoders to capture semantic information and "
    "builds a heterogeneous graph over documents words and labels experimental "
    "results on benchmark datasets demonstrate the effectiveness of our approach "
    "in particular we show that label-specific attention improves performance "
    "on long-tail categories while maintaining competitive accuracy on head "
    "labels furthermore our asymmetric loss function mitigates the negative "
    "flooding problem common in multi-label scenarios"
).split()


def random_text(rng: random.Random) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(rng.randint(50, 200)))


def random_labels(rng: random.Random, num_labels: int, avg_pos: float = 2.4) -> str:
    """Multi-hot string with `avg_pos` expected positives per sample."""
    p = avg_pos / num_labels
    return "".join("1" if rng.random() < p else "0" for _ in range(num_labels))


def write_split(path: Path, n_samples: int, num_labels: int, seed: int) -> None:
    rng = random.Random(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for _ in range(n_samples):
            # TAB separated — src/data.load_tsv splits on '\t'.
            f.write(f"{random_labels(rng, num_labels)}\t{random_text(rng)}\n")
    print(f"  Wrote {n_samples} samples to {path}")


def main():
    enable_utf8_output()
    parser = argparse.ArgumentParser(description="Write a synthetic TSV dataset")
    parser.add_argument("--config", type=str, required=True,
                        help="Config whose dataset paths / num_labels to mirror")
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-val", type=int, default=100)
    parser.add_argument("--n-test", type=int, default=100)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg["dataset"]
    num_labels = data_cfg["num_labels"]

    print(f"Writing synthetic {data_cfg.get('name', '?')} data "
          f"({num_labels} labels)...")
    write_split(Path(data_cfg["train_path"]), args.n_train, num_labels, seed=42)
    write_split(Path(data_cfg["val_path"]),   args.n_val,   num_labels, seed=43)
    write_split(Path(data_cfg["test_path"]),  args.n_test,  num_labels, seed=44)

    print("\nDone. Smoke-test the pipeline with:")
    print(f"  python tools/sanity_check.py --config {args.config}")
    print(f"  python scripts/01_train_phase1.py --config {args.config}")


if __name__ == "__main__":
    main()
