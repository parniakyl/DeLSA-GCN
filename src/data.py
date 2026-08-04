"""Data loading for multi-label TSV files (AAPD / RMSC-V2 / PHATE share one format).

File format (one line per sample):
    <multi-hot binary string>\t<raw text>

Example line (AAPD, 54 labels):
    000100...0<TAB>we study the problem of learning from labeled and unlabeled data

The multi-hot string length must equal `num_labels`
(54 for AAPD, 22 for RMSC-V2, 4 for PHATE).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class MultiLabelSample:
    text: str
    labels: np.ndarray  # shape (num_labels,), dtype float32


def load_tsv(path: str, num_labels: int) -> list[MultiLabelSample]:
    """Read a TSV file in the AAPD/SGM multi-hot format."""
    samples: list[MultiLabelSample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            label_str = parts[0]
            text = parts[1] if len(parts) > 1 else ""
            # Multi-hot string may be 0/1; convert to float array.
            label_vec = np.array([float(c) for c in label_str], dtype=np.float32)
            if label_vec.shape[0] != num_labels:
                raise ValueError(
                    f"Line has {label_vec.shape[0]} labels, expected {num_labels}"
                )
            samples.append(MultiLabelSample(text=text, labels=label_vec))
    return samples


def load_split(data_dir: str, split: str, num_labels: int) -> list[MultiLabelSample]:
    """Load a named split from `data_dir`.

    Looks for `<split>.tsv` first, then `<dataset-prefixed>_<split>.tsv`
    (e.g. `aapd_train.tsv`).
    """
    candidates = [
        os.path.join(data_dir, f"{split}.tsv"),
        os.path.join(data_dir, f"aapd_{split}.tsv"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return load_tsv(path, num_labels=num_labels)
    raise FileNotFoundError(
        f"Could not find {split} split in {data_dir}: tried {candidates}"
    )


def load_split_file(path: str, num_labels: int) -> list[MultiLabelSample]:
    """Load a TSV file directly from its full path.

    This is the entry point used by every script, since the configs already
    specify full paths (e.g. `data/aapd/aapd_train.tsv`).
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    return load_tsv(path, num_labels=num_labels)


class MultiLabelDataset(Dataset):
    """Dataset wrapping a MultiLabelSample list. Tokenization happens in collate_fn."""

    def __init__(self, samples: list[MultiLabelSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> MultiLabelSample:
        return self.samples[idx]


def make_collate_fn(tokenizer, max_seq_len: int = 256):
    """Return a collate_fn that batches samples into tokenizer tensors."""

    def collate(batch: Iterable[MultiLabelSample]):
        texts = [s.text for s in batch]
        labels = np.stack([s.labels for s in batch], axis=0)
        enc = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_seq_len,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels":         torch.from_numpy(labels),
        }

    return collate


def label_matrix(samples: list[MultiLabelSample]) -> np.ndarray:
    """Stack a sample list into an (N, num_labels) multi-hot matrix."""
    return np.stack([s.labels for s in samples], axis=0)
