"""Initial node features for the heterogeneous graph.

Three node types, three feature sources — all 768-d, all computed once and
cached to disk by `scripts/05_compute_node_features.py`:

  1. Document nodes: pooled encoder embedding from the frozen Phase-2 model
  2. Word nodes:     mean of the embeddings of documents containing that word
  3. Label nodes:    encoded label description text

Word features are averaged document embeddings rather than fresh forward passes
— semantically meaningful and essentially free, given the doc embeddings.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


@torch.no_grad()
def compute_doc_embeddings(
    model: nn.Module,
    samples: list,
    tokenizer,
    max_seq_len: int = 256,
    batch_size: int = 32,
    device: torch.device = None,
    desc: str = "Encoding documents",
) -> np.ndarray:
    """Pooled document embeddings from the model's encoder.

    Returns: (N_docs, hidden_size) float array.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    all_embeddings = []
    for start in tqdm(range(0, len(samples), batch_size), desc=desc):
        batch_samples = samples[start:start + batch_size]
        texts = [s.text for s in batch_samples]
        enc = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_seq_len,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        out = model(input_ids=input_ids, attention_mask=attention_mask)
        if "pooled" in out:
            pooled = out["pooled"]
        else:
            # Fallback: mean-pool the encoder's last hidden state directly.
            outputs = model.encoder(input_ids=input_ids, attention_mask=attention_mask,
                                    return_dict=True)
            last_hidden = outputs.last_hidden_state
            mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
            pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

        all_embeddings.append(pooled.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


@torch.no_grad()
def compute_doc_logits(
    model: nn.Module,
    samples: list,
    tokenizer,
    max_seq_len: int = 256,
    batch_size: int = 32,
    device: torch.device = None,
    desc: str = "Computing logits",
) -> np.ndarray:
    """Phase-2 logits for every document, cached for the test-time ensemble.

    Returns: (N_docs, num_labels) float array.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    all_logits = []
    for start in tqdm(range(0, len(samples), batch_size), desc=desc):
        batch_samples = samples[start:start + batch_size]
        texts = [s.text for s in batch_samples]
        enc = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_seq_len,
            return_tensors="pt",
        )
        out = model(
            input_ids=enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device),
        )
        all_logits.append(out["logits"].cpu().numpy())

    return np.concatenate(all_logits, axis=0)


def compute_word_embeddings(
    doc_embeddings: np.ndarray,
    tokenized_docs: list,
    word_id_map: dict,
    vocab_size: int,
    desc: str = "Computing word embeddings",
) -> np.ndarray:
    """Word features = mean embedding of the documents containing each word.

    `tokenized_docs` MUST come from the same tokenizer used to build the graph
    vocabulary (see `src/tokenization.make_word_tokenizer`), otherwise word ids
    will not line up with the graph's word nodes.

    Returns: (vocab_size, hidden_size) float array.
    """
    hidden_size = doc_embeddings.shape[1]
    word_embeddings = np.zeros((vocab_size, hidden_size), dtype=np.float32)
    word_counts = np.zeros(vocab_size, dtype=np.float32)

    for doc_idx, doc_words in enumerate(tqdm(tokenized_docs, desc=desc)):
        unique_words = {word_id_map[w] for w in doc_words if w in word_id_map}
        for wid in unique_words:
            word_embeddings[wid] += doc_embeddings[doc_idx]
            word_counts[wid] += 1

    # Words absent from every document keep their zero vector (count clamped to 1).
    word_counts = np.maximum(word_counts, 1.0)
    return word_embeddings / word_counts[:, None]


def compute_label_embeddings(
    encoder_name: str,
    label_descriptions: list,
    device: torch.device = None,
    max_seq_len: int = 32,
) -> np.ndarray:
    """Label features = mean-pooled encoding of each label's description.

    Returns: (num_labels, hidden_size) float array.
    """
    from .models.lsa_classifier import encode_label_descriptions

    return encode_label_descriptions(
        encoder_name=encoder_name,
        label_descriptions=label_descriptions,
        device=device,
        max_seq_len=max_seq_len,
    ).numpy()
