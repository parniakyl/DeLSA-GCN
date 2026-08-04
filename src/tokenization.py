"""Word-level tokenization used to build the heterogeneous graph.

This is *not* the model's subword tokenizer for classification — it is the
tokenizer that defines the **word nodes** of the graph, and it must be identical
in `scripts/04_build_graph.py` (which builds the vocabulary) and
`scripts/05_compute_node_features.py` (which averages document embeddings per
word). That is the only reason this lives in one shared module.

Two modes, selected by the dataset language:

  english          whitespace + punctuation stripping (S-GCN / TextGCN style)
  chinese/persian  the encoder's own WordPiece tokenizer

Whitespace splitting is wrong for Chinese (no spaces between words — it would
degenerate to characters) and lossy for Persian (ZWNJ-joined compounds), which
is why those datasets reuse the HuggingFace tokenizer instead:

  whitespace on Chinese:  "双主唱好棒" -> ['双主唱好棒']  (one meaningless blob)
  chinese-roberta-wwm:    "双主唱好棒" -> ['双', '主唱', '好棒']
"""

from __future__ import annotations

import re
import string
from typing import Callable

# Precompiled once — this runs over every document in the corpus.
_NON_WORD = re.compile(r"[^A-Za-z0-9(),!?'`]")
_CONTRACTIONS = [
    (re.compile(r"\'s"),  " 's"),
    (re.compile(r"\'ve"), " 've"),
    (re.compile(r"n\'t"), " n't"),
    (re.compile(r"\'re"), " 're"),
    (re.compile(r"\'d"),  " 'd"),
    (re.compile(r"\'ll"), " 'll"),
    (re.compile(r","),    " , "),
    (re.compile(r"!"),    " ! "),
    (re.compile(r"\("),   " ( "),
    (re.compile(r"\)"),   " ) "),
    (re.compile(r"\?"),   " ? "),
]
_MULTISPACE = re.compile(r"\s{2,}")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)

ENGLISH_LANGUAGES = {"", "en", "english"}


def whitespace_tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace (matches S-GCN).

    Stopwords are deliberately *not* removed — the transformer encoder handles
    that better than a hardcoded list.
    """
    s = _NON_WORD.sub(" ", text)
    for pattern, repl in _CONTRACTIONS:
        s = pattern.sub(repl, s)
    s = _MULTISPACE.sub(" ", s)
    s = s.strip().lower()
    s = s.translate(_PUNCT_TABLE)
    return s.split()


def make_word_tokenizer(language: str | None, encoder_name: str) -> Callable[[str], list[str]]:
    """Return the word-node tokenizer for a dataset language.

    Args:
        language:     `model.language` from the config ('english' if absent)
        encoder_name: HuggingFace encoder id, used for non-English languages
    """
    lang = (language or "english").lower()
    if lang in ENGLISH_LANGUAGES:
        print(f"  Word tokenizer: whitespace (language={lang})")
        return whitespace_tokenize

    from transformers import AutoTokenizer

    print(f"  Word tokenizer: {encoder_name} WordPiece (language={lang})")
    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    # `tokenize` returns subword strings without [CLS]/[SEP].
    return tokenizer.tokenize


def build_vocab(tokenized_docs: list[list[str]], min_freq: int) -> tuple[list[str], dict]:
    """Build the graph vocabulary from tokenized documents.

    Returns (word_list, word_id_map) where word_id_map[word] == index in word_list.
    """
    word_freq: dict[str, int] = {}
    for doc_words in tokenized_docs:
        for w in doc_words:
            word_freq[w] = word_freq.get(w, 0) + 1

    word_list = sorted([w for w, f in word_freq.items() if f >= min_freq])
    word_id_map = {w: i for i, w in enumerate(word_list)}

    print(f"  Vocabulary: {len(word_list)} words (min_freq={min_freq})")
    return word_list, word_id_map
