from __future__ import annotations

from collections import Counter
import hashlib
from typing import Any


STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "while", "with", "without", "to", "of", "in", "on",
    "for", "from", "by", "as", "at", "is", "are", "was", "were", "be", "been", "this", "that", "these",
    "those", "it", "its", "we", "our", "you", "your", "they", "their", "can", "could", "should", "would",
}


def tokenize(text: str) -> list[str]:
    tokens = []
    for token in text.lower().split():
        cleaned = "".join(ch for ch in token if ch.isalnum())
        if cleaned and cleaned not in STOPWORDS:
            tokens.append(cleaned)
    return tokens


def build_sparse_vector(text: str) -> dict[str, Any]:
    tokens = tokenize(text)
    counts = Counter(tokens)
    terms = sorted(counts.items())
    return {
        "indices": [_term_index(term) for term, _ in terms],
        "values": [float(freq) for _, freq in terms],
        "terms": [term for term, _ in terms],
    }


def _term_index(term: str) -> int:
    digest = hashlib.sha1(term.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)
