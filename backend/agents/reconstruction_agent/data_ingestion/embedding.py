from __future__ import annotations

import math
import os
from collections import Counter

EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "256"))


def k_mer_encode(sequence: str, k: int = 4) -> list[float]:
    """Encode a DNA sequence into a normalized k-mer frequency vector."""
    sequence = (sequence or "").upper()
    if k <= 0:
        return [0.0] * EMBEDDING_DIM

    alphabet = ["A", "C", "G", "T", "N"]
    dim = EMBEDDING_DIM if k == 4 else 4**k
    counts = Counter(
        sequence[i : i + k]
        for i in range(0, len(sequence) - k + 1)
        if len(sequence[i : i + k]) == k
    )
    vector = [0.0] * dim
    for mer, value in counts.items():
        index = 0
        for base in mer:
            index = index * 5 + (alphabet.index(base) if base in alphabet else 4)
        vector[index % dim] += value

    norm = math.sqrt(sum(v * v for v in vector))
    if norm > 0:
        normalized = [v / norm for v in vector]
        normalized[-1] = math.sqrt(max(0.0, 1.0 - sum(v * v for v in normalized[:-1])))
        return normalized
    return [0.0] * dim
