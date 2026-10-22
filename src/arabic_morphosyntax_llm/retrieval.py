"""Reproducible in-context demonstration selection."""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import Any


RETRIEVAL_METHODS = (
    "random",
    "chrf-high",
    "chrf-low",
    "cosine-high",
    "cosine-low",
)


@dataclass(frozen=True)
class Selection:
    records: list[dict[str, Any]]
    metadata: list[dict[str, Any]]


def retrieval_text(record: dict[str, Any], task: str) -> str:
    if task in {"morphology", "parse-gold"}:
        tokens = record.get("tokens")
        if not isinstance(tokens, list):
            raise ValueError(f"record {record.get('id')}: tokens required for retrieval")
        return " ".join(tokens)
    text = record.get("text")
    if not isinstance(text, str):
        raise ValueError(f"record {record.get('id')}: text required for retrieval")
    return text


def _chrf(candidate: str, reference: str) -> float:
    """Experiment chrF++ approximation: mean character and word n-gram P/R."""
    precisions: list[float] = []
    recalls: list[float] = []
    for text_mode, orders in (("char", 6), ("word", 2)):
        left = candidate if text_mode == "char" else candidate.split()
        right = reference if text_mode == "char" else reference.split()
        local_p: list[float] = []
        local_r: list[float] = []
        for order in range(1, orders + 1):
            a = Counter(tuple(left[i:i + order]) for i in range(len(left) - order + 1))
            b = Counter(tuple(right[i:i + order]) for i in range(len(right) - order + 1))
            if not a and not b:
                continue
            overlap = sum((a & b).values())
            local_p.append(overlap / sum(b.values()) if b else 0.0)
            local_r.append(overlap / sum(a.values()) if a else 0.0)
        precisions.append(sum(local_p) / len(local_p) if local_p else 0.0)
        recalls.append(sum(local_r) / len(local_r) if local_r else 0.0)
    p = sum(precisions) / 2
    r = sum(recalls) / 2
    return 5 * p * r / (4 * p + r) if p + r else 0.0


def _cosine(left: Any, right: Any) -> float:
    if not isinstance(left, list) or not isinstance(right, list) or len(left) != len(right):
        raise ValueError("cosine retrieval requires equal-length embedding arrays")
    if not left:
        raise ValueError("embedding arrays cannot be empty")
    numerator = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def select_demonstrations(
    query: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    task: str,
    shots: int,
    method: str,
    seed: int,
) -> Selection:
    if method not in RETRIEVAL_METHODS:
        raise ValueError(f"Unknown retrieval method {method!r}")
    if shots < 0:
        raise ValueError("shots cannot be negative")
    count = min(shots, len(candidates))
    if count == 0:
        return Selection([], [])

    if method == "random":
        rng = random.Random(f"{seed}:{query['id']}")
        selected = [(index, None) for index in rng.sample(range(len(candidates)), count)]
    else:
        scored: list[tuple[int, float]] = []
        if method.startswith("chrf"):
            query_value = retrieval_text(query, task)
            for index, candidate in enumerate(candidates):
                scored.append((index, _chrf(query_value, retrieval_text(candidate, task))))
        else:
            query_embedding = query.get("embedding")
            for index, candidate in enumerate(candidates):
                scored.append((index, _cosine(query_embedding, candidate.get("embedding"))))
        reverse = method.endswith("high")
        scored.sort(key=lambda item: ((-item[1]) if reverse else item[1], item[0]))
        selected = scored[:count]

    records = [candidates[index] for index, _ in selected]
    metadata = [
        {"id": candidates[index]["id"], "score": score, "candidate_index": index}
        for index, score in selected
    ]
    return Selection(records, metadata)
