"""Random keyword-subspace sampling used by KRSB heads."""

from __future__ import annotations

import random
from typing import Mapping, Sequence


def ensure_keyword_list(value) -> list[str]:
    """Normalize a cell of keywords to a cleaned list of strings."""
    if value is None:
        return []
    if isinstance(value, float) and value != value:  # NaN
        return []
    if isinstance(value, (list, tuple)):
        return [str(token).strip() for token in value if str(token).strip()]
    text = str(value).strip()
    if not text:
        return []
    for sep in (";", "|", ","):
        if sep in text:
            return [part.strip() for part in text.split(sep) if part.strip()]
    return [text]


def unique_keep_order(keywords: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for keyword in keywords:
        key = keyword.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(keyword)
    return unique


def sample_keywords_for_row(
    keywords_by_method: Mapping[str, Sequence[str]],
    rng: random.Random,
    method_names: Sequence[str],
    methods_per_model: int = 4,
    total_k: int = 40,
    per_method_min: int = 2,
    add_method_tags: bool = True,
    method_tags: Mapping[str, str] | None = None,
) -> str:
    """Build one keyword-text for a document, matching the original KRSB sampler.

    Steps (same as the SciBERT notebooks):

    1. Shuffle extractor names and keep ``methods_per_model`` of them.
    2. Deduplicate each method's list while preserving rank order.
    3. Split ``total_k`` across the chosen methods, at least ``per_method_min`` each.
    4. Sample without replacement from every method pool.
    5. Optionally prefix phrases with a method tag such as ``[YAKE]``.
    6. Shuffle the mixed phrases and join them with ``"; "``.
    """
    tags = dict(method_tags or {})
    cols = list(method_names)
    rng.shuffle(cols)
    cols = cols[: max(1, methods_per_model)]

    per_method: dict[str, list[str]] = {}
    for col in cols:
        per_method[col] = unique_keep_order(ensure_keyword_list(keywords_by_method.get(col)))

    n_methods = len(cols)
    base = max(per_method_min, total_k // n_methods)
    counts = {col: base for col in cols}
    remainder = max(0, total_k - base * n_methods)
    for _ in range(remainder):
        counts[rng.choice(cols)] += 1

    sampled: list[str] = []
    for col in cols:
        pool = per_method[col]
        if not pool:
            continue
        k = min(counts[col], len(pool))
        picked = rng.sample(pool, k)
        if add_method_tags:
            tag = tags.get(col, "[KW]")
            sampled.extend(f"{tag} {phrase}" for phrase in picked)
        else:
            sampled.extend(picked)

    rng.shuffle(sampled)
    return "; ".join(sampled)
