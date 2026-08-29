"""Сэмплирование случайного подпространства ключевых фраз для одной головы KRSB."""

from __future__ import annotations

import random
from typing import Mapping, Sequence


def ensure_keyword_list(value) -> list[str]:
    """Привести ячейку (list / строка / NaN) к списку непустых фраз."""
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
    """Убрать дубликаты без учёта регистра, сохранив исходный порядок (ранг)."""
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
    """Собрать keyword-текст одного документа для конкретной головы.

    1. Перемешать имена экстракторов и оставить ``methods_per_model``.
    2. Дедуплицировать пул каждого метода, сохранив порядок.
    3. Разделить бюджет ``total_k`` по выбранным методам (не меньше ``per_method_min``).
    4. Сэмплировать без возвращения из пула каждого метода.
    5. При необходимости префиксовать фразы тегом источника, например ``[YAKE]``.
    6. Перемешать смешанный список и склеить через ``"; "``.
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
