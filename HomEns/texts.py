"""Сборка keyword-текста из фраз одного экстрактора (без подпространства)."""

from __future__ import annotations

from typing import Sequence

from KRSB.sampling import ensure_keyword_list


def method_keywords_to_text(
    phrases: Sequence[str] | str | None,
    tag: str = "[KW]",
    add_method_tags: bool = True,
) -> str:
    """Склеить фразы одного метода в keyword-текст для головы HomEns.

    Берутся все фразы столбца, опционально с тегом источника, через ``"; "``.
    Подпространства нет: разнообразие даёт сам экстрактор, а не сэмплер.
    """
    keywords = ensure_keyword_list(phrases)
    if not keywords:
        return "[EMPTY]"
    if add_method_tags:
        parts = [f"{tag} {phrase}" for phrase in keywords]
    else:
        parts = keywords
    return "; ".join(parts)
