"""Таблица ключевых фраз: один столбец на экстрактор, одна строка на документ."""

from __future__ import annotations

from typing import Sequence

from tqdm.auto import tqdm

from .base import KeywordExtractor
from .sampling import ensure_keyword_list


class KeywordBank:
    """Выровненные списки keyphrases для ``n`` документов и нескольких методов.

    Это общий вход для ансамблей: имя метода → список длины ``n_docs``,
    внутри — ранжированные фразы конкретного документа.
    """

    def __init__(
        self,
        keywords: dict[str, list[list[str]]],
        tags: dict[str, str] | None = None,
        texts: Sequence[str] | None = None,
    ):
        if not keywords:
            raise ValueError("KeywordBank needs at least one extractor column")
        lengths = {name: len(values) for name, values in keywords.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Extractor columns have different lengths: {lengths}")
        self.keywords = {
            name: [ensure_keyword_list(row) for row in rows]
            for name, rows in keywords.items()
        }
        self.tags = dict(tags or {name: f"[{name.upper()}]" for name in self.keywords})
        self.texts = None if texts is None else list(texts)
        if self.texts is not None and len(self.texts) != self.n_docs:
            raise ValueError("texts length must match keyword columns")

    @property
    def method_names(self) -> list[str]:
        """Имена экстракторов в порядке столбцов."""
        return list(self.keywords)

    @property
    def n_docs(self) -> int:
        """Число документов (длина любого столбца)."""
        first = next(iter(self.keywords.values()))
        return len(first)

    def __len__(self) -> int:
        return self.n_docs

    def row(self, index: int) -> dict[str, list[str]]:
        """Срез всех методов для одного документа: ``{method: [phrases...]}``."""
        return {name: values[index] for name, values in self.keywords.items()}

    def subset(self, indices: Sequence[int]) -> KeywordBank:
        """Новый банк по списку индексов (с повторами — для bootstrap)."""
        idx = list(indices)
        keywords = {name: [rows[i] for i in idx] for name, rows in self.keywords.items()}
        texts = None if self.texts is None else [self.texts[i] for i in idx]
        return KeywordBank(keywords=keywords, tags=self.tags, texts=texts)

    @classmethod
    def from_extractors(
        cls,
        texts: Sequence[str],
        extractors: Sequence[KeywordExtractor],
        top_n: int = 15,
        show_progress: bool = True,
    ) -> KeywordBank:
        """Прогнать экстракторы по корпусу и собрать банк."""
        if not extractors:
            raise ValueError("Provide at least one KeywordExtractor")
        keywords: dict[str, list[list[str]]] = {}
        tags: dict[str, str] = {}
        iterator = extractors
        if show_progress:
            iterator = tqdm(extractors, desc="extractors")
        for extractor in iterator:
            name = extractor.name
            tags[name] = extractor.tag
            keywords[name] = extractor.extract_many(
                texts, top_n=top_n, show_progress=show_progress
            )
        return cls(keywords=keywords, tags=tags, texts=texts)
