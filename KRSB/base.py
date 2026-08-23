"""Общие абстракции для ансамблей классификации по ключевым словам."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin


class KeywordExtractor(ABC):
    """Экстрактор ключевых фраз: документ → упорядоченный список keyphrases.

    Достаточно реализовать :meth:`extract`. Пакетный :meth:`extract_many`
    по умолчанию просто вызывает его в цикле; его можно переопределить,
    если метод умеет работать сразу по корпусу (например TF-IDF).
    """

    name: str
    tag: str

    @abstractmethod
    def extract(self, text: str, top_n: int = 15) -> list[str]:
        """Вернуть не более ``top_n`` ключевых фраз одного документа."""

    def extract_many(
        self,
        texts: Sequence[str],
        top_n: int = 15,
        show_progress: bool = True,
    ) -> list[list[str]]:
        """Извлечь ключевые фразы для списка документов."""
        iterator: Sequence[str] = texts
        if show_progress:
            try:
                from tqdm.auto import tqdm

                iterator = tqdm(texts, desc=self.name, leave=False)
            except ImportError:
                iterator = texts
        return [self.extract(text, top_n=top_n) for text in iterator]


class Encoder(ABC):
    """Преобразует keyword-тексты (или любые строки) в матрицу признаков.

    Замороженный нейросетевой энкодер помечают ``shared=True`` — лес
    переиспользует один экземпляр. Векторизаторы с обучением словаря
    оставляют ``shared=False``: каждая голова получает свой clone.
    """

    shared: bool = False

    def fit(self, texts: Sequence[str]) -> Encoder:
        """Подогнать энкодер по текстам. Для замороженных моделей — no-op."""
        return self

    @abstractmethod
    def encode(self, texts: Sequence[str], batch_size: int = 64):
        """Матрица ``(n_texts, n_features)`` — плотная ndarray или sparse."""


class KeywordEnsemble(ClassifierMixin, BaseEstimator, ABC):
    """Sklearn-классификатор, который голосует по keyword-представлениям документов."""

    @abstractmethod
    def fit(self, X, y):
        """Обучить на сырых текстах или на :class:`~KRSB.bank.KeywordBank`."""

    @abstractmethod
    def predict_proba(self, X) -> NDArray[np.float64]:
        """Мягкое голосование, форма ``(n_samples, n_classes)``."""

    def predict(self, X) -> NDArray:
        """Метка класса с максимальной усреднённой вероятностью."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
