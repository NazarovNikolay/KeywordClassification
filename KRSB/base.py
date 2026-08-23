"""Shared abstractions for keyword-based classification ensembles."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin


class KeywordExtractor(ABC):
    """Turns a document into an ordered list of keyphrases.

    Concrete extractors (YAKE, RAKE, TopicRank, LLM prompts, ...) only need
    to implement :meth:`extract`. Batch extraction has a default loop that
    subclasses may override for vectorization or caching.
    """

    name: str
    tag: str

    @abstractmethod
    def extract(self, text: str, top_n: int = 15) -> list[str]:
        """Return up to ``top_n`` keyphrases for a single document."""

    def extract_many(
        self,
        texts: Sequence[str],
        top_n: int = 15,
        show_progress: bool = True,
    ) -> list[list[str]]:
        iterator: Sequence[str] = texts
        if show_progress:
            try:
                from tqdm.auto import tqdm

                iterator = tqdm(texts, desc=self.name, leave=False)
            except ImportError:
                iterator = texts
        return [self.extract(text, top_n=top_n) for text in iterator]


class Encoder(ABC):
    """Maps keyword-texts (or any strings) to a feature matrix.

    Frozen neural encoders set ``shared=True`` so the ensemble reuses one
    copy. Fitted vectorizers set ``shared=False`` so each forest head gets
    its own clone and vocabulary.
    """

    shared: bool = False

    def fit(self, texts: Sequence[str]) -> Encoder:
        return self

    @abstractmethod
    def encode(self, texts: Sequence[str], batch_size: int = 64):
        """Return a 2-D array or sparse matrix of shape ``(n_texts, n_features)``."""


class KeywordEnsemble(ClassifierMixin, BaseEstimator, ABC):
    """Sklearn-style classifier that votes over keyword views of documents."""

    @abstractmethod
    def fit(self, X, y):
        """Fit on raw texts (``Sequence[str]``) or a :class:`~KRSB.bank.KeywordBank`."""

    @abstractmethod
    def predict_proba(self, X) -> NDArray[np.float64]:
        """Soft votes, shape ``(n_samples, n_classes)``."""

    def predict(self, X) -> NDArray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
