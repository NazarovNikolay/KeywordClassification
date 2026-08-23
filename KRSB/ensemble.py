"""Keyword Random Subspace Bagging (KRSB) ensemble."""

from __future__ import annotations

import random
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from tqdm.auto import tqdm

from .bank import KeywordBank
from .base import Encoder, KeywordEnsemble, KeywordExtractor
from .sampling import sample_keywords_for_row


class KRSB(KeywordEnsemble):
    """Ensemble of lightweight heads over random keyword subspaces.

    Each estimator:

    1. Draws a bootstrap of documents.
    2. For every document, samples a random subset of extractors and
       ``total_k`` keyphrases (the original KRSB sampler).
    3. Encodes the resulting keyword-text.
    4. Fits a cheap classifier (logistic regression by default).

    Prediction averages class probabilities (soft voting). Extractor tags
    such as ``[YAKE]`` are kept so a neural encoder can see which method
    produced each phrase — same protocol as the SciBERT notebooks.
    """

    def __init__(
        self,
        encoder: Encoder | None = None,
        extractors: Sequence[KeywordExtractor] | None = None,
        n_estimators: int = 10,
        bootstrap_ratio: float = 1.0,
        methods_per_model: int = 4,
        total_k: int = 40,
        per_method_min: int = 2,
        add_method_tags: bool = True,
        top_n_keywords: int = 15,
        head_estimator=None,
        seed: int = 42,
        encode_batch_size: int = 64,
    ):
        self.encoder = encoder
        self.extractors = None if extractors is None else list(extractors)
        self.n_estimators = n_estimators
        self.bootstrap_ratio = bootstrap_ratio
        self.methods_per_model = methods_per_model
        self.total_k = total_k
        self.per_method_min = per_method_min
        self.add_method_tags = add_method_tags
        self.top_n_keywords = top_n_keywords
        self.head_estimator = head_estimator
        self.seed = seed
        self.encode_batch_size = encode_batch_size

    def fit(self, X, y):
        bank = self._as_bank(X, fit_extractors=True)
        y = np.asarray(y)
        if len(y) != bank.n_docs:
            raise ValueError("X and y must have the same number of documents")
        if self.encoder is None:
            from .encoders import TfidfEncoder

            self.encoder = TfidfEncoder()

        self.classes_ = np.unique(y)
        self._class_to_idx_ = {label: i for i, label in enumerate(self.classes_)}
        self.method_names_ = bank.method_names
        self.method_tags_ = dict(bank.tags)
        self.heads_ = []

        n = bank.n_docs
        indices = np.arange(n)
        template = self.head_estimator
        if template is None:
            template = LogisticRegression(max_iter=2000, C=5.0)

        for t in tqdm(range(self.n_estimators), desc="estimators"):
            est_seed = self.seed + t * 1009
            bs_n = max(1, int(n * self.bootstrap_ratio))
            bs_idx = np.random.RandomState(est_seed).choice(indices, size=bs_n, replace=True)
            boot = bank.subset(bs_idx)
            y_boot = y[bs_idx]
            texts = self._make_texts(boot, est_seed)
            encoder = self._encoder_for_head(texts)
            features = encoder.encode(texts, batch_size=self.encode_batch_size)
            clf = clone(template)
            clf.fit(features, y_boot)
            self.heads_.append((est_seed, clf, encoder))
        return self

    def predict_proba(self, X) -> NDArray[np.float64]:
        if not getattr(self, "heads_", None):
            raise RuntimeError("KRSB must be fitted before predict_proba")
        bank = self._as_bank(X, fit_extractors=False)
        n = bank.n_docs
        n_classes = len(self.classes_)
        proba_sum = np.zeros((n, n_classes), dtype=np.float64)
        for est_seed, clf, encoder in self.heads_:
            texts = self._make_texts(bank, est_seed)
            features = encoder.encode(texts, batch_size=self.encode_batch_size)
            proba_sum += self._align_proba(clf, clf.predict_proba(features))
        return proba_sum / len(self.heads_)

    # Alias used in the original notebooks.
    @property
    def heads(self):
        return getattr(self, "heads_", [])

    def _encoder_for_head(self, texts: Sequence[str]) -> Encoder:
        encoder = self.encoder
        if encoder is None:
            raise RuntimeError("encoder is not set")
        if getattr(encoder, "shared", False):
            return encoder.fit(texts)
        cloned = clone(encoder)
        return cloned.fit(texts)

    def _make_texts(self, bank: KeywordBank, est_seed: int) -> list[str]:
        texts: list[str] = []
        for i in range(len(bank)):
            rng = random.Random(est_seed * 1_000_003 + i * 1_000_033)
            text = sample_keywords_for_row(
                keywords_by_method=bank.row(i),
                rng=rng,
                method_names=self.method_names_,
                methods_per_model=self.methods_per_model,
                total_k=self.total_k,
                per_method_min=self.per_method_min,
                add_method_tags=self.add_method_tags,
                method_tags=self.method_tags_,
            )
            texts.append(text if text else "[EMPTY]")
        return texts

    def _align_proba(self, clf, proba) -> NDArray[np.float64]:
        aligned = np.zeros((proba.shape[0], len(self.classes_)), dtype=np.float64)
        for src, label in enumerate(clf.classes_):
            dst = self._class_to_idx_.get(label)
            if dst is not None:
                aligned[:, dst] = proba[:, src]
        return aligned

    def _as_bank(self, X, fit_extractors: bool) -> KeywordBank:
        if isinstance(X, KeywordBank):
            return X
        if not self.extractors:
            raise TypeError(
                "Pass a KeywordBank or provide extractors=... to fit/predict on raw texts"
            )
        extractors = list(self.extractors)
        if fit_extractors:
            for extractor in extractors:
                fit = getattr(extractor, "fit", None)
                if callable(fit):
                    fit(X)
        return KeywordBank.from_extractors(
            texts=list(X),
            extractors=extractors,
            top_n=self.top_n_keywords,
        )


KeywordForestHeads = KRSB
