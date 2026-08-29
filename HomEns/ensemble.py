"""HomEns: гомогенный ансамбль — одна голова на каждый экстрактор ключевых слов."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from KRSB.bank import KeywordBank
from KRSB.base import Encoder, KeywordEnsemble, KeywordExtractor

from .texts import method_keywords_to_text
from .weighting import compute_weights, soft_weighted_proba


@dataclass
class HomEnsHead:
    """Базовый классификатор, обученный только на фразах одного экстрактора."""

    method: str
    classifier: Any
    encoder: Encoder
    val_f1: float
    weight: float = 0.0


class HomEns(KeywordEnsemble):
    """Гомогенный ансамбль: отдельная голова на каждый метод извлечения.

    Каждая голова:

    1. Берёт **все** ключевые фразы своего экстрактора (без подпространства).
    2. Собирает keyword-текст с тегом источника, как ``[YAKE] orbit; [YAKE] nasa``.
    3. Кодирует текст и учит дешёвый классификатор (по умолчанию логистическая регрессия).
    4. Получает вес ``w ∝ 1 / (1 - F1_val)``.

    Предсказание — взвешенная сумма вероятностей (soft weighted voting).
    """

    def __init__(
        self,
        encoder: Encoder | None = None,
        extractors: Sequence[KeywordExtractor] | None = None,
        methods: Sequence[str] | None = None,
        val_size: float = 0.15,
        add_method_tags: bool = True,
        top_n_keywords: int = 15,
        head_estimator=None,
        seed: int = 42,
        encode_batch_size: int = 64,
    ):
        self.encoder = encoder
        self.extractors = None if extractors is None else list(extractors)
        self.methods = None if methods is None else list(methods)
        self.val_size = val_size
        self.add_method_tags = add_method_tags
        self.top_n_keywords = top_n_keywords
        self.head_estimator = head_estimator
        self.seed = seed
        self.encode_batch_size = encode_batch_size

    def fit(self, X, y, X_val=None, y_val=None):
        """Обучить по одной голове на метод; веса считаются по валидации.

        ``X`` / ``X_val`` — сырые тексты или :class:`~KRSB.bank.KeywordBank`.
        Если ``X_val`` не задан и ``val_size > 0``, валидация отщепляется от train.
        """
        train_bank, y_train, val_bank, y_val_arr = self._prepare_splits(X, y, X_val, y_val)
        if self.encoder is None:
            from KRSB.encoders import TfidfEncoder

            self.encoder = TfidfEncoder()

        y_all = np.concatenate([y_train, y_val_arr])
        self.classes_ = np.unique(y_all)
        self._class_to_idx_ = {label: i for i, label in enumerate(self.classes_)}
        self.method_names_ = list(self.methods) if self.methods else list(train_bank.method_names)
        missing = [name for name in self.method_names_ if name not in train_bank.keywords]
        if missing:
            raise ValueError(f"Unknown methods for this KeywordBank: {missing}")
        self.method_tags_ = dict(train_bank.tags)

        template = self.head_estimator
        if template is None:
            template = LogisticRegression(max_iter=2000, C=5.0)

        heads: list[HomEnsHead] = []
        val_f1_by_method: dict[str, float] = {}
        for method in tqdm(self.method_names_, desc="homens heads"):
            train_texts = self._method_texts(train_bank, method)
            val_texts = self._method_texts(val_bank, method)
            encoder = self._encoder_for_head(train_texts)
            features = encoder.encode(train_texts, batch_size=self.encode_batch_size)
            clf = clone(template)
            clf.fit(features, y_train)
            val_features = encoder.encode(val_texts, batch_size=self.encode_batch_size)
            val_proba = self._align_proba(clf, clf.predict_proba(val_features))
            val_pred = self.classes_[np.argmax(val_proba, axis=1)]
            val_f1 = float(f1_score(y_val_arr, val_pred, average="macro", zero_division=0))
            val_f1_by_method[method] = val_f1
            heads.append(
                HomEnsHead(
                    method=method,
                    classifier=clf,
                    encoder=encoder,
                    val_f1=val_f1,
                )
            )

        weights = compute_weights(self.method_names_, val_f1_by_method)
        for head in heads:
            head.weight = weights[head.method]
        self.heads_ = heads
        self.weights_ = weights
        return self

    def predict_proba(self, X) -> NDArray[np.float64]:
        """Взвешенная сумма вероятностей голов; каждая голова видит только свой метод."""
        if not getattr(self, "heads_", None):
            raise RuntimeError("HomEns must be fitted before predict_proba")
        bank = self._as_bank(X, fit_extractors=False)
        proba_by_method = {
            head.method: self._head_proba(head, bank) for head in self.heads_
        }
        return soft_weighted_proba(proba_by_method, self.weights_, self.method_names_)

    def evaluate_combinations(self, X, y) -> pd.DataFrame:
        """Перебрать все непустые подмножества голов и пересчитать веса."""
        if not getattr(self, "heads_", None):
            raise RuntimeError("HomEns must be fitted before evaluate_combinations")
        bank = self._as_bank(X, fit_extractors=False)
        y = np.asarray(y)
        proba_by_method = {
            head.method: self._head_proba(head, bank) for head in self.heads_
        }
        val_f1_by_method = {head.method: head.val_f1 for head in self.heads_}
        rows = []
        methods = list(self.method_names_)
        for rank in range(1, len(methods) + 1):
            for combo in itertools.combinations(methods, rank):
                combo_methods = list(combo)
                weights = compute_weights(combo_methods, val_f1_by_method)
                subset_proba = {name: proba_by_method[name] for name in combo_methods}
                ensemble_proba = soft_weighted_proba(subset_proba, weights, combo_methods)
                pred = self.classes_[np.argmax(ensemble_proba, axis=1)]
                row = {
                    "methods": "+".join(combo_methods),
                    "n_methods": len(combo_methods),
                    "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
                    "accuracy": float(accuracy_score(y, pred)),
                }
                for name in combo_methods:
                    row[f"w_{name}"] = weights[name]
                    row[f"val_f1_{name}"] = val_f1_by_method[name]
                rows.append(row)
        return pd.DataFrame(rows).sort_values("f1_macro", ascending=False).reset_index(drop=True)

    @property
    def heads(self) -> list[HomEnsHead]:
        return getattr(self, "heads_", [])

    def _head_proba(self, head: HomEnsHead, bank: KeywordBank) -> NDArray[np.float64]:
        texts = self._method_texts(bank, head.method)
        features = head.encoder.encode(texts, batch_size=self.encode_batch_size)
        return self._align_proba(head.classifier, head.classifier.predict_proba(features))

    def _method_texts(self, bank: KeywordBank, method: str) -> list[str]:
        tag = self.method_tags_.get(method, bank.tags.get(method, "[KW]"))
        return [
            method_keywords_to_text(
                bank.row(i)[method],
                tag=tag,
                add_method_tags=self.add_method_tags,
            )
            for i in range(len(bank))
        ]

    def _encoder_for_head(self, texts: Sequence[str]) -> Encoder:
        encoder = self.encoder
        if encoder is None:
            raise RuntimeError("encoder is not set")
        if getattr(encoder, "shared", False):
            return encoder.fit(texts)
        cloned = clone(encoder)
        return cloned.fit(texts)

    def _align_proba(self, clf, proba) -> NDArray[np.float64]:
        aligned = np.zeros((proba.shape[0], len(self.classes_)), dtype=np.float64)
        for src, label in enumerate(clf.classes_):
            dst = self._class_to_idx_.get(label)
            if dst is not None:
                aligned[:, dst] = proba[:, src]
        return aligned

    def _prepare_splits(self, X, y, X_val, y_val):
        y = np.asarray(y)
        if X_val is not None:
            if y_val is None:
                raise ValueError("y_val is required when X_val is provided")
            train_bank = self._as_bank(X, fit_extractors=True)
            val_bank = self._as_bank(X_val, fit_extractors=False)
            y_val_arr = np.asarray(y_val)
            if len(y) != train_bank.n_docs or len(y_val_arr) != val_bank.n_docs:
                raise ValueError("X/y and X_val/y_val must have matching document counts")
            return train_bank, y, val_bank, y_val_arr

        if self.val_size and self.val_size > 0:
            if isinstance(X, KeywordBank):
                if len(y) != X.n_docs:
                    raise ValueError("X and y must have the same number of documents")
                train_idx, val_idx = self._split_indices(np.arange(X.n_docs), y)
                return X.subset(train_idx), y[train_idx], X.subset(val_idx), y[val_idx]
            texts = list(X)
            if len(y) != len(texts):
                raise ValueError("X and y must have the same number of documents")
            x_train, x_val, y_train, y_val_arr = self._split_xy(texts, y)
            train_bank = self._as_bank(x_train, fit_extractors=True)
            val_bank = self._as_bank(x_val, fit_extractors=False)
            return train_bank, y_train, val_bank, y_val_arr

        bank = self._as_bank(X, fit_extractors=True)
        if len(y) != bank.n_docs:
            raise ValueError("X and y must have the same number of documents")
        return bank, y, bank, y

    def _split_indices(self, indices: NDArray, y: NDArray) -> tuple[NDArray, NDArray]:
        try:
            return train_test_split(
                indices, test_size=self.val_size, random_state=self.seed, stratify=y
            )
        except ValueError:
            return train_test_split(indices, test_size=self.val_size, random_state=self.seed)

    def _split_xy(self, texts: list[str], y: NDArray):
        try:
            return train_test_split(
                texts, y, test_size=self.val_size, random_state=self.seed, stratify=y
            )
        except ValueError:
            return train_test_split(texts, y, test_size=self.val_size, random_state=self.seed)

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
