"""Тесты HomEns: веса, keyword-текст одного метода и гомогенный ансамбль."""

from __future__ import annotations

import unittest

import numpy as np

from HomEns import HomEns, compute_weights, method_keywords_to_text, soft_weighted_proba
from KRSB.bank import KeywordBank
from KRSB.base import KeywordExtractor
from KRSB.encoders import TfidfEncoder


class DummyExtractor(KeywordExtractor):
    """Детерминированный экстрактор: оставляет только слова из заданного словаря."""

    def __init__(self, name: str, tag: str, vocab: list[str]):
        self.name = name
        self.tag = tag
        self.vocab = vocab

    def extract(self, text: str, top_n: int = 15) -> list[str]:
        tokens = [tok.lower() for tok in text.split() if tok.isalpha()]
        picked = [tok for tok in tokens if tok in self.vocab]
        return picked[:top_n]


class WeightingTests(unittest.TestCase):
    def test_compute_weights_prefers_higher_f1(self):
        weights = compute_weights(
            ["yake", "rake"],
            {"yake": 0.9, "rake": 0.1},
        )
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=6)
        self.assertGreater(weights["yake"], weights["rake"])

    def test_soft_weighted_proba(self):
        proba = {
            "a": np.array([[1.0, 0.0], [0.0, 1.0]]),
            "b": np.array([[0.0, 1.0], [1.0, 0.0]]),
        }
        out = soft_weighted_proba(proba, {"a": 0.75, "b": 0.25}, ["a", "b"])
        np.testing.assert_allclose(out[0], [0.75, 0.25])

    def test_method_keywords_to_text_tags_and_empty(self):
        text = method_keywords_to_text(["orbit", "nasa"], tag="[YAKE]")
        self.assertEqual(text, "[YAKE] orbit; [YAKE] nasa")
        self.assertEqual(method_keywords_to_text([]), "[EMPTY]")


class HomEnsTests(unittest.TestCase):
    def setUp(self):
        self.texts = [
            "nasa launches a rover to mars planet orbit",
            "the patient received medicine in the hospital clinic",
            "the car engine and brakes failed on the highway",
            "nasa satellite orbit around planet mars",
            "hospital clinic treats patient with medicine",
            "car engine repair on the highway brakes",
        ]
        self.y = np.array([0, 1, 2, 0, 1, 2])
        self.extractors = [
            DummyExtractor("space", "[SP]", ["nasa", "mars", "orbit", "planet", "satellite"]),
            DummyExtractor("med", "[MD]", ["hospital", "patient", "medicine", "clinic"]),
            DummyExtractor("auto", "[AU]", ["car", "engine", "brakes", "highway"]),
        ]

    def test_fit_predict_on_texts(self):
        model = HomEns(
            encoder=TfidfEncoder(),
            extractors=self.extractors,
            val_size=0.0,
            seed=0,
        )
        model.fit(self.texts, self.y)
        proba = model.predict_proba(self.texts)
        pred = model.predict(self.texts)
        self.assertEqual(proba.shape, (6, 3))
        self.assertEqual(pred.shape, (6,))
        self.assertTrue(np.allclose(proba.sum(axis=1), 1.0, atol=1e-6))
        self.assertEqual(len(model.heads), 3)
        self.assertAlmostEqual(sum(model.weights_.values()), 1.0, places=6)
        self.assertGreaterEqual((pred == self.y).mean(), 0.5)

    def test_fit_from_bank_and_explicit_val(self):
        bank = KeywordBank.from_extractors(
            self.texts, self.extractors, top_n=4, show_progress=False
        )
        train_bank = bank.subset([0, 1, 2, 3])
        val_bank = bank.subset([4, 5])
        model = HomEns(encoder=TfidfEncoder(), val_size=0.0, seed=1)
        model.fit(train_bank, self.y[:4], X_val=val_bank, y_val=self.y[4:])
        pred = model.predict(bank)
        self.assertEqual(len(pred), 6)
        self.assertTrue(all(head.val_f1 >= 0.0 for head in model.heads))

    def test_evaluate_combinations(self):
        bank = KeywordBank.from_extractors(
            self.texts, self.extractors, top_n=4, show_progress=False
        )
        model = HomEns(encoder=TfidfEncoder(), val_size=0.0, seed=0)
        model.fit(bank, self.y)
        table = model.evaluate_combinations(bank, self.y)
        self.assertEqual(len(table), 7)
        self.assertIn("f1_macro", table.columns)
        self.assertEqual(table.iloc[0]["f1_macro"], table["f1_macro"].max())


if __name__ == "__main__":
    unittest.main()
