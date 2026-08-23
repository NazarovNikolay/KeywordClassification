"""Тесты сэмплера, KeywordBank и леса KRSB на синтетических документах."""

from __future__ import annotations

import unittest

import numpy as np

from KRSB.bank import KeywordBank
from KRSB.base import KeywordExtractor
from KRSB.encoders import TfidfEncoder
from KRSB.ensemble import KRSB
from KRSB.extractors import TfidfKeywordExtractor, TopicRankExtractor
from KRSB.sampling import sample_keywords_for_row


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


class SamplingTests(unittest.TestCase):
    def test_sample_respects_tags_and_budget(self):
        import random

        rng = random.Random(0)
        text = sample_keywords_for_row(
            keywords_by_method={
                "yake": ["solar panel", "orbit", "nasa"],
                "rake": ["space shuttle", "launch"],
                "topicrank": ["mars", "rover", "planet"],
            },
            rng=rng,
            method_names=["yake", "rake", "topicrank"],
            methods_per_model=2,
            total_k=4,
            per_method_min=1,
            method_tags={"yake": "[YAKE]", "rake": "[RAKE]", "topicrank": "[TR]"},
        )
        self.assertTrue(text)
        self.assertIn("[", text)
        parts = [p.strip() for p in text.split(";") if p.strip()]
        self.assertLessEqual(len(parts), 4)


class BankAndEnsembleTests(unittest.TestCase):
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

    def test_keyword_bank_from_extractors(self):
        bank = KeywordBank.from_extractors(
            self.texts,
            [
                DummyExtractor("a", "[A]", ["nasa", "mars", "orbit", "planet"]),
                DummyExtractor("b", "[B]", ["hospital", "patient", "medicine", "clinic"]),
            ],
            top_n=4,
            show_progress=False,
        )
        self.assertEqual(bank.n_docs, 6)
        self.assertEqual(bank.method_names, ["a", "b"])
        self.assertEqual(len(bank.row(0)["a"]), 4)

    def test_krsb_fit_predict_on_texts(self):
        model = KRSB(
            encoder=TfidfEncoder(),
            extractors=[
                DummyExtractor("space", "[SP]", ["nasa", "mars", "orbit", "planet", "satellite"]),
                DummyExtractor("med", "[MD]", ["hospital", "patient", "medicine", "clinic"]),
                DummyExtractor("auto", "[AU]", ["car", "engine", "brakes", "highway"]),
            ],
            n_estimators=3,
            methods_per_model=2,
            total_k=6,
            seed=0,
        )
        model.fit(self.texts, self.y)
        proba = model.predict_proba(self.texts)
        pred = model.predict(self.texts)
        self.assertEqual(proba.shape, (6, 3))
        self.assertEqual(pred.shape, (6,))
        self.assertTrue(np.allclose(proba.sum(axis=1), 1.0, atol=1e-6))
        self.assertGreaterEqual((pred == self.y).mean(), 0.5)

    def test_krsb_from_precomputed_bank(self):
        bank = KeywordBank(
            keywords={
                "yake": [["nasa", "mars"], ["hospital", "clinic"], ["car", "engine"]] * 2,
                "rake": [["orbit"], ["medicine"], ["brakes"]] * 2,
            },
            tags={"yake": "[YAKE]", "rake": "[RAKE]"},
        )
        model = KRSB(
            encoder=TfidfEncoder(),
            n_estimators=2,
            methods_per_model=2,
            total_k=4,
            seed=1,
        )
        model.fit(bank, self.y)
        pred = model.predict(bank)
        self.assertEqual(len(pred), 6)

    def test_tfidf_and_topicrank_extractors_smoke(self):
        tfidf = TfidfKeywordExtractor()
        phrases = tfidf.extract_many(self.texts, top_n=5)
        self.assertEqual(len(phrases), 6)
        self.assertTrue(any(phrases))

        tr = TopicRankExtractor()
        got = tr.extract(self.texts[0], top_n=5)
        self.assertTrue(got)


if __name__ == "__main__":
    unittest.main()
