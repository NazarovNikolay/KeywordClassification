"""HomEns — Homogeneous Ensemble of keyword classifiers.

Гомогенный ансамбль: на каждый экстрактор ключевых слов — своя голова,
затем мягкое голосование с весами по macro-F1 на валидации.
"""

from KRSB.bank import KeywordBank
from KRSB.base import Encoder, KeywordEnsemble, KeywordExtractor

from .ensemble import HomEns, HomEnsHead
from .texts import method_keywords_to_text
from .weighting import compute_weights, soft_weighted_proba

__all__ = [
    "Encoder",
    "HomEns",
    "HomEnsHead",
    "KeywordBank",
    "KeywordEnsemble",
    "KeywordExtractor",
    "compute_weights",
    "method_keywords_to_text",
    "soft_weighted_proba",
]
