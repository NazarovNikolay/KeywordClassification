"""KRSB — Keyword Random Subspace Bagging.

Ансамбль классификации текстов по ключевым словам: несколько экстракторов
строят представления документа, каждая голова леса берёт bootstrap документов
и случайное подпространство методов, кодирует получившийся keyword-текст
и голосует вероятностями класса.
"""

from .bank import KeywordBank
from .base import Encoder, KeywordEnsemble, KeywordExtractor
from .encoders import BertEncoder, TfidfEncoder
from .ensemble import KRSB, KeywordForestHeads
from .extractors import RakeExtractor, TfidfKeywordExtractor, TopicRankExtractor, YakeExtractor

__all__ = [
    "BertEncoder",
    "Encoder",
    "KRSB",
    "KeywordBank",
    "KeywordEnsemble",
    "KeywordExtractor",
    "KeywordForestHeads",
    "RakeExtractor",
    "TfidfEncoder",
    "TfidfKeywordExtractor",
    "TopicRankExtractor",
    "YakeExtractor",
]
