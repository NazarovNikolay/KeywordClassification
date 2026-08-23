"""Keyword extractors used to build the views of a KRSB forest."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Iterable, Sequence

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from .base import KeywordExtractor

_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9'+-]*")
_MAX_CHARS = 4000


def _truncate(text: str, max_chars: int = _MAX_CHARS) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _tokens(text: str) -> list[str]:
    return [tok.lower() for tok in _TOKEN_RE.findall(text)]


class YakeExtractor(KeywordExtractor):
    """YAKE statistical keyphrase extractor."""

    name = "yake"
    tag = "[YAKE]"

    def __init__(self, language: str = "en", ngram: int = 3, window_size: int = 1):
        self.language = language
        self.ngram = ngram
        self.window_size = window_size
        self._extractor = None

    def _get_extractor(self, top_n: int):
        try:
            import yake
        except ImportError as exc:
            raise ImportError("YakeExtractor requires the 'yake' package") from exc
        return yake.KeywordExtractor(
            lan=self.language,
            n=self.ngram,
            dedupLim=0.9,
            windowsSize=self.window_size,
            top=top_n,
        )

    def extract(self, text: str, top_n: int = 15) -> list[str]:
        text = _truncate(text)
        if not text:
            return []
        extractor = self._get_extractor(top_n)
        # YAKE returns (phrase, score) with lower score = more important.
        scored = extractor.extract_keywords(text)
        return [phrase for phrase, _score in scored[:top_n]]


class RakeExtractor(KeywordExtractor):
    """RAKE keyphrase extractor (rake-nltk)."""

    name = "rake"
    tag = "[RAKE]"

    def __init__(self, language: str = "english"):
        self.language = language

    @staticmethod
    def _ensure_nltk():
        import nltk

        for resource, path in (
            ("stopwords", "corpora/stopwords"),
            ("punkt", "tokenizers/punkt"),
            ("punkt_tab", "tokenizers/punkt_tab"),
        ):
            try:
                nltk.data.find(path)
            except LookupError:
                nltk.download(resource, quiet=True)

    def extract(self, text: str, top_n: int = 15) -> list[str]:
        try:
            from rake_nltk import Rake
        except ImportError as exc:
            raise ImportError("RakeExtractor requires the 'rake-nltk' package") from exc
        text = _truncate(text)
        if not text:
            return []
        self._ensure_nltk()
        rake = Rake(language=self.language)
        rake.extract_keywords_from_text(text)
        phrases = rake.get_ranked_phrases()
        seen: set[str] = set()
        out: list[str] = []
        for phrase in phrases:
            cleaned = " ".join(phrase.split())
            key = cleaned.lower()
            if not cleaned or key in seen:
                continue
            seen.add(key)
            out.append(cleaned)
            if len(out) >= top_n:
                break
        return out


class TopicRankExtractor(KeywordExtractor):
    """Simplified TopicRank: cluster n-gram candidates and rank topics.

    If ``pke`` is installed the original TopicRank implementation is used.
    Otherwise a lightweight graph fallback (token n-grams + PageRank) runs
    with no extra dependencies beyond scikit-learn.
    """

    name = "topicrank"
    tag = "[TR]"

    def __init__(
        self,
        ngram_range: tuple[int, int] = (1, 3),
        cluster_threshold: float = 0.25,
        use_pke: bool = False,
    ):
        self.ngram_range = ngram_range
        self.cluster_threshold = cluster_threshold
        self.use_pke = use_pke

    def extract(self, text: str, top_n: int = 15) -> list[str]:
        text = _truncate(text)
        if not text:
            return []
        if self.use_pke:
            phrases = self._extract_pke(text, top_n)
            if phrases is not None:
                return phrases
        return self._extract_fallback(text, top_n)

    def _extract_pke(self, text: str, top_n: int) -> list[str] | None:
        try:
            import pke
        except ImportError:
            return None
        extractor = pke.unsupervised.TopicRank()
        extractor.load_document(input=text, language="en")
        extractor.candidate_selection()
        extractor.candidate_weighting()
        return [phrase for phrase, _score in extractor.get_n_best(n=top_n)]

    def _extract_fallback(self, text: str, top_n: int) -> list[str]:
        tokens = [tok for tok in _tokens(text) if tok not in ENGLISH_STOP_WORDS and len(tok) > 2]
        if not tokens:
            return []
        min_n, max_n = self.ngram_range
        freq: dict[str, int] = {}
        order: list[str] = []
        for n in range(min_n, max_n + 1):
            for i in range(len(tokens) - n + 1):
                phrase = " ".join(tokens[i : i + n])
                if phrase not in freq:
                    order.append(phrase)
                freq[phrase] = freq.get(phrase, 0) + 1
        # Cap the graph size so TopicRank stays usable on long 20 Newsgroups posts.
        ranked = sorted(order, key=lambda p: (-freq[p], len(p.split()), p))
        candidates = ranked[:80]
        if not candidates:
            return []

        clusters = _cluster_by_jaccard(candidates, self.cluster_threshold)
        positions = {phrase: i for i, phrase in enumerate(candidates)}
        n_topics = len(clusters)
        weights = [[0.0] * n_topics for _ in range(n_topics)]
        for i, left in enumerate(clusters):
            for j, right in enumerate(clusters):
                if i >= j:
                    continue
                dist = _topic_distance(left, right, positions)
                if dist <= 0:
                    continue
                w = 1.0 / dist
                weights[i][j] = w
                weights[j][i] = w
        ranks = _pagerank(weights)
        order = sorted(range(n_topics), key=lambda i: ranks[i], reverse=True)
        phrases: list[str] = []
        for topic_i in order:
            representative = min(clusters[topic_i], key=lambda p: positions.get(p, math.inf))
            phrases.append(representative)
            if len(phrases) >= top_n:
                break
        return phrases


class TfidfKeywordExtractor(KeywordExtractor):
    """Corpus-level TF-IDF n-grams: useful as a third cheap view of a document."""

    name = "tfidf"
    tag = "[TFIDF]"

    def __init__(self, ngram_range: tuple[int, int] = (1, 2), max_features: int = 50_000):
        self.ngram_range = ngram_range
        self.max_features = max_features
        self._vectorizer: TfidfVectorizer | None = None
        self._fitted_on: int | None = None

    def fit(self, texts: Sequence[str]) -> TfidfKeywordExtractor:
        self._vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            stop_words="english",
            min_df=1,
        )
        self._vectorizer.fit(texts)
        self._fitted_on = len(texts)
        return self

    def extract(self, text: str, top_n: int = 15) -> list[str]:
        if self._vectorizer is None:
            # Single-document fallback: treat the text as a tiny corpus.
            self.fit([text, text])
        assert self._vectorizer is not None
        matrix = self._vectorizer.transform([_truncate(text)])
        if matrix.nnz == 0:
            return []
        row = matrix.tocsr()
        indices = row.indices
        data = row.data
        order = data.argsort()[::-1][:top_n]
        names = self._vectorizer.get_feature_names_out()
        return [str(names[indices[i]]) for i in order]

    def extract_many(
        self,
        texts: Sequence[str],
        top_n: int = 15,
        show_progress: bool = True,
    ) -> list[list[str]]:
        if self._vectorizer is None:
            self.fit(texts)
            self.fit(texts)
        assert self._vectorizer is not None
        matrix = self._vectorizer.transform(texts)
        names = self._vectorizer.get_feature_names_out()
        out: list[list[str]] = []
        csr = matrix.tocsr()
        for i in range(csr.shape[0]):
            start, end = csr.indptr[i], csr.indptr[i + 1]
            if start == end:
                out.append([])
                continue
            data = csr.data[start:end]
            indices = csr.indices[start:end]
            order = data.argsort()[::-1][:top_n]
            out.append([str(names[indices[j]]) for j in order])
        return out


def _cluster_by_jaccard(phrases: Sequence[str], threshold: float) -> list[list[str]]:
    """Greedy average-linkage style clustering on token Jaccard overlap."""
    token_sets = [set(p.split()) for p in phrases]
    parent = list(range(len(phrases)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(phrases)):
        for j in range(i + 1, len(phrases)):
            a, b = token_sets[i], token_sets[j]
            union = len(a | b)
            if union == 0:
                continue
            if len(a & b) / union >= threshold:
                pi, pj = find(i), find(j)
                if pi != pj:
                    parent[pj] = pi
    buckets: dict[int, list[str]] = defaultdict(list)
    for i, phrase in enumerate(phrases):
        buckets[find(i)].append(phrase)
    return list(buckets.values())


def _topic_distance(left: Iterable[str], right: Iterable[str], positions: dict[str, int]) -> float:
    best = math.inf
    for a in left:
        pa = positions.get(a)
        if pa is None:
            continue
        for b in right:
            pb = positions.get(b)
            if pb is None:
                continue
            best = min(best, abs(pa - pb) + 1)
    return best if best is not math.inf else 0.0


def _pagerank(weights: list[list[float]], damping: float = 0.85, iters: int = 30) -> list[float]:
    n = len(weights)
    if n == 0:
        return []
    rank = [1.0 / n] * n
    out_sum = [sum(row) for row in weights]
    for _ in range(iters):
        new = [(1.0 - damping) / n] * n
        for j in range(n):
            if out_sum[j] == 0:
                share = damping * rank[j] / n
                for i in range(n):
                    new[i] += share
                continue
            for i in range(n):
                if weights[j][i]:
                    new[i] += damping * rank[j] * weights[j][i] / out_sum[j]
        rank = new
    return rank
