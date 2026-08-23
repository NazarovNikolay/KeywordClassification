"""Text encoders that turn keyword-strings into feature matrices."""

from __future__ import annotations

from typing import Sequence

from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer

from .base import Encoder


class TfidfEncoder(BaseEstimator, Encoder):
    """Hashable bag-of-n-grams encoder. Each forest head typically clones it."""

    shared = False

    def __init__(
        self,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 1,
        max_features: int | None = 50_000,
    ):
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_features = max_features
        self.vectorizer_ = TfidfVectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            max_features=max_features,
            token_pattern=r"(?u)\S+",
        )

    def fit(self, texts: Sequence[str]) -> TfidfEncoder:
        self.vectorizer_.fit(texts)
        return self

    def encode(self, texts: Sequence[str], batch_size: int = 64):
        return self.vectorizer_.transform(list(texts))


class BertEncoder(Encoder):
    """Frozen HuggingFace encoder; CLS pooling, same as the original KRSB notebooks."""

    shared = True

    def __init__(self, model_name_or_path: str, max_length: int = 128, device: str | None = None):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError("BertEncoder requires 'torch' and 'transformers'") from exc

        self.model_name_or_path = model_name_or_path
        self.max_length = max_length
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def encode(self, texts: Sequence[str], batch_size: int = 64):
        import numpy as np
        from tqdm.auto import tqdm

        torch = self._torch
        embs = []
        texts = list(texts)
        ranges = range(0, len(texts), batch_size)
        for start in tqdm(ranges, desc="encode", leave=False):
            batch = texts[start : start + batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                out = self.model(**enc)
            cls = out.last_hidden_state[:, 0, :]
            embs.append(cls.detach().cpu().numpy())
        return np.vstack(embs) if embs else np.zeros((0, 0), dtype=np.float32)
