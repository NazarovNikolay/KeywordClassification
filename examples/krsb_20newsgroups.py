"""KRSB on 20 Newsgroups with YAKE, RAKE and TopicRank.

Downloads the dataset automatically via scikit-learn, extracts keyphrases
with three unsupervised methods, then fits Keyword Random Subspace Bagging
(a forest of logistic heads over random keyword subspaces).

Run from the repository root::

    pip install -e .[extractors]
    python examples/krsb_20newsgroups.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from KRSB import KRSB, RakeExtractor, TfidfEncoder, TopicRankExtractor, YakeExtractor
from KRSB.bank import KeywordBank

CATEGORIES = [
    "sci.space",
    "sci.med",
    "rec.autos",
    "talk.politics.misc",
]
SAMPLES_PER_CLASS = 80
RANDOM_STATE = 42


def load_subset():
    data = fetch_20newsgroups(
        subset="all",
        categories=CATEGORIES,
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    texts, labels = [], []
    counts = {i: 0 for i in range(len(CATEGORIES))}
    for text, label in zip(data.data, data.target):
        if counts[label] >= SAMPLES_PER_CLASS:
            continue
        cleaned = " ".join(text.split())
        if len(cleaned) < 80:
            continue
        texts.append(cleaned)
        labels.append(label)
        counts[label] += 1
        if all(v >= SAMPLES_PER_CLASS for v in counts.values()):
            break
    return texts, labels, list(data.target_names)


def main():
    texts, labels, target_names = load_subset()
    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    print(f"train={len(x_train)} test={len(x_test)} classes={target_names}")

    extractors = [
        YakeExtractor(ngram=2),
        RakeExtractor(),
        TopicRankExtractor(),
    ]
    print("Extracting keywords for the training split...")
    train_bank = KeywordBank.from_extractors(x_train, extractors, top_n=12)
    print("Extracting keywords for the test split...")
    test_bank = KeywordBank.from_extractors(x_test, extractors, top_n=12)

    print("Train document 0 / YAKE:", train_bank.row(0)["yake"][:5])
    print("Train document 0 / RAKE:", train_bank.row(0)["rake"][:5])
    print("Train document 0 / TopicRank:", train_bank.row(0)["topicrank"][:5])

    model = KRSB(
        encoder=TfidfEncoder(),
        n_estimators=8,
        methods_per_model=2,
        total_k=20,
        per_method_min=2,
        seed=RANDOM_STATE,
    )
    model.fit(train_bank, y_train)
    pred = model.predict(test_bank)
    print(classification_report(y_test, pred, target_names=target_names, digits=3))


if __name__ == "__main__":
    main()
