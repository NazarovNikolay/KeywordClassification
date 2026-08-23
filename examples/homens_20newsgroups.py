"""Пример HomEns на 20 Newsgroups: YAKE, RAKE и TopicRank.

Датасет скачивается через scikit-learn. На каждый экстрактор — своя голова,
веса считаются по macro-F1 на валидации, предсказание — soft weighted voting.

Запуск из корня репозитория::

    pip install -e .[extractors]
    python examples/homens_20newsgroups.py
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

from HomEns import HomEns
from KRSB import RakeExtractor, TfidfEncoder, TopicRankExtractor, YakeExtractor
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
    """Скачать 20 Newsgroups и взять по ``SAMPLES_PER_CLASS`` документов на класс."""
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
    x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )
    print(f"train={len(x_train)} val={len(x_val)} test={len(x_test)} classes={target_names}")

    extractors = [
        YakeExtractor(ngram=2),
        RakeExtractor(),
        TopicRankExtractor(),
    ]
    print("Extracting keywords for the training split...")
    train_bank = KeywordBank.from_extractors(x_train, extractors, top_n=12)
    print("Extracting keywords for the validation split...")
    val_bank = KeywordBank.from_extractors(x_val, extractors, top_n=12)
    print("Extracting keywords for the test split...")
    test_bank = KeywordBank.from_extractors(x_test, extractors, top_n=12)

    print("Train document 0 / YAKE:", train_bank.row(0)["yake"][:5])
    print("Train document 0 / RAKE:", train_bank.row(0)["rake"][:5])
    print("Train document 0 / TopicRank:", train_bank.row(0)["topicrank"][:5])

    model = HomEns(encoder=TfidfEncoder(), seed=RANDOM_STATE)
    model.fit(train_bank, y_train, X_val=val_bank, y_val=y_val)
    print("Head weights:", {head.method: round(head.weight, 3) for head in model.heads})
    print("Head val F1:", {head.method: round(head.val_f1, 3) for head in model.heads})

    pred = model.predict(test_bank)
    print(classification_report(y_test, pred, target_names=target_names, digits=3))

    combos = model.evaluate_combinations(test_bank, y_test)
    print("Best combination on test:")
    print(combos.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
