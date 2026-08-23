# Keyword Classification

Библиотека ансамблей для **классификации текстов по ключевым словам**. Документ представляется набором keyphrases от
нескольких экстракторов; дальше эти представления комбинируются в ансамбль.

## Методы

| Метод                      | Описание                                                                                          |
|----------------------------|---------------------------------------------------------------------------------------------------|
| [KRSB](KRSB/METHOD.md)     | Keyword Random Subspace Bagging: лес голов над случайными подпространствами ключевых фраз         |
| [HomEns](HomEns/METHOD.md) | Homogeneous Ensemble: одна голова на экстрактор, веса по val F1, soft weighted voting             |

Общие абстракции (`KeywordExtractor`, `Encoder`, `KeywordBank`, `KeywordEnsemble`) живут в `KRSB` и
переиспользуются ансамблями.

## Быстрый старт

```bash
pip install -e ".[extractors]"
python examples/krsb_20newsgroups.py
python examples/homens_20newsgroups.py
```

Примеры на 20 Newsgroups: YAKE + RAKE + TopicRank, датасет скачивается через scikit-learn. Те же сценарии — в
`examples/krsb_20newsgroups.ipynb` и `examples/homens_20newsgroups.ipynb`.

## Лицензия

Apache License 2.0.
