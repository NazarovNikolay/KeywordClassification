# Keyword Classification

Библиотека ансамблей для **классификации текстов по ключевым словам**. Документ представляется набором keyphrases от
нескольких экстракторов; дальше эти представления комбинируются в ансамбль.

## Методы

| Метод                  | Описание                                                                                  |
|------------------------|-------------------------------------------------------------------------------------------|
| [KRSB](KRSB/METHOD.md) | Keyword Random Subspace Bagging: лес голов над случайными подпространствами ключевых фраз |

## Быстрый старт

```bash
pip install -e ".[extractors]"
python examples/krsb_20newsgroups.py
```

Пример на 20 Newsgroups: YAKE + RAKE + TopicRank, датасет скачивается через scikit-learn. Тот же сценарий — в
`examples/krsb_20newsgroups.ipynb`.

## Лицензия

Apache License 2.0.
