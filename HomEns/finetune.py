"""Опциональное дообучение HuggingFace-классификатора на фразах одного метода.

На каждый экстрактор — свой sequence classifier; тег метода добавляется
в tokenizer как special token.
"""

from __future__ import annotations

import numpy as np

try:
    from torch.utils.data import Dataset
except ImportError:  # torch нужен только для этого модуля
    class Dataset:  # type: ignore[no-redef]
        pass

from KRSB.bank import KeywordBank

from .texts import method_keywords_to_text


class SingleMethodKWDataset(Dataset):
    """Dataset: документ → keyword-текст **одного** экстрактора."""

    def __init__(
        self,
        bank: KeywordBank,
        labels,
        tokenizer,
        method: str,
        max_length: int = 128,
        add_method_tags: bool = True,
    ):
        self.bank = bank
        self.labels = np.asarray(labels)
        self.tokenizer = tokenizer
        self.method = method
        self.max_length = max_length
        self.add_method_tags = add_method_tags
        self.tag = bank.tags.get(method, "[KW]")

    def __len__(self) -> int:
        return len(self.bank)

    def __getitem__(self, index: int):
        text = method_keywords_to_text(
            self.bank.row(index)[self.method],
            tag=self.tag,
            add_method_tags=self.add_method_tags,
        )
        encoded = self.tokenizer(text, truncation=True, max_length=self.max_length)
        encoded["labels"] = int(self.labels[index])
        return encoded


def train_base_classifier(
    bank: KeywordBank,
    y,
    val_bank: KeywordBank,
    y_val,
    method: str,
    num_labels: int,
    model_name: str = "distilbert-base-uncased",
    output_dir: str = "./homens_runs",
    seed: int = 42,
    epochs: int = 3,
    batch_size: int = 16,
    max_length: int = 128,
    learning_rate: float = 2e-5,
):
    """Дообучить sequence classifier на keyword-текстах одного метода.

    Возвращает ``(trainer, tokenizer, val_f1)``.
    """
    try:
        import torch
        from sklearn.metrics import accuracy_score, f1_score
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        raise ImportError("train_base_classifier requires torch, transformers and scikit-learn") from exc

    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tag = bank.tags.get(method, "[KW]")
    tokenizer.add_special_tokens({"additional_special_tokens": [tag]})

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.resize_token_embeddings(len(tokenizer))

    train_ds = SingleMethodKWDataset(bank, y, tokenizer, method, max_length=max_length)
    val_ds = SingleMethodKWDataset(val_bank, y_val, tokenizer, method, max_length=max_length)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="macro"),
        }

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()

    out = trainer.predict(val_ds)
    val_pred = np.argmax(out.predictions, axis=1)
    val_f1 = float(f1_score(np.asarray(y_val), val_pred, average="macro"))
    return trainer, tokenizer, val_f1
