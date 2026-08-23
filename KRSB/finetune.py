"""Опциональное дообучение HuggingFace-энкодера на keyword-текстах.

Соответствует этапу ``finetune_encoder`` в исходных ноутбуках KRSB:
на каждой эпохе ``epoch_salt`` меняет подпространство фраз, теги методов
добавляются в tokenizer как special tokens, затем веса сохраняются и
могут быть загружены в :class:`~KRSB.encoders.BertEncoder`.
"""

from __future__ import annotations

import random

import numpy as np

try:
    from torch.utils.data import Dataset
except ImportError:  # torch нужен только для этого модуля
    class Dataset:  # type: ignore[no-redef]
        pass

from .bank import KeywordBank
from .sampling import sample_keywords_for_row


class KWFinetuneDataset(Dataset):
    """Dataset, который при каждом ``__getitem__`` заново собирает keyword-текст.

    ``epoch_salt`` сдвигает RNG, поэтому на разных эпохах голова видит
    другое подпространство методов и фраз.
    """

    def __init__(
        self,
        bank: KeywordBank,
        labels,
        tokenizer,
        base_seed: int,
        methods_per_model: int = 4,
        total_k: int = 40,
        max_length: int = 128,
        add_method_tags: bool = True,
    ):
        self.bank = bank
        self.labels = np.asarray(labels)
        self.tokenizer = tokenizer
        self.base_seed = base_seed
        self.methods_per_model = methods_per_model
        self.total_k = total_k
        self.max_length = max_length
        self.add_method_tags = add_method_tags
        self.epoch_salt = 0

    def set_epoch(self, epoch: int) -> None:
        """Сменить соль подпространства перед новой эпохой."""
        self.epoch_salt = int(epoch)

    def __len__(self) -> int:
        return len(self.bank)

    def __getitem__(self, index: int):
        rng = random.Random(self.base_seed * 1_000_003 + index * 1_000_033 + self.epoch_salt * 9_999_937)
        text = sample_keywords_for_row(
            keywords_by_method=self.bank.row(index),
            rng=rng,
            method_names=self.bank.method_names,
            methods_per_model=self.methods_per_model,
            total_k=self.total_k,
            add_method_tags=self.add_method_tags,
            method_tags=self.bank.tags,
        )
        encoded = self.tokenizer(text, truncation=True, max_length=self.max_length)
        encoded["labels"] = int(self.labels[index])
        return encoded


def finetune_encoder(
    bank: KeywordBank,
    y,
    val_bank: KeywordBank,
    y_val,
    num_labels: int,
    model_name: str = "distilbert-base-uncased",
    output_dir: str = "./ft_kw_encoder",
    seed: int = 42,
    epochs: int = 2,
    methods_per_model: int = 4,
    total_k: int = 40,
    max_length: int = 128,
):
    """Дообучить sequence classifier на сэмплированных keyword-текстах и сохранить encoder."""
    try:
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
        raise ImportError("finetune_encoder requires transformers and scikit-learn") from exc

    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    special = sorted(set(bank.tags.values()))
    tokenizer.add_special_tokens({"additional_special_tokens": special})

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.resize_token_embeddings(len(tokenizer))

    train_ds = KWFinetuneDataset(
        bank, y, tokenizer, base_seed=seed,
        methods_per_model=methods_per_model, total_k=total_k, max_length=max_length,
    )
    val_ds = KWFinetuneDataset(
        val_bank, y_val, tokenizer, base_seed=seed + 123,
        methods_per_model=methods_per_model, total_k=total_k, max_length=max_length,
    )

    class EpochShuffleTrainer(Trainer):
        def train(self, *args, **kwargs):
            for epoch in range(int(self.args.num_train_epochs)):
                train_ds.set_epoch(epoch)
                val_ds.set_epoch(epoch)
            return super().train(*args, **kwargs)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="macro"),
        }

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=100,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none",
    )
    trainer = EpochShuffleTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir
