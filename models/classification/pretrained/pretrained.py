from pathlib import Path
from typing import Any, List, Union
import numpy as np
import torch
from datasets import Dataset, ClassLabel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)

from models.classification.base import BaseClassificationModel


class PretrainedClassifier(BaseClassificationModel):

    def __init__(self, model_path: Path, **kwargs: Any) -> None:

        super().__init__(model_path)

        self.model_checkpoint: str = kwargs.pop(
            "model_checkpoint", "distilbert-base-uncased"
        )
        self.epochs: int = kwargs.pop("epochs", 3)
        self.batch_size: int = kwargs.pop("batch_size", 8)
        self.patience: int = kwargs.pop("patience", 2)
        self.device: str = kwargs.pop("device", "cuda")
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
            self.model_checkpoint
        )
        self.model: AutoModelForSequenceClassification = (
            AutoModelForSequenceClassification.from_pretrained(
                self.model_checkpoint, num_labels=5
            )
        )
        self.model.to(self.device)
        self.model.eval()

    def fit(self, X_train: Any, y_train: Any, X_val: Any, y_val: Any) -> None:

        random_seed: int = 42
        class_label = ClassLabel(num_classes=5)

        train_data = Dataset.from_dict(
            {"review": X_train.tolist(), "labels": (y_train - 1).tolist()}
        )
        eval_data = Dataset.from_dict(
            {"review": X_val.tolist(), "labels": (y_val - 1).tolist()}
        )
        train_data = train_data.cast_column("labels", class_label)
        eval_data = eval_data.cast_column("labels", class_label)

        def tokenize_batch(batch):
            return self.tokenizer(
                batch["review"], padding=True, truncation=True, max_length=512
            )

        train_data = train_data.map(
            tokenize_batch, batched=True, remove_columns=["review"]
        )
        eval_data = eval_data.map(
            tokenize_batch, batched=True, remove_columns=["review"]
        )

        train_data.set_format(
            type="torch", columns=["input_ids", "attention_mask", "labels"]
        )
        eval_data.set_format(
            type="torch", columns=["input_ids", "attention_mask", "labels"]
        )

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=1)
            return self.evaluate(None, labels, preds)

        training_args = TrainingArguments(
            output_dir=self.model_path / "classifier_output",
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            eval_strategy="epoch",
            logging_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            seed=random_seed,
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            greater_is_better=True,
        )

        def collate_without_token_type(batch):
            batch = DataCollatorWithPadding(self.tokenizer)(batch)
            batch.pop("token_type_ids", None)
            return batch

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=eval_data,
            tokenizer=self.tokenizer,
            data_collator=collate_without_token_type,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=self.patience)],
        )

        print(f"Model is on device: {next(self.model.parameters()).device}")

        self.model.train()
        trainer.train()
        self.model.eval()

        metrics = trainer.evaluate()
        print(f"Validation: {metrics}")
        self.save(self.model_path / "checkpoint")

    def predict(self, texts: Union[str, List[str], np.ndarray]) -> List[int]:

        if hasattr(texts, "tolist"):
            texts = texts.tolist()

        if isinstance(texts, str):
            texts = [texts]

        batch_size = 32
        all_preds = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            self.model.eval()
            with torch.no_grad():
                out = self.model(**enc)
                logits = out.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds.tolist())

            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        ratings = [int(p) + 1 for p in all_preds]

        return ratings

    def save(self, save_path: str) -> None:
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    def load(self, load_path: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(load_path)
        self.model.eval()
