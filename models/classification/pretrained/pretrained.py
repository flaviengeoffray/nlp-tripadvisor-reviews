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
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
            self.model_checkpoint
        )
        self.model: AutoModelForSequenceClassification = (
            AutoModelForSequenceClassification.from_pretrained(
                self.model_checkpoint, num_labels=5
            )
        )
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

    def predict(self, texts: Union[str, List[str]]) -> Union[int, List[int]]:

        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False

        # Tokenize input texts and move to model device
        encodings = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        )
        encodings = {k: v.to(self.model.device) for k, v in encodings.items()}
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**encodings)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()

        ratings = (preds + 1).tolist()
        return ratings[0] if single_input else ratings

    def save(self, save_path: str) -> None:
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    def load(self, load_path: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(load_path)
        self.model.eval()
