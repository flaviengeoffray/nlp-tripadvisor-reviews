from pathlib import Path
from typing import Any, List, Union
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)

from models.generative.base import BaseGenerativeModel
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class PretrainedGenerator(BaseGenerativeModel):
    def __init__(self, model_path: Path, **kwargs: Any) -> None:
        super().__init__(model_path)

        self.model_checkpoint: str = kwargs.pop("model_checkpoint", "t5-base")
        self.epochs: int = kwargs.pop("epochs", 3)
        self.batch_size: int = kwargs.pop("batch_size", 8)
        self.patience: int = kwargs.pop("patience", 2)
        requested_device: str = kwargs.pop("device", "cuda")

        if requested_device == "mps":
            if not torch.backends.mps.is_available():
                logger.warning(
                    "Warning: MPS device requested but not available. Using CPU instead."
                )
                self.device = "cpu"
            else:
                self.device = "mps"
                logger.info(
                    "Warning: MPS backend has limited memory. If you encounter OOM errors, set device='cpu'."
                )
        elif requested_device == "cuda":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_checkpoint)
        self.model.to(self.device)
        self.model.eval()

    def fit(
        self,
        X_train: Any,
        y_train: Any,
        X_val: Any,
        y_val: Any,
    ) -> None:
        random_seed: int = 42

        train_data = Dataset.from_dict(
            {
                "overall": y_train.tolist(),
                "title": [None] * len(X_train),
                "text": X_train.tolist(),
            }
        )
        eval_data = Dataset.from_dict(
            {
                "overall": y_val.tolist(),
                "title": [None] * len(X_val),
                "text": X_val.tolist(),
            }
        )

        def make_input_target(example):
            rating = int(example["overall"])
            keywords = example["title"] if example["title"] is not None else ""
            instruction = f"rating: {rating} keywords: {keywords}"
            return {"input_text": instruction, "target_text": str(example["text"])}

        train_data = train_data.map(
            make_input_target, remove_columns=["overall", "title", "text"]
        )
        eval_data = eval_data.map(
            make_input_target, remove_columns=["overall", "title", "text"]
        )

        max_input_length = 128
        max_target_length = 256

        def tokenize_batch(batch):
            model_inputs = self.tokenizer(
                batch["input_text"], max_length=max_input_length, truncation=True
            )
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    batch["target_text"], max_length=max_target_length, truncation=True
                )
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        train_data = train_data.map(
            tokenize_batch, batched=True, remove_columns=["input_text", "target_text"]
        )
        eval_data = eval_data.map(
            tokenize_batch, batched=True, remove_columns=["input_text", "target_text"]
        )

        # Set up Trainer with seq2seq data collator
        training_args = TrainingArguments(
            output_dir="generator_output",
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            save_strategy="no",
            predict_with_generate=True,
            seed=random_seed,
            dataloader_pin_memory=False if self.device == "mps" else True,
        )
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=eval_data,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

        self.model.train()
        trainer.train()
        self.model.eval()

        val_metrics = trainer.evaluate()
        logger.info(f"Validation loss: {val_metrics.get('eval_loss'):.4f}")
        self.eval_data = eval_data

    def generate(
        self, inputs: Union[str, List[str]], max_length: int = 100
    ) -> List[str]:
        if isinstance(inputs, str):
            inputs = [inputs]
            single_input = True
        else:
            single_input = False

        encodings = self.tokenizer(
            inputs, return_tensors="pt", padding=True, truncation=True
        )
        encodings = {k: v.to(self.device) for k, v in encodings.items()}

        self.model.eval()

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=encodings["input_ids"],
                attention_mask=encodings.get("attention_mask"),
                max_length=max_length,
            )

        generated_texts = [
            self.tokenizer.decode(seq, skip_special_tokens=True) for seq in outputs
        ]
        return generated_texts[0] if single_input else generated_texts

    def evaluate(self, test_dataset=None):
        # TODO: Use evaluate of parent

        import evaluate

        bleu_metric = evaluate.load("sacrebleu")
        wer_metric = evaluate.load("wer")
        cer_metric = evaluate.load("cer")

        # Use provided test dataset or the stored eval_data
        ds = (
            test_dataset
            if test_dataset is not None
            else getattr(self, "eval_data", None)
        )
        if ds is None:
            raise ValueError(
                "No dataset available for evaluation. Provide a test_dataset or run fit() first."
            )
        # If external test dataset is given, prepare it to have input_text and target_text
        if test_dataset is not None:
            ds = ds.remove_columns(
                [
                    col
                    for col in ds.column_names
                    if col not in ["overall", "title", "text"]
                ]
            )
            ds = ds.map(
                lambda ex: {
                    "input_text": f"rating: {int(ex['overall'])} keywords: {ex['title'] if ex['title'] else ''}",
                    "target_text": str(ex["text"]),
                },
                remove_columns=["overall", "title", "text"],
            )

        # Get lists of input instructions and reference texts
        inputs = ds["input_text"]
        references = ds["target_text"]
        # Generate predictions in batches to manage memory
        self.model.eval()
        predictions = []
        batch_size = 8
        for i in range(0, len(inputs), batch_size):
            batch_inputs = inputs[i : i + batch_size]
            # Tokenize batch of instructions
            enc = self.tokenizer(
                batch_inputs, return_tensors="pt", padding=True, truncation=True
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                outs = self.model.generate(
                    input_ids=enc["input_ids"],
                    attention_mask=enc.get("attention_mask"),
                    max_length=256,
                )
            batch_preds = [
                self.tokenizer.decode(seq, skip_special_tokens=True) for seq in outs
            ]
            predictions.extend(batch_preds)

        # Compute evaluation metrics
        bleu_score = bleu_metric.compute(
            predictions=predictions, references=[[ref] for ref in references]
        )["score"]
        wer_score = wer_metric.compute(predictions=predictions, references=references)
        cer_score = cer_metric.compute(predictions=predictions, references=references)
        return {"BLEU": bleu_score, "WER": wer_score, "CER": cer_score}

    def save(self, save_path: str) -> None:
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    def load(self, load_path: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(load_path)
        self.model.to(self.device)
        self.model.eval()
