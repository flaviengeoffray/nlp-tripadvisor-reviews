from pathlib import Path
from typing import Any
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)

from models.generative.base import BaseGenerativeModel


class TripAdvisorReviewGenerator(BaseGenerativeModel):

    def __init__(self, model_path: Path, **kwargs: Any) -> None:
        super().__init__(model_path)

        self.model_checkpoint: str = kwargs.pop("model_checkpoint", "t5-base")
        self.epochs: int = kwargs.pop("epochs", 3)
        self.batch_size: int = kwargs.pop("batch_size", 8)
        self.patience: int = kwargs.pop("patience", 2)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_checkpoint)
        self.model.eval()

    def fit(
        self,
        X_train: Any,
        y_train: Any,
        X_val: Any,
        y_val: Any,
        # self,
        # epochs: int = 3,
        # batch_size: int = 8,
        # split_ratio: float = 0.1,
        # random_seed: int = 42,
    ):
        """
        Fine-tune the seq2seq model on the TripAdvisor review dataset.
        Uses "rating: X keywords: Y" as input and the review text as output for training.
        """
        # Load dataset
        from datasets import load_dataset

        dataset = load_dataset("jniimi/tripadvisor-review-rating")
        data = dataset["train"]

        # Keep only relevant columns and prepare input/target texts
        data = data.remove_columns(
            [
                col
                for col in data.column_names
                if col not in ["overall", "title", "text"]
            ]
        )

        # Define a function to create the input and target strings for each example
        def make_input_target(example):
            rating = int(example["overall"])
            keywords = example["title"] if example["title"] is not None else ""
            # Construct the instruction: e.g., "rating: 5 keywords: great service, clean rooms"
            instruction = f"rating: {rating} keywords: {keywords}"
            return {"input_text": instruction, "target_text": str(example["text"])}

        data = data.map(make_input_target, remove_columns=["overall", "title", "text"])

        # Split into training and validation sets
        data_split = data.train_test_split(test_size=split_ratio, seed=random_seed)
        train_data = data_split["train"]
        eval_data = data_split["test"]

        # Tokenize inputs and targets
        max_input_length = 128
        max_target_length = 256

        def tokenize_batch(batch):
            # Tokenize the input instruction
            model_inputs = self.tokenizer(
                batch["input_text"], max_length=max_input_length, truncation=True
            )
            # Tokenize the target review text
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
        # (Note: we don't set_format to torch here; we'll use a data collator to handle dynamic padding)

        # Set up Trainer with seq2seq data collator
        training_args = TrainingArguments(
            output_dir="generator_output",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            save_strategy="no",
            predict_with_generate=True,  # enable generation for evaluation if needed
            seed=random_seed,
        )
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=eval_data,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            # Note: no compute_metrics during training to avoid slowdown (we'll use evaluate() separately)
        )

        # Fine-tune the model
        self.model.train()
        trainer.train()
        self.model.eval()

        # Evaluate on the validation set (compute loss only, since no compute_metrics in Trainer)
        val_metrics = trainer.evaluate()
        print(f"Validation loss: {val_metrics.get('eval_loss'):.4f}")
        # Store trainer and eval_data for later use in generation or evaluation
        self.eval_data = eval_data

    def generate(self, inputs, max_length: int = 100):
        """
        Generate review text from an instruction input.
        `inputs` can be a single string or a list of strings, each in the format "rating: X keywords: Y".
        Returns the generated review text(s).
        """
        if isinstance(inputs, str):
            inputs = [inputs]
            single_input = True
        else:
            single_input = False

        # Tokenize the input instruction(s)
        encodings = self.tokenizer(
            inputs, return_tensors="pt", padding=True, truncation=True
        )
        encodings = {k: v.to(self.model.device) for k, v in encodings.items()}
        self.model.eval()
        # Generate output sequences
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=encodings["input_ids"],
                attention_mask=encodings.get("attention_mask"),
                max_length=max_length,
            )
        # Decode the generated tokens to text
        generated_texts = [
            self.tokenizer.decode(seq, skip_special_tokens=True) for seq in outputs
        ]
        return generated_texts[0] if single_input else generated_texts

    def evaluate(self, test_dataset=None):
        """
        Evaluate the model on a test dataset (or the held-out validation set if none provided).
        Generates reviews for each input and computes BLEU, WER, and CER metrics against the reference texts.
        """
        # Lazy import of evaluation metrics
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
            ).to(self.model.device)
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
        self.model.eval()
