from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import evaluate
import numpy as np
from bert_score import BERTScorer
from torch import Tensor

from data.tripadvisor_dataset import extract_keywords
from models.base import BaseModel


class BaseGenerativeModel(BaseModel, ABC):
    """
    Base class for generative models.
    """

    def __init__(self, model_path: Path, **kwargs: Any) -> None:
        super().__init__(model_path, **kwargs)

        self._bleu: evaluate.EvaluationModule = evaluate.load("sacrebleu")
        self._rouge: evaluate.EvaluationModule = evaluate.load("rouge")
        self._bert_scorer: BERTScorer = BERTScorer(
            lang="en", rescale_with_baseline=True
        )

    @abstractmethod
    def generate(self, input_data: Optional[Any] = None) -> Any:
        """
        Generate a sentence from an input data.
        """
        raise NotImplementedError

    def evaluate(
        self,
        X: Union[np.ndarray, Tensor, List[Any]],
        y: Union[np.ndarray, Tensor, List[int]],
        y_pred: Optional[Union[np.ndarray, Tensor, List[int]]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate the model's performance.

        :param Union[np.ndarray, Tensor, List[Any]] X: Input data (list of sentences)
        :param Union[np.ndarray, Tensor, List[int]] y: True labels (list of target texts)
        :param Optional[Union[np.ndarray, Tensor, List[int]]] y_pred:
            Predicted labels (if already computed to avoid recomputation)
        :return Dict[str, float]: Dictionary with BLEU, ROUGE, and BERTScore metrics
        """
        if isinstance(X, Tensor):
            X = X.tolist()
        elif isinstance(X, np.ndarray):
            X = X.tolist()
        elif y_pred is None:
            raise ValueError(f"X is not supported: {type(X)}")

        if isinstance(y, (Tensor, np.ndarray)):
            targets = [str(v) for v in y.tolist()]
        elif isinstance(y, list):
            targets = y
        else:
            raise ValueError(f"y is not supported: {type(y)}")

        if y_pred is None:
            targets = X

            preds: List[str] = []
            for text, rating in zip(targets, y):
                keywords = extract_keywords(text)
                prompt = f"{rating}: {keywords}"
                out = self.generate(prompt)
                preds.extend(out)

        else:
            if isinstance(y_pred, (Tensor, np.ndarray)):
                preds = [str(v) for v in y_pred.tolist()]
            elif isinstance(y_pred, list):
                preds = y_pred
            else:
                raise ValueError(f"y_pred is not supported: {type(y_pred)}")

        bleu_res = self._bleu.compute(
            predictions=preds,
            references=[[t] for t in targets],
        )
        bleu_score = bleu_res["score"] / 100.0

        rouge_res = self._rouge.compute(predictions=preds, references=targets)
        rouge1 = rouge_res["rouge1"]
        rouge2 = rouge_res["rouge2"]
        rougeL = rouge_res["rougeL"]

        P, R, F1 = self._bert_scorer.score(preds, targets)
        bert_p = P.mean().item()
        bert_r = R.mean().item()
        bert_f1 = F1.mean().item()

        return {
            "bleu": bleu_score,
            "rouge1": rouge1,
            "rouge2": rouge2,
            "rougeL": rougeL,
            "bertscore_precision": bert_p,
            "bertscore_recall": bert_r,
            "bertscore_f1": bert_f1,
        }
