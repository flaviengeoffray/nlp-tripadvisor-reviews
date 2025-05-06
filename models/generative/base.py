from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

import numpy as np
from torch import Tensor
import torchmetrics

# import torch
# from torch import Tensor
from models.base import BaseModel


class BaseGenerativeModel(BaseModel, ABC):
    """
    Base class for generative models.
    """

    @abstractmethod
    def generate(self, input_data: Optional[Any] = None) -> Any:
        """
        Generate a sentence from an input data.
        """
        raise NotImplementedError

    def evaluate(
        self,
        X: Union[np.ndarray, Tensor],
        y: Union[np.ndarray, Tensor],
        y_pred: Union[np.ndarray, Tensor] = None,
    ) -> Dict[str, float]:
        metric = torchmetrics.CharErrorRate()
        cer = metric(y_pred, y)

        metric = torchmetrics.WordErrorRate()
        wer = metric(y_pred, y)

        metric = torchmetrics.BLEUScore()
        bleu = metric(y_pred, y)

        return {
            "cer": cer,
            "wer": wer,
            "bleu": bleu,
        }
