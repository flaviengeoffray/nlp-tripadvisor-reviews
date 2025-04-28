from abc import ABC, abstractmethod
from typing import Any, Optional

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
