from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
from overrides import override
from torch.nn import Parameter


class ModelBase(nn.Module):
    """Base class for all models."""

    @override
    def __init__(self):
        super().__init__()

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Forward pass logic.

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self) -> str:
        """Model prints with number of trainable parameters."""
        model_parameters: Iterator[Parameter] = filter(
            lambda p: p.requires_grad, self.parameters()
        )
        params: int = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + f"\nTrainable parameters: {params}"
