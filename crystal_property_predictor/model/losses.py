import torch
import torch.nn.functional as F


def nll_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.nll_loss(output, target)


def mse_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(output, target)
