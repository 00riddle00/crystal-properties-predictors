import torch
import torch.nn as nn
from torcheval.metrics.functional import mean_squared_error, r2_score


def top_1_acc(output: torch.Tensor, target: torch.Tensor) -> float:
    return top_k_acc(output, target, k=1)


def top_3_acc(output: torch.Tensor, target: torch.Tensor) -> float:
    return top_k_acc(output, target, k=3)


def top_k_acc(output: torch.Tensor, target: torch.Tensor, k: int) -> float:
    pred: torch.Tensor = torch.topk(output, k, dim=1)[1]
    assert pred.shape[0] == len(target)
    correct: float = 0.0  # but it will always take on whole positive number values
    i: int
    for i in range(k):
        correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def mae_score(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    _output: torch.Tensor = torch.flatten(torch.transpose(output, 0, 1))
    _target: torch.Tensor = torch.flatten(torch.transpose(target, 0, 1))
    mean_absolute_error: nn.L1Loss = nn.L1Loss()
    return mean_absolute_error(_output, _target)


def mse_score(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    _output: torch.Tensor = torch.flatten(torch.transpose(output, 0, 1))
    _target: torch.Tensor = torch.flatten(torch.transpose(target, 0, 1))
    return mean_squared_error(_output, _target)


def r_2_score(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    _output: torch.Tensor = torch.flatten(torch.transpose(output, 0, 1))
    _target: torch.Tensor = torch.flatten(torch.transpose(target, 0, 1))
    return r2_score(_output, _target)
