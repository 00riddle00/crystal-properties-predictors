import torch
from torcheval.metrics.functional import mean_squared_error, r2_score


def top_1_acc(output, target):
    return top_k_acc(output, target, k=1)


def top_3_acc(output, target):
    return top_k_acc(output, target, k=3)


def top_k_acc(output, target, k):
    pred = torch.topk(output, k, dim=1)[1]
    assert pred.shape[0] == len(target)
    correct = 0.0
    for i in range(k):
        correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def mae_score(output, target):
    _output = torch.flatten(torch.transpose(output, 0, 1))
    _target = torch.flatten(torch.transpose(target, 0, 1))
    mean_absolute_error = torch.nn.L1Loss()
    return mean_absolute_error(_output, _target)


def mse_score(output, target):
    _output = torch.flatten(torch.transpose(output, 0, 1))
    _target = torch.flatten(torch.transpose(target, 0, 1))
    return mean_squared_error(_output, _target)


def r_2_score(output, target):
    _output = torch.flatten(torch.transpose(output, 0, 1))
    _target = torch.flatten(torch.transpose(target, 0, 1))
    return r2_score(_output, _target)
