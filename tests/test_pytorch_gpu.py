import torch


def test_pytorch_gpu():
    dev = torch.device("cuda:0")
    x = torch.zeros(3, 4, device=dev)
    x += 1
    print(x)
    pass
