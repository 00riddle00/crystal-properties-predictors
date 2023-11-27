import torch
import torch.nn as nn
import torch.nn.functional as F
from overrides import override

from crystal_property_predictor.base import ModelBase
from crystal_property_predictor.utils import setup_logger

log = setup_logger(__name__)


class MnistModel(ModelBase):
    @override
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

        log.info(f"<init>: \n{self}")

    @override
    def forward(self, input_):
        x = F.relu(F.max_pool2d(self.conv1(input_), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class CrystalPropModel(ModelBase):
    """A neural network with two linear layers."""

    @override
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.layer_1: nn.Linear = nn.Linear(
            in_features=input_dim, out_features=input_dim // 2, bias=True
        )
        self.layer_2: nn.Linear = nn.Linear(
            in_features=input_dim // 2, out_features=1, bias=True
        )
        nn.init.kaiming_uniform_(self.layer_1.weight)
        nn.init.kaiming_uniform_(self.layer_2.weight)

    @override
    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        res1: torch.Tensor = F.relu(self.layer_1(input_))
        res2: torch.Tensor = F.relu(self.layer_2(res1))
        res = F.relu(res2)
        return res
