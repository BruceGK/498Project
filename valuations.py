from abc import ABC
from typing import Any

import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Module):
    def __init__(self):
        super(Linear, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(6000, 3)
        )

    def forward(self, x):
        y = self.layers(x)
        return y


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(6000, 8000),
            nn.ReLU(),
            nn.Linear(8000, 8000),
            nn.ReLU(),
            nn.Linear(8000, 8000),
            nn.ReLU(),
            nn.Linear(8000, 8000),
            nn.ReLU(),
            nn.Linear(8000, 3)
        )

    def forward(self, x):
        y = self.layers(x)
        return y
