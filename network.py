import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class DenseNet(nn.Module):
    def __init__(self, input_size):
        super(DenseNet, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(input_size, 96)
        self.fc2 = nn.Linear(96, 96)
        self.fc3 = nn.Linear(96, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
