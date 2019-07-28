import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from Quantize import Quantize 

class QLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, num_bits=8):
        super(QLinear, self).__init__(in_features, out_features, bias)
        self.num_bits = 8

    def forward(self, x):
        W = Quantize(self.weight, num_bits=self.num_bits)
        output = F.linear(x, W)
        return output


