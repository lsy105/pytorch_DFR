import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from Quantize import Quantize 

class QLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, num_bits=8):
        super(QLinear, self).__init__(in_features, out_features, bias)
        self.num_bits = num_bits
        self.register_buffer('w_min_max', torch.zeros(2))
        self.w_min_max[0] = -1.0
        self.w_min_max[1] = 1.0

    def forward(self, x):
        W = Quantize(self.weight, min_data = self.w_min_max[0], max_data = self.w_min_max[1], \
                     num_bits = self.num_bits)
        output = F.linear(x, W)
        return output


