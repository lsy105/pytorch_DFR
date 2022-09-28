import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from Quantize import * 

class QLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, num_bits=8):
        super(QLinear, self).__init__(in_features, out_features, bias)
        self.num_bits = 8
        self.weight_q = torch.quantization.FakeQuantize(observer=torch.quantization.observer.MovingAverageMinMaxObserver,
                                                        quant_min=0, quant_max=255)
    def forward(self, x):
        W = self.weight_q(self.weight) 
        output = F.linear(x, W)
        return output


class FixedQLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, num_bits=8):
        super(FixedQLinear, self).__init__(in_features, out_features, bias)
        nn.init.xavier_uniform_(self.weight)
        self.register_buffer('W_min_max', torch.zeros(2))
        self.W_min_max[0] = -1
        self.W_min_max[1] = 1
        #nn.init.uniform_(self.weight, -, 0.01)
        self.num_bits = 8
        self.act = nn.Softsign()
        
    def forward(self, x):
        #W = self.act(self.weight)
        #W = LogQuantize(W, self.W_min_max[0], self.W_min_max[1])
        W = Quantize(self.weight)
        #print(new_weight, W)
        output = F.linear(x, W)
        return output


