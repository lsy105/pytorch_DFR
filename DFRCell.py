import torch
import numpy as np
import torch.nn as nn
from Quantize import Quantize 

class DFRCell(nn.Module):
    def __init__(self, n_hidden=10):
        super(DFRCell, self).__init__()
        self.act = torch.nn.ReLU()
        self.mask = torch.nn.Parameter(data=torch.Tensor(1, n_hidden), requires_grad=False)
        nn.init.uniform_(self.mask, a=0.0, b=1.0)
        self.mask = Quantize(self.mask)
                
    def forward(self, x, prev_output):
        x = Quantize(x)
        vec_x = torch.matmul(x, self.mask) 
        output = self.act(vec_x + 0.8 * prev_output)
        output = Quantize(output)
        return output


