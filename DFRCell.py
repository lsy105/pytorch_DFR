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
        self.register_buffer('l1_min_max', torch.zeros(2))
                
    def forward(self, x, prev_output):
        vec_x = torch.matmul(x, self.mask)
        output = self.act(vec_x + 0.8 * prev_output)
        self.l1_min_max[0] = torch.min(self.l1_min_max[0], torch.min(output))
        self.l1_min_max[1] = torch.max(self.l1_min_max[1], torch.max(output))
        return output


class QDFRCell(nn.Module):
    def __init__(self, n_hidden=10):
        super(QDFRCell, self).__init__()
        self.act = torch.nn.ReLU()
        self.mask = torch.nn.Parameter(data=torch.Tensor(1, n_hidden), requires_grad=False)
        nn.init.uniform_(self.mask, a=0.0, b=1.0)
        self.register_buffer('in_min_max', torch.zeros(2))
        self.in_min_max[1] = 0.497 
        self.register_buffer('l1_min_max', torch.zeros(2))
        self.l1_min_max[1] = 2.0 
        self.register_buffer('maskout_min_max', torch.zeros(2))
        self.maskout_min_max[1] = 2.0 
                
    def forward(self, x, prev_output):
        #update min and max of input
        x = Quantize(x, self.in_min_max[0], self.in_min_max[1])
        q_mask = Quantize(self.mask)
        vec_x = torch.matmul(x, q_mask)
        q_vec_x = Quantize(vec_x, self.maskout_min_max[0], self.maskout_min_max[1])
        bias = Quantize(0.2 * prev_output, self.l1_min_max[0], self.l1_min_max[1])
        output = self.act(q_vec_x + prev_output - bias)
        output = Quantize(output, self.l1_min_max[0], self.l1_min_max[1])
        return output


