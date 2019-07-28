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
                
    def forward(self, x, prev_output):
        vec_x = torch.matmul(x, self.mask)
        output = self.act(vec_x + 0.8 * prev_output)
        return output


class QDFRCell(nn.Module):
    def __init__(self, n_hidden=10):
        super(QDFRCell, self).__init__()
        self.act = torch.nn.ReLU()
        self.mask = torch.nn.Parameter(data=torch.Tensor(1, n_hidden), requires_grad=False)
        nn.init.uniform_(self.mask, a=0.0, b=1.0)
        self.register_buffer('in_min', torch.zeros(1))
        self.register_buffer('in_max', torch.ones(1))
        self.register_buffer('l1_min', torch.zeros(1))
        self.register_buffer('l1_max', torch.zeros(1))
        self.register_buffer('out_min', torch.zeros(1))
        self.register_buffer('out_max', torch.zeros(1))
                
    def forward(self, x, prev_output):
        #update min and max of input
        #self.in_min = 0.99 * self.in_min + 0.01 * torch.min(x)
        #self.in_max = 0.99 * self.in_max + 0.01 * torch.max(x)
        x = Quantize(x, self.in_min, self.in_max)
        q_mask = Quantize(self.mask)
        vec_x = torch.matmul(x, q_mask)
        #update mask_out min and max
        self.l1_min = 0.99 * self.l1_min + 0.01 * torch.min(vec_x)
        self.l1_max = 0.99 * self.l1_max + 0.01 * torch.max(vec_x)
        q_vec_x = Quantize(vec_x, self.l1_min, self.l1_max)
        output = self.act(q_vec_x + 0.8 * prev_output)
        #update output min and max
        self.out_min = 0.99 * self.out_min + 0.01 * torch.min(output)
        self.out_max = 0.99 * self.out_max + 0.01 * torch.max(output)
        #print(self.in_min, self.in_max, self.l1_min, self.l1_max, self.out_min, self.out_max)
        output = Quantize(output, self.out_min, self.out_max)
        return output


