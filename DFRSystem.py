import torch
import numpy as np
import torch.nn as nn
from DFRCell import DFRCell, QDFRCell
from Quantize import Quantize
from Linear import QLinear

class DFRSystem(nn.Module):
    def __init__(self, n_hidden=10):
        super(DFRSystem, self).__init__()
        #self.fc1 = QLinear(n_hidden, 1, bias=False, num_bits=8)
        self.fc1 = nn.Linear(n_hidden, 1, bias=False)
        self.DFRCell = DFRCell(n_hidden)
                
    def forward(self, x, prev_out):
        cell_out = self.DFRCell(x, prev_out)
        output = self.fc1(cell_out)
        return output, cell_out
