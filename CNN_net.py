import torch
import numpy as np
import torch.nn as nn
from DFRCell import *
from Quantize import Quantize
from Linear import *

class CNNSystem(nn.Module):
    def __init__(self, time_step=10):
        super(CNNSystem, self).__init__()
        self.l1 = nn.Conv1d(1, 16, 3, stride=1, padding=1) 
        self.a1 = nn.ReLU()
        self.l2 = nn.Conv1d(16, 32, 3, stride=1, padding=1)
        self.a2 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(time_step * 32, 2, bias=False)
                
    def forward(self, x):
        output = self.l1(x)
        output = self.a1(output)
        output = self.l2(output)
        output = self.a2(output)
        output = self.flatten(output)
        output = self.fc1(output)
        return output
