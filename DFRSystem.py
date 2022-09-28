import torch
import numpy as np
import torch.nn as nn
from DFRCell import *
from Quantize import Quantize
from Linear import *

class DFRSystem(nn.Module):
    def __init__(self, n_hidden=10):
        super(DFRSystem, self).__init__()
        self.fc1 = QLinear(n_hidden, 1, bias=False, num_bits=8)
        #self.fc1 = nn.Linear(n_hidden, 1, bias=False)
        self.DFRCell = QDFRCell(n_hidden)
                
    def forward(self, x, prev_out):
        cell_out = self.DFRCell(x, prev_out)
        output = self.fc1(cell_out)
        return output, cell_out

class NarmaDFRSystem(nn.Module):
    def __init__(self, n_hidden=10):
        super(NarmaDFRSystem, self).__init__()
        #self.fc1 = QLinear(n_hidden, 1, bias=False, num_bits=8)
        self.fc1 = nn.Linear(n_hidden, 1, bias=False)
        self.DFRCell = NarmaDFRCell(n_hidden)
                
    def forward(self, x, prev_out):
        cell_out = self.DFRCell(x, prev_out)
        output = self.fc1(cell_out)
        return output, cell_out

class NewFloatDFRSystem(nn.Module):
    def __init__(self, n_hidden=10, n_fc=20):
        super(NewFloatDFRSystem, self).__init__()
        self.fc1 = nn.Linear(n_hidden, n_fc, bias=False)
        self.fc2 = nn.Linear(n_fc, 2)
        self.DFRCell = NewDFRCell(n_hidden=n_hidden)
        self.sigmoid = nn.Sigmoid()
        self.act = nn.ReLU()
                
    def forward(self, batch_seq_x, prev_out):
        batch_size = batch_seq_x.size(0) 
        time_step = batch_seq_x.size(1)
        for t in range(time_step):
            cell_out, prev_out = self.DFRCell(batch_seq_x[:, t], prev_out)
        output = self.fc1(cell_out)
        output = self.act(output)
        output = self.fc2(output)
        return output

class OriFloatDFRSystem(nn.Module):
    def __init__(self, n_hidden=10, n_fc=20):
        super(OriFloatDFRSystem, self).__init__()
        self.fc1 = nn.Linear(n_hidden, n_fc, bias=False)
        self.fc2 = nn.Linear(n_fc, 2)
        self.DFRCell = OriDFRCell(n_hidden=n_hidden)
        self.sigmoid = nn.Sigmoid()
        self.act = nn.ReLU()
                
    def forward(self, batch_seq_x, prev_out):
        batch_size = batch_seq_x.size(0) 
        time_step = batch_seq_x.size(1)
        for t in range(time_step):
            cell_out, prev_out = self.DFRCell(batch_seq_x[:, t], prev_out)
        output = self.fc1(cell_out)
        output = self.act(output)
        output = self.fc2(output)
        return output

class TOriFloatDFRSystem(nn.Module):
    def __init__(self, n_hidden=10, n_fc=20):
        super(TOriFloatDFRSystem, self).__init__()
        self.fc1 = nn.Linear(n_hidden, n_fc, bias=False)
        self.fc2 = nn.Linear(n_fc, 2)
        self.DFRCell = OriDFRCell(n_hidden=n_hidden)
        self.sigmoid = nn.Sigmoid()
        self.act = nn.ReLU()
                
    def forward(self, batch_seq_x, prev_out):
        batch_size = batch_seq_x.size(0) 
        time_step = batch_seq_x.size(1)
        for t in range(time_step):
            cell_out, prev_out = self.DFRCell(batch_seq_x[:, t], prev_out)
        med = self.fc1(cell_out)
        output = self.act(med)
        output = self.fc2(output)
        return output, med

class OriQDFRSystem(nn.Module):
    def __init__(self, n_hidden=10, n_fc=20):
        super(OriQDFRSystem, self).__init__()
        #self.fc1 = nn.Linear(n_hidden, n_fc, bias=False)
        self.fc1 = QLinear(n_hidden, n_fc, bias=False)
        self.fc2 = nn.Linear(n_fc, 2)
        self.DFRCell = OriQDFRCell(n_hidden=n_hidden)
        self.sigmoid = nn.Sigmoid()
        self.act = nn.ReLU()
        self.in_q = torch.quantization.FakeQuantize(observer=torch.quantization.observer.MovingAverageMinMaxObserver,
                                                    quant_min=0, quant_max=255)
                
    def forward(self, batch_seq_x, prev_out):
        batch_seq_x = self.in_q(batch_seq_x)
        batch_size = batch_seq_x.size(0) 
        time_step = batch_seq_x.size(1)
        for t in range(time_step):
            cell_out, prev_out = self.DFRCell(batch_seq_x[:, t], prev_out)
        output = self.fc1(cell_out)
        output = self.act(output)
        output = self.fc2(output)
        return output

class TOriQDFRSystem(nn.Module):
    def __init__(self, n_hidden=10, n_fc=20):
        super(TOriQDFRSystem, self).__init__()
        #self.fc1 = nn.Linear(n_hidden, n_fc, bias=False)
        self.fc1 = QLinear(n_hidden, n_fc, bias=False)
        self.fc2 = nn.Linear(n_fc, 2)
        self.DFRCell = OriQDFRCell(n_hidden=n_hidden)
        self.sigmoid = nn.Sigmoid()
        self.act = nn.ReLU()
        self.in_q = torch.quantization.FakeQuantize(observer=torch.quantization.observer.MovingAverageMinMaxObserver,
                                                    quant_min=0, quant_max=255)
                
    def forward(self, batch_seq_x, prev_out):
        batch_seq_x = self.in_q(batch_seq_x)
        batch_size = batch_seq_x.size(0) 
        time_step = batch_seq_x.size(1)
        for t in range(time_step):
            cell_out, prev_out = self.DFRCell(batch_seq_x[:, t], prev_out)
        med = self.fc1(cell_out)
        output = self.act(med)
        output = self.fc2(output)
        return output, med

class FloatDFRSystem(nn.Module):
    def __init__(self, n_hidden=10):
        super(FloatDFRSystem, self).__init__()
        self.fc1 = nn.Linear(n_hidden, 20, bias=False)
        self.fc2 = nn.Linear(20, 2)
        self.DFRCell = ADFRCell(n_hidden=n_hidden)
        self.sigmoid = nn.Sigmoid()
        self.act = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
                
    def forward(self, x, prev_out):
        cell_out = self.DFRCell(x, prev_out)
        output = self.fc1(cell_out)
        output = self.act(output)
        output = self.fc2(output)
        return output, cell_out 

class QDFRSystem(nn.Module):
    def __init__(self, n_hidden=10):
        super(QDFRSystem, self).__init__()
        self.fc1 = FixedQLinear(n_hidden, 20)
        #self.fc1 = nn.Linear(n_hidden, 20, bias=False)
        self.fc2 = nn.Linear(20, 2, bias=True)
        self.DFRCell = AQDFRCell(n_hidden=n_hidden)
        self.act = nn.ReLU()
                
    def forward(self, x, prev_out):
        cell_out = self.DFRCell(x, prev_out)
        output = self.fc1(cell_out)
        output = self.act(output)
        output = self.fc2(output)
        return output, cell_out


class ImageDFRSystem(nn.Module):
    def __init__(self, n_hidden=10, num_DFR=6):
        super(ImageDFRSystem, self).__init__()
        #self.conv1 = nn.Conv2d(1, 16, 3, stride=2, padding=1)
        #self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        #self.conv3 = nn.Conv2d(32, 1, 3, stride=1, padding=1)
        #self.conv5 = nn.Conv2d(1, 1, 3, stride=1, padding=1)
        self.fc_p1 = nn.Linear(14*14, 7*7, bias=False)
        self.fc1 = nn.Linear(n_hidden, 8, bias=False)
        self.fc2 = nn.Linear(8, 10, bias=True)
        self.DFR = [ParallelDFRCell(n_in=1, n_hidden=n_hidden)]
        self.FC = [nn.Linear(1, n_hidden, bias=False)]
        for i in range(num_DFR - 1):
            self.DFR.append(ParallelDFRCell(n_in=n_hidden, n_hidden=n_hidden))
            self.FC.append(nn.Linear(n_hidden, n_hidden))
        self.DFR = nn.Sequential(*self.DFR)
        self.FC = nn.Sequential(*self.FC)
        self.act = nn.ReLU()
        #self.Pre = nn.Sequential(self.conv1d_1, self.act, self.conv1d_2)
                
    def forward(self, x, cell_out):
        x = x.view(x.shape[0], -1)
        x = self.fc_p1(x)
        #print(x)
        batch_size, data_len = x.shape
        for i in range(data_len):
            pixel = x[:, i].view(-1, 1)
            cell_out[0] = self.DFR[0](pixel, cell_out[0])
            for i in range(1, len(self.DFR)):
                cell_out[i] = self.DFR[i](cell_out[i - 1], cell_out[i])
        output = self.fc1(cell_out[-1])
        output = self.act(output)
        output = self.fc2(output)
        return output  

class QImageDFRSystem(nn.Module):
    def __init__(self, n_hidden=10, num_DFR=6):
        super(QImageDFRSystem, self).__init__()
        #self.conv1 = nn.Conv2d(1, 16, 3, stride=2, padding=1)
        #self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        #self.conv3 = nn.Conv2d(32, 1, 3, stride=1, padding=1)
        #self.conv5 = nn.Conv2d(1, 1, 3, stride=1, padding=1)
        #self.fc_p1 = nn.Linear(14*14, 7*7)
        self.fc_p1 = QLinear(14*14, 7*7)
        self.fc1 = QLinear(n_hidden, 8, bias=False)
        self.fc2 = nn.Linear(8, 10)
        self.DFR = [QParallelDFRCell(n_in=1, n_hidden=n_hidden)]
        self.FC = [nn.Linear(1, n_hidden)]
        self.bn1 = nn.BatchNorm1d(7 * 7, affine=False)
        for i in range(num_DFR - 1):
            self.DFR.append(QParallelDFRCell(n_in=n_hidden, n_hidden=n_hidden))
            self.FC.append(nn.Linear(n_hidden, n_hidden))
        self.DFR = nn.Sequential(*self.DFR)
        self.FC = nn.Sequential(*self.FC)
        self.act = nn.ReLU()
        self.act1 = nn.Tanh()
        #self.Pre = nn.Sequential(self.conv1d_1, self.act, self.conv1d_2)
        self.register_buffer('in_min_max', torch.zeros(2))
        self.in_min_max[0] = -0.45 
        self.in_min_max[1] = 3.55
        #self.register_buffer('fc0', torch.zeros(2))
        #self.fc0[0].requires_grad_(False)
        #self.fc0[1].requires_grad_(False)
        #self.fc0[0] = -60 
        #self.fc0[1] = 60

    def forward(self, x, cell_out):
        #self.in_min_max[0] = torch.min(self.in_min_max[0], torch.min(x))
        #self.in_min_max[1] = torch.max(self.in_min_max[0], torch.max(x))
        #print(self.in_min_max)
        x = x.view(x.shape[0], -1)
        x = Quantize(x, self.in_min_max[0], self.in_min_max[1])
        x = self.fc_p1(x)
        #self.fc0[0] = 0.99 * self.fc0[0] + 0.01 * torch.min(x)
        #self.fc0[1] = 0.99 * self.fc0[1] + 0.01 * torch.max(x)
        #print(self.fc0[0], self.fc0[1])
        #print(torch.min(x), torch.max(x))
        #x = LogQuantize(x, self.fc0[0].detach(), self.fc0[1].detach())
        batch_size, data_len = x.shape
        for i in range(data_len):
            pixel = x[:, i].view(-1, 1)
            cell_out[0] = self.DFR[0](pixel, cell_out[0])
            for i in range(1, len(self.DFR)):
                cell_out[i] = self.DFR[i](cell_out[i - 1], cell_out[i])
        output = self.fc1(cell_out[-1])
        output = self.act(output)
        output = self.fc2(output)
        return output  

class ResidualBlock(nn.Module):
    def __init__(self, n_in, n_out):
        super(ResidualBlock, self).__init__()
        self.DFRLayer1 = ADFRCell(n_in=n_in, n_hidden=n_out)
        self.DFRLayer2 = ADFRCell(n_in=n_out, n_hidden=n_out)
        self.act1 = nn.Hardtanh(min_val=0.0, max_val=2.0) 
        self.act2 = nn.Hardtanh(min_val=0.0, max_val=2.0) 

    def forward(self, x, prev1, prev2):
        residual = 4
        x = self.DFRLayer1(x, prev1)
        x = self.act1(x)
        prev1 = x
        x = self.DFRLayer2(x, prev2)
        #x += residual
        x = self.act2(x)
        prev2 = x
        return x, prev1, prev2
        

class DeepFloatDFRSystem(nn.Module):
    def __init__(self, block, num_blocks, n_in=1, n_hidden=10):
        super(DeepFloatDFRSystem, self).__init__()
        self.DFRLayer1 = ADFRCell(n_in=n_in, n_hidden=n_hidden)
        self.DFRLayer2 = ADFRCell(n_in=n_hidden, n_hidden=n_hidden)
        self.DFRLayer3 = ADFRCell(n_in=n_hidden, n_hidden=n_hidden)
        self.Layers = []
        for i in range(num_blocks):
            self.Layers.append(block(n_hidden, n_hidden))
        self.Layers = nn.Sequential(*self.Layers)
        self.fc1 = nn.Linear(n_hidden, 20)
        self.fc2 = nn.Linear(20, 2)
        self.act = nn.Hardtanh(min_val=0.0, max_val=2.0)
        self.act1 = nn.ReLU()

    def forward(self, x, prev):
        prev_out = []
        x = self.DFRLayer1(x, prev[0])
        prev_out.append(x)
        x = self.DFRLayer2(x, prev[1])
        prev_out.append(x)
        #x = self.DFRLayer3(x, prev[2])
        #prev_out.append(x)
        #for i in range(len(self.Layers)):
        #    idx1, idx2 = i * 2 + 1, i * 2 + 2
        #x, prev1, prev2 = self.Layers[0](x, prev[1], prev[2])

        #prev_out.append(prev1)
        #prev_out.append(prev2)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        return x, prev_out 
