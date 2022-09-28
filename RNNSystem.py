import torch
import numpy as np
import torch.nn as nn
from RNN import RNN, LSTM, GRU

class RNNSystem(nn.Module):
    def __init__(self, n_in=1, n_hidden=10):
        super(RNNSystem, self).__init__()
        self.rnn = RNN(n_in, n_hidden)
        self.act = nn.ReLU()
        self.fc1 = nn.Linear(n_hidden, 20, bias=False) 
        self.fc2 = nn.Linear(20, 2) 
        #torch.nn.init.xavier_normal_(self.fc1.weight, gain=1.0)
        #torch.nn.init.xavier_normal_(self.fc2.weight, gain=1.0)
                
    def forward(self, x, hx1):
        batch_size, seq_size = x.shape
        for i in range(seq_size):
            hx1 = self.rnn(x[:, i:i+1], hx1)
        output = self.fc1(hx1)
        output = self.act(output)
        #output = self.dropout1(output)
        output = self.fc2(output)
        return output


class ImageRNNSystem(nn.Module):
    def __init__(self, n_hidden=10, num_RNN=6):
        super(ImageRNNSystem, self).__init__()
        #self.conv1 = nn.Conv2d(1, 16, 3, stride=2, padding=1)
        #self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        #self.conv3 = nn.Conv2d(32, 1, 3, stride=1, padding=1)
        #self.conv5 = nn.Conv2d(1, 1, 3, stride=1, padding=1)
        self.fc_p1 = nn.Linear(14*14, 7*7, bias=False)
        self.fc1 = nn.Linear(n_hidden, 8, bias=False)
        self.fc2 = nn.Linear(8, 10)
        self.RNN = [RNN(1, n_hidden)]
        self.FC = [nn.Linear(1, n_hidden)]
        for i in range(num_RNN - 1):
            self.RNN.append(RNN(n_in=n_hidden, hidden_size=n_hidden))
            self.FC.append(nn.Linear(n_hidden, n_hidden))
        self.RNN = nn.Sequential(*self.RNN)
        self.FC = nn.Sequential(*self.FC)
        self.act = nn.ReLU()
        #self.Pre = nn.Sequential(self.conv1d_1, self.act, self.conv1d_2)

    def forward(self, x, cell_out):
        x = x.view(x.shape[0], -1)
        x = self.fc_p1(x)
        batch_size, data_len = x.shape
        for i in range(data_len):
            pixel = x[:, i].view(-1, 1)
            cell_out[0] = self.RNN[0](pixel, cell_out[0])
            for i in range(1, len(self.RNN)):
                cell_out[i] = self.RNN[i](cell_out[i - 1], cell_out[i])
        output = self.fc1(cell_out[-1])
        output = self.act(output)
        output = self.fc2(output)
        return output
