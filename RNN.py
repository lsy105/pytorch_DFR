import torch
import numpy as np
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, n_in=1, hidden_size=10):
        super(RNN, self).__init__()
        self.RNN1 = nn.RNNCell(n_in, hidden_size, bias=False, nonlinearity='tanh')

    def forward(self, x, hidden):
        ho = self.RNN1(x, hidden)
        return ho

class LSTM(nn.Module):
    def __init__(self, n_hidden=10):
        super(LSTM, self).__init__()
        self.lstm1 = nn.LSTMCell(1, n_hidden)

    def forward(self, x, hx, cx):
        return self.lstm1(x, (hx, cx))

class GRU(nn.Module):
    def __init__(self, n_hidden=10):
        super(GRU, self).__init__()
        self.GRU1 = nn.GRUCell(1, n_hidden)
        self.GRU2 = nn.GRUCell(n_hidden, n_hidden)
        self.GRU3 = nn.GRUCell(n_hidden, n_hidden)
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
        self.dropout3 = nn.Dropout(p=0.5)

    def forward(self, x, hx1, hx2):
        ho1 = self.GRU1(x, hx1)
        #ho1 = self.dropout1(ho1)
        ho2 = self.GRU2(ho1, hx2)
        #ho2 = self.dropout2(ho2)
        return ho1, ho2



