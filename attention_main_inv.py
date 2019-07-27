import torch
from ESNAttention import ESNCell 
import torch.optim as optim
import torch.nn as nn
import numpy as np
from NARMADataset import NARMADataset  
from torch.utils.data.dataloader import DataLoader
from collections import deque

lr = 0.0001
batch_size = 1
epochs = 1 
train_cycles=4000 
test_cycles=1000
warmup_cycles=100
n_mask = 1
n_hidden = 800 
n_out = 1
net = ESNCell(n_mask, n_hidden, n_out)
# NARMA30 dataset
narma10_train_dataset = NARMADataset(train_cycles, epochs, system_order=10, seed=1)
narma10_test_dataset = NARMADataset(test_cycles, 1, system_order=10, seed=10)
# Data loader
trainloader = DataLoader(narma10_train_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
testloader = DataLoader(narma10_test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
criterion = nn.MSELoss()
time_step = 10 

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.constant_(m.weight.data, 0.0)

net.apply(weights_init)

def Eval(testloader, net):
    result = []
    Y_test = []
    for data in testloader:
        x, y = data
        x = x.view(-1)
        y = y.view(-1)
        x_t = torch.zeros(n_hidden)
        output = torch.zeros(n_out)
        xp_list = np.zeros((time_step, n_hidden), dtype=np.float32)
        for idx in range(len(x)):
            xp = torch.from_numpy(xp_list)
            output, x_t = net(x[idx], xp, output)
            xp_list = np.concatenate((xp[1:, :], x_t.view(1, -1)), axis=0)
            result.append(output.item())
            Y_test.append(y[idx].item())
    result = np.asarray(result)
    Y_test = np.asarray(Y_test)
    NRMSE = np.sqrt(np.divide(                          \
            np.mean(np.square(result - Y_test)),   \
            np.var(Y_test)))
    print NRMSE 

def InvSolve(X, y):
    #print np.linalg.pinv(X).shape, y.shape, np.dot(np.linalg.pinv(X), y).shape
    return np.dot(np.linalg.pinv(X), y)
 
X = [] 
Y = []

for data in trainloader:
    x, y = data
    x = x.view(-1)
    y = y.view(-1)
    x_t = torch.zeros(n_hidden)
    output = torch.zeros(n_out)
    xp_list = np.zeros((time_step, n_hidden), dtype=np.float32)
    net.zero_grad()
    for idx in range(len(x)):
        xp = torch.from_numpy(xp_list)
        output, x_t = net(x[idx], xp, output)
        xp_list = np.concatenate((xp[1:, :], x_t.view(1, -1)), axis=0)
        X.append(x_t.detach().numpy())
        Y.append(y[idx].numpy())
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    weight = torch.from_numpy(InvSolve(X, Y)).view(1, -1)
    for p in net.parameters():
        p.data = weight
    X = []
    Y = []
Eval(testloader, net)
