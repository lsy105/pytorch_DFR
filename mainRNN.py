import torch
from RNNSystem import RNNSystem
import torch.optim as optim
import torch.nn as nn
import numpy as np
from Dataset import Dataset  
from torch.utils.data.dataloader import DataLoader
from Q import QParams
import copy

# NARMA30 dataset
x_train = np.asarray([line.strip() for line in open('train_data.txt', 'r').readlines()], dtype=float)
y_train = np.asarray([line.strip() for line in open('train_label.txt', 'r').readlines()], dtype=float)
x_test = np.asarray([line.strip() for line in open('test_data.txt', 'r').readlines()], dtype=float)
y_test = np.asarray([line.strip() for line in open('test_label.txt', 'r').readlines()], dtype=float)
#parameters
batch_size = 1 
epochs = 100 
node_size = 10
in_size = 1
sample_size = 300
time_step = sample_size
lr = 0.002
print(np.max(x_train), np.min(x_train))
print(np.max(x_test), np.min(x_test))
#change input dim to node_size
x_train = np.reshape(x_train, (len(x_train), 1, in_size))
x_test = np.reshape(x_test, (len(x_test), 1, in_size))
# Data loader
train_data = Dataset(x_train, y_train)
test_data = Dataset(x_test, y_test)
trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=1)
testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=1)

#create system
net = RNNSystem(n_hidden=node_size).float()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
criterion = nn.MSELoss()

def Eval(testloader, net, hx1, hx2, hx3):
    def NMSE(y_true, y_pred):
        abs_diff = y_true - y_pred
        norm1 = np.sum(abs_diff * abs_diff)
        norm2 = np.sum(y_true * y_true)
        res = np.sqrt(norm1) / np.sqrt(norm2)
        return res * res

    y_true, y_pred = [], []
    for idx, data in enumerate(testloader):
        x, y = data
        x = x.view(-1, 1).float()
        y = y.view(-1, 1).float()
        output, hx1, hx2, hx3 = net(x, hx1, hx2, hx3)
        y_true.append(y.detach().numpy())
        y_pred.append(output.detach().numpy())
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    return NMSE(y_true, y_pred) 

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
#net.apply(weights_init)
running_loss = 0.0
for epoch in range(epochs):
    print("epoch:", epoch)
    scheduler.step()
    hx1 = torch.zeros(1, node_size)
    hx2 = torch.zeros(1, node_size)
    hx3 = torch.zeros(1, node_size)
    for idx, data in enumerate(trainloader):
        x, y = data
        x = x.view(-1, 1).float()
        y = y.view(-1, 1).float()
        batch_loss = 0
        optimizer.zero_grad()
        output, hx1, hx2, hx3 = net(x, hx1, hx2, hx3)
        loss = criterion(output, y)
        running_loss += loss.item()
        loss.backward(retain_graph=True)
        optimizer.step()
    print("NMSE:", Eval(testloader, net, hx1, hx2, hx3))

#quantize input
for name, buffer in net.named_buffers():
    temp_Q = QParams(name)
    print(name, buffer)
    if "in_min_max" in name:
        print(buffer[0].item(), buffer[1].item())
        temp_Q.Quantize(A=x_test, minval=buffer[0].item(), maxval=buffer[1].item(), row=len(x_test), col=1)
    elif "l1_min_max" in name:
        temp_Q.Quantize(minval=buffer[0].item(), maxval=buffer[1].item())
        Init_Q = QParams("init")
        print(x_t)
        Init_Q.Quantize(x_t.detach().numpy(), minval=buffer[0].item(), maxval=buffer[1].item(), row=1, col=10)
    else:
        temp_Q.Quantize(minval=buffer[0].item(), maxval=buffer[1].item())

for name, weight in net.named_parameters():
    print(name)
    temp_Q = QParams(name)
    W = weight.detach().numpy()
    print(W, name)
    if 'mask' in name:
        temp_Q.Quantize(W, row=1, col=10)
    else:
        temp_Q.Quantize(W, row=10, col=1)
