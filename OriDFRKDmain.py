import torch
from DFRSystem import * 
import torch.optim as optim
import torch.nn as nn
import numpy as np
from Dataset import Dataset, DatasetMixed 
from torch.utils.data.dataloader import DataLoader
from Q import QParams
import copy
import torch.backends.cudnn as cudnn
from Loss import KDLoss 

cudnn.benchmark = True
#device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# NARMA30 dataset
import scipy.io
x = scipy.io.loadmat('./Kian_data/recieved1_20db_4_4.mat')
y = scipy.io.loadmat('./Kian_data/target.mat')
x_data = x['recieved1']
y_data = y['target']

x_train = np.asarray(x_data[0:1000], dtype=float)
y_train = np.asarray(y_data[0:1000], dtype=float)
x_test = np.asarray(x_data[1000:], dtype=float)
y_test = np.asarray(y_data[1000:], dtype=float)
#parameters
batch_size = 32
grad_batch = 1
epochs = 100 
node_size = 8 
n_fc = 20
in_size = 1
sequence_length = 8 
lr = 0.01
#change input dim to node_size
x_train = np.reshape(x_train, (len(x_train)))
x_test = np.reshape(x_test, (len(x_test)))
# Data loader
train_data = Dataset(x_train, y_train, sequence_length)
test_data = Dataset(x_test, y_test, sequence_length)
trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1)
testloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=1)

#create system
net = OriFloatDFRSystem(n_hidden=node_size, n_fc=n_fc).float().to(device)
T_net = OriFloatDFRSystem(n_hidden=node_size, n_fc=n_fc).float().to(device)
pre_trained = torch.load("Teachermodel99.pch")
T_net.load_state_dict(pre_trained.state_dict(), strict=True)
#optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.000, nesterov=True)
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.0)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.1)
W = torch.FloatTensor([1.0, 1.0]).to(device)
criterion = KDLoss(W=W, alpha=0.2, T=30)
T_net.eval()
def accuracy(output, target):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        batch_size = target.size(0)
        pred = torch.argmax(output, dim=1)
        res = torch.sum(pred.eq(target)).item() / batch_size
        return res

def Eval(testloader, net):
    num_batch = 0
    acc = 0.0
    with torch.no_grad():
        for data in testloader:
            x, y = data
            x = x.float().to(device)
            y = y.float().to(device)
            hx = torch.zeros(x.size(0), node_size).to(device)
            optimizer.zero_grad()
            output = net(x, hx)
            acc += accuracy(output, y)
            num_batch += 1
    print(acc / num_batch)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        m.weight.data.uniform_(0.0, 0.02)
#net.apply(weights_init)

for epoch in range(epochs):
    print("epoch:", epoch)
    running_loss = 0.0
    count, count1 = 0, 0
    for idx, data in enumerate(trainloader):
        x, y = data
        x = x.float().to(device)
        y = y.float().to(device)
        hx = torch.zeros(x.size(0), node_size).to(device)
        T_hx = torch.zeros(x.size(0), node_size).to(device)
        optimizer.zero_grad()
        output = net(x, hx)
        T_output = T_net(x, T_hx) 
        loss = criterion(output, y.long(), T_output)
        running_loss = loss.item()
        loss.backward()
        optimizer.step()
    scheduler.step()
    Eval(testloader, net)
Eval(testloader, net)
torch.save(net, "KDmodel" + str(epoch) + ".pch")

"""
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
"""
