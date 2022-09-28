import torch
from DFRSystem import * 
import torch.optim as optim
import torch.nn as nn
import numpy as np
from Dataset import Dataset  
from torch.utils.data.dataloader import DataLoader
from Q import QParams
import copy
from sklearn.metrics import roc_auc_score
import torch.backends.cudnn as cudnn
from Loss import * 

cudnn.benchmark = True
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# NARMA30 dataset
import scipy.io
x = scipy.io.loadmat('recieved1.mat')
y = scipy.io.loadmat('target.mat')
x_data = x['recieved1']
y_data = y['target']

x_train = np.asarray(x_data[0:1000], dtype=float)
y_train = np.asarray(y_data[0:1000], dtype=float)
x_test = np.asarray(x_data[1000:], dtype=float)
y_test = np.asarray(y_data[1000:], dtype=float)
mean = np.mean(x_train)
std = np.std(x_train)
x_train -= mean
x_train /= std
x_test -= mean
x_test /= std
#parameters
batch_size = 1
grad_batch = 1
epochs = 100
node_size = 8
in_size = 1
lr = 0.001
#change input dim to node_size
x_train = np.reshape(x_train, (len(x_train), 1, in_size))
x_test = np.reshape(x_test, (len(x_test), 1, in_size))
# Data loader
train_data = Dataset(x_train, y_train, True)
test_data = Dataset(x_test, y_test)
trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=1)
testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=1)

#create system
net = FloatDFRSystem(n_hidden=node_size).float().to(device)
#optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.000, nesterov=True)
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.0)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.1)
W = torch.FloatTensor([1.2, 1.0]).to(device)
#criterion = KDLoss(W=W, alpha=0.3, T=70)
criterion = nn.CrossEntropyLoss(weight=W)

def Eval(testloader, net, hx1):
    count = 0
    pred, gt = [], []
    net = net.eval()
    for idx, data in enumerate(testloader):
        x, y = data
        x = x.view(-1, 1).float().to(device)
        y = y.view(-1, 1).float().to(device)
        with torch.no_grad():
            output, hx1 = net.forward(x, hx1)
            output = torch.argmax(output, dim=1)
            pred.append(output.item()) 
            gt.append(y.item())
    acc = np.mean(np.array(pred) == np.array(gt))
    print(acc)
    #print(roc_auc_score(gt, pred))
    
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
teacher = torch.load("teacher.pch")
teacher.eval()

for epoch in range(epochs):
    print("epoch:", epoch)
    hx1 = torch.zeros(node_size).to(device)
    running_loss = 0.0
    count, count1 = 0, 0
    for idx, data in enumerate(trainloader):
        x, y = data
        x = x.view(-1, 1).float().to(device)
        y = y.view(-1).long().to(device)
        optimizer.zero_grad()
        output, hx1 = net(x, hx1)
        hx1.detach_()
        loss = criterion(output, y)
        loss.backward()
        if idx != 0 and idx % grad_batch == 0:
            optimizer.step()
    scheduler.step()
    #print(running_loss / len(trainloader))
    #print(count / len(trainloader))
    #print(count1 / len(trainloader))
hx1_cp1 = hx1.clone()
Eval(testloader, net, hx1_cp1)
#Eval(trainloader, net, hx1)
torch.save(net, "teacher" + str(epoch) + ".pch")

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
