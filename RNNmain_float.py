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
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from RNNSystem import * 
from sklearn.metrics import roc_auc_score
from evaluation import Eval

cudnn.benchmark = True
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# NARMA30 dataset
import scipy.io
#x = scipy.io.loadmat('Kian_data/recieved-30db_6.mat')
#y = scipy.io.loadmat('Kian_data/target-30db_6.mat')
x = scipy.io.loadmat('Kian_data/recieved1_20db_6_6.mat')
y = scipy.io.loadmat('Kian_data/target.mat')
#x = scipy.io.loadmat('recieved1.mat')
#y = scipy.io.loadmat('target.mat')
x_data = x['recieved1']
y_data = y['target']
x_train = np.asarray(x_data[0:1000], dtype=float)
y_train = np.asarray(y_data[0:1000], dtype=float)
x_test = np.asarray(x_data[1000:], dtype=float)
y_test = np.asarray(y_data[1000:], dtype=float)
mean = np.mean(x_data)
#t_max = np.mean(np.sort(x_data, axis=None)[-500:])
#t_min = np.mean(np.sort(x_data, axis=None)[:500])
#diff = t_max - t_min 
#x_train = (x_train - mean) / diff
#x_test = (x_test - mean) / diff
mean = np.mean(x_data)
std = np.std(x_data)
x_train -= mean
x_train /= std
x_test -= mean
x_test /= std
#parameters
batch_size = 1
grad_batch = 1
epochs = 30
node_size = 10
in_size = 1
lr = 0.001
#change input dim to node_size
t_data = QParams("Qtest_data")
t_data.GetFloat(x_test, row=len(x_test), col=1)
t_label = QParams("Qtest_label")
t_label.GetFloat(y_test, row=len(y_test), col=1)
x_train = np.reshape(x_train, (len(x_train), 1, in_size))
x_test = np.reshape(x_test, (len(x_test), 1, in_size))
# Data loader
train_data = Dataset(x_train, y_train, True)
test_data = Dataset(x_test, y_test)
trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=1)
testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=1)

#create system
net = RNNSystem(n_hidden=node_size).float().to(device)
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.99, weight_decay=0.0001, nesterov=True)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
#optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.0001)
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
W = torch.FloatTensor([1.0, 1.0]).to(device)
#criterion = KDLoss(W=W, alpha=0.1, T=30)
criterion = nn.CrossEntropyLoss(weight=W)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        m.weight.data.uniform_(0.0, 0.02)
net.apply(weights_init)
teacher = torch.load("teacher.pch")
teacher.eval()

for epoch in range(epochs):
    print("epoch:", epoch)
    hx1 = torch.zeros(batch_size, node_size).to(device)
    t_hx1 = torch.zeros(batch_size, node_size).to(device)
    running_loss = 0.0
    count, count1 = 0, 0
    for idx, data in enumerate(trainloader):
        x, y = data
        x = x.view(-1, 1).float().to(device)
        y = y.view(-1).long().to(device)
        optimizer.zero_grad()
        output, hx1 = net(x, hx1)
        #t_output, t_hx1 = teacher(x, t_hx1) 
        hx1.detach_()
        #t_hx1.detach_()
        #loss = criterion(output, y, t_output)
        loss = criterion(output, y)
        #running_loss += loss.item()
        loss.backward()
        if idx != 0 and idx % grad_batch == 0:
            optimizer.step()
    scheduler.step()
    #print(running_loss / len(trainloader))
    #print(count / len(trainloader))
    #print(count1 / len(trainloader))
hx1_cp1 = hx1.clone()
hx1_cp2 = hx1.clone()
#Eval(testloader, teacher, hx1_cp1)
Eval(testloader, net, hx1_cp2)
#Eval(trainloader, net, hx1)
torch.save(net, "RNNmodel" + str(epoch) + ".pch")
temp_Q = QParams("hx1")
W = hx1.detach().cpu().numpy()
temp_Q.GetFloat(W, row=1, col=hx1.shape[1]) 
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
"""
for name, weight in net.named_parameters():
    print(name, weight.shape)
    if "fc1.weight" in name:
        temp_Q = QParams(name)
        print(weight.shape, name)
        W = weight.detach().cpu().numpy().transpose()
        print(W.shape, name)
        temp_Q.GetFloat(W, row=W.shape[0], col=W.shape[1]) 
    elif "fc2.weight" in name:
        temp_Q = QParams(name)
        W = weight.detach().cpu().numpy().transpose()
        temp_Q.GetFloat(W, row=W.shape[0], col=W.shape[1])
    elif "weight_ih" in name:
        temp_Q = QParams(name)
        W = weight.detach().cpu().numpy().transpose()
        temp_Q.GetFloat(W, row=W.shape[0], col=W.shape[1])
    elif "weight_hh" in name:
        temp_Q = QParams(name)
        W = weight.detach().cpu().numpy().transpose()
        temp_Q.GetFloat(W, row=W.shape[0], col=W.shape[1])
    else:
        temp_Q = QParams(name)
        W = weight.detach().cpu().numpy()
        temp_Q.GetFloat(W, row=1, col=W.shape[0])

