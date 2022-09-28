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
import argparse

cudnn.benchmark = True
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

teacher = torch.load("teacher.pch")
teacher.eval()

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

def Run(alpha_in, T_in, W_in):
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
    net = QDFRSystem(n_hidden=node_size).float().to(device)
    #optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.000, nesterov=True)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.0)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.1)
    W = torch.FloatTensor([W_in, 1.0]).to(device)
    criterion = KDLoss(W=W, alpha=alpha_in, T=T_in)
    #criterion = nn.CrossEntropyLoss(weight=W)

    for epoch in range(epochs):
        hx1 = torch.zeros(node_size).to(device)
        t_hx1 = torch.zeros(node_size).to(device)
        running_loss = 0.0
        count, count1 = 0, 0
        for idx, data in enumerate(trainloader):
            x, y = data
            x = x.view(-1, 1).float().to(device)
            y = y.view(-1).long().to(device)
            optimizer.zero_grad()
            output, hx1 = net(x, hx1)
            t_output, t_hx1 = teacher(x, t_hx1) 
            hx1.detach_()
            t_hx1.detach_()
            loss = criterion(output, y, t_output)
            #loss = criterion(output, y)
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
#    Eval(testloader, teacher, hx1_cp1)
    Eval(testloader, net, hx1_cp2)
    

if __name__ == "__main__":
    alphas = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.50]
    Ts = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    Ws = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    for alpha in alphas:
        for T in Ts:
            for W in Ws:
                print(alpha, T, W)
                Run(alpha, T, W)
