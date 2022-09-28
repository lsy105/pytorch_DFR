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
from sklearn.metrics import roc_auc_score

cudnn.benchmark = True
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# NARMA30 dataset
def Run(t_w, alpha_t, T_t):
    import scipy.io
    x = scipy.io.loadmat('recieved1.mat')
    y = scipy.io.loadmat('target.mat')
    x_data = x['recieved1']
    y_data = y['target']

    x_train = np.asarray(x_data[0:1000], dtype=float)
    y_train = np.asarray(y_data[0:1000], dtype=float)
    x_test = np.asarray(x_data[1000:], dtype=float)
    y_test = np.asarray(y_data[1000:], dtype=float)
    #t_max = np.mean(np.sort(x_data, axis=None)[-500:])
    #t_min = np.mean(np.sort(x_data, axis=None)[:500])
    #x_train = -1 + (x_train - t_min) * 2/(t_max - t_min)
    #x_test = -1 + (x_test - t_min) * 2/(t_max - t_min)
    mean = np.mean(x_train)
    std = np.std(x_train)
    x_train -= mean
    x_train /= std
    x_test -= mean
    x_test /= std
    x_sorted = np.sort(x_train, axis=None)
    a_max = np.mean(np.min(x_sorted[-100:]))
    a_min = np.mean(np.min(x_sorted[:100]))
    #parameters
    batch_size = 1
    grad_batch = 1
    epochs = 30
    node_size = 8
    in_size = 1
    lr = 0.001
    #change input dim to node_size
    t_data = QParams("Qtest_data")
    t_data.Quantize(x_test, minval=-1.0, maxval=1.0, row=len(x_test), col=1)
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
    net = QDFRSystem(n_hidden=node_size).float().to(device)
    pre_trained = torch.load("teacher.pch")
    net.load_state_dict(pre_trained.state_dict(), strict=False)
    #optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.000, nesterov=True)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.0)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    W = torch.FloatTensor([t_w, 1.0]).to(device)
    criterion = KDLoss(W=W, alpha=alpha_t, T=T_t)
    #criterion = nn.CrossEntropyLoss(weight=W)
    def Eval(testloader, net, hx1):
        count = 0
        pred, gt = [], []
        net = net.eval()
        for idx, data in enumerate(testloader):
            x, y = data
            x = x.view(-1, 1).float().to(device)
            y = y.view(-1).long().to(device)
            #y = torch.nn.functional.one_hot(y, 2)
            with torch.no_grad():
                output, hx1 = net.forward(x, hx1)
                output = nn.functional.softmax(output, dim=1)
                output = torch.argmax(output, dim=1)
                pred.append(output.detach().cpu().numpy()) 
                gt.append(y.detach().cpu().numpy())
        #pred = np.array(pred).reshape((-1, 2))
        #gt = np.array(gt).reshape((-1, 2))
        #print(roc_auc_score(gt, pred))
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
    #Eval(testloader, teacher, hx1_cp1)
    Eval(testloader, net, hx1_cp2)
    #Eval(trainloader, net, hx1)
    torch.save(net, "KDmodel" + str(epoch) + ".pch")
    temp_Q = QParams("hx1")
    W = hx1.detach().cpu().numpy()
    temp_Q.Quantize(W, minval=-1.0, maxval=1.0, row=1, col=hx1.shape[1])

if __name__ == '__main__':
    #W = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2.0]
    W = [0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2.0]
    alphas = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
    Ts = [1.5, 2, 3, 4.5, 6.0, 8.0, 10.0, 20.0]
    #for alpha in alphas:
    #    for T in Ts:
    for w in W:
        print(w, 0.99, 1.5)
        Run(w, 0.99, 1.5)

