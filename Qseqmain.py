import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from DFRSystem import *
import torch.optim as optim
import numpy as np
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import roc_auc_score
import torch.backends.cudnn as cudnn
from Loss import *
from Q import QParams

cudnn.benchmark = True
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

#parameters
batch_size = 512 
test_batch_size = 1
grad_batch = 1
epochs = 60 
node_size = 8 
in_size = 1
lr = 0.01
num_DFR = 1 

trainloader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                   transforms.Resize((14,14)),
                   transforms.ToTensor(),
                   transforms.Normalize((0.1307,), (0.3081,))
                  ])),
                   drop_last=True, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                   transforms.Resize((14,14)),
                   transforms.ToTensor(),
                   transforms.Normalize((0.1307,), (0.3081,))
                   ])),
                   drop_last=True, batch_size=test_batch_size, shuffle=False)

#create system
net = QImageDFRSystem(n_hidden=node_size, num_DFR=num_DFR).float().to(device)
#optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.000, nesterov=True)
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.0)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
#criterion = nn.CrossEntropyLoss()
W = torch.FloatTensor([0.25, 1.0]).to(device)
criterion = KDLoss(W=W, alpha=0.1, T=25)

def Eval(testloader, net, hx):
    count = 0
    pred, gt = [], []
    net = net.eval()
    save_x, save_y = [], []
    for idx, data in enumerate(testloader):
        x, y = data
        #x = x.view(x.shape[0], -1).float().to(device)
        #y = y.view(-1).float().to(device)
        x = x.float().to(device)
        y = y.view(-1).float().to(device)
        save_x.append(x.detach().cpu().numpy().reshape((-1)))
        save_y.append(y.detach().cpu().numpy().reshape((-1)))
        with torch.no_grad():
            output = net.forward(x, hx)
            output = torch.argmax(output, dim=1)
            pred.append(output.cpu().numpy())
            gt.append(y.cpu().numpy())
    temp_Q = QParams("test_data")
    W = np.array(save_x)
    print(W.shape)
    temp_Q.Quantize(W, minval=-1, maxval=1.0, row=W.shape[0], col=W.shape[1])
    temp_Q = QParams("test_label")
    W = np.array(save_y) 
    temp_Q.GetFloat(W, row=W.shape[0], col=W.shape[1])
    acc = np.mean(np.array(pred) == np.array(gt))
    print(acc)


for epoch in range(epochs):
    print("epoch:", epoch)
    running_loss = 0.0
    count, count1 = 0, 0
    for idx, data in enumerate(trainloader):
        hx = [torch.zeros(batch_size, node_size).to(device) for _ in range(num_DFR)]
        x, y = data
        #x = x.view(x.shape[0], -1).float().to(device)
        #y = y.view(-1).long().to(device)
        x = x.float().to(device)
        y = y.view(-1).long().to(device)
        optimizer.zero_grad()
        output = net(x, hx) 
        loss = criterion(output, y)
        #loss = criterion(output, y)
        #running_loss += loss.item()
        loss.backward()
        optimizer.step()
    scheduler.step()
hx = [torch.zeros(batch_size, node_size).to(device) for _ in range(num_DFR)]
Eval(testloader, net, hx)
#Eval(trainloader, net, hx1)
torch.save(net, "KDmodel" + str(epoch) + ".pch")

for name, weight in net.named_parameters():
    print(name)
    if "fc1.weight" in name:
        temp_Q = QParams(name)
        W = weight.detach().cpu().numpy().transpose()
        temp_Q.Quantize(W, row=W.shape[0], col=W.shape[1])
    elif "fc2.weight" in name:
        temp_Q = QParams(name)
        W = weight.detach().cpu().numpy().transpose()
        temp_Q.GetFloat(W, row=W.shape[0], col=W.shape[1])
    elif "fc2.bias" in name:
        temp_Q = QParams(name)
        print(weight.shape)
        W = weight.detach().cpu().numpy().transpose().reshape((1, -1))
        print(W.shape)
        temp_Q.GetFloat(W, row=W.shape[0], col=W.shape[1])
    elif "DFR.0.mask" in name:
        temp_Q = QParams(name)
        W = weight.detach().cpu().numpy()
        temp_Q.Quantize(W, minval= -0.5, maxval = 0.5, row=W.shape[0], col=W.shape[1])
    elif "fc_p1.weight" in name:
        temp_Q = QParams(name)
        W = weight.detach().cpu().numpy().transpose()
        temp_Q.Quantize(W, row=W.shape[0], col=W.shape[1])
