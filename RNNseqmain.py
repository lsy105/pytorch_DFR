import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from RNNSystem import *
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
epochs = 30 
node_size = 8
in_size = 1
lr = 0.01
num_RNN = 3 

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
net = ImageRNNSystem(n_hidden=node_size, num_RNN=num_RNN).float().to(device)
#optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.000, nesterov=True)
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.0)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
criterion = nn.CrossEntropyLoss()

def Eval(testloader, net, hx):
    count = 0
    pred, gt = [], []
    net = net.eval()
    for idx, data in enumerate(testloader):
        x, y = data
        #x = x.view(x.shape[0], -1).float().to(device)
        #y = y.view(-1).float().to(device)
        x = x.float().to(device)
        y = y.view(-1).float().to(device)
        with torch.no_grad():
            output = net.forward(x, hx)
            output = torch.argmax(output, dim=1)
            pred.append(output.cpu().numpy())
            gt.append(y.cpu().numpy())
    acc = np.mean(np.array(pred) == np.array(gt))
    print(acc)


for epoch in range(epochs):
    print("epoch:", epoch)
    running_loss = 0.0
    count, count1 = 0, 0
    for idx, data in enumerate(trainloader):
        hx = [torch.zeros(batch_size, node_size).to(device) for _ in range(num_RNN)]
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
hx = [torch.zeros(test_batch_size, node_size).to(device) for _ in range(num_RNN)]
Eval(testloader, net, hx)
#Eval(trainloader, net, hx1)
torch.save(net, "RNNseqmodel" + str(epoch) + ".pch")

for name, weight in net.named_parameters():
    if "fc1.weight" in name:
        temp_Q = QParams(name)
        W = weight.detach().cpu().numpy().transpose()
        temp_Q.GetFloat(W, row=W.shape[0], col=W.shape[1])
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
    elif "RNN.0.RNN1.weight_ih" in name:
        temp_Q = QParams(name)
        W = weight.detach().cpu().numpy().transpose()
        temp_Q.GetFloat(W, row=W.shape[0], col=W.shape[1])
    elif "RNN.0.RNN1.weight_hh" in name:
        temp_Q = QParams(name)
        W = weight.detach().cpu().numpy().transpose()
        temp_Q.GetFloat(W, row=W.shape[0], col=W.shape[1])
    elif "fc_p1.weight" in name:
        temp_Q = QParams(name)
        W = weight.detach().cpu().numpy().transpose()
        temp_Q.GetFloat(W, row=W.shape[0], col=W.shape[1])

