import torch
from DFRSystem import * 
import torch.optim as optim
import torch.nn as nn
import numpy as np
from Dataset import Dataset  
from torch.utils.data.dataloader import DataLoader
from Q import QParams
import copy
import torch.backends.cudnn as cudnn
from Loss import CustomBCE 
from np2array import * 
import math 

#cudnn.benchmark = True
#device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

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
batch_size = 512 
grad_batch = 1
epochs = -1 
node_size = 8 
n_fc = 20
in_size = 1
sequence_length = 8 
lr = 0.01
model_dir = "Qmodel/"

#change input dim to node_size
x_train = np.reshape(x_train, (len(x_train)))
x_test = np.reshape(x_test, (len(x_test)))

# Data loader
train_data = Dataset(x_train, y_train, sequence_length)
test_data = Dataset(x_test, y_test, sequence_length)
trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1)
testloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=1)

#create system
net = OriQDFRSystem(n_hidden=node_size, n_fc=n_fc).float().to(device)
#pre_trained = torch.load("Teachermodel99.pch")
#net.load_state_dict(pre_trained.state_dict(), strict=True)
#optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.000, nesterov=True)
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.0)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.1)
W = torch.FloatTensor([1.2, 1.0]).to(device)
criterion = nn.CrossEntropyLoss(weight=W)

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
        print(x.shape, y.shape)
        x = x.float().to(device)
        y = y.float().to(device)
        hx = torch.zeros(x.size(0), node_size).to(device)
        optimizer.zero_grad()
        output = net(x, hx)
        loss = criterion(output, y.long())
        running_loss = loss.item()
        loss.backward()
        optimizer.step()
    scheduler.step()
import time
start = time.time()
Eval(testloader, net)
end = time.time()
print("time: ", end - start)
torch.save(net, "rawmodel" + ".pch")
#get q parameters
target_dict = {}
torch.quantization.get_observer_dict(net, target_dict)
param_name = ['fc1.weight', 'DFRCell.mask', 'DFRCell.output', 'input'] 
q_params = {}
for i, key in enumerate(target_dict):
    scale, z_point = target_dict[key].calculate_qparams()
    scale, z_point = scale.cpu().numpy()[0].item(), z_point.cpu().numpy()[0].astype(int).item()
    q_params[param_name[i]] = (scale, z_point)

for key in q_params:
    print(key, q_params[key])

#calculate scaling factors for reservoir layer s1 * s2 / s3
scale_in_mask_param = q_params['input'][0] * q_params['DFRCell.mask'][0] / q_params['DFRCell.output'][0]
print("input_mask_out_scale: ", scale_in_mask_param, round(-math.log2(scale_in_mask_param)))

#calculate scaling factors for fc1 layer s1 * s2
scale_in_mask_param = q_params['DFRCell.output'][0] * q_params['fc1.weight'][0] 
print("fc1_output_scale: ", scale_in_mask_param)


#write input to file
fin = open(model_dir + 'input.txt', 'w')
x_test = torch.quantize_per_tensor(torch.from_numpy(x_test).float(), scale, z_point, torch.quint8)
fin.write(ToCArray(x_test.int_repr().numpy().reshape(-1, 1)))
fin.close()
fin = open(model_dir + 'label.txt', 'w')
fin.write(ToCArray(y_test))
fin.close()

for name, param in net.named_parameters():
    fin = open(model_dir + name + ".txt", 'w')
    if 'weight' in name:
        if 'fc1' in name:
            scale, z_point = q_params['fc1.weight']
            param = torch.quantize_per_tensor(param.cpu(), scale, z_point, torch.quint8)
            fin.write(ToCArray(param.transpose(0, 1).int_repr().numpy()))
        else:
            fin.write(ToCArray(param.transpose(0, 1).detach().cpu().numpy()))
    else:
        if 'DFRCell.mask' in name:
            scale, z_point = q_params['DFRCell.mask']
            param = torch.quantize_per_tensor(param.cpu(), scale, z_point, torch.quint8)
            fin.write(ToCArray(param.int_repr().numpy()))
        else:
            fin.write(ToCArray(param.detach().cpu().numpy()))
    fin.close()

