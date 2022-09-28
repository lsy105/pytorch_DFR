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

def Eval(testloader, net, hx1, classes=2):
    count = 0
    num_classes = classes
    pred, gt = [], []
    p_out = []
    y_out = []
    net = net.eval()
    for idx, data in enumerate(testloader):
        x, y = data
        x = x.view(-1, 1).float().to(device)
        y = y.view(-1, 1).float().to(device)
        y_onehot = torch.FloatTensor(y.shape[0], num_classes).to(device)
        y_onehot.zero_()
        y_onehot.scatter_(1, y.long(), 1)
        y_out.append(y_onehot.cpu().numpy())
        with torch.no_grad():
            output, hx1 = net.forward(x, hx1)
            p = nn.functional.softmax(output)
            output = torch.argmax(output, dim=1)
            pred.append(output.item())
            p_out.append(p.cpu().numpy())
            gt.append(y.item())
    acc = np.mean(np.array(pred) == np.array(gt))
    print(np.sum(np.array(pred) == np.array(gt)))
    print(acc)
    from sklearn.metrics import roc_auc_score
    p_out = np.array(p_out).reshape((-1, num_classes))
    y_out = np.array(y_out).reshape((-1, num_classes))
    print(y_out.shape)
    print(roc_auc_score(y_out, p_out))

