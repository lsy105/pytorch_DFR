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

def Run(t_alpha, t_T):
    #parameters
    batch_size = 512 
    test_batch_size = 512 
    grad_batch = 1
    epochs = 30 
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
    pre_trained = torch.load("SeqNoKD.pch")
    net.load_state_dict(pre_trained.state_dict(), strict=True)
    #optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.000, nesterov=True)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.0)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    #criterion = nn.CrossEntropyLoss()
    #W = torch.FloatTensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).to(device)
    criterion = KDLoss(alpha=t_alpha, T=t_T)
     
    def Eval(testloader, net, hx):
        count = 0
        pred, gt = [], []
        net = net.eval()
        save_x, save_y = [], []
        class_true = [0 for _ in range(10)]
        class_count = [0 for _ in range(10)]
        for idx, data in enumerate(testloader):
            x, y = data
            #x = x.view(x.shape[0], -1).float().to(device)
            #y = y.view(-1).float().to(device)
            x = x.float().to(device)
            y = y.view(-1).float().to(device)
            #save_x.append(x.detach().cpu().numpy().reshape((-1)))
            #save_y.append(y.detach().cpu().numpy().reshape((-1)))
            with torch.no_grad():
                output = net.forward(x, hx)
                output = torch.argmax(output, dim=1)
                #gt_t = int(y.cpu().item())
                #pred_t = int(output.cpu().item())
                #class_count[gt_t] += 1
                #if pred_t == gt_t:
                #    class_true[gt_t] += 1
                pred.append(output.cpu().numpy())
                gt.append(y.cpu().numpy())
        """
        temp_Q = QParams("test_data")
        W = np.array(save_x)
        temp_Q.Quantize(W, minval=-2, maxval=2, row=W.shape[0], col=W.shape[1], num_bits=10)
        #temp_Q.GetFloat(W, row=W.shape[0], col=W.shape[1])
        temp_Q = QParams("test_label")
        W = np.array(save_y) 
        temp_Q.GetFloat(W, row=W.shape[0], col=W.shape[1])
        """
        #print(class_true, class_count, np.array(class_true) / np.array(class_count))
        acc = np.mean(np.array(pred) == np.array(gt))
        print(acc)
    
    teacher = torch.load("RNNteacher95.pch")
    #teacher = torch.load("ImageDFRfloat93.pch")
    teacher.eval()
    
    for epoch in range(epochs):
        running_loss = 0.0
        count, count1 = 0, 0
        for idx, data in enumerate(trainloader):
            hx = [torch.zeros(batch_size, node_size).to(device) for _ in range(num_DFR)]
            t_hx = [torch.zeros(batch_size, node_size).to(device) for _ in range(num_DFR)]
            x, y = data
            #x = x.view(x.shape[0], -1).float().to(device)
            #y = y.view(-1).long().to(device)
            x = x.float().to(device)
            y = y.view(-1).long().to(device)
            optimizer.zero_grad()
            t_output = teacher(x, t_hx)
            output = net(x, hx) 
            loss = criterion(output, y, t_output)
            #loss = criterion(output, y)
            #running_loss += loss.item()
            loss.backward()
            optimizer.step()
        scheduler.step()
    hx = [torch.zeros(test_batch_size, node_size).to(device) for _ in range(num_DFR)]
    t_hx = [torch.zeros(test_batch_size, node_size).to(device) for _ in range(num_DFR)]
    Eval(testloader, net, hx)
    #Eval(testloader, teacher, t_hx)
    #Eval(trainloader, net, hx1)
    torch.save(net, "KDmodel" + str(epochs) + ".pch")


if __name__ == "__main__":
    alphas = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.99]
    Ts = [1.5, 2.0, 3.0, 5, 8, 10, 15, 20]
    for alpha in alphas:
        for T in Ts:
            print(alpha, T)
            Run(alpha, T)
