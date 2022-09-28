import torch
import torch.nn as nn
import torch.nn.functional as F
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = ((1-pt)**self.gamma) * logpt
        loss = F.nll_loss(logpt, target, self.weight)
        return loss

class KDLoss():
    def __init__(self, W = None, alpha = 0.1, T = 100, **kwargs):
        self.alpha = alpha
        self.T = T
        self.W = W
        self.FCLoss = FocalLoss(weight=W)

    def __call__(self, output, label, teacher_output):
        T = self.T
        alpha = self.alpha
        W = self.W
        #loss1 = nn.KLDivLoss(reduction='none')(logpt, F.softmax(teacher_output/T, dim=1))
        #print(F.softmax(teacher_output / T, dim=1))
        loss1 = nn.KLDivLoss(reduction='none')(F.log_softmax(output/T, dim=1), F.softmax(teacher_output/T, dim=1))
        loss1 *= self.W 
        #p = F.softmax(output/T, dim=1)
        #pt = F.softmax(teacher_output/T, dim=1)
        #loss1 *= ((p - pt) / pt) ** 2
        loss1 = torch.mean(loss1, axis=1) * (alpha * T * T)
        #loss2 = self.FCLoss(output, label)
        loss2 = F.cross_entropy(output, label, weight=self.W) * (1. - alpha)
        #loss2 = self.FCLoss(output, label) * (1. - alpha)
        #loss1 = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(output/T, dim=1),
        #                     F.softmax(teacher_output/T, dim=1)) * (alpha * T * T)
        
        KD_loss = loss1 + loss2
        #KD_loss = self.FCLoss(output, label)
        return KD_loss.mean()

class NewKDLoss():
    def __init__(self, W = None, alpha = 0.1, T = 100, **kwargs):
        self.alpha = alpha
        self.T = T
        self.W = W
        self.FCLoss = FocalLoss(weight=W)

    def __call__(self, output, label, teacher_output):
        T = self.T
        alpha = self.alpha
        W = self.W
        T_pred = torch.argmax(teacher_output, dim=1)
        loss1, loss2 = 0.0, 0.0
        for i in range(output.shape[0]):
            model_out = output[i].view(1, -1)
            T_model_out= teacher_output[i].view(1, -1)
            cur_label = label[i].view(-1)
            if T_pred[i].item() == label[i].item():
                temp_loss = nn.KLDivLoss(reduction='none')(F.log_softmax(model_out/T, dim=1), F.softmax(T_model_out/T, dim=1))
                loss1 += torch.mean(temp_loss, axis=1) * (alpha * T * T)
                loss2 += F.cross_entropy(model_out, cur_label, weight=W) * (1. - alpha)
            else:
                loss2 += F.cross_entropy(model_out, cur_label, weight=W)
        KD_loss = loss1 + loss2
        return KD_loss.mean()
    
class CustomBCE():
    def __init__(self, class_weights=None, **kwargs):
        self.class_weights = class_weights

    def __call__(self, output, target):
        output = torch.sigmoid(output)

        if self.class_weights is not None:
            loss = self.class_weights[1] * (target * torch.log(output)) + \
                   self.class_weights[0] * ((1 - target) * torch.log(1 - output))
        else:
            loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

        loss = torch.neg(torch.mean(loss))
        return loss
