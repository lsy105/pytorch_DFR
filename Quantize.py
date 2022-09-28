import torch
import torch.nn
import sys
from torch.autograd.function import Function

class FakeQuantize(Function):
    @staticmethod
    def forward(ctx, in_data, num_bits, min_data=None, max_data=None):
        if min_data is None or max_data is None:
            min_data, max_data = torch.min(in_data), torch.max(in_data)
        min_data = torch.clamp(min_data, max=0)
        max_data = torch.clamp(max_data, min=0)
        qmin = 0
        qmax = 2**num_bits - 1
        scale = (max_data - min_data) / (qmax - qmin)
        zero_point = int(qmin - min_data / scale)
        new_data = torch.clamp(in_data, min=min_data.item(), max=max_data.item()) 
        output = torch.round((new_data - min_data) / scale) * scale  + min_data
        #output = torch.clamp(output, min=min_data.item(), max=max_data.item()) 
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None 

class LogFakeQuantize(Function):
    @staticmethod
    def forward(ctx, in_data, min_data=None, max_data=None, num_bits=8):
        if min_data is None or max_data is None:
            min_data, max_data = torch.min(in_data), torch.max(in_data)
        min_data = torch.clamp(min_data, max=0)
        max_data = torch.clamp(max_data, min=0)
        qmin = 0
        qmax = 2**num_bits - 1
        scale = (max_data - min_data) / (qmax - qmin)
        scale = torch.pow(2, torch.round(torch.log2(scale)))
        zero_point = int(qmin - min_data / scale)
        new_data = torch.clamp(in_data, min=min_data.item(), max=max_data.item()) 
        output = torch.round((new_data - min_data) / scale) * scale  + min_data
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None 

class FixedQuantize(Function):
    @staticmethod
    def forward(ctx, in_data):
        #min_data, max_data = -1, 1
        #scale = 0.5
        #output = torch.round((in_data - min_data) / scale) * scale + min_data
        #return output
        return torch.sign(in_data)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None 

def Quantize(x, min_data=None, max_data=None, num_bits=8):
    #return FixedQuantize().apply(x)
    if min_data is not None:
        return FakeQuantize().apply(x, num_bits, min_data, max_data)
    else:
        return FakeQuantize().apply(x, num_bits)

def LogQuantize(x, min_data=None, max_data=None, num_bits=8):
    #return FixedQuantize().apply(x)
    if min_data is not None:
        return LogFakeQuantize().apply(x, min_data, max_data, num_bits)
    else:
        return LogFakeQuantize().apply(x)

def NewQuantize(x, min_data=None, max_data=None, num_bits=8):
    return FixedQuantize().apply(x)
