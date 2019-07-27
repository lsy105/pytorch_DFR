import torch
import torch.nn
from torch.autograd.function import Function

class FakeQuantize(Function):
    @staticmethod
    def forward(ctx, in_data, num_bits=8):
        min_data, max_data = torch.min(in_data), torch.max(in_data)
        min_data = torch.clamp(min_data, max=0)
        max_data = torch.clamp(max_data, min=0)
        qmin = 0
        qmax = 2**num_bits - 1
        scale = (max_data - min_data) / (qmax - qmin)
        zero_point = int(qmin - min_data / scale)
        new_data = torch.clamp(in_data, min=min_data, max=max_data) 
        output = torch.round((new_data - min_data) / scale) + min_data
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None 

def Quantize(x, num_bits=8):
    return FakeQuantize().apply(x, num_bits)

