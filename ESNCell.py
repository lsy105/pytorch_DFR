import torch
import numpy as np
import torch.nn as nn

class ESNCell(nn.Module):
    def __init__(self, n_in, n_hidden, n_output, sparsity=0.5):
        super(ESNCell, self).__init__()
        self.linear = nn.Linear(n_hidden, n_output, bias=False)
        self.act = torch.nn.ReLU()
        self.sparsity = sparsity
        self.register_buffer('w_in', self.generate_w_in(n_hidden, n_in))
        self.register_buffer('w_rev', self.generate_w_rev(n_hidden, n_hidden))
        self.register_buffer('w_fb', self.generate_w_fb(n_hidden, n_output))
        self.register_buffer('w_bias', self.generate_bias(n_hidden))
  
    def spectral_radius(self, m):
        return torch.max(torch.abs(torch.eig(m)[0]))

    def generate_w_in(self, n_in, n_out):
        w = np.random.choice([0, -1.0, 1.0], \
                             (n_in, n_out), \
                             p = [1 - self.sparsity,  \
                                  self.sparsity / 2,
                                  self.sparsity / 2]).astype(np.float32)
        #w[w == 1] = np.random.rand(len(w[w == 1])) * 2.0 - 1.0
        w = torch.from_numpy(w)
        return w

    def generate_w_rev(self, n_in, n_out):
        w = np.random.choice([0, 1], 
                             (n_in, n_out), 
                             p = [1 - self.sparsity, 
                                  self.sparsity]).astype(np.float32)
        #for spectral_radius
        w[w == 1] = np.random.rand(len(w[w == 1])) * 2.0 - 1.0
        w = torch.from_numpy(w)
        w *= 1.2 / self.spectral_radius(w)
        return w
           
    def generate_w_fb(self, n_in, n_out):
        return torch.rand(n_in, n_out) * 2 - 1.0 

    def generate_bias(self, n_in):
        # creates a mask with either value either being m_amp or - m_amp
        bias = torch.rand(n_in) * 2 - 1.0 
        return bias
                
    def forward(self, u, x_t_1, y_t_1):
        x_in = (self.w_in * u).view(-1)
        x_p = torch.mv(self.w_rev, x_t_1)
        x_y = torch.mv(self.w_fb, y_t_1)
        act_x = 0.5 * x_p + 0.5 * self.act(x_in + x_p)
        return self.linear(act_x), act_x
