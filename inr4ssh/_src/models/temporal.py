"""
Taken From: https://github.com/mbilos/stribor/blob/master/stribor/net/time_net.py
* Identity
* Linear
* Tanh
* Log
* Fourier 
* Random Fourier Features
* Sinusoidal Positional Encoding
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeIdentity(nn.Module):
    def __init__(self, out_dim: int, **kwargs):
        super().__init__()
        self.out_dim = out_dim

    def forward(self, t):
        return t.repeat_interleave(self.out_dim, dim=-1)

    # def derivative(self, t):
    #     return torch.ones_like(t).repeat_interleave(self.out_dim, dim=-1)
