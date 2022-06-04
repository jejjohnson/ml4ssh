import torch
import torch.nn as nn
import torch.nn.functional as F


class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return sine(x, self.w0)


class GELU(nn.Module):
    '''GELU activation (https://arxiv.org/abs/1606.08415) as used in Sparse Transformers (https://arxiv.org/abs/1904.10509).'''

    def forward(self, input):
        return gelu(input)


class Swish(nn.Module):
    '''Swish activation (https://arxiv.org/abs/1710.05941).'''

    def forward(self, input):
        return swish(input)


class ConcatReLU(nn.Module):
    '''Concatenated ReLU (http://arxiv.org/abs/1603.05201).'''

    def forward(self, input):
        return concat_relu(input)


class ConcatELU(nn.Module):
    '''Like concatenated ReLU (http://arxiv.org/abs/1603.05201), but with ELU instead.'''

    def forward(self, input):
        return concat_elu(input)


class GatedTanhUnit(nn.Module):
    '''Gated Tanh activation.'''

    def __init__(self, dim=-1):
        super(GatedTanhUnit, self).__init__()
        self.dim = dim

    def forward(self, x):
        return gated_tanh(x, dim=self.dim)


def sine(x, w0):
    '''GELU activation (https://arxiv.org/abs/1606.08415) as used in Sparse Transformers (https://arxiv.org/abs/1904.10509).'''
    return torch.sin(w0 * x)

def gelu(x):
    '''GELU activation (https://arxiv.org/abs/1606.08415) as used in Sparse Transformers (https://arxiv.org/abs/1904.10509).'''
    return x * torch.sigmoid(1.702 * x)


def swish(x):
    '''Swish activation (https://arxiv.org/abs/1710.05941).'''
    return x * torch.sigmoid(x)


def concat_relu(x):
    '''Concatenated ReLU (http://arxiv.org/abs/1603.05201).'''
    return F.relu(torch.cat([x, -x], dim=1))


def concat_elu(x):
    '''Like concatenated ReLU (http://arxiv.org/abs/1603.05201), but with ELU instead.'''
    return F.elu(torch.cat([x, -x], dim=1))


def gated_tanh(x, dim):
    '''Gated Tanh activation.'''
    x_tanh, x_sigmoid = torch.chunk(x, 2, dim=dim)
    return torch.tanh(x_tanh) * torch.sigmoid(x_sigmoid)