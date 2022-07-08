"""
Taken From: https://github.com/boschresearch/multiplicative-filter-networks
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from .utils import exists, cast_tuple
from typing import Callable, Optional


class MFNBase(nn.Module):
    """
    Multiplicative filter network base class.
    Expects the child class to define the 'filters' attribute, which should be
    a nn.ModuleList of n_layers+1 filters with output equal to hidden_size.
    """

    def __init__(
        self, dim_hidden: int, dim_out: int, num_layers: int, weight_scale: float, use_bias: bool=True, final_activation: Optional[nn.Module] = None
    ):
        super().__init__()

        self.linear = nn.ModuleList(
            [nn.Linear(dim_hidden, dim_hidden, use_bias) for _ in range(num_layers)]
        )
        self.output_linear = nn.Linear(dim_hidden, dim_out)
        self.output_act = nn.Identity() if not exists(final_activation) else final_activation

        for lin in self.linear:
            lin.weight.data.uniform_(
                -np.sqrt(weight_scale / dim_hidden),
                np.sqrt(weight_scale / dim_hidden),
            )

    def forward(self, x):
        out = self.filters[0](x)
        for i in range(1, len(self.filters)):
            out = self.filters[i](x) * self.linear[i - 1](out)
        out = self.output_linear(out)

        out = self.output_act(out)

        return out


class FourierLayer(nn.Module):
    """
    Sine filter as used in FourierNet.
    """

    def __init__(self, dim_in: int, dim_out: int, weight_scale: float):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.linear.weight.data *= weight_scale  # gamma
        self.linear.bias.data.uniform_(-np.pi, np.pi)

    def forward(self, x):
        return torch.sin(self.linear(x))


class FourierNet(MFNBase):
    def __init__(
        self,
        dim_in: int,
        dim_hidden: int,
        dim_out: int,
        num_layers=3,
        input_scale=256.0,
        weight_scale=1.0,
        use_bias=True,
        final_activation=False,
    ):
        super().__init__(
            dim_hidden, dim_out, num_layers, weight_scale, use_bias, final_activation
        )
        self.filters = nn.ModuleList(
            [
                FourierLayer(dim_in, dim_hidden, input_scale / np.sqrt(num_layers + 1))
                for _ in range(num_layers + 1)
            ]
        )

class GaborLayer(nn.Module):
    """
    Gabor-like filter as used in GaborNet.
    """

    def __init__(self, dim_in: int, dim_out: int, weight_scale: float, alpha: float=1.0, beta: float=1.0):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.mu = nn.Parameter(2 * torch.rand(dim_out, dim_in) - 1)
        self.gamma = nn.Parameter(
            torch.distributions.gamma.Gamma(alpha, beta).sample((dim_out,))
        )
        self.linear.weight.data *= weight_scale * torch.sqrt(self.gamma[:, None])
        self.linear.bias.data.uniform_(-np.pi, np.pi)

    def forward(self, x):
        D = (
            (x ** 2).sum(-1)[..., None]
            + (self.mu ** 2).sum(-1)[None, :]
            - 2 * x @ self.mu.T
        )
        return torch.sin(self.linear(x)) * torch.exp(-0.5 * D * self.gamma[None, :])


class GaborNet(MFNBase):
    def __init__(
        self,
        dim_in: int,
        dim_hidden: int,
        dim_out: int,
        num_layers: int=3,
        input_scale: float=256.0,
        weight_scale: float=1.0,
        alpha=6.0,
        beta=1.0,
        use_bias: bool=True,
        final_activation: Optional[nn.Module] = None,
    ):
        super().__init__(
            dim_hidden, dim_out, num_layers, weight_scale, use_bias, final_activation
        )
        self.filters = nn.ModuleList(
            [
                GaborLayer(
                    dim_in,
                    dim_hidden,
                    input_scale / np.sqrt(num_layers + 1),
                    alpha / (num_layers + 1),
                    beta,
                )
                for _ in range(num_layers + 1)
            ]
        )