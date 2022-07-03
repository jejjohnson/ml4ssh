"""
Taken from: https://github.com/kklemon/gon-pytorch/blob/master/gon_pytorch/modules.py
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import copy
from functools import partial
from typing import Optional, List, Callable
from math import pi, sqrt


class CoordinateEncoding(nn.Module):
    def __init__(self, proj_matrix, is_trainable=False):
        super().__init__()
        if is_trainable:
            self.register_parameter('proj_matrix', nn.Parameter(proj_matrix))
        else:
            self.register_buffer('proj_matrix', proj_matrix)
        self.in_dim = self.proj_matrix.size(0)
        self.out_dim = self.proj_matrix.size(1) * 2

    def forward(self, x):
        shape = x.shape
        channels = shape[-1]

        assert channels == self.in_dim, f'Expected input to have {self.in_dim} channels (got {channels} channels)'

        x = x.reshape(-1, channels)
        x = x @ self.proj_matrix

        x = x.view(*shape[:-1], -1)
        x = 2 * pi * x

        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


class IdentityPositionalEncoding(CoordinateEncoding):
    def __init__(self, in_dim):
        super().__init__(torch.eye(in_dim))
        self.out_dim = in_dim

    def forward(self, x):
        return x


class NeRFPositionalEncoding(CoordinateEncoding):
    def __init__(self, in_dim, n=10):
        super().__init__((2.0 ** torch.arange(n))[None, :])
        self.out_dim = n * 2 * in_dim

    def forward(self, x):
        shape = x.shape
        x = x.unsqueeze(-1) * self.proj_matrix
        x = pi * x
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        x = x.view(*shape[:-1], -1)
        return x


class GaussianFourierFeatureTransform(CoordinateEncoding):
    def __init__(self, in_dim: int, mapping_size: int = 32, sigma: float = 1.0, is_trainable: bool = False, seed=None):
        super().__init__(self.get_transform_matrix(in_dim, mapping_size, sigma, seed=seed), is_trainable=is_trainable)
        self.mapping_size = mapping_size
        self.sigma = sigma
        self.seed = seed

    @classmethod
    def get_transform_matrix(cls, in_dim, mapping_size, sigma, seed=None):
        generator = None
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)
        return torch.normal(mean=0, std=sigma, size=(in_dim, mapping_size), generator=generator)

    @classmethod
    def from_matrix(cls, projection_matrix):
        in_dim, mapping_size = projection_matrix.shape
        feature_transform = cls(in_dim, mapping_size)
        feature_transform.projection_matrix.data = projection_matrix
        return feature_transform

    def __repr__(self):
        return f'{self.__class__.__name__}(in_dim={self.in_dim}, mapping_size={self.mapping_size}, sigma={self.sigma})'