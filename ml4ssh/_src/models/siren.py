"""
Taken From: https://github.com/lucidrains/siren-pytorch
"""

import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from .activations import Sine
from .utils import exists, cast_tuple

from .operators import mod_additive, mod_multiplicative
from typing import Callable, Optional

# helpers



# class Siren(nn.Module):
#     def __init__(self, dim_in, dim_out, w0 = 1., c = 6., is_first = False, use_bias = True, activation = None):
#         super().__init__()
#         self.dim_in = dim_in
#         self.is_first = is_first

#         weight = torch.zeros(dim_out, dim_in)
#         bias = torch.zeros(dim_out) if use_bias else None
#         self.init_(weight, bias, c = c, w0 = w0)

#         self.weight = nn.Parameter(weight)
#         self.bias = nn.Parameter(bias) if use_bias else None
#         self.activation = Sine(w0) if activation is None else activation

#     def init_(self, weight, bias, c, w0):
#         dim = self.dim_in

#         w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
#         weight.uniform_(-w_std, w_std)

#         if exists(bias):
#             bias.uniform_(-w_std, w_std)

#     def forward(self, x):
#         out =  F.linear(x, self.weight, self.bias)
#         out = self.activation(out)
#         return out

class Siren(nn.Module):
    def __init__(self, dim_in, dim_out, w0 = 1., c = 6., is_first = False, use_bias = True, activation = None, resnet = False):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c = c, w0 = w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation
        self.resnet = resnet

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if exists(bias):
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out =  F.linear(x, self.weight, self.bias)
        out = self.activation(out)
        if self.resnet:
            out = 0.5 * ( x + out )
        return out



class SirenNet(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, w0 = 1., w0_initial = 30., c = 6.0, use_bias = True, final_activation = None, resnet = False):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden

        self.layers = nn.ModuleList([])
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden
            res_first = False

            self.layers.append(Siren(
                dim_in = layer_dim_in,
                dim_out = dim_hidden,
                w0 = layer_w0,
                c = c,
                use_bias = use_bias,
                is_first = is_first,
                resnet = True if resnet and res_first else False
            ))
            if res_first:
                res_first = False

        final_activation = nn.Identity() if not exists(final_activation) else final_activation
        self.last_layer = Siren(dim_in = dim_hidden, dim_out = dim_out, w0 = w0, use_bias = use_bias, activation = final_activation)

    def forward(self, x: torch.Tensor, mods: Optional[torch.Tensor] = None):
        mods = cast_tuple(mods, self.num_layers)

        for layer, mod in zip(self.layers, mods):
            x = layer(x)

            if exists(mod):
                
                x *= rearrange(mod, 'd -> () d')

        return self.last_layer(x)


# modulatory feed forward

class Modulator(nn.Module):
    def __init__(self, dim_in, dim_hidden, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([])

        for ind in range(num_layers):
            is_first = ind == 0
            dim = dim_in if is_first else (dim_hidden + dim_in)

            self.layers.append(nn.Sequential(
                nn.Linear(dim, dim_hidden),
                nn.ReLU()
            ))

    def forward(self, z):
        x = z
        hiddens = []
        for layer in self.layers:
            x = layer(x)
            hiddens.append(x)
            x = torch.cat((x, z))

        return tuple(hiddens)


class ModulatedSirenNet(nn.Module):
    def __init__(self, 
                 dim_in, 
                 dim_hidden, 
                 dim_out, 
                 num_layers: int=5, 
                 latent_dim: int=512,
                 num_layers_latent: int=3,
                 operation: str="multiply",
                 w0: float = 1.,
                 w0_initial: float = 30., 
                 c: float = 6.0, 
                 use_bias: bool = True, 
                 final_activation: Optional[nn.Module] = None, 
                 resnet: bool = False
                ):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden

        self.layers = nn.ModuleList([])
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden
            res_first = False

            self.layers.append(Siren(
                dim_in = layer_dim_in,
                dim_out = dim_hidden,
                w0 = layer_w0,
                c = c,
                use_bias = use_bias,
                is_first = is_first,
                resnet = True if resnet and res_first else False
            ))
            if res_first:
                res_first = False

                
            self.modulator = Modulator(
                dim_in=latent_dim,
                dim_hidden=dim_hidden,
                num_layers=num_layers_latent,
            )
        if operation in ["mult", "multiply", "multiplicative"]:
            operation = lambda x, z: x * z
        elif operation in ["add", "addition", "additive"]:
            operation = lambda x, z: x + z
        else:
            raise ValueError(f"Unrecognized operation: {operation}")

        self.operation = operation

        final_activation = nn.Identity() if not exists(final_activation) else final_activation
        self.last_layer = Siren(dim_in = dim_hidden, dim_out = dim_out, w0 = w0, use_bias = use_bias, activation = final_activation)

    def forward(self, x, latent):
        
        mods = self.modulator(latent)
        
        mods = cast_tuple(mods, self.num_layers)
        

        for layer, mod in zip(self.layers, mods):
            x = layer(x)
            mod = rearrange(mod, 'd -> () d')
            x = self.operation(x, mod)

        return self.last_layer(x)

class SirenWrapper(nn.Module):
    def __init__(self, net, image_width, image_height, latent_dim = None):
        super().__init__()
        assert isinstance(net, SirenNet), 'SirenWrapper must receive a Siren network'

        self.net = net
        self.image_width = image_width
        self.image_height = image_height

        self.modulator = None
        if exists(latent_dim):
            self.modulator = Modulator(
                dim_in = latent_dim,
                dim_hidden = net.dim_hidden,
                num_layers = net.num_layers
            )

        tensors = [torch.linspace(-1, 1, steps = image_height), torch.linspace(-1, 1, steps = image_width)]
        mgrid = torch.stack(torch.meshgrid(*tensors, indexing = 'ij'), dim=-1)
        mgrid = rearrange(mgrid, 'h w c -> (h w) c')
        self.register_buffer('grid', mgrid)

    def forward(self, img = None, *, latent = None):
        modulate = exists(self.modulator)
        assert not (modulate ^ exists(latent)), 'latent vector must be only supplied if `latent_dim` was passed in on instantiation'

        mods = self.modulator(latent) if modulate else None

        coords = self.grid.clone().detach().requires_grad_()
        out = self.net(coords, mods)
        out = rearrange(out, '(h w) c -> () c h w', h = self.image_height, w = self.image_width)

        if exists(img):
            return F.mse_loss(img, out)

        return out


