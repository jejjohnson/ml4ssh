import torch
import torch.nn as nn
from torch.nn import ReLU
from .utils import exists


class MLP(nn.Module):
    def __init__(
            self,
            dim_in,
            dim_hidden,
            dim_out,
            num_layers,
            activation = None,
            use_bias = True,
            final_activation = None,
    ):
        super().__init__()
        self.dim_in = dim_in

        self.layers = nn.ModuleList([])

        for ilayer in range(num_layers):
            is_first = ilayer == 0
            layer_dim_in = dim_in if is_first else dim_hidden

            self.layers.append(
                nn.Linear(
                    in_features=layer_dim_in,
                    out_features=dim_hidden,
                    bias=use_bias,

                )
            )

        self.activation = ReLU() if activation is None else activation
        final_activation = nn.Identity() if not exists(final_activation) else final_activation
        self.last_layer = nn.Linear(in_features=dim_hidden, out_features=dim_out)

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)

        return self.last_layer(x)

