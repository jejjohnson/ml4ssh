import torch
import torch.nn as nn
from torch.nn import ReLU
from .utils import exists
from .mlp import MLP

class FourierFeatureMLP(MLP):
    def __init__(
            self,
            dim_hidden,
            dim_out,
            num_layers,
            encoder,
            activation = None,
            use_bias = True,
            final_activation = None,
    ):

        super().__init__(
            dim_in=encoder.out_dim,
            dim_hidden=dim_hidden,
            dim_out=dim_out,
            num_layers=num_layers,
            activation=activation,
            use_bias=use_bias,
            final_activation=final_activation
         )
        self.encoder = encoder

    def forward(self, x):

        # encode the inputs
        x = self.encoder(x)

        # loop through the layers
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)

        return self.last_layer(x)