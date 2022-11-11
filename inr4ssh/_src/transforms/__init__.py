import torch
import torch.nn as nn


class Identity(nn.Module):
    """A transformation that converts from degrees to radians.
    It also includes an optional scaling factor.
    """

    def __init__(self):
        """_summary_

        Args:
            scaler (torch.Tensor, optional): an optional scaling factor.
        """
        super().__init__()

    def forward(self, x):
        return x

    def inverse(self, x):
        return x
