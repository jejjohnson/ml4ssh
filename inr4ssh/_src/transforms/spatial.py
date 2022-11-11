import torch
import torch.nn as nn


class SpatialDegree2Rads(nn.Module):
    """A transformation that converts from degrees to radians.
    It also includes an optional scaling factor.
    """

    def __init__(self, scaler=1):
        """_summary_

        Args:
            scaler (torch.Tensor, optional): an optional scaling factor.
        """
        super().__init__()
        self.register_buffer("scaler", torch.FloatTensor(scaler))
        self.output_dim = 3

    def forward(self, x):
        return self.scaler * torch.deg2rad(x)

    def inverse(self, x):
        return torch.rad2deg(x / self.scaler)
