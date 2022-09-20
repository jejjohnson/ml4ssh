import torch
import torch.nn as nn


class InputScalingTransform(nn.Module):
    def __init__(self, x_min, x_max):
        super().__init__()

        self.register_buffer("x_min", torch.FloatTensor(x_min))
        self.register_buffer("x_max", torch.FloatTensor(x_max))

    def forward(self, x, inverse=False):
        if not inverse:
            return self.transform(x)
        else:
            return self.inverse_transform(x)

    def transform(self, x):
        return (x - self.x_min) / (self.x_max - self.x_min)

    def inverse_transform(self, x):
        return x * (self.x_max - self.x_min) + self.x_min
