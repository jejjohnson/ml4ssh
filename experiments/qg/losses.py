import sys, os
from pyprojroot import here

# spyder up to find the root
root = here(project_files=[".root"])

# append to path
sys.path.append(str(root))

import torch
import torch.nn as nn
import torch.nn.functional as F
from inr4ssh._src.operators import differential_simp as diffops_simp


def initialize_data_loss(config):

    if config.loss == "mse":
        loss_fn = nn.MSELoss(reduction=config.reduction)
    elif config.loss == "mae":
        loss_fn = nn.SmoothL1Loss(reduction=config.reduction)
    else:
        raise ValueError(f"Unrecognized loss: {config.loss}")

    return loss_fn


class RegQG(nn.Module):
    def __init__(self, alpha: float = 1e-4):
        super().__init__()

        alpha = torch.Tensor([alpha])

        self.register_buffer("alpha", alpha)

    def forward(self, x, f):
        with torch.set_grad_enabled(True):
            x = torch.autograd.Variable(x, requires_grad=True)

            u = f(x)

            # 𝛁𝑢
            grad_nn = diffops_simp.gradient(u, x)

            # div𝛁𝑢 = ∂𝑥𝛁𝑢 + ∂𝑦𝛁𝑢 = △𝑢
            q_nn = diffops_simp.divergence(grad_nn, x, [0, 1])

            #
            dlaplacU = diffops_simp.gradient(q_nn, x)
            Jacob_U_laplacU = (
                grad_nn[:, 0] * dlaplacU[:, 1] - grad_nn[:, 1] * dlaplacU[:, 0]
            )

            pde_loss = F.mse_loss(
                dlaplacU[:, 2] + Jacob_U_laplacU, torch.zeros_like(Jacob_U_laplacU)
            )
            return self.alpha * pde_loss
