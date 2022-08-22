# from ..operators.differential_simp import gradient
from ..operators.differential import grad as gradient
import torch
import torch.nn as nn


class QGRegularization(nn.Module):
    def __init__(
        self, f: float = 1e-4, g: float = 9.81, Lr: float = 1.0, reduction: str = "mean"
    ):
        super().__init__()

        self.f = f
        self.g = g
        self.Lr = Lr
        self.reduction = reduction

    def forward(self, out, x):

        x = x.requires_grad_(True)

        # gradient, nabla x
        out_jac = gradient(out, x)
        assert out_jac.shape == x.shape

        # calculate term 1
        loss1 = _qg_term1(out_jac, x, self.f, self.g, self.Lr)
        # calculate term 2
        loss2 = _qg_term2(out_jac, self.f, self.g, self.Lr)

        loss = (loss1 + loss2).square()

        if self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "mean":
            return loss.mean()
        else:
            raise ValueError(f"Unrecognized reduction: {self.reduction}")


def qg_constants(f, g, L_r):
    c_1 = g / f
    c_2 = 1 / L_r**2
    c_3 = c_1 * c_2
    return c_1, c_2, c_3


def qg_loss(
    ssh, x, f: float = 1e-4, g: float = 9.81, Lr: float = 1.0, reduction: str = "mean"
):
    # gradient, nabla x
    # x = x.detach().clone().requires_grad_(True)
    # print(x.shape, ssh.shape)
    ssh_jac = gradient(ssh, x)
    assert ssh_jac.shape == x.shape

    # calculate term 1
    loss1 = _qg_term1(ssh_jac, x, f, g, Lr)
    # calculate term 2
    loss2 = _qg_term2(ssh_jac, f, g, Lr)

    loss = (loss1 + loss2).square()

    if reduction == "sum":
        return loss.sum()
    elif reduction == "mean":
        return loss.mean()
    else:
        raise ValueError(f"Unrecognized reduction: {reduction}")


def _qg_term1(u_grad, x_var, f: float = 1.0, g: float = 1.0, L_r: float = 1.0):
    """
    t1 = ∂𝑡∇2𝑢 + 𝑐1 ∂𝑥𝑢 ∂𝑦∇2𝑢 − 𝑐1 ∂𝑦𝑢 ∂𝑥∇2𝑢
    Parameters:
    ----------
    u_grad: torch.Tensor, (B, Nx, Ny, T)
    x_var: torch.Tensor, (B,
    f: float, (,)
    g: float, (,)
    Lr: float, (,)

    Returns:
    --------
    loss : torch.Tensor, (B,)
    """

    x_var = x_var.requires_grad_(True)
    c_1, c_2, c_3 = qg_constants(f, g, L_r)

    # get partial derivatives | partial x, y, t
    u_x, u_y, u_t = torch.split(u_grad, [1, 1, 1], dim=1)

    # jacobian^2 x2, ∇2
    u_grad2 = gradient(u_grad, x_var)
    assert u_grad2.shape == x_var.shape

    # split jacobian -> partial x, partial y, partial t
    u_xx, u_yy, u_tt = torch.split(u_grad2, [1, 1, 1], dim=1)
    assert u_xx.shape == u_yy.shape == u_tt.shape

    # laplacian (spatial), nabla^2
    u_lap = u_xx + u_yy
    assert u_lap.shape == u_xx.shape == u_yy.shape

    # gradient of laplacian, ∇ ∇2
    u_lap_grad = gradient(u_lap, x_var)
    assert u_lap_grad.shape == x_var.shape

    # split laplacian into partials
    u_lap_grad_x, u_lap_grad_y, u_lap_grad_t = torch.split(u_lap_grad, [1, 1, 1], dim=1)
    assert u_lap_grad_x.shape == u_lap_grad_y.shape == u_lap_grad_t.shape

    # term 1
    loss = u_lap_grad_t + c_1 * u_x * u_lap_grad_y - c_1 * u_y * u_lap_grad_x
    assert loss.shape == u_lap_grad_t.shape == u_lap_grad_y.shape == u_lap_grad_x.shape

    return loss


def _qg_term2(u_grad, f: float = 1.0, g: float = 1.0, Lr: float = 1.0):
    """
    t2 = 𝑐2 ∂𝑡(𝑢) + 𝑐3 ∂𝑥(𝑢) ∂𝑦(𝑢) − 𝑐3 ∂𝑦(𝑢) ∂𝑥(𝑢)

    Parameters:
    ----------
    ssh_grad: torch.Tensor, (B, Nx, Ny, T)
    f: float, (,)
    g: float, (,)
    Lr: float, (,)

    Returns:
    --------
    loss : torch.Tensor, (B,)
    """
    _, c_2, c_3 = qg_constants(f, g, Lr)

    # get partial derivatives | partial x, y, t
    *_, u_t = torch.split(u_grad, [1, 1, 1], dim=1)

    # calculate term 2
    loss = -c_2 * u_t

    return loss
