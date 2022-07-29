import torch


def l2_loss(y_true, y_pred, mask=None):

    if mask is not None:
        return (mask * (y_pred - y_true) ** 2).mean()
    else:
        return ((y_pred - y_true) ** 2).mean()


def l1_loss(y_true, y_pred, mask=None):

    if mask is not None:
        return (mask * torch.abs(y_pred - y_true)).mean()
    else:
        return torch.abs(y_pred - y_true).mean()
