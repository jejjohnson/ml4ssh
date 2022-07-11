import torch.nn as nn


def loss_factory(config):

    if config.losses.loss == "mse":
        return nn.MSELoss(reduction=config.losses.reduction)

    elif config.losses.loss == "nll":
        return nn.GaussianNLLLoss(reduction=config.losses.reduction)

    elif config.losses.loss == "kld":
        return nn.KLDivLoss(reduction=config.losses.reduction)

    else:
        raise ValueError(f"Unrecognized loss: {config.loss}")

def regularization_factory(config):
    if config.losses.reg == "qg":
        raise NotImplementedError()
    else:
        raise ValueError(f"Unrecognized ")