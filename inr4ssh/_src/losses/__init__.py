import torch.nn as nn
from ml_collections import config_dict


def default_loss_config():
    config = config_dict.ConfigDict()
    config.loss = "mse"
    config.reduction = "reduction"
    return config


def loss_factory(config=None):
    if config is None:
        config = default_loss_config()

    if config.loss == "mse":
        return nn.MSELoss(reduction=config.reduction)

    elif config.loss == "nll":
        return nn.GaussianNLLLoss(reduction=config.reduction)

    elif config.loss == "kld":
        return nn.KLDivLoss(reduction=config.reduction)

    else:
        raise ValueError(f"Unrecognized loss: {config.loss}")


def regularization_factory(config):
    if config.losses.reg == "qg":
        raise NotImplementedError()
    else:
        raise ValueError(f"Unrecognized ")
