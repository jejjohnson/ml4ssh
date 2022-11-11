import torch
from ml_collections import config_dict
from functools import partial


def default_optimizer_config():
    config = config_dict.ConfigDict()
    config.optimizer = "adam"
    config.learning_rate = 1e-4
    return config


def optimizer_factory(config=None):
    if config is None:
        config = default_optimizer_config()

    if config.optimizer == "adam":
        return partial(torch.optim.Adam, lr=config.learning_rate)
    elif config.optimizer == "adamw":
        return partial(torch.optim.AdamW, lr=config.learning_rate)
    elif config.optimizer == "adamax":
        return partial(torch.optim.Adamax, lr=config.learning_rate)
    else:
        raise ValueError(f"Unrecognized optimizer: {config.optimizer}")
