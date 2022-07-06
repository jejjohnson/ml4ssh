from skorch.callbacks import EarlyStopping, LRScheduler, WandbLogger
import torch

def get_optimizer(config):

    if config.optimizer == "adam":
        return torch.optim.Adam
    elif config.optimizer == "adamw":
        return torch.optim.AdamW
    elif config.optimizer == "adamax":
        return torch.optim.Adamax
    else:
        raise ValueError(f"Unrecognized optimizer: {config.optimizer}")


def get_lr_scheduler(config):

    if config.lr_scheduler == "reduce":
        lr_scheduler = LRScheduler(
            policy="ReduceLROnPlateau",
            monitor="valid_loss",
            mode="min",
            factor=config.factor,
            patience=config.patience
        )
        return lr_scheduler
    elif config.lr_scheduler == "cosine":
        raise NotImplementedError()
    elif config.lr_scheduler == "hey":
        raise NotImplementedError()
    else:
        raise ValueError(f"Unrecognized learning rate scheduler: {config.lr_scheduler}")
