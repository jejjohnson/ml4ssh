from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR, MultiStepLR
import torch
from functools import partial

def optimizer_factory(config):

    if config.optimizer.optimizer == "adam":
        return partial(torch.optim.Adam, lr=config.optimizer.learning_rate)
    elif config.optimizer.optimizer == "adamw":
        return partial(torch.optim.AdamW, lr=config.optimizer.learning_rate)
    elif config.optimizer.optimizer == "adamax":
        return partial(torch.optim.Adamax, lr=config.optimizer.learning_rate)
    else:
        raise ValueError(f"Unrecognized optimizer: {config.optimizer}")


def lr_scheduler_factory(config):

    if config.lr_scheduler.lr_scheduler == "reduce":
        lr_scheduler = partial(
            ReduceLROnPlateau,
            patience=config.lr_scheduler.patience,
            factor=config.lr_scheduler.factor,
            mode="min"
        )
        return lr_scheduler

    elif config.lr_scheduler.lr_scheduler == "cosine":
        return partial(
            CosineAnnealingLR,
            T_max=config.dataloader.batch_size * config.optimizer.num_epochs,
            eta_min=config.lr_scheduler.min_learning_rate,
        )

    elif config.lr_scheduler.lr_scheduler == "onecyle":
        raise NotImplementedError()

    elif config.lr_scheduler.lr_scheduler == "step":
        return partial(
            StepLR,
            step_size=config.lr_scheduler.steps,
            gamma=config.lr_scheduler.gamma
        )
    elif config.lr_scheduler.lr_scheduler == "multistep":
        return partial(MultiStepLR,
                       milestones=config.lr_scheduler.milestones,
                       gamma=config.lr_scheduler.gamma
                       )
    else:
        raise ValueError(f"Unrecognized learning rate scheduler: {config.lr_scheduler}")
