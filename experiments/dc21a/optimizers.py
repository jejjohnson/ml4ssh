import torch
from functools import partial
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau,
    CosineAnnealingLR,
    StepLR,
    MultiStepLR,
)


def optimizer_factory(config):

    if config.optimizer == "adam":
        return partial(torch.optim.Adam, lr=config.learning_rate)
    elif config.optimizer == "adamw":
        return partial(torch.optim.AdamW, lr=config.learning_rate)
    elif config.optimizer == "adamax":
        return partial(torch.optim.Adamax, lr=config.learning_rate)
    else:
        raise ValueError(f"Unrecognized optimizer: {config.optimizer}")


def lr_scheduler_factory(config):

    if config.lr_scheduler == "reduce":
        lr_scheduler = partial(
            ReduceLROnPlateau,
            patience=config.patience,
            factor=config.factor,
            mode="min",
        )
        return lr_scheduler

    elif config.lr_scheduler == "cosine":
        return partial(
            CosineAnnealingLR,
            T_max=config.dataloader.batch_size * config.optimizer.num_epochs,
            eta_min=config.min_learning_rate,
        )

    elif config.lr_scheduler == "onecyle":
        raise NotImplementedError()

    elif config.lr_scheduler == "step":
        return partial(StepLR, step_size=config.steps, gamma=config.gamma)
    elif config.lr_scheduler == "multistep":
        return partial(MultiStepLR, milestones=config.milestones, gamma=config.gamma)

    elif config.lr_scheduler == "warmcosine":
        return partial(
            LinearWarmupCosineAnnealingLR,
            warmup_epochs=config.warmup_epochs,
            max_epochs=config.max_epochs,
            warmup_start_lr=config.warmup_lr,
            eta_min=config.eta_min,
        )
    else:
        raise ValueError(f"Unrecognized learning rate scheduler: {config.lr_scheduler}")
