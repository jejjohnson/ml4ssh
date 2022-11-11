from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau,
    CosineAnnealingLR,
    StepLR,
    MultiStepLR,
)
from functools import partial
from ml_collections import config_dict


def default_lr_scheduler_config():
    config = config_dict.ConfigDict()
    config.lr_scheduler = "warmcosine"
    # STEP LR
    config.steps = 100
    # Step LR, MultiStep LR
    config.gamma = 1e-4
    # Cosine Annealing
    config.min_learning_rate = 1e-10
    config.T_max = 1000
    # WarmUp Cosine
    config.num_epochs = 100
    config.warmup_epochs = 10
    config.warmup_lr = 1e-10
    # Warmup Cosine + Cosine Annealing
    config.eta_min = 0
    # MULTISTEP
    config.milestones = [10, 100, 1_000]
    # Reduce LR
    config.patience = 10
    config.factor = 1
    config.mode = "min"
    return config


def lr_scheduler_factory(config=None):
    if config is None:
        config = default_lr_scheduler_config()

    if config.lr_scheduler == "reduce":
        lr_scheduler = partial(
            ReduceLROnPlateau,
            patience=config.patience,
            factor=config.factor,
            mode=config.mode,
        )
        return lr_scheduler

    elif config.lr_scheduler == "cosine":
        return partial(
            CosineAnnealingLR,
            T_max=config.T_max,
            eta_min=config.eta_min,
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
