from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pathlib import Path
def get_callbacks(config, wandb_logger=None):
    callbacks = []
    if config.callbacks.early_stopping is True:
        cb = EarlyStopping(
            monitor="valid_loss",
            mode="min",
            patience=config.callbacks.patience,
        )
        callbacks.append(cb)
    if config.callbacks.model_checkpoint is True:
        if wandb_logger is not None:
            log_dir = wandb_logger.experiment.dir
        else:
            log_dir = "./"
        cb = ModelCheckpoint(
            dirpath=str(Path(log_dir).joinpath("checkpoints")),
            monitor="valid_loss",
            mode="min",
            save_top_k=2,
        )
        callbacks.append(cb)
    return callbacks

# from skorch.callbacks import EarlyStopping, LRScheduler, WandbLogger
#
# def get_callbacks(config, wandb_logger=None):
#     callbacks = []
#     if config.wandb is True:
#         cb = EarlyStopping(
#             monitor="valid_loss",
#             patience=config.patience,
#         )
#         callbacks.append(("early_stopping", cb))
#     if config.early_stopping is True:
#         cb = WandbLogger(
#             wandb_run=wandb_logger,
#             save_model=config.save_model
#         )
#         callbacks.append(("wandb", cb))
#     return callbacks
