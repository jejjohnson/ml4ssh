from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pathlib import Path

# wandb
from pytorch_lightning.loggers import LoggerCollection, WandbLogger
import wandb

# pytorch
from pytorch_lightning import Callback
import pytorch_lightning as pl
import torch

# others
import glob
import os


def get_wandb_logger(trainer: pl.Trainer) -> WandbLogger:
    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    if isinstance(trainer.logger, LoggerCollection):
        for logger in trainer.logger:
            if isinstance(logger, WandbLogger):
                return logger

    raise Exception(
        "You are using wandb related callback, but WandbLogger was not found for some reason..."
    )


class UploadCheckpointsToWandbAsArtifact(Callback):
    """Upload experiment checkpoints to wandb as an artifact at the end of training."""

    def __init__(self, ckpt_dir: str = "checkpoints/", upload_best_only: bool = False):
        self.ckpt_dir = ckpt_dir
        self.upload_best_only = upload_best_only

    def on_train_end(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        ckpts = wandb.Artifact("experiment-ckpts", type="checkpoints")

        if self.upload_best_only:
            ckpts.add_file(trainer.checkpoint_callback.best_model_path)
        else:
            for path in glob.glob(
                os.path.join(self.ckpt_dir, "**/*.ckpt"), recursive=True
            ):
                ckpts.add_file(path)

        experiment.log_artifact(ckpts)


class WatchModelWithWandb(Callback):
    """Make WandbLogger watch model at the beginning of the run."""

    def __init__(self, log: str = "gradients", log_freq: int = 100):
        self.log = log
        self.log_freq = log_freq

    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        logger.watch(model=trainer.model, log=self.log, log_freq=self.log_freq)


def get_callbacks(config, wandb_logger=None):
    callbacks = []
    if config.early_stopping is True:
        cb = EarlyStopping(
            monitor="valid_loss",
            mode="min",
            patience=config.patience,
        )
        callbacks.append(cb)
    if config.model_checkpoint is True:
        if wandb_logger is not None:
            log_dir = wandb_logger.experiment.dir
        else:
            log_dir = "./"
        cb = ModelCheckpoint(
            dirpath=str(Path(log_dir).joinpath("checkpoints")),
            monitor="valid_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        )
        callbacks.append(cb)

        cb_2 = UploadCheckpointsToWandbAsArtifact(
            ckpt_dir=str(Path(log_dir).joinpath("checkpoints")), upload_best_only=False
        )

        callbacks.append(cb_2)

    if config.watch_model is True:

        cb = WatchModelWithWandb()

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
