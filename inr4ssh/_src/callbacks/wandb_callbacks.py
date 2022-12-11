# wandb
from typing import List
from pytorch_lightning.loggers import WandbLogger
import wandb

# pytorch
from pytorch_lightning import Callback
import pytorch_lightning as pl
import torch

# others
import glob
import os


def get_wandb_logger(trainer: pl.Trainer) -> WandbLogger:

    for logger in trainer.loggers:
        if isinstance(logger, WandbLogger):
            return trainer.logger

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
