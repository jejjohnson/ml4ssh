from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar
from inr4ssh._src.callbacks.wandb_callbacks import (
    WatchModelWithWandb,
    UploadCheckpointsToWandbAsArtifact,
)


def get_callbacks(config, wandb_logger=None):
    callbacks = []
    if config.early_stopping is True:
        cb = EarlyStopping(
            monitor="val_loss",
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
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        )
        callbacks.append(cb)

    if config.wandb_artifact is True:

        cb = UploadCheckpointsToWandbAsArtifact(
            ckpt_dir=str(Path(log_dir).joinpath("checkpoints")), upload_best_only=False
        )

        callbacks.append(cb)

    if config.watch_model is True:

        cb = WatchModelWithWandb()

        callbacks.append(cb)

    if config.tqdm is True:
        cb = TQDMProgressBar(refresh_rate=config.tqdm_refresh)

        callbacks.append(cb)

    return callbacks
