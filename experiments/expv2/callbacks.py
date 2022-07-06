from skorch.callbacks import EarlyStopping, LRScheduler, WandbLogger

def get_callbacks(config, wandb_logger=None):
    callbacks = []
    if config.wandb is True:
        cb = EarlyStopping(
            monitor="valid_loss",
            patience=config.patience,
        )
        callbacks.append(("early_stopping", cb))
    if config.early_stopping is True:
        cb = WandbLogger(
            wandb_run=wandb_logger,
            save_model=config.save_model
        )
        callbacks.append(("wandb", cb))
    return callbacks
