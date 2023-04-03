import ml_collections
from loguru import logger
from pytorch_lightning.loggers import WandbLogger


def init_wandb_logger(config):
    logger.info("Initializaing Logger...")
    logger.debug(f"Log directory: {config.log.log_dir}")
    wandb_logger = WandbLogger(
        config=config.to_dict(),
        mode=config.log.mode,
        project=config.log.project,
        entity=config.log.entity,
        save_dir=config.log.log_dir,
        resume=False,
        log_model=False,
    )
    return wandb_logger


def init_datamodule(self):
    pass


def init_model(self):
    pass


def init_transforms(self):
    pass
