import sys

from pyprojroot import here

# spyder up to find the root
root = here(project_files=[".root"])

# append to path
sys.path.append(str(root))

import ml_collections

import tqdm
import wandb
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pathlib import Path

import numpy as np
from loguru import logger


from inr4ssh._src.datamodules.qg_sim import QGSimulation
from losses import RegQG, initialize_data_loss
from model import INRModel
from figures import plot_maps
from utils import (
    initialize_siren_model,
    initialize_callbacks,
    load_model_from_cpkt,
    get_physical_arrays,
    plot_physical_quantities,
)

from inr4ssh._src.io import get_wandb_config, get_wandb_model
from inr4ssh._src.models.siren import SirenNet


seed_everything(123)


def train(config: ml_collections.ConfigDict, workdir: str, savedir: str):

    # initialize logger
    logger.info("Initializaing Logger...")
    wandb_logger = WandbLogger(
        config=config.to_dict(),
        mode=config.log.mode,
        project=config.log.project,
        entity=config.log.entity,
        dir=config.log.log_dir,
        resume=False,
        log_model="all",
    )

    # initialize dataloader
    logger.info("Initializing data module...")
    dm = QGSimulation(config)
    dm.setup()
    logger.info(f"Number of Data Points: {len(dm.ds_predict)}...")
    logger.info(f"Number of Training: {len(dm.ds_train)}...")
    logger.info(f"Number of Validation: {len(dm.ds_valid)}...")

    x_init, y_init = dm.ds_train[:10]

    # initialize model
    if config.pretrained.checkpoint is True:
        logger.info("Initializing Model from checkpoint...")
        old_config = get_wandb_config(config.pretrained.run_path)
        old_config = ml_collections.config_dict.ConfigDict(old_config).model
        net = initialize_siren_model(old_config.model, x_init, y_init)
    else:
        logger.info("Initializing SIREN Model...")
        net = initialize_siren_model(config.model, x_init, y_init)

    # initialize regularization
    if config.loss.qg:
        logger.info("Initializing PDE REG...")
        reg_loss = RegQG(config.loss.alpha)
    else:
        reg_loss = None

    # initialize dataloss
    logger.info("Initializing Data Loss...")
    data_loss = initialize_data_loss(config.loss)

    # initialize learner
    if config.pretrained.checkpoint is True:
        logger.info("Initializing PL Model (pretrained)...")
        learn = INRModel.load_from_checkpoint(
            checkpoint_path=load_model_from_cpkt(config.pretrained),
            model=net,
            reg_pde=reg_loss,
            loss_data=data_loss,
            learning_rate=config.optim.learning_rate,
            warmup=config.optim.warmup,
            num_epochs=config.optim.num_epochs,
            alpha=config.loss.alpha,
            qg=config.loss.qg,
        )
    else:
        logger.info("Initializing PL Model...")
        learn = INRModel(
            model=net,
            reg_pde=reg_loss,
            loss_data=data_loss,
            learning_rate=config.optim.learning_rate,
            warmup=config.optim.warmup,
            warmup_start_lr=config.optim.warmup_start_lr,
            eta_min=config.optim.eta_min,
            num_epochs=config.optim.num_epochs,
            alpha=config.loss.alpha,
            qg=config.loss.qg,
        )

    # initialize callbacks
    logger.info("Initializing Callbacks...")
    callbacks = initialize_callbacks(config, wandb_logger.experiment.dir)

    # initialize trainer
    logger.info("Initializing PL Trainer...")
    trainer = Trainer(
        min_epochs=1,
        max_epochs=config.optim.num_epochs,
        accelerator=config.trainer.accelerator,
        devices=config.trainer.devices,
        enable_progress_bar=True,
        logger=wandb_logger,
        callbacks=callbacks,
        accumulate_grad_batches=config.trainer.grad_batches,
    )

    # train
    logger.info("Initializing PL Trainer...")
    trainer.fit(
        learn,
        datamodule=dm,
    )

    logger.info("Getting physical quantities...")
    xr_data = get_physical_arrays(learn.model, dm)

    logger.info("Plotting physical quantities...")
    plot_physical_quantities(xr_data, wandb_logger, name=None)

    wandb.finish()
