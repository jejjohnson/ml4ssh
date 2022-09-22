import sys, os


from pyprojroot import here

# spyder up to find the root
root = here(project_files=[".root"])

# append to path
sys.path.append(str(root))

import ml_collections
import copy
import tqdm
import wandb
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


from losses import RegQG, initialize_data_loss
from model import INRModel
from inr4ssh._src.datamodules.qg_sim import QGSimulation
from utils import (
    initialize_siren_model,
    initialize_callbacks,
    load_model_from_cpkt,
    get_physical_arrays,
    plot_physical_quantities,
)

seed_everything(123)


def train(config: ml_collections.ConfigDict, workdir: str, savedir: str):

    # ==================================
    # EXPERIMENT I: QG IMAGE
    # ==================================

    # initialize logger
    logger.info("Initializaing Logger...")
    wandb_logger = WandbLogger(
        config=config.to_dict(),
        mode=config.log.mode,
        project=config.log.project,
        entity=config.log.entity,
        dir=config.log.log_dir,
        resume=False,
    )

    # initialize dataloader
    logger.info("Initializing data module...")
    dm = QGSimulation(config)
    dm.setup()
    logger.info(f"Number of Data Points: {len(dm.ds_predict)}...")
    logger.info(f"Number of Training: {len(dm.ds_train)}...")
    logger.info(f"Number of Validation: {len(dm.ds_valid)}...")
    # ==================================
    # EXPERIMENT Ia: SIREN + NO QG LOSS
    # ==================================

    x_init, y_init = dm.ds_train[:10]

    # initialize model
    logger.info("Initializing SIREN Model...")
    net = initialize_siren_model(config.model, x_init, y_init)

    # initialize dataloss
    logger.info("Initializing Data Loss...")
    data_loss = initialize_data_loss(config.loss)

    logger.info("Initializing PL Model...")
    learn = INRModel(
        model=net,
        loss_data=data_loss,
        reg_pde=None,
        learning_rate=config.optim.learning_rate,
        warmup=config.optim.warmup,
        warmup_start_lr=config.optim.warmup_start_lr,
        eta_min=config.optim.eta_min,
        num_epochs=config.optim.num_epochs,
        alpha=0.0,
        qg=False,
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
    plot_physical_quantities(xr_data, wandb_logger, name="img")

    # ==================================
    # EXPERIMENT Ib: PRE-SIREN + QG LOSS
    # ==================================

    # initialize regularization
    logger.info("Initializing QG PDE REG...")
    reg_loss = RegQG(config.loss.alpha)

    # initialize dataloss
    logger.info("Initializing Loss...")
    data_loss = initialize_data_loss(config.loss)

    logger.info("Initializing PL Model (with Pretrained)...")
    learn_qg = INRModel(
        model=copy.deepcopy(learn.model),
        loss_data=data_loss,
        reg_pde=reg_loss,
        learning_rate=config.optim_qg.learning_rate,
        warmup=config.optim_qg.warmup,
        warmup_start_lr=config.optim_qg.warmup_start_lr,
        eta_min=config.optim_qg.eta_min,
        num_epochs=config.optim.num_epochs,
        alpha=config.loss.alpha,
        qg=True,
    )

    # initialize callbacks
    logger.info("Initializing Callbacks...")
    callbacks = initialize_callbacks(config, wandb_logger.experiment.dir, "qg")

    # initialize trainer
    logger.info("Initializing PL Trainer...")
    trainer = Trainer(
        min_epochs=1,
        max_epochs=config.optim_qg.num_epochs,
        accelerator=config.trainer_qg.accelerator,
        devices=config.trainer_qg.devices,
        enable_progress_bar=True,
        logger=wandb_logger,
        callbacks=callbacks,
        accumulate_grad_batches=config.trainer.grad_batches,
    )

    # train
    logger.info("Starting training (QG)...")
    trainer.fit(
        learn_qg,
        datamodule=dm,
    )

    logger.info("Getting physical quantities...")
    xr_data = get_physical_arrays(learn_qg.model, dm)

    logger.info("Plotting physical quantities...")
    plot_physical_quantities(xr_data, wandb_logger, name="img_qg")

    logger.info("Finishing...")
    wandb.finish()
