import sys, os

import ml_collections

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from pyprojroot import here

# spyder up to find the root
root = here(project_files=[".root"])

# append to path
sys.path.append(str(root))

import config
from pathlib import Path
from simple_parsing import ArgumentParser
import time
from loguru import logger

import torch

from inr4ssh._src.io import save_object
from inr4ssh._src.datamodules.ssh_obs import SSHAltimetry
from inr4ssh._src.metrics.psd import compute_psd_scores
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader

from callbacks import get_callbacks
from utils import (
    get_interpolation_alongtrack_prediction_ds,
    get_alongtrack_prediction_ds,
)
from utils import (
    plot_psd_figs,
    get_grid_stats,
    postprocess_predictions,
    get_alongtrack_stats,
)
from losses import loss_factory, regularization_factory
from optimizers import optimizer_factory, lr_scheduler_factory
from models import model_factory, CoordinatesLearner

import torch.nn as nn
import pytorch_lightning as pl

import wandb
from inr4ssh._src.io import simpleargs_2_ndict

seed_everything(123)


def train(config: ml_collections.ConfigDict, workdir, savedir):

    # INITIALIZE LOGGER
    logger.info("Initializaing Logger...")
    wandb_logger = WandbLogger(
        config=config.to_dict(),
        mode=config.log.mode,
        project=config.log.project,
        entity=config.log.entity,
        dir=config.log.log_dir,
        resume=False,
        log_model=False,
    )

    # DATA MODULE
    logger.info("Initializing data module...")
    dm = SSHAltimetry(
        data=config.data,
        preprocess=config.preprocess,
        traintest=config.split,
        features=config.features,
        dataloader=config.dataloader,
        eval=config.eval_data,
    )

    dm.setup()

    logger.info(f"Number of Data Points: {len(dm.ds_predict)}...")
    logger.info(f"Number of Training: {len(dm.ds_train)}...")
    logger.info(f"Number of Validation: {len(dm.ds_valid)}...")

    # objects
    logger.info("Saving scaler transform...")
    path_scaler = Path(wandb_logger.experiment.dir).joinpath(f"scaler.pickle")

    # models to save
    save_object(dm.scaler, path_scaler)

    # save with wandb
    wandb_logger.experiment.save(str(path_scaler))

    logger.info("Extracing train and test dims...")
    x_train, y_train = dm.ds_train[:]

    dim_in = x_train.shape[1]
    dim_out = y_train.shape[1]

    # update params
    logger.info("Adding dims to params...")
    params = {"dim_in": dim_in, "dim_out": dim_out}
    wandb_logger.experiment.config.update(params, allow_val_change=True)

    logger.info(f"Creating {config.model.model} neural network...")
    net = model_factory(
        model=config.model.model, dim_in=dim_in, dim_out=dim_out, config=config.model
    )
    #
    logger.info("Initializing callbacks...")
    callbacks = get_callbacks(config.callbacks, wandb_logger)

    # ============================
    # PYTORCH LIGHTNING CLASS
    # ============================

    logger.info("Initializing trainer class...")
    learn = CoordinatesLearner(
        model=net,
        params_loss=config.loss,
        params_optim=config.optimizer,
        params_lr=config.lr_scheduler,
    )

    # start trainer
    logger.info("Initializing trainer...")
    trainer = Trainer(
        min_epochs=config.optimizer.min_epochs,
        max_epochs=config.optimizer.num_epochs,
        accelerator=config.trainer.accelerator,
        devices=config.trainer.devices,
        enable_progress_bar=True,
        logger=wandb_logger,
        callbacks=callbacks,
        accumulate_grad_batches=config.trainer.grad_batches,
        strategy=config.trainer.strategy,
        num_nodes=config.trainer.num_nodes,
    )

    logger.info("Training...")
    trainer.fit(learn, datamodule=dm)

    # ==============================
    # GRID PREDICTIONS
    # ==============================

    logger.info("GRID STATS...")

    # TESTING
    logger.info("Making predictions (grid)...")
    t0 = time.time()
    with torch.inference_mode():
        predictions = trainer.predict(learn, datamodule=dm, return_predictions=True)
        predictions = torch.cat(predictions)
        predictions = predictions.numpy()
    t1 = time.time() - t0

    logger.info(f"Time Taken for {dm.ds_predict[:][0].shape[0]} points: {t1:.4f} secs")
    wandb_logger.log_metrics(
        {
            "time_predict_grid": t1,
        }
    )

    ds_oi = postprocess_predictions(predictions, dm, config.data.ref_data_dir, logger)

    alongtracks, tracks = get_interpolation_alongtrack_prediction_ds(
        ds_oi, config.data.test_data_dir, config.eval_data, logger
    )

    logger.info("Getting RMSE Metrics (GRID)...")
    rmse_metrics = get_grid_stats(
        alongtracks, config.metrics, None, wandb_logger.log_metrics
    )

    logger.info(f"Grid Stats: {rmse_metrics}")

    # compute scores
    logger.info("Computing PSD Scores (Grid)...")
    psd_metrics = compute_psd_scores(
        ssh_true=tracks.ssh_alongtrack,
        ssh_pred=tracks.ssh_map,
        delta_x=config.metrics.velocity * config.metrics.delta_t,
        npt=tracks.npt,
        scaling="density",
        noverlap=0,
    )
    #
    logger.info(f"Grid PSD: {psd_metrics}")

    logger.info(f"Resolved scale (grid): {psd_metrics.resolved_scale:.2f}")
    wandb_logger.log_metrics(
        {
            "resolved_scale_grid": psd_metrics.resolved_scale,
        }
    )

    logger.info(f"Plotting PSD Score and Spectrum (Grid)...")
    plot_psd_figs(psd_metrics, logger, wandb_logger.experiment.log, method="grid")
    logger.info("Finished GRID Script...!")

    # ==============================
    # ALONGTRACK PREDICTIONS
    # ==============================

    logger.info("ALONGTRACK STATS...")

    X_test, y_test = get_alongtrack_prediction_ds(
        dm, config.data.test_data_dir, config.preprocess, config.eval_data, logger
    )

    # initialize dataset
    ds_test = TensorDataset(
        torch.FloatTensor(X_test)
        # torch.Tensor(y_test)
    )
    # initialize dataloader
    dl_test = DataLoader(
        ds_test,
        batch_size=config.dataloader.batchsize_test,
        shuffle=False,
        num_workers=config.dataloader.num_workers,
        pin_memory=config.dataloader.pin_memory,
    )

    logger.info(f"Predicting alongtrack data...")
    t0 = time.time()
    with torch.inference_mode():
        predictions = trainer.predict(
            learn, dataloaders=dl_test, return_predictions=True
        )
        predictions = torch.cat(predictions)
        predictions = predictions.numpy()
    t1 = time.time() - t0

    wandb_logger.log_metrics(
        {
            "time_predict_alongtrack": t1,
        }
    )

    logger.info("Calculating stats (alongtrack)...")
    get_alongtrack_stats(y_test, predictions, logger, wandb_logger.log_metrics)

    # PSD
    logger.info(f"Getting PSD Scores (alongtrack)...")
    psd_metrics = compute_psd_scores(
        ssh_true=y_test.squeeze(),
        ssh_pred=predictions.squeeze(),
        delta_x=config.metrics.velocity * config.metrics.delta_t,
        npt=None,
        scaling="density",
        noverlap=0,
    )

    logger.info(f"Resolved scale (alongtrack): {psd_metrics.resolved_scale:.2}")
    wandb_logger.log_metrics(
        {
            "resolved_scale_alongtrack": psd_metrics.resolved_scale,
        }
    )

    logger.info(f"Plotting PSD Score and Spectrum (AlongTrack)...")
    plot_psd_figs(psd_metrics, logger, wandb_logger.experiment.log, method="alongtrack")

    wandb.finish()
