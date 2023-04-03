import sys, os

import ml_collections

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from pyprojroot import here

# spyder up to find the root
root = here(project_files=[".root"])

# append to path
sys.path.append(str(root))

from pathlib import Path
from simple_parsing import ArgumentParser
import time
from loguru import logger

import torch


from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger


import torch.nn as nn
import pytorch_lightning as pl

import wandb
from inr4ssh._src.datamodules.dc20a import AlongTrackDataModule

from pathlib import Path
from inr4ssh._src.logging.wandb import load_wandb_checkpoint, load_wandb_run_config
from ml_collections import config_dict
import utils

seed_everything(123, workers=True)


def inference(config: ml_collections.ConfigDict, savedir):

    # update config
    config.model = utils.update_config_pretrain(config.model)

    # INITIALIZE LOGGER
    logger.info("Initializaing Logger...")
    wandb_logger = WandbLogger(
        config=config.to_dict(),
        mode=config.log.mode,
        project=config.log.project,
        entity=config.log.entity,
        dir=config.log.log_dir,
        resume=config.log.resume,
        log_model=False,
    )

    # DATA MODULE
    logger.info("Initializing data module...")
    # initialize data module
    dm = AlongTrackDataModule(
        root=None,
        config=config,
        download=False,
    )

    # initialize datamodule params
    dm.setup()

    logger.info(f"Number of Data Points: {len(dm.ds_predict)}...")
    logger.info(f"Number of Training: {len(dm.ds_train)}...")
    logger.info(f"Number of Validation: {len(dm.ds_valid)}...")

    logger.info("Extracing train and test dims...")
    data = dm.ds_train[:10]

    x_init = torch.cat([data["spatial"], data["temporal"]], dim=1)
    y_init = data["output"]

    dim_in = x_init.shape[1]
    dim_out = y_init.shape[1]

    # update params
    logger.info("Adding dims to params...")
    params = {"dim_in": dim_in, "dim_out": dim_out}
    wandb_logger.experiment.config.update(params, allow_val_change=True)

    logger.info(f"Creating {config.model.model} neural network...")
    from inr4ssh._src.models.models_factory import model_factory

    net = model_factory(
        model=config.model.model,
        # dim_in=x_train.shape[1],
        dim_in=dim_in,
        # dim_out=y_train.shape[1],
        dim_out=dim_out,
        config=config.model,
    )

    logger.info(f"Testing forward pass...")
    out = net(x_init)
    assert out.shape == y_init.shape

    logger.info("Initializing spatial-temporal encoders...")
    from inr4ssh._src.transforms.utils import (
        spatial_transform_factory,
        temporal_transform_factory,
    )

    spatial_transform = spatial_transform_factory(config.encoder_spatial)
    temporal_transform = temporal_transform_factory(config.encoder_temporal)

    logger.info("Initializing callbacks...")
    from inr4ssh._src.callbacks.utils import get_callbacks

    callbacks = get_callbacks(config.callbacks, wandb_logger)

    # ============================
    # PYTORCH LIGHTNING CLASS
    # ============================

    logger.info("Initializing Learning class...")
    from inr4ssh._src.trainers.nerf import INRModel

    # update the number of epochs
    config.lr_scheduler.max_epochs = config.trainer.num_epochs

    learn = INRModel.load_from_checkpoint(
        checkpoint_path=config.model.pretrain_checkpoint_file,
        model=net,
        spatial_transform=spatial_transform,
        temporal_transform=temporal_transform,
        optimizer_config=config.optimizer,
        lr_scheduler_config=config.lr_scheduler,
        loss_config=config.loss,
    )

    # # overwrite new criteria
    # learn.spatial_transform = spatial_transform
    # learn.temporal_transform = temporal_transform
    # learn.optimizer_config = config.optimizer
    # learn.loss_config = config.loss
    # learn.lr_scheduler = config.lr_scheduler

    # start trainer
    logger.info("Initializing trainer...")
    trainer = Trainer(
        min_epochs=1,
        max_epochs=config.trainer.num_epochs,
        accelerator=config.trainer.accelerator,
        devices=config.trainer.devices,
        enable_progress_bar=True,
        logger=wandb_logger,
        callbacks=callbacks,
        accumulate_grad_batches=config.trainer.grad_batches,
        strategy=config.trainer.strategy,
        num_nodes=config.trainer.num_nodes,
        deterministic=config.trainer.deterministic,
    )

    # ==============================
    # GRID PREDICTIONS
    # ==============================

    logger.info("GRID STATS...")

    logger.info("Making predictions (grid)...")
    t0 = time.time()

    with torch.inference_mode():
        predictions = trainer.predict(
            learn, dataloaders=dm.test_dataloader(), return_predictions=True
        )
        predictions = torch.cat(predictions)

    t1 = time.time() - t0
    logger.info(
        f"Time Taken for {dm.ds_predict[:]['spatial'].shape[0]} points: {t1:.4f} secs"
    )
    wandb_logger.log_metrics(
        {
            "time_predict_grid": t1,
        }
    )

    logger.info("Creating results xr.Dataset...")
    df_pred = dm.ds_test.create_predict_df(predictions.detach().numpy())
    ds_pred = (
        df_pred.reset_index().set_index(["longitude", "latitude", "time"]).to_xarray()
    )

    # save file
    ds_pred.to_netcdf(savedir)

    # RMSE STATS

    from inr4ssh._src.metrics.field.stats import nrmse_spacetime, rmse_space, nrmse_time

    logger.info("nRMSE Stats...")
    nrmse_xyt = nrmse_spacetime(
        ds_pred["ssh_model_predict"], ds_pred["ssh_model"]
    ).values
    logger.info(f"Leaderboard SSH RMSE score =  {nrmse_xyt:.2f}")
    wandb_logger.log_metrics(
        {
            "nrmse_mu": nrmse_xyt,
        }
    )

    rmse_t = nrmse_time(ds_pred["ssh_model_predict"], ds_pred["ssh_model"])

    err_var_time = rmse_t.std().values
    logger.info(f"Error Variability =  {err_var_time:.2f}")
    wandb_logger.log_metrics(
        {
            "nrmse_std": err_var_time,
        }
    )

    # PSD STATS
    from inr4ssh._src.metrics.psd import (
        psd_isotropic_score,
        psd_spacetime_score,
        wavelength_resolved_spacetime,
        wavelength_resolved_isotropic,
    )
    import numpy as np

    time_norm = np.timedelta64(1, "D")
    # mean psd of signal
    ds_pred["time"] = (ds_pred.time - ds_pred.time[0]) / time_norm

    # Time-Longitude (Lat avg) PSD Score
    ds_field = ds_pred.chunk(
        {
            "time": 1,
            "longitude": ds_pred["longitude"].size,
            "latitude": ds_pred["latitude"].size,
        }
    ).compute()

    # Time-Longitude (Lat avg) PSD Score
    psd_score = psd_spacetime_score(
        ds_field["ssh_model_predict"], ds_field["ssh_model"]
    )

    logger.info("PSD Space-time statistics...")
    spatial_resolved, time_resolved = wavelength_resolved_spacetime(psd_score)
    logger.info(
        f"Shortest Spatial Wavelength Resolved = {spatial_resolved:.2f} (degree lon)"
    )
    logger.info(f"Shortest Temporal Wavelength Resolved = {time_resolved:.2f} (days)")

    wandb_logger.log_metrics(
        {
            "wavelength_space_deg": spatial_resolved,
        }
    )
    wandb_logger.log_metrics(
        {
            "wavelength_time_days": time_resolved,
        }
    )

    # Isotropic (Time avg) PSD Score
    logger.info("PSD Isotropic statistics...")
    psd_iso_score = psd_isotropic_score(
        ds_pred["ssh_model_predict"], ds_pred["ssh_model"]
    )

    space_iso_resolved = wavelength_resolved_isotropic(psd_iso_score, level=0.5)
    logger.info(
        f"Shortest Spatial Wavelength Resolved = {space_iso_resolved:.2f} (degree)"
    )
    wandb_logger.log_metrics(
        {
            "wavelength_iso_degree": space_iso_resolved,
        }
    )

    import pandas as pd

    data = [
        [
            "SIREN GF/GF",
            nrmse_xyt,
            err_var_time,
            spatial_resolved,
            time_resolved,
            space_iso_resolved,
            "GF/GF",
            "eval_siren.ipynb",
        ]
    ]

    Leaderboard = pd.DataFrame(
        data,
        columns=[
            "Method",
            "µ(RMSE) ",
            "σ(RMSE)",
            "λx (degree)",
            "λt (days)",
            "λr (degree)",
            "Notes",
            "Reference",
        ],
    )
    print("Summary of the leaderboard metrics:")
    print(Leaderboard.to_markdown())

    wandb.finish()
