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


def update_config_pretrain(config):

    if config.pretrain:

        # load previous config
        logger.info(f"Loading previous wandb config...")
        logger.info(
            f"wandb run: {config.pretrain_entity}/{config.pretrain_project}/{config.pretrain_id}"
        )
        prev_config = load_wandb_run_config(
            entity=config.pretrain_entity,
            project=config.pretrain_project,
            id=config.pretrain_id,
        )
        # print(prev_config)
        # prev_config = config_dict.ConfigDict(prev_config)

        # load previous checkpoint
        logger.info(f"Downloading prev run checkpoint...")
        logger.info(f"Prev Run: {config.pretrain_reference}")
        checkpoint_dir = load_wandb_checkpoint(
            entity=config.pretrain_entity,
            project=config.pretrain_project,
            reference=config.pretrain_reference,
            mode="online",
        )

        checkpoint_file = Path(checkpoint_dir).joinpath(config.pretrain_checkpoint)
        logger.info(f"Checkpoint file: {checkpoint_file}")

        # TODO: fix hack for pretraining config params
        logger.info(f"Hack: copying prev config pretrain params...")
        pretrain = True
        pretrain_id = config.pretrain_id
        pretrain_checkpoint = config.pretrain_checkpoint
        pretrain_reference = config.pretrain_reference

        # overwrite config
        logger.info(f"Overwriting previous config...")

        config = config_dict.ConfigDict(prev_config["model"])
        config.pretrain = pretrain
        config.pretrain_id = pretrain_id
        config.pretrain_checkpoint = pretrain_checkpoint
        config.pretrain_reference = pretrain_reference
        config.pretrain_checkpoint_file = checkpoint_file

    return config


seed_everything(123, workers=True)


def train(config: ml_collections.ConfigDict, workdir, savedir):

    # update config
    config.model = update_config_pretrain(config.model)

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
    from inr4ssh._src.trainers.osse_2020a import INRModel

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

    logger.info("Training...")
    trainer.fit(learn, datamodule=dm)

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
    psd_score = psd_spacetime_score(ds_pred["ssh_model"], ds_pred["ssh_model_predict"])

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
        ds_pred["ssh_model"], ds_pred["ssh_model_predict"]
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

    # # TESTING
    # logger.info("Making predictions (grid)...")
    # t0 = time.time()
    # with torch.inference_mode():
    #     predictions = trainer.predict(learn, datamodule=dm, return_predictions=True)
    #     predictions = torch.cat(predictions)
    #     predictions = predictions.numpy()
    # t1 = time.time() - t0

    # logger.info(f"Time Taken for {dm.ds_predict[:][0].shape[0]} points: {t1:.4f} secs")
    # wandb_logger.log_metrics(
    #     {
    #         "time_predict_grid": t1,
    #     }
    # )

    # ds_oi = postprocess_predictions(predictions, dm, config.data.ref_data_dir, logger)

    # alongtracks, tracks = get_interpolation_alongtrack_prediction_ds(
    #     ds_oi, config.data.test_data_dir, config.eval_data, logger
    # )

    # logger.info("Getting RMSE Metrics (GRID)...")
    # rmse_metrics = get_grid_stats(
    #     alongtracks, config.metrics, None, wandb_logger.log_metrics
    # )

    # logger.info(f"Grid Stats: {rmse_metrics}")

    # # compute scores
    # logger.info("Computing PSD Scores (Grid)...")
    # psd_metrics = compute_psd_scores(
    #     ssh_true=tracks.ssh_alongtrack,
    #     ssh_pred=tracks.ssh_map,
    #     delta_x=config.metrics.velocity * config.metrics.delta_t,
    #     npt=tracks.npt,
    #     scaling="density",
    #     noverlap=0,
    # )
    # #
    # logger.info(f"Grid PSD: {psd_metrics}")

    # logger.info(f"Resolved scale (grid): {psd_metrics.resolved_scale:.2f}")
    # wandb_logger.log_metrics(
    #     {
    #         "resolved_scale_grid": psd_metrics.resolved_scale,
    #     }
    # )

    # logger.info(f"Plotting PSD Score and Spectrum (Grid)...")
    # plot_psd_figs(psd_metrics, logger, wandb_logger.experiment.log, method="grid")
    # logger.info("Finished GRID Script...!")

    # # ==============================
    # # ALONGTRACK PREDICTIONS
    # # ==============================

    # logger.info("ALONGTRACK STATS...")

    # X_test, y_test = get_alongtrack_prediction_ds(
    #     dm, config.data.test_data_dir, config.preprocess, config.eval_data, logger
    # )

    # # initialize dataset
    # ds_test = TensorDataset(
    #     torch.FloatTensor(X_test)
    #     # torch.Tensor(y_test)
    # )
    # # initialize dataloader
    # dl_test = DataLoader(
    #     ds_test,
    #     batch_size=config.dataloader.batchsize_test,
    #     shuffle=False,
    #     num_workers=config.dataloader.num_workers,
    #     pin_memory=config.dataloader.pin_memory,
    # )

    # logger.info(f"Predicting alongtrack data...")
    # t0 = time.time()
    # with torch.inference_mode():
    #     predictions = trainer.predict(
    #         learn, dataloaders=dl_test, return_predictions=True
    #     )
    #     predictions = torch.cat(predictions)
    #     predictions = predictions.numpy()
    # t1 = time.time() - t0

    # wandb_logger.log_metrics(
    #     {
    #         "time_predict_alongtrack": t1,
    #     }
    # )

    # logger.info("Calculating stats (alongtrack)...")
    # get_alongtrack_stats(y_test, predictions, logger, wandb_logger.log_metrics)

    # # PSD
    # logger.info(f"Getting PSD Scores (alongtrack)...")
    # psd_metrics = compute_psd_scores(
    #     ssh_true=y_test.squeeze(),
    #     ssh_pred=predictions.squeeze(),
    #     delta_x=config.metrics.velocity * config.metrics.delta_t,
    #     npt=None,
    #     scaling="density",
    #     noverlap=0,
    # )

    # logger.info(f"Resolved scale (alongtrack): {psd_metrics.resolved_scale:.2}")
    # wandb_logger.log_metrics(
    #     {
    #         "resolved_scale_alongtrack": psd_metrics.resolved_scale,
    #     }
    # )

    # logger.info(f"Plotting PSD Score and Spectrum (AlongTrack)...")
    # plot_psd_figs(psd_metrics, logger, wandb_logger.experiment.log, method="alongtrack")

    wandb.finish()
