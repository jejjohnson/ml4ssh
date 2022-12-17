import os
import sys

import ml_collections

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from pyprojroot import here

# spyder up to find the root
root = here(project_files=[".root"])

# append to path
sys.path.append(str(root))

from pathlib import Path
import time
from loguru import logger

import torch

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger

import wandb
from inr4ssh._src.datamodules.dc21a import DC21AlongTrackDM
from inr4ssh._src.transforms.utils import (
    spatial_transform_factory,
    temporal_transform_factory,
)
from inr4ssh._src.models.models_factory import model_factory
from inr4ssh._src.trainers.nerf import INRModel
from inr4ssh._src.logging.wandb import load_wandb_checkpoint, load_wandb_run_config
from ml_collections import config_dict
from utils import update_config_pretrain

seed_everything(123, workers=True)


def train(config: ml_collections.ConfigDict, workdir, savedir):

    if config.model.pretrain:
        config.model = update_config_pretrain(config.model)

    # INITIALIZE LOGGER
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

    # DATA MODULE
    logger.info("Initializing data module...")
    # initialize data module
    dm = DC21AlongTrackDM(
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

    spatial_transform = spatial_transform_factory(config.encoder_spatial)
    temporal_transform = temporal_transform_factory(config.encoder_temporal)

    logger.info("Initializing callbacks...")
    from inr4ssh._src.callbacks.utils import get_callbacks

    callbacks = get_callbacks(config.callbacks, wandb_logger)

    # ============================
    # PYTORCH LIGHTNING CLASS
    # ============================

    # update the number of epochs
    config.lr_scheduler.max_epochs = config.trainer.num_epochs

    if config.model.pretrain:
        logger.info("Initializing Learning class from pretrained...")
        learn = INRModel.load_from_checkpoint(
            checkpoint_path=config.model.pretrain_checkpoint_file,
            model=net,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            optimizer_config=config.optimizer,
            lr_scheduler_config=config.lr_scheduler,
            loss_config=config.loss,
        )
    else:
        logger.info("Initializing Learning class...")
        learn = INRModel(
            model=net,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            optimizer_config=config.optimizer,
            lr_scheduler_config=config.lr_scheduler,
            loss_config=config.loss,
        )

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

    # # ==============================
    # # ALONGTRACK PREDICTIONS
    # # ==============================
    # logger.info("AlongTrack STATS...")
    #
    # logger.info("Making predictions (alongtrack)...")
    # t0 = time.time()
    #
    # with torch.inference_mode():
    #     predictions = trainer.predict(
    #         learn, dataloaders=dm.test_dataloader(), return_predictions=True
    #     )
    #     predictions = torch.cat(predictions)
    #
    # t1 = time.time() - t0
    # logger.info(
    #     f"Time Taken for {dm.ds_test[:]['spatial'].shape[0]} points: {t1:.4f} secs"
    # )
    # wandb_logger.log_metrics(
    #     {
    #         "time_predict_alongtrack": t1,
    #     }
    # )

    # ==============================
    # GRID PREDICTIONS
    # ==============================

    logger.info("GRID STATS...")

    logger.info("Making predictions (grid)...")
    t0 = time.time()

    with torch.inference_mode():
        predictions = trainer.predict(
            learn, dataloaders=dm.predict_dataloader(), return_predictions=True
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
    df_pred = dm.ds_predict.create_predict_df(predictions.detach().numpy())
    ds_pred = (
        df_pred.reset_index().set_index(["longitude", "latitude", "time"]).to_xarray()
    )

    ds_pred = ds_pred.rename({"predict": "ssh"})

    # add corrections to SSH
    ds_pred["ssh"] = dm.correct_ssh(ds_pred["ssh"])

    ds_test_alongtrack = dm.get_data_test()

    # ==================================
    # ALONGTRACK INTERPOLATION
    # ==================================

    from inr4ssh._src.interp import interp_on_alongtrack
    from inr4ssh._src.metrics.psd import compute_psd_scores, select_track_segments

    # TODO: change function arg from DS to SSH
    # TODO: write an SSH calculation function

    from inr4ssh._src.preprocess.spatial import convert_lon_360_180, convert_lon_180_360

    ds_pred["longitude"] = convert_lon_180_360(ds_pred.longitude)
    ds_test_alongtrack["longitude"] = convert_lon_180_360(ds_test_alongtrack.longitude)
    alongtracks = interp_on_alongtrack(
        gridded_dataset=ds_pred,
        ds_alongtrack=ds_test_alongtrack,
        lon_min=convert_lon_180_360(config.evaluation.lon_min),
        lon_max=convert_lon_180_360(config.evaluation.lon_max),
        lat_min=config.evaluation.lat_min,
        lat_max=config.evaluation.lat_max,
        time_min=config.evaluation.time_min,
        time_max=config.evaluation.time_max,
    )

    # ===================================
    # STATS - RMSE
    # ===================================

    from inr4ssh._src.metrics.stats import calculate_nrmse

    logger.info("Getting RMSE Metrics (GRID)...")
    rmse_metrics = calculate_nrmse(
        true=alongtracks.ssh_alongtrack,
        pred=alongtracks.ssh_map,
        time_vector=alongtracks.time,
        dt_freq=config.metrics.bin_time_step,
        min_obs=config.metrics.min_obs,
    )
    logger.info(f"Grid Stats: \n{rmse_metrics}")
    wandb_logger.log_metrics(
        {
            "rmse_mean_grid": rmse_metrics.rmse_mean,
            "rmse_std_grid": rmse_metrics.rmse_std,
            "nrmse_mean_grid": rmse_metrics.nrmse_mean,
            "nrmse_std_grid": rmse_metrics.nrmse_std,
        }
    )

    # ======================================
    # STATS - PSD
    # ======================================
    logger.info("Selecting track segments...")
    tracks = select_track_segments(
        time_alongtrack=alongtracks.time,
        lat_alongtrack=alongtracks.lat,
        lon_alongtrack=convert_lon_360_180(alongtracks.lon),
        ssh_alongtrack=alongtracks.ssh_alongtrack,
        ssh_map_interp=alongtracks.ssh_map,
    )

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

    logger.info(f"Grid PSD: \n{psd_metrics}")
    logger.info(f"Resolved scale (grid): {psd_metrics.resolved_scale:.2f}")

    wandb.finish()
