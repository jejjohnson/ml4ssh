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
import xarray as xr


from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger


import torch.nn as nn
import pytorch_lightning as pl

import wandb
from inr4ssh._src.datamodules.osse_2020a import AlongTrackDataModule

from pathlib import Path
from inr4ssh._src.logging.wandb import load_wandb_checkpoint, load_wandb_run_config
from ml_collections import config_dict
import utils
from inr4ssh._src.preprocess.coords import (
    correct_coordinate_labels,
    correct_longitude_domain,
)
from inr4ssh._src.preprocess.regrid import oi_regrid

seed_everything(123, workers=True)


def metrics(config: ml_collections.ConfigDict, savedir, variable_name=None):

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
        id=config.log.id,
        log_model=False,
    )
    logger.info(f"Loading reference dataset")
    ds_filenames = Path(config.datadir.clean.ref_dir).joinpath(
        "NATL60-CJM165_GULFSTREAM_y*"
    )
    logger.info(f"Dataset: {ds_filenames}")
    ds_ref = xr.open_mfdataset(str(ds_filenames), engine="netcdf4")

    logger.info(f"Loading predictions xr dataset...")
    logger.debug(f"{savedir}")
    ds_pred = xr.open_dataset(savedir, engine="netcdf4")

    logger.info("Postprocessing data...")

    logger.info(f"Correcting spatial coord labels...")
    ds_pred = correct_coordinate_labels(ds_pred)
    ds_ref = correct_coordinate_labels(ds_ref)

    logger.info("Subsetting datasets (temporal)...")
    ds_pred = ds_pred.sel(
        time=slice(config.evaluation.time_min, config.evaluation.time_max),
        longitude=slice(config.evaluation.lon_min, config.evaluation.lon_max),
        latitude=slice(config.evaluation.lat_min, config.evaluation.lat_max),
        drop=True,
    )
    ds_ref = ds_ref.sel(
        time=slice(config.evaluation.time_min, config.evaluation.time_max),
        longitude=slice(config.evaluation.lon_min, config.evaluation.lon_max),
        latitude=slice(config.evaluation.lat_min, config.evaluation.lat_max),
        drop=True,
    )

    if config.evaluation.time_resample is not "":
        ds_ref = ds_ref.resample(time=config.evaluation.time_resample).mean()

    logger.info(f"Correcting variable labels...")
    ds_ref = ds_ref.rename({"sossheig": "ssh"})
    ds_pred = ds_pred.rename({variable_name: "ssh"})

    logger.info(f"Correcting longitude domain...")
    ds_ref = correct_longitude_domain(ds_ref)
    ds_pred = correct_longitude_domain(ds_pred)

    ds_pred = ds_pred.transpose("time", "latitude", "longitude")

    logger.info(f"Regridding predictions to reference...")
    ds_ref["ssh_predict"] = oi_regrid(ds_pred["ssh"], ds_ref["ssh"])

    # RMSE STATS

    from inr4ssh._src.metrics.field.stats import nrmse_spacetime, rmse_space, nrmse_time

    logger.info("nRMSE Stats...")
    nrmse_xyt = nrmse_spacetime(ds_ref["ssh_predict"], ds_ref["ssh"]).values
    logger.info(f"Leaderboard SSH RMSE score =  {nrmse_xyt:.2f}")
    wandb_logger.log_metrics(
        {
            "nrmse_mu": nrmse_xyt,
        }
    )

    rmse_t = nrmse_time(ds_ref["ssh_predict"], ds_ref["ssh"])

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

    # get the temporal normalization criteria
    logger.info(f"Normalizing temporal domain...")
    time_norm = np.timedelta64(config.evaluation.dt_freq, config.evaluation.dt_unit)
    logger.info(f"Normalizing: {time_norm}")

    # temporally normalize
    ds_ref["time"] = (ds_ref.time - ds_ref.time[0]) / time_norm

    #### Degrees
    # %%
    # Time-Longitude (Lat avg) PSD Score
    ds_field = ds_ref.chunk(
        {
            "time": 1,
            "longitude": ds_ref["longitude"].size,
            "latitude": ds_ref["latitude"].size,
        }
    ).compute()

    # Time-Longitude (Lat avg) PSD Score
    psd_score = psd_spacetime_score(ds_field["ssh_predict"], ds_field["ssh"])

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
    psd_iso_score = psd_isotropic_score(ds_ref["ssh_predict"], ds_ref["ssh"])

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
