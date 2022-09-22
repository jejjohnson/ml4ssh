import sys, os

from pyprojroot import here

# spyder up to find the root
root = here(project_files=[".root"])

# append to path
sys.path.append(str(root))

import time
import argparse
import imageio
from pathlib import Path
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from inr4ssh._src.data.ssh_obs import (
    load_ssh_altimetry_data_test,
    load_ssh_altimetry_data_train,
    load_ssh_correction,
)
from inr4ssh._src.datamodules.ssh_obs import SSHAltimetry
from inr4ssh._src.features.data_struct import df_2_xr
from inr4ssh._src.interp import interp_on_alongtrack
from inr4ssh._src.metrics.psd import compute_psd_scores, select_track_segments
from inr4ssh._src.metrics.stats import (
    calculate_nrmse,
    calculate_nrmse_elementwise,
    calculate_rmse_elementwise,
)
from inr4ssh._src.models.activations import get_activation

from inr4ssh._src.models.siren import ModulatedSirenNet, Modulator, Siren, SirenNet
from inr4ssh._src.postprocess.ssh_obs import postprocess
from inr4ssh._src.preprocess.coords import (
    correct_coordinate_labels,
    correct_longitude_domain,
)
from inr4ssh._src.preprocess.subset import spatial_subset, temporal_subset
from inr4ssh._src.viz.psd import plot_psd_score, plot_psd_spectrum
from loguru import logger
from skorch import NeuralNetRegressor
from skorch.callbacks import EarlyStopping, LRScheduler, WandbLogger, Checkpoint
from skorch.dataset import ValidSplit
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.notebook import tqdm as tqdm

pl.seed_everything(123)

import matplotlib.pyplot as plt
import seaborn as sns
from inr4ssh._src.viz.movie import create_movie


def postprocess_predictions(predictions, dm, ref_data_dir, logger):

    # POSTPROCESS
    # convert to da
    logger.info("Convert data to xarray ds...")
    ds_oi = dm.X_pred_index
    ds_oi["ssh"] = predictions
    ds_oi = df_2_xr(ds_oi)

    # open correction dataset
    logger.info("Loading SSH corrections...")
    ds_correct = load_ssh_correction(ref_data_dir)

    # correct predictions
    logger.info("Correcting SSH predictions...")
    ds_oi = postprocess(ds_oi, ds_correct)

    return ds_oi


def get_interpolation_alongtrack_prediction_ds(
    ds_oi, test_data_dir, config_eval_data, logger
):
    # open along track dataset
    logger.info("Loading test dataset...")
    ds_alongtrack = load_ssh_altimetry_data_test(test_data_dir)

    # interpolate along track
    logger.info("Interpolating alongtrack obs...")
    alongtracks = interp_on_alongtrack(
        gridded_dataset=ds_oi,
        ds_alongtrack=ds_alongtrack,
        lon_min=config_eval_data.lon_min,
        lon_max=config_eval_data.lon_max,
        lat_min=config_eval_data.lat_min,
        lat_max=config_eval_data.lat_max,
        time_min=config_eval_data.time_min,
        time_max=config_eval_data.time_max,
    )

    logger.info("Selecting track segments...")
    tracks = select_track_segments(
        time_alongtrack=alongtracks.time,
        lat_alongtrack=alongtracks.lat,
        lon_alongtrack=alongtracks.lon,
        ssh_alongtrack=alongtracks.ssh_alongtrack,
        ssh_map_interp=alongtracks.ssh_map,
    )

    return alongtracks, tracks


def get_grid_stats(alongtracks, args, logger, wandb_fn=None):

    rmse_metrics = calculate_nrmse(
        true=alongtracks.ssh_alongtrack,
        pred=alongtracks.ssh_map,
        time_vector=alongtracks.time,
        dt_freq=args.bin_time_step,
        min_obs=args.min_obs,
    )

    if wandb_fn is not None:
        wandb_fn(
            {
                "rmse_mean_grid": rmse_metrics.rmse_mean,
                "rmse_std_grid": rmse_metrics.rmse_std,
                "nrmse_mean_grid": rmse_metrics.nrmse_mean,
                "nrmse_std_grid": rmse_metrics.nrmse_std,
            }
        )

    return rmse_metrics


def get_alongtrack_prediction_ds(
    dm, test_data_dir, config_preprocess, config_eval_data, logger
):
    # ==================================
    # PREDICTIONS - ALONGTRACK
    # ==================================

    logger.info(f"Opening alongtrack dataset...")
    ds_alongtrack = load_ssh_altimetry_data_test(test_data_dir)

    logger.info(f"Correcting coordinate labels...")
    ds_alongtrack = correct_coordinate_labels(ds_alongtrack)

    logger.info(f"correcting longitudal domain...")
    ds_alongtrack = correct_longitude_domain(ds_alongtrack)

    # temporal subset
    logger.info(f"Temporal subset...")
    ds_alongtrack = temporal_subset(
        ds_alongtrack,
        time_min=np.datetime64(config_preprocess.time_min),
        time_max=np.datetime64(config_preprocess.time_max),
        time_buffer=config_preprocess.time_buffer,
    )

    logger.info(f"Spatial subset...")
    ds_alongtrack = spatial_subset(
        ds_alongtrack,
        lon_min=config_eval_data.lon_min,
        lon_max=config_eval_data.lon_max,
        lon_buffer=config_eval_data.lon_buffer,
        lat_min=config_eval_data.lat_min,
        lat_max=config_eval_data.lat_max,
        lat_buffer=config_eval_data.lat_buffer,
    )

    logger.info(f"Converting to a dataframe...")
    ds_alongtrack = ds_alongtrack.to_dataframe().reset_index().dropna()

    logger.info(f"Feature transformation for alongtrack data...")
    X_test = dm.scaler.transform(ds_alongtrack)
    y_test = ds_alongtrack["sla_unfiltered"]
    return X_test, y_test


def get_alongtrack_stats(y_test, predictions, logger, wandb_fn=None):

    # STATS
    logger.info(f"Calculating alongtrack RMSE...")
    rmse_mean, rmse_std = calculate_rmse_elementwise(y_test, predictions)

    if wandb_fn is not None:
        wandb_fn(
            {
                f"rmse_mean_alongtrack": rmse_mean,
                f"rmse_std_alongtrack": rmse_std,
            }
        )

    logger.info(f"RMSE: {rmse_mean}\nRMSE (stddev): {rmse_std}")

    # NORMALIZED metrics
    logger.info(f"Calculating alongtrack NRMSE...")
    metrics = ["custom", "std", "mean", "minmax", "iqr"]

    for imetric in metrics:
        nrmse_mean, nrmse_std = calculate_nrmse_elementwise(
            y_test, predictions, imetric
        )

        logger.info(
            f"RMSE ({imetric}): mean - {nrmse_mean:.4f}, stddev - {nrmse_std:.4f}"
        )

        if wandb_fn is not None:
            wandb_fn(
                {
                    f"nrmse_mean_alongtrack_{imetric}": nrmse_mean,
                    f"nrmse_std_alongtrack_{imetric}": nrmse_std,
                }
            )
    return None


def plot_psd_figs(psd_metrics, logger, wandb_fn=None, method: str = "alongtrack"):

    if wandb_fn is not None:
        wandb_fn(
            {
                "resolved_scale_alongtrack": psd_metrics.resolved_scale,
            }
        )

    # PLOTS

    fig, ax = plot_psd_score(
        psd_diff=psd_metrics.psd_diff,
        psd_ref=psd_metrics.psd_ref,
        wavenumber=psd_metrics.wavenumber,
        resolved_scale=psd_metrics.resolved_scale,
    )

    if wandb_fn is not None:
        wandb_fn(
            {
                f"psd_score_{method}": wandb.Image(fig),
            }
        )

    fig, ax = plot_psd_spectrum(
        psd_study=psd_metrics.psd_study,
        psd_ref=psd_metrics.psd_ref,
        wavenumber=psd_metrics.wavenumber,
    )

    if wandb_fn is not None:
        wandb_fn(
            {
                f"psd_spectrum_{method}": wandb.Image(fig),
            }
        )
    return None
