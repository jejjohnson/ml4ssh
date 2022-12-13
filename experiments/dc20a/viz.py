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
from inr4ssh._src.metrics.psd import psd_isotropic
from inr4ssh._src.viz.psd.isotropic import plot_psd_isotropic
from inr4ssh._src.metrics.psd import psd_isotropic_score, wavelength_resolved_isotropic

import matplotlib.pyplot as plt
import seaborn as sns

sns.reset_defaults()
sns.set_context(context="talk", font_scale=0.7)

seed_everything(123, workers=True)


def viz(
    config: ml_collections.ConfigDict,
    figure: str = "density",
    resultsfile: str = None,
    savedir: str = None,
    variable_name: str = "ssh",
) -> None:

    figure = figure.lower()

    if figure == "density":
        raise NotImplementedError()
        # density(config, resultsfile, savedir, variable_name)
    elif figure == "psd_iso":
        psd_iso(config, resultsfile, savedir, variable_name)
    elif figure == "psd":
        psd(config, resultsfile, savedir, variable_name)
    elif figure == "gif":
        raise NotImplementedError()
    elif figure == "stats":
        stats(config, resultsfile, savedir, variable_name)
    else:
        raise ValueError(f"Unrecognized figure type: {figure}")

    return None


def get_gridded_data(config, resultsfile, variable_name):
    logger.info(f"Loading reference dataset")
    ds_filenames = Path(config.datadir.clean.ref_dir).joinpath(
        "NATL60-CJM165_GULFSTREAM_y*"
    )
    logger.info(f"Dataset: {ds_filenames}")
    ds_ref = xr.open_mfdataset(str(ds_filenames), engine="netcdf4")

    logger.info(f"Loading predictions xr dataset...")
    logger.debug(f"{resultsfile}")
    ds_pred = xr.open_dataset(resultsfile, engine="netcdf4")

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

    if config.evaluation.time_resample != "":
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

    return ds_ref


def plot_st_psd(ds: xr.DataArray, label: str = "ssh", units: str = "degrees"):

    from inr4ssh._src.viz.psd.spacetime import plot_psd_spacetime_wavelength

    logger.info(f"Plotting Spatial-Temporal PSD SSH Predictions...")
    factor = 1.0 if units == "degrees" else 1e3
    fig, ax, cbar = plot_psd_spacetime_wavelength(
        ds.freq_longitude * factor,
        ds.freq_time,
        ds,
    )
    if units == "degrees":
        units = r"$^{\circ}$"
        ax.set_xlabel(r"Wavelength [degrees]")
    else:
        ax.set_xlabel(f"Wavelength [km]")

    if label == "ssh":
        cbar.ax.set_ylabel(f"PSD [{units}" + r"$^2$s$^{-2}$/cyles/" + f"{units}]")
    elif label == "ke":
        cbar.ax.set_ylabel(f"PSD [{units}" + r"$^2$s$^{-2}$/cyles/" + f"{units}]")
    elif label == "enstropy":
        cbar.ax.set_ylabel(r"PSD [s$^{-1}$/cyles/" + f"{units}")

    return fig, ax


def plot_st_psd_score(ds: xr.DataArray, units: str = "degrees"):

    from inr4ssh._src.viz.psd.spacetime import plot_psd_spacetime_score_wavelength

    logger.info(f"Plotting Spatial-Temporal PSD Score SSH Predictions...")
    factor = 1.0 if units == "degrees" else 1e3
    fig, ax, cbar = plot_psd_spacetime_score_wavelength(
        ds.freq_longitude * factor,
        ds.freq_time,
        ds,
    )

    ax.set_xlabel(f"Wavelength [{units}]")

    return fig, ax


def stats(config: ml_collections.ConfigDict, resultsfile, savedir, variable_name=None):

    t0 = time.time()
    if savedir is None:
        savedir = Path(root).joinpath("figures")
        Path(savedir).mkdir(parents=True, exist_ok=True)
    else:
        savedir = Path(savedir)

    ds_ref = get_gridded_data(config, resultsfile, variable_name)

    # RMSE STATS

    from inr4ssh._src.metrics.field.stats import nrmse_spacetime, rmse_space, nrmse_time

    logger.info("nRMSE Stats...")
    nrmse_xyt = nrmse_spacetime(ds_ref["ssh_predict"], ds_ref["ssh"]).values
    logger.info(f"Leaderboard SSH RMSE score =  {nrmse_xyt:.2f}")

    # =================================
    # ERROR VARIABILITY (TEMPORAL)
    # =================================

    rmse_t = nrmse_time(ds_ref["ssh_predict"], ds_ref["ssh"])

    err_var_time = rmse_t.std().values
    logger.info(f"Error Variability =  {err_var_time:.2f}")

    logger.info(f"Saving statistics plot...")
    logger.debug(f"{savedir}")
    fig, ax = plt.subplots()

    rmse_t.plot(ax=ax, color="red")

    ax.set(xlabel="Time", ylabel="nRMSE")
    ax.set_ylim((0, 1.0))
    plt.tight_layout()
    fig.savefig(savedir.joinpath("temporal_error.png"))
    plt.close()

    # =================================
    # ERROR VARIABILITY (SPATIAL)
    # =================================

    rmse_xy = rmse_space(ds_ref["ssh_predict"], ds_ref["ssh"])
    err_var_space = rmse_xy.std().values
    logger.info(f"Error Variability (Spatial)=  {err_var_space:.2f}")

    fig, ax = plt.subplots()

    rmse_xy.transpose("latitude", "longitude").plot.imshow(ax=ax)
    ax.set(xlabel="Longitude", ylabel="Latitude")
    fig.savefig(savedir.joinpath("spatial_error.png"))

    plt.tight_layout()

    # ==================================
    # MULTIVARIATE STATS (TEMPORAL)
    # ==================================
    from tqdm.notebook import tqdm
    from hyppo.independence import RV
    from hyppo.d_variate import dHsic
    import numpy as np

    times = []
    stats = {
        "rv": list(),
        "rvd": list(),
        "hsic": list(),
        "energy": list(),
    }

    for idata in tqdm(ds_ref.groupby("time")):
        # do statistic
        stats["rv"].append(
            RV().statistic(
                idata[1]["ssh"].values.flatten()[:, None],
                idata[1]["ssh_predict"].values.flatten()[:, None],
            )
        )

        # do statistic
        stats["rvd"].append(
            RV().statistic(idata[1]["ssh"].values, idata[1]["ssh_predict"].values)
        )
        # stats["energy"].append(
        #     Energy().statistic(idata[1]["ssh"].values, idata[1]["ssh_predict"].values)
        # )
        stats["hsic"].append(
            dHsic().statistic(idata[1]["ssh"].values, idata[1]["ssh_predict"].values)
            / (
                np.sqrt(
                    dHsic().statistic(idata[1]["ssh"].values, idata[1]["ssh"].values)
                )
                * np.sqrt(
                    dHsic().statistic(
                        idata[1]["ssh_predict"].values, idata[1]["ssh_predict"].values
                    )
                )
            )
        )
        times.append(idata[0])

    fig, ax = plt.subplots()

    rmse_t.plot(ax=ax, color="red", label="nRMSE")
    ax.plot(times, stats["rv"], color="orange", label="RV Coeff. (Flatten)")
    ax.plot(times, stats["rvd"], color="blue", label="RV Coeff.")
    ax.plot(times, stats["hsic"], color="green", label="nHSIC")

    ax.set(xlabel="Time", ylabel="Score")
    ax.set_ylim((0, 1.0))
    plt.legend()
    plt.tight_layout()
    fig.savefig(savedir.joinpath("temporal_multivar_error.png"))
    plt.close()

    logger.info(f"Time taken: {time.time()-t0:.2f} secs")


def psd(config: ml_collections.ConfigDict, resultsfile, savedir, variable_name=None):

    t0 = time.time()
    if savedir is None:
        savedir = Path(root).joinpath("figures")
        Path(savedir).mkdir(parents=True, exist_ok=True)
    else:
        savedir = Path(savedir)

    ds_ref = get_gridded_data(config, resultsfile, variable_name)

    import numpy as np

    # get the temporal normalization criteria
    logger.info(f"Normalizing temporal domain...")
    time_norm = np.timedelta64(config.evaluation.dt_freq, config.evaluation.dt_unit)
    logger.info(f"Normalizing: {time_norm}")

    # temporally normalize
    ds_ref["time"] = (ds_ref.time - ds_ref.time[0]) / time_norm

    # change units
    logger.info(f"Calculating longitude units..")
    ds_ref["longitude"] = ds_ref.longitude * 111e3
    ds_ref["latitude"] = ds_ref.latitude * 111e3

    from inr4ssh._src.operators.finite_diff import (
        calculate_gradient,
        calculate_laplacian,
    )

    logger.info("Calculating Kinetic Energy...")
    ds_ref["ssh_grad"] = calculate_gradient(ds_ref["ssh"], "longitude", "latitude")

    ds_ref["ssh_grad_predict"] = calculate_gradient(
        ds_ref["ssh_predict"], "longitude", "latitude"
    )

    logger.info("Calculating Enstropy...")
    ds_ref["ssh_lap"] = (
        0.5 * calculate_laplacian(ds_ref["ssh"], "longitude", "latitude") ** 2
    )
    ds_ref["ssh_lap_predict"] = (
        0.5 * calculate_laplacian(ds_ref["ssh_predict"], "longitude", "latitude") ** 2
    )

    # =======================================================
    # SSH SPATIAL-TEMPORAL PSD (Degrees)
    # =======================================================
    from inr4ssh._src.metrics.psd import psd_spacetime_score, psd_spacetime

    # Time-Longitude (Lat avg) PSD Score
    ds_field = ds_ref.chunk(
        {
            "time": 1,
            "longitude": ds_ref["longitude"].size,
            "latitude": ds_ref["latitude"].size,
        }
    ).compute()

    logger.info("Calculating PSD (NATL60)...")
    ds_field_psd = psd_spacetime(ds_field["ssh"])
    logger.info("Calculating PSD (Predictions)...")
    ds_predict_psd = psd_spacetime(ds_field["ssh_predict"])

    # PLOT TRUTH
    fig, ax = plot_st_psd(ds_field_psd, label="ssh", units="m")

    plt.tight_layout()
    fig.savefig(savedir.joinpath("psd_st_ssh_true.png"))
    plt.close()

    # PLOT PREDICTIONS
    fig, ax = plot_st_psd(ds_predict_psd, label="ssh", units="m")

    plt.tight_layout()
    fig.savefig(savedir.joinpath("psd_st_ssh_predict.png"))
    plt.close()

    # =======================================================
    # SSH SPATIAL-TEMPORAL PSD SCORE (Degrees)
    # =======================================================

    logger.info(f"Calculating Spatial-Temporal PSD Score...")
    ds_field_ = ds_field.chunk(
        {
            "time": 1,
            "longitude": ds_field["longitude"].size,
            "latitude": ds_field["latitude"].size,
        }
    ).compute()
    psd_score = psd_spacetime_score(ds_field_["ssh_predict"], ds_field_["ssh"])

    fig, ax = plot_st_psd_score(psd_score, units="m")

    plt.tight_layout()
    fig.savefig(savedir.joinpath("psd_score_st_ssh.png"))
    plt.close()

    logger.info(f"Time taken: {time.time() - t0:.2f} secs")

    # =======================================================
    # KINETIC ENERGY SPATIAL-TEMPORAL PSD (Meters)
    # =======================================================
    from inr4ssh._src.metrics.psd import psd_spacetime_score, psd_spacetime

    # =======================================================
    # KINETIC ENERGY SPATIAL-TEMPORAL PSD (Meters)
    # =======================================================
    # Time-Longitude (Lat avg) PSD Score
    ds_field = ds_field.chunk(
        {
            "time": 1,
            "longitude": ds_field["longitude"].size,
            "latitude": ds_field["latitude"].size,
        }
    ).compute()

    # ------------------
    # KINETIC ENERGY
    # ------------------

    logger.info("Calculating KE PSD (NATL60)...")
    ds_field_psd = psd_spacetime(ds_field["ssh_grad"])
    logger.info("Calculating KE PSD (Predictions)...")
    ds_predict_psd = psd_spacetime(ds_field["ssh_grad_predict"])

    # PLOT TRUTH
    fig, ax = plot_st_psd(ds_field_psd, label="ke", units="m")

    plt.tight_layout()
    fig.savefig(savedir.joinpath("psd_st_ke_true.png"))
    plt.close()

    # PLOT PREDICTIONS
    fig, ax = plot_st_psd(ds_predict_psd, label="ke", units="m")

    plt.tight_layout()
    fig.savefig(savedir.joinpath("psd_st_ke_predict.png"))
    plt.close()

    # ------------------
    # ENSTROPY
    # ------------------

    logger.info("Calculating KE PSD (NATL60)...")
    ds_field_psd = psd_spacetime(ds_field["ssh_lap"])
    logger.info("Calculating KE PSD (Predictions)...")
    ds_predict_psd = psd_spacetime(ds_field["ssh_lap_predict"])

    # PLOT TRUTH
    fig, ax = plot_st_psd(ds_field_psd, label="enstropy", units="m")

    plt.tight_layout()
    fig.savefig(savedir.joinpath("psd_st_ens_true.png"))
    plt.close()

    # PLOT PREDICTIONS
    fig, ax = plot_st_psd(ds_predict_psd, label="enstropy", units="m")

    plt.tight_layout()
    fig.savefig(savedir.joinpath("psd_st_ens_predict.png"))
    plt.close()

    # =======================================================
    # KINETIC ENERGY SPATIAL-TEMPORAL PSD SCORE (Degrees)
    # =======================================================

    logger.info(f"Calculating Kinetic Energy Spatial-Temporal PSD Score...")

    psd_score = psd_spacetime_score(
        ds_field_["ssh_grad_predict"], ds_field_["ssh_grad"]
    )

    fig, ax = plot_st_psd_score(psd_score, units="m")

    plt.tight_layout()
    fig.savefig(savedir.joinpath("psd_score_st_ke.png"))
    plt.close()

    logger.info(f"Time taken: {time.time() - t0:.2f} secs")

    # =======================================================
    # ENSTROPY SPATIAL-TEMPORAL PSD SCORE (Degrees)
    # =======================================================

    logger.info(f"Calculating Enstropy Spatial-Temporal PSD Score...")

    psd_score = psd_spacetime_score(ds_field_["ssh_lap_predict"], ds_field_["ssh_lap"])

    fig, ax = plot_st_psd_score(psd_score, units="m")

    plt.tight_layout()
    fig.savefig(savedir.joinpath("psd_score_st_ens.png"))
    plt.close()

    logger.info(f"Time taken: {time.time() - t0:.2f} secs")


def psd_iso(
    config: ml_collections.ConfigDict, resultsfile, savedir, variable_name=None
):

    t0 = time.time()
    if savedir is None:
        savedir = Path(root).joinpath("figures")
        Path(savedir).mkdir(parents=True, exist_ok=True)
    else:
        savedir = Path(savedir)

    ds_ref = get_gridded_data(config, resultsfile, variable_name)

    import numpy as np

    # get the temporal normalization criteria
    logger.info(f"Normalizing temporal domain...")
    time_norm = np.timedelta64(config.evaluation.dt_freq, config.evaluation.dt_unit)
    logger.info(f"Normalizing: {time_norm}")

    # temporally normalize
    ds_ref["time"] = (ds_ref.time - ds_ref.time[0]) / time_norm

    # change units
    logger.info(f"Calculating longitude units..")
    ds_ref["longitude"] = ds_ref.longitude * 111e3
    ds_ref["latitude"] = ds_ref.latitude * 111e3

    from inr4ssh._src.operators.finite_diff import (
        calculate_gradient,
        calculate_laplacian,
    )

    logger.info("Calculating Kinetic Energy...")
    ds_ref["ssh_grad"] = calculate_gradient(ds_ref["ssh"], "longitude", "latitude")

    ds_ref["ssh_grad_predict"] = calculate_gradient(
        ds_ref["ssh_predict"], "longitude", "latitude"
    )

    logger.info("Calculating Enstropy...")
    ds_ref["ssh_lap"] = (
        0.5 * calculate_laplacian(ds_ref["ssh"], "longitude", "latitude") ** 2
    )
    ds_ref["ssh_lap_predict"] = (
        0.5 * calculate_laplacian(ds_ref["ssh_predict"], "longitude", "latitude") ** 2
    )

    # =======================================================
    # SSH ISOTROPIC PSD (Degrees)
    # =======================================================
    from inr4ssh._src.metrics.psd import psd_spacetime_score, psd_spacetime

    # Time-Longitude (Lat avg) PSD Score
    ds_field = ds_ref.chunk(
        {
            "time": 1,
            "longitude": ds_ref["longitude"].size,
            "latitude": ds_ref["latitude"].size,
        }
    ).compute()

    logger.info("Calculating PSD (NATL60)...")
    ds_field_psd = psd_isotropic(ds_field["ssh"])
    logger.info("Calculating PSD (Predictions)...")
    ds_predict_psd = psd_isotropic(ds_field["ssh_predict"])

    # PLOT TRUTH
    fig, ax, secax = plot_psd_isotropic(
        ds_field_psd.freq_r.values * 1e3, ds_field_psd.values, color="black"
    )

    ax.plot(ds_predict_psd.freq_r.values * 1e3, ds_predict_psd.values, color="red")
    plt.xlim(
        (
            np.ma.min(np.ma.masked_invalid(ds_predict_psd.freq_r.values * 1e3)),
            np.ma.max(np.ma.masked_invalid(ds_predict_psd.freq_r.values * 1e3)),
        )
    )
    plt.legend(["Reference", "Comparison"])
    plt.tight_layout()
    fig.savefig(savedir.joinpath("psd_iso_ssh.png"))
    plt.close()

    # =======================================================
    # SSH ISOTROPIC PSD SCORE (Degrees)
    # =======================================================

    logger.info(f"Calculating Spatial-Temporal PSD Score...")
    psd_iso_score = psd_isotropic_score(ds_field["ssh_predict"], ds_field["ssh"])
    space_iso_resolved = wavelength_resolved_isotropic(psd_iso_score, level=0.5)
    logger.info(
        f"Shortest Spatial Wavelength Resolved = {space_iso_resolved:.2f} (degrees)"
    )

    fig, ax, secax = plot_psd_isotropic(
        psd_iso_score.freq_r.values * 1e3, psd_iso_score.values, color="black"
    )

    ax.set(ylabel="PSD Score", yscale="linear")
    plt.ylim((0, 1.0))
    plt.xlim(
        (
            np.ma.min(np.ma.masked_invalid(psd_iso_score.freq_r.values * 1e3)),
            np.ma.max(np.ma.masked_invalid(psd_iso_score.freq_r.values * 1e3)),
        )
    )

    # plot the graph point
    resolved_scale = 1 / (space_iso_resolved * 1e-3)
    ax.vlines(
        x=resolved_scale, ymin=0, ymax=0.5, color="green", linewidth=2, linestyle="--"
    )
    ax.hlines(
        y=0.5,
        xmin=np.ma.min(np.ma.masked_invalid(psd_iso_score.freq_r.values * 1e3)),
        xmax=resolved_scale,
        color="green",
        linewidth=2,
        linestyle="--",
    )

    label = f"Resolved Scales \n $\lambda$ > {int(space_iso_resolved * 1e-3)} km"
    plt.scatter(
        resolved_scale, 0.5, color="green", marker=".", linewidth=5, label=label
    )
    plt.legend()
    plt.tight_layout()
    fig.savefig(savedir.joinpath("psd_score_iso_ssh.png"))
    plt.close()

    # # =======================================================
    # # KINETIC ENERGY SPATIAL-TEMPORAL PSD (Meters)
    # # =======================================================
    # from inr4ssh._src.metrics.psd import psd_spacetime_score, psd_spacetime
    #
    #
    #
    # # =======================================================
    # # KINETIC ENERGY SPATIAL-TEMPORAL PSD (Meters)
    # # =======================================================
    # # Time-Longitude (Lat avg) PSD Score
    # ds_field = ds_field.chunk(
    #     {
    #         "time": 1,
    #         "longitude": ds_field["longitude"].size,
    #         "latitude": ds_field["latitude"].size,
    #     }
    # ).compute()
    #
    # # ------------------
    # # KINETIC ENERGY
    # # ------------------
    #
    # logger.info("Calculating KE PSD (NATL60)...")
    # ds_field_psd = psd_spacetime(ds_field["ssh_grad"])
    # logger.info("Calculating KE PSD (Predictions)...")
    # ds_predict_psd = psd_spacetime(ds_field["ssh_grad_predict"])
    #
    #
    # # PLOT TRUTH
    # fig, ax = plot_st_psd(ds_field_psd, label="ke", units="m")
    #
    # plt.tight_layout()
    # fig.savefig(savedir.joinpath("psd_st_ke_true.png"))
    # plt.close()
    #
    # # PLOT PREDICTIONS
    # fig, ax = plot_st_psd(ds_predict_psd, label="ke", units="m")
    #
    # plt.tight_layout()
    # fig.savefig(savedir.joinpath("psd_st_ke_predict.png"))
    # plt.close()
    #
    # # ------------------
    # # ENSTROPY
    # # ------------------
    #
    # logger.info("Calculating KE PSD (NATL60)...")
    # ds_field_psd = psd_spacetime(ds_field["ssh_lap"])
    # logger.info("Calculating KE PSD (Predictions)...")
    # ds_predict_psd = psd_spacetime(ds_field["ssh_lap_predict"])
    #
    #
    # # PLOT TRUTH
    # fig, ax = plot_st_psd(ds_field_psd, label="enstropy", units="m")
    #
    # plt.tight_layout()
    # fig.savefig(savedir.joinpath("psd_st_ens_true.png"))
    # plt.close()
    #
    # # PLOT PREDICTIONS
    # fig, ax = plot_st_psd(ds_predict_psd, label="enstropy", units="m")
    #
    # plt.tight_layout()
    # fig.savefig(savedir.joinpath("psd_st_ens_predict.png"))
    # plt.close()
    #
    # # =======================================================
    # # KINETIC ENERGY SPATIAL-TEMPORAL PSD SCORE (Degrees)
    # # =======================================================
    #
    # logger.info(f"Calculating Kinetic Energy Spatial-Temporal PSD Score...")
    #
    # psd_score = psd_spacetime_score(ds_field_["ssh_grad_predict"], ds_field_["ssh_grad"])
    #
    # fig, ax = plot_st_psd_score(psd_score, units="m")
    #
    # plt.tight_layout()
    # fig.savefig(savedir.joinpath("psd_score_st_ke.png"))
    # plt.close()
    #
    # logger.info(f"Time taken: {time.time() - t0:.2f} secs")
    #
    # # =======================================================
    # # ENSTROPY SPATIAL-TEMPORAL PSD SCORE (Degrees)
    # # =======================================================
    #
    # logger.info(f"Calculating Enstropy Spatial-Temporal PSD Score...")
    #
    # psd_score = psd_spacetime_score(ds_field_["ssh_lap_predict"], ds_field_["ssh_lap"])
    #
    # fig, ax = plot_st_psd_score(psd_score, units="m")
    #
    # plt.tight_layout()
    # fig.savefig(savedir.joinpath("psd_score_st_ens.png"))
    # plt.close()

    logger.info(f"Time taken: {time.time() - t0:.2f} secs")
