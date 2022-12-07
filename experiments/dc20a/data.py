import sys, os

import ml_collections

from pyprojroot import here

# spyder up to find the root
root = here(project_files=[".root"])

# append to path
sys.path.append(str(root))
import ml_collections
from loguru import logger
import time
from typing import List, Optional, Union
from pathlib import Path
import xarray as xr
from inr4ssh._src.data.natl60 import download_obs, download_ref, URL_REF, URL_OBS
from inr4ssh._src.data.natl60 import check_osse_files, get_raw_altimetry_files
from inr4ssh._src.data.utils import load_xr_datasets_list, load_alongtrack_parallel
from inr4ssh._src.preprocess.swot import preprocess_karin_swot


def download(datadir: str = None, dataset: str = "obs") -> None:

    if dataset.lower() == "obs":
        logger.info(f"Downloading obs data from:")
        logger.info(f"{URL_OBS}")
        logger.info(f"To directory: {datadir}")
        download_obs(datadir)
        logger.info("Done!")

    elif dataset.lower() == "ref":
        logger.info(f"Downloading natl60 data from:")
        logger.info(f"{URL_REF}")
        logger.info(f"To directory: {datadir}")
        download_ref(datadir)
        logger.info("Done!")

    else:
        raise ValueError(f"unrecognized dataset: {dataset.lower()}")


def preprocess(config: ml_collections.ConfigDict) -> None:

    logger.info("Running preprocess (clean) script...")
    t0 = time.time()

    obs_dir = Path(config.datadir.raw.obs_dir)

    logger.info("checking osse files...")
    logger.debug(f"Checking file: {str(obs_dir)}")
    check_osse_files(obs_dir, None, "obs")

    logger.info("Get NADIR track files...")
    alongtrack_files = get_raw_altimetry_files(obs_dir)

    # load them in parallel
    logger.info("Load NADIR tracks (parallel)...")
    ds_nadirs = load_xr_datasets_list(alongtrack_files)

    for ifilename, ids in ds_nadirs.items():
        # save dataset
        Path(config.datadir.clean.obs_dir).mkdir(parents=True, exist_ok=True)
        ifilename = Path(config.datadir.clean.obs_dir).joinpath(ifilename)
        logger.info(f"Saving NADIR: {ifilename}...")
        ids.to_netcdf(ifilename)

    logger.info("Get SWOT track files...")
    file_path = get_raw_altimetry_files(obs_dir, "swot")

    logger.info("Open KARIN SWOT dataset...")
    ds_karin_swot = xr.open_dataset(file_path[0])

    logger.info("Preprocess KARIN SWOT dataset...")
    ds_karin_swot = preprocess_karin_swot(ds_karin_swot, author="")

    logger.debug("Checking size of swot data...")
    assert ds_karin_swot.coords["time"].shape == (8_412_216,)

    ifilename = Path(config.datadir.clean.obs_dir).joinpath(file_path[0].name)
    logger.info(f"Saving KARIN SWOT: {ifilename}...")
    ds_karin_swot.to_netcdf(ifilename)

    # SWOT NADIR DATA
    logger.info("Get SWOT track files...")
    file_path = get_raw_altimetry_files(obs_dir, "swotnadir")

    logger.info("Open KARIN SWOT NADIR dataset...")
    ds_karin_swot_nadir = xr.open_dataset(file_path[0])

    # select the first cycle
    ds_karin_swot_nadir = ds_karin_swot_nadir.isel(cycle=0)

    logger.debug("Checking size of swot nadir data...")
    assert ds_karin_swot_nadir.coords["time"].shape == (161_333,)

    ifilename = Path(config.datadir.clean.obs_dir).joinpath(file_path[0].name)
    logger.info(f"Saving KARIN SWOT NADIR, {ifilename}...")
    ds_karin_swot_nadir.to_netcdf(ifilename)

    logger.info(f"Done!")
    logger.debug(f"Time Taken: {time.time()-t0:.2f} secs")


def ml_ready(config: ml_collections.ConfigDict, experiment: str) -> None:
    from inr4ssh._src.io import list_all_files
    from inr4ssh._src.data.natl60 import get_swot_obs_setup_files

    logger.info(f"Starting preprocess (ml_ready) script...")
    logger.info(f"Dataset: {experiment}...")
    t0 = time.time()

    logger.info("Getting files in directory...")
    all_files = list_all_files(config.datadir.clean.obs_dir)

    logger.info(f"Loading files for dataset: {experiment}")
    setup_files = get_swot_obs_setup_files(all_files, experiment)

    # choose the variables
    variables = ["ssh_obs", "ssh_model", "lon", "lat"]
    logger.info(f"Selecting variables:")
    logger.info(f"{variables}")

    import numpy as np
    from inr4ssh._src.preprocess.spatial import convert_lon_360_180
    from inr4ssh._src.data.utils import load_alongtrack_parallel

    def preprocess(x):
        x = x[variables]

        # subset temporal
        time_min = np.datetime64(config.preprocess.subset_time.time_min)
        time_max = np.datetime64(config.preprocess.subset_time.time_max)

        x = x.sel(time=slice(time_min, time_max))

        # correct longitude dimensions
        x["lon"] = convert_lon_360_180(x["lon"])

        # subset region
        x = x.where(
            (x["lon"] >= config.preprocess.subset_spatial.lon_min)
            & (x["lon"] <= config.preprocess.subset_spatial.lon_max)
            & (x["lat"] >= config.preprocess.subset_spatial.lat_min)
            & (x["lat"] <= config.preprocess.subset_spatial.lat_max),
            drop=True,
        )

        return x

    logger.info("Loading preprocessing script...")
    ds_swot = load_alongtrack_parallel(setup_files, preprocess=preprocess)

    # sort by time
    logger.info("Sorting by time...")
    ds_swot = ds_swot.sortby("time").compute()

    logger.info("Saving data...")
    Path(config.datadir.staging.staging_dir).mkdir(parents=True, exist_ok=True)
    save_name = Path(config.datadir.staging.staging_dir).joinpath(f"{experiment}.nc")
    logger.info(f"{save_name}")
    ds_swot.to_netcdf(save_name)

    logger.info(f"Done!")
    logger.debug(f"Time Taken: {time.time()-t0:.2f} secs")
