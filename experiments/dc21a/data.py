from pathlib import Path
import ml_collections
import time
import numpy as np
import yaml
from loguru import logger

from inr4ssh._src.io import check_if_file, check_if_directory, list_all_files, runcmd
from inr4ssh._src.data.dc21a import (
    download_obs,
    download_correction,
    download_results,
    get_dc21a_obs_setup_files,
)
from inr4ssh._src.preprocess.spatial import convert_lon_360_180
from inr4ssh._src.data.utils import load_alongtrack_parallel


def download(datadir: str, creds_file: str, dataset: str = "obs") -> None:
    """The script to download the datasets.

    Args:
        datadir (str): the directory to store the dataset
        creds_file (str): the .yaml file with the credentials for the user/pwd
        dataset (str): the dataset to download
            options = {"obs", "correction", "results"}
    """

    # check if directory
    dataset = dataset.lower()

    logger.debug(f"Data directory: {datadir}")
    check_if_directory(datadir)

    # check if credentials file exists
    logger.debug(f"Credentials file: {creds_file}")
    check_if_file(creds_file)

    # extract credentials from file
    logger.info("Loading yaml file...")
    with open(creds_file, "r") as file:
        creds = yaml.safe_load(file)
        username = creds["username"]
        password = creds["password"]

    if dataset == "obs":
        # create obs directory
        datadir = Path(datadir).joinpath("raw/obs")
        datadir.mkdir(parents=True, exist_ok=True)

        # download the data
        logger.info("Downloading altimetry tracks...")
        logger.debug(f"{datadir}")
        download_obs(str(datadir), username=username, password=password)

        logger.info("Removing dead files...")
        runcmd(f"rm {datadir}/._*")

    elif dataset == "correction":
        # create obs directory
        datadir = Path(datadir).joinpath("raw/correction")
        datadir.mkdir(parents=True, exist_ok=True)

        # download the data
        logger.info("Downloading correction dataset...")
        logger.debug(f"{datadir}")
        download_correction(str(datadir), username=username, password=password)

        logger.info("Removing dead files...")
        runcmd(f"rm {datadir}/._*")

    elif dataset == "evaluation":
        # check directory
        # make reference directory
        # download the data
        raise NotImplementedError()

    elif dataset == "results":
        # create obs directory
        datadir = Path(datadir).joinpath("results")
        datadir.mkdir(parents=True, exist_ok=True)

        # download the data
        logger.info("Downloading results dataset...")
        logger.debug(f"{datadir}")
        download_results(str(datadir), username=username, password=password)

    else:
        raise ValueError(f"Unrecognized dataset: {dataset}")


def ml_ready(config: ml_collections.ConfigDict) -> None:

    logger.info("Starting preprocess (ml ready) script...")
    t0 = time.time()

    logger.info("Getting files in directory...")
    all_files = list_all_files(config.datadir.raw.obs_dir)

    logger.info("Loading files for dataset (train)...")
    setup_files = get_dc21a_obs_setup_files(all_files, setup="train")

    def preprocess(x):
        x = x["sla_unfiltered"]

        # correct longitude dimensions
        x["longitude"] = convert_lon_360_180(x["longitude"])

        # subset region
        x = x.where(
            (x["longitude"] >= config.preprocess.subset_spatial.lon_min)
            & (x["longitude"] <= config.preprocess.subset_spatial.lon_max)
            & (x["latitude"] >= config.preprocess.subset_spatial.lat_min)
            & (x["latitude"] <= config.preprocess.subset_spatial.lat_max),
            drop=True,
        )

        # subset temporal
        time_min = np.datetime64(config.preprocess.subset_time.time_min)
        time_max = np.datetime64(config.preprocess.subset_time.time_max)

        x = x.sel(time=slice(time_min, time_max))

        return x

    logger.info("Loading train alongtrack data...")
    ds_alongtracks = load_alongtrack_parallel(setup_files, preprocess=preprocess)

    # sort by time
    logger.info("Sorting by time...")
    ds_alongtracks = ds_alongtracks.sortby("time").compute()

    logger.info("Saving data...")
    Path(config.datadir.staging.staging_dir).mkdir(parents=True, exist_ok=True)
    save_name = Path(config.datadir.staging.staging_dir).joinpath(f"train.nc")
    logger.info(f"{save_name}")
    ds_alongtracks.to_netcdf(save_name)

    logger.info("Loading files for dataset (test)...")
    setup_files = get_dc21a_obs_setup_files(all_files, setup="test")

    logger.info("Loading test alongtrack data...")
    ds_alongtracks = load_alongtrack_parallel(setup_files, preprocess=preprocess)

    # sort by time
    logger.info("Sorting by time...")
    ds_alongtracks = ds_alongtracks.sortby("time").compute()

    logger.info("Saving data...")
    Path(config.datadir.staging.staging_dir).mkdir(parents=True, exist_ok=True)
    save_name = Path(config.datadir.staging.staging_dir).joinpath(f"test.nc")
    logger.info(f"{save_name}")
    ds_alongtracks.to_netcdf(save_name)

    logger.info(f"Done!")
    logger.debug(f"Time Taken: {time.time()-t0:.2f} secs")
