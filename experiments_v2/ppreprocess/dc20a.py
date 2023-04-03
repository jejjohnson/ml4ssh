import loguru
import time
from pathlib import Path

import ml_collections
from loguru import logger
import time
from typing import List, Optional, Union
from pathlib import Path
import xarray as xr
import numpy as np
from inr4ssh._src.data.natl60 import download_obs, download_ref, URL_REF, URL_OBS
from inr4ssh._src.data.natl60 import check_dc20a_files, get_raw_altimetry_files
from inr4ssh._src.data.utils import load_xr_datasets_list, load_alongtrack_parallel
from inr4ssh._src.preprocess.swot import preprocess_karin_swot
from inr4ssh._src.io import list_all_files
from inr4ssh._src.data.natl60 import get_swot_obs_setup_files
from inr4ssh._src.preprocess.spatial import convert_lon_360_180
from inr4ssh._src.data.utils import load_alongtrack_parallel

pyrootutils.setup_root(__file__, indicator=".root", pythonpath=True)


def preprocess_dc20a(obs_dir: str) -> None:
    loguru.logger.info("Running preprocess (clean) script...")
    t0 = time.time()

    obs_dir = Path(obs_dir)

    loguru.logger.info("checking osse files...")
    loguru.logger.debug(f"Checking file: {str(obs_dir)}")
    check_dc20a_files(obs_dir, None, "obs")

    # logger.info("Get NADIR track files...")
    # alongtrack_files = get_raw_altimetry_files(obs_dir)

    # # load them in parallel
    # logger.info("Load NADIR tracks (parallel)...")
    # ds_nadirs = load_xr_datasets_list(alongtrack_files)

    # for ifilename, ids in ds_nadirs.items():
    #     # save dataset
    #     Path(config.datadir.clean.obs_dir).mkdir(parents=True, exist_ok=True)
    #     ifilename = Path(config.datadir.clean.obs_dir).joinpath(ifilename)
    #     logger.info(f"Saving NADIR: {ifilename}...")
    #     ids.to_netcdf(ifilename)

    # logger.info("Get SWOT track files...")
    # file_path = get_raw_altimetry_files(obs_dir, "swot")

    # logger.info("Open KARIN SWOT dataset...")
    # logger.debug(f"File path: \n{file_path}")
    # ds_karin_swot = xr.open_dataset(file_path[0], engine="netcdf4")

    # logger.info("Preprocess KARIN SWOT dataset...")
    # ds_karin_swot = preprocess_karin_swot(ds_karin_swot, author="")

    # logger.debug("Checking size of swot data...")
    # assert ds_karin_swot.coords["time"].shape == (8_412_216,)

    # ifilename = Path(config.datadir.clean.obs_dir).joinpath(file_path[0].name)
    # logger.info(f"Saving KARIN SWOT: {ifilename}...")
    # ds_karin_swot.to_netcdf(ifilename)

    # # SWOT NADIR DATA
    # logger.info("Get SWOT track files...")
    # file_path = get_raw_altimetry_files(obs_dir, "swotnadir")

    # logger.info("Open KARIN SWOT NADIR dataset...")
    # logger.debug(f"File path: \n{file_path}")
    # ds_karin_swot_nadir = xr.open_dataset(file_path[0], engine="netcdf4")

    # # select the first cycle
    # ds_karin_swot_nadir = ds_karin_swot_nadir.isel(cycle=0)

    # logger.debug("Checking size of swot nadir data...")
    # assert ds_karin_swot_nadir.coords["time"].shape == (161_333,)

    # ifilename = Path(config.datadir.clean.obs_dir).joinpath(file_path[0].name)
    # logger.info(f"Saving KARIN SWOT NADIR, {ifilename}...")
    # ds_karin_swot_nadir.to_netcdf(ifilename)

    logger.info(f"Done!")
    logger.debug(f"Time Taken: {time.time()-t0:.2f} secs")
