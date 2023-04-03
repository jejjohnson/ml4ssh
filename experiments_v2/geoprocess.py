import loguru
import time
from pathlib import Path
import pyrootutils
from functools import reduce

pyrootutils.setup_root(__file__, indicator=".root", pythonpath=True)

import hydra
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
from inr4ssh._src.files import make_directory


def alongtrack(cfg) -> None:
    loguru.logger.info("Running geoprocessing script...")
    t0 = time.time()

    logger.info("Getting files in directory...")
    all_files = list_all_files(cfg.clean_obs_dir)

    logger.info(f"Loading files for dataset: {cfg.experiment}")
    setup_files = get_swot_obs_setup_files(all_files, cfg.case)

    # ds_karin_swot_nadir.to_netcdf(ifilename)

    # choose the variables
    logger.info(f"Selecting variables:")
    logger.info(f"{cfg.variables}")

    transforms: List[callable] = hydra.utils.instantiate(cfg.transforms)

    import functools

    def compose(*functions):
        return functools.reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)

    preprocess = compose(*transforms) if transforms is not None else None

    logger.info("Loading preprocessing script...")
    ds_swot = load_alongtrack_parallel(setup_files, preprocess=preprocess)

    # sort by time
    logger.info("Sorting by time...")
    ds_swot = ds_swot.sortby("time").compute()

    logger.info("Saving data...")
    Path(cfg.staging_dir).mkdir(parents=True, exist_ok=True)
    save_name = Path(cfg.staging_dir).joinpath(f"{cfg.case}.nc")
    logger.info(f"Save Name: {save_name}")
    ds_swot.to_netcdf(str(save_name))

    logger.info(f"Done!")
    logger.debug(f"Time Taken: {time.time()-t0:.2f} secs")


def grid(cfg) -> None:
    pass


@hydra.main(config_path="../config", config_name="main", version_base="1.2")
def main(cfg):
    dataset = cfg.get("dataset")
    loguru.logger.info(f"Initializing Geoprocessing. Dataset: {dataset}")

    if dataset == "alongtrack":
        alongtrack(cfg.geoprocess)
    elif dataset == "grid":
        raise NotImplementedError(f"")
    elif dataset == "hybrid":
        raise NotImplementedError()
    else:
        raise ValueError(f"Unrecognized experiment: {dataset}")


if __name__ == "__main__":
    main()
