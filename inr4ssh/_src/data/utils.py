from typing import Callable, Optional, Dict, Any, List
import pickle
import tqdm
import xarray as xr
from dataclasses import asdict
import wandb
import subprocess
from pathlib import Path
import warnings
from dask.array.core import PerformanceWarning


def load_xr_datasets_list(files: List[str]):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=PerformanceWarning)
        # Note: there is an annoying performance memory due to the chunking
        ds = dict()

        for ifile in files:
            ds[str(Path(ifile).name)] = xr.open_dataset(ifile)

    return ds


def load_alongtrack_parallel(
    files: List[str], concat_dim: str = "time", preprocess=None
) -> xr.Dataset:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=PerformanceWarning)
        # Note: there is an annoying performance memory due to the chunking

        ds = xr.open_mfdataset(
            files,
            combine="nested",
            concat_dim=concat_dim,
            parallel=True,
            preprocess=preprocess,
            engine="netcdf4",
        )
        ds = ds.sortby(concat_dim)

    return ds
