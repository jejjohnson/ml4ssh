from typing import List
from pathlib import Path
import xarray as xr
import tqdm


def get_altimetry_train_filenames():

    return [
        "dt_gulfstream_alg_phy_l3_20161201-20180131_285-315_23-53.nc",
        "dt_gulfstream_h2g_phy_l3_20161201-20180131_285-315_23-53.nc",
        "dt_gulfstream_j2g_phy_l3_20161201-20180131_285-315_23-53.nc",
        "dt_gulfstream_j2n_phy_l3_20161201-20180131_285-315_23-53.nc",
        "dt_gulfstream_j3_phy_l3_20161201-20180131_285-315_23-53.nc",
        "dt_gulfstream_s3a_phy_l3_20161201-20180131_285-315_23-53.nc",
    ]


def get_altimetry_test_filename():
    return "dt_gulfstream_c2_phy_l3_20161201-20180131_285-315_23-53.nc"


def get_altimetry_correction_filename():
    return "mdt.nc"


def check_data_dir(train: bool=True):
    # TODO: check directory for train/test/correction files.
    return None

def load_ssh_altimetry_data_train(data_dir: str):
    list_of_datasets = []

    files = get_altimetry_train_filenames()

    for ifile in tqdm.tqdm(files):
        ids = xr.open_dataset(Path(data_dir).joinpath(ifile))
        list_of_datasets.append(ids)

    # concatenate
    ds_obs = xr.concat(list_of_datasets, dim='time')

    # sort by time
    ds_obs = ds_obs.sortby("time")

    return ds_obs


def load_ssh_altimetry_data_test(data_dir: str):

    # get correction filename
    filename = get_altimetry_test_filename()

    # get directory
    ref_dir = Path(data_dir).joinpath(filename)

    # load dataset
    ds_test = xr.open_dataset(ref_dir)
    
    # sort by time
    ds_test = ds_test.sortby("time")

    return ds_test

def load_ssh_correction(data_dir: str):

    # get correction filename
    filename = get_altimetry_correction_filename()

    # get directory
    ref_dir = Path(data_dir).joinpath(filename)

    # load dataset
    ds_correct = xr.open_dataset(ref_dir)

    # rename values
    ds_correct = ds_correct.rename({"lat": "latitude", "lon": "longitude"})

    return ds_correct

