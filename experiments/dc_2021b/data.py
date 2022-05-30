import tqdm
from pathlib import Path
import xarray as xr

def get_data_args(parser):
    parser.add_argument('--train-data-dir', type=str, default="/home/johnsonj/data/dc_2021/raw/train")
    parser.add_argument('--ref-data-dir', type=str, default="/home/johnsonj/data/dc_2021/raw/ref/")
    parser.add_argument('--test-data-dir', type=str, default="/home/johnsonj/data/dc_2021/raw/test/")
    return parser


def get_train_filenames():

    return [
        "dt_gulfstream_alg_phy_l3_20161201-20180131_285-315_23-53.nc",
        "dt_gulfstream_h2g_phy_l3_20161201-20180131_285-315_23-53.nc",
        "dt_gulfstream_j2g_phy_l3_20161201-20180131_285-315_23-53.nc",
        "dt_gulfstream_j2n_phy_l3_20161201-20180131_285-315_23-53.nc",
        "dt_gulfstream_j3_phy_l3_20161201-20180131_285-315_23-53.nc",
        "dt_gulfstream_s3a_phy_l3_20161201-20180131_285-315_23-53.nc",
    ]

def get_ref_filename():

    return "mdt.nc"
def load_data(config):

    list_of_datasets = []

    files = get_train_filenames()

    for ifile in tqdm.tqdm(files):
        
        ids = xr.open_dataset(Path(config.train_data_dir).joinpath(ifile))
        list_of_datasets.append(ids)
        
    # concatenate
    ds_obs = xr.concat(list_of_datasets, dim='time')

    # sort by time
    ds_obs = ds_obs.sortby("time")

    return ds_obs


