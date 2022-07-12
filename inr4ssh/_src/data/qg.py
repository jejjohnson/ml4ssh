from pathlib import Path
import xarray as xr


def get_qg_filename():

    return "qg_sim.nc"


def load_qg_data(data_dir):

    filename = get_qg_filename()

    return xr.open_dataset(Path(data_dir).joinpath(filename))

