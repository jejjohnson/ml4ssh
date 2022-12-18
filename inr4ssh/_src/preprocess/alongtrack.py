import numpy as np
from typing import List, Union
import pyinterp
from tqdm import tqdm
import xarray as xr
from inr4ssh._src.preprocess.coords import Coordinates2DT


def alongtrack_bin_from_coords(
    ds_obs: xr.Dataset,
    variable: str,
    coords: Coordinates2DT,
    time_buffer: np.timedelta64,
) -> xr.Dataset:

    # print("Coords:")
    # print(lon_coords.shape, lat_coords.shape, time_coords.shape)

    # create binning object
    binning = pyinterp.Binning2D(
        pyinterp.Axis(coords.lon_coords), pyinterp.Axis(coords.lat_coords)
    )

    # initialize datasets
    ds_obs_binned = []

    for itime in tqdm(coords.time_coords):
        binning.clear()

        # get all indices within timestamp + buffer
        ids = np.where((np.abs(ds_obs.time.values - itime) < 2.0 * time_buffer))[0]

        # extract lat,lon,values
        values = np.ravel(ds_obs[variable].values[ids])
        lons = np.ravel(ds_obs.longitude.values[ids])
        lats = np.ravel(ds_obs.latitude.values[ids])

        # mask all nans
        msk = np.isfinite(values)

        # push finite values through
        binning.push(lons[msk], lats[msk], values[msk])

        gridded = (
            ("time", "latitude", "longitude"),
            binning.variable("mean").T[None, ...],
        )

        # create gridded dataset
        ds_obs_binned.append(
            xr.Dataset(
                {variable: gridded},
                {
                    "time": [itime],
                    "latitude": np.array(binning.y),
                    "longitude": np.array(binning.x),
                },
            ).astype("float32", casting="same_kind")
        )

    # concatenate final dataset
    ds_obs_binned = xr.concat(ds_obs_binned, dim="time")
    return ds_obs_binned


def alongtrack_bin_from_da(
    ds_obs: xr.Dataset, ds_ref: xr.Dataset, variable: str, time_buffer: np.timedelta64
) -> xr.Dataset:

    coords = Coordinates2DT(
        lon_coords=ds_ref.longitude.values,
        lat_coords=ds_ref.latitude.values,
        time_coords=ds_ref.time.values,
    )

    return alongtrack_bin_from_coords(
        ds_obs=ds_obs, variable=variable, coords=coords, time_buffer=time_buffer
    )
