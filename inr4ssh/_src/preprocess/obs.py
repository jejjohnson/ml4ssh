import numpy as np
from typing import List, Union
import pyinterp
from tqdm import tqdm
import xarray as xr


def add_noise(
    data, sigma: float = 0.01, noise: str = "gauss", seed: int = 123, df: int = 3
):

    rng = np.random.RandomState(seed)

    if noise == "gauss":
        return data + sigma * rng.standard_normal(size=data.shape)
    elif noise == "cauchy":
        return data + sigma * rng.standard_cauchy(size=data.shape)

    elif noise == "tstudent":
        return data + sigma * rng.standard_t(df=df, size=data.shape)

    elif noise == "exp":
        return data + sigma * rng.standard_exponential(size=data.shape)
    else:
        raise ValueError(f"Unrecognized noise: {noise}")


def get_streaming_lines(
    data: np.ndarray, rate: int = 10, width: int = 4, axis: str = "x", init: int = 0
) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Parameters
    ----------
    data : np.ndarray, (Nt, Nx, Ny)
        a 2D Spatio-Temporal array
    rate : int, default=10
        the speed of the tracks wrt to time.
        (best values are between 1-10)
    width : int, default=4
        the width of the track, SWOT=30, Nadir=2-4
    axis : str, default="x"
        (note, I dont think this does anything)

    Returns
    -------
    obs : np.ndarray, (Nt, Nx, Ny)
        an array where the values outside the track are masked out
    inds : List[np.ndarray]
        the indices that are masked per time step.
    """

    nt, nx, ny = data.shape

    data = data.reshape((nt, nx * ny))

    y_obs = np.empty_like(data)
    y_obs[:] = np.nan

    index_obs = []

    if axis == "x":
        loop_axis = nx
        n_points = ny
    elif axis == "y":
        loop_axis = ny
        n_points = nx
    else:
        raise ValueError(f"Unrecognized loop axis")

    for i in range(nt):

        index = []

        for j in range(loop_axis):
            start = init + n_points * (j - 1) + rate + j + (rate * np.mod(i, n_points))
            index.extend(np.arange(start, start + width))

            start = init + n_points * (j - 1) + rate - j - (rate * np.mod(i, n_points))
            index.extend(np.arange(start - width, start))

        index = np.array(index)

        idx = np.unique(np.where((index < (nx * ny)) & (index >= 0))[0])

        index = index[idx]

        index_obs.append(index)
        y_obs[i, index] = data[i, index]

    y_obs = y_obs.reshape((nt, nx, ny))

    return y_obs, index_obs


def add_obs_tracks(obs1, obs2):

    obs = np.nan

    obs = 0.5 * (np.nan_to_num(obs1) + np.nan_to_num(obs2))

    obs[obs == 0] = np.nan

    return obs


def bin_observations_coords(
    ds_obs: xr.Dataset,
    variable: str,
    lon_coords: np.ndarray,
    lat_coords: np.ndarray,
    time_coords: np.ndarray,
    time_buffer: np.timedelta64,
) -> xr.Dataset:

    # print("Coords:")
    # print(lon_coords.shape, lat_coords.shape, time_coords.shape)

    # create binning object
    binning = pyinterp.Binning2D(pyinterp.Axis(lon_coords), pyinterp.Axis(lat_coords))

    # initialize datasets
    ds_obs_binned = []

    for itime in tqdm(time_coords):
        binning.clear()

        # get all indices within timestamp + buffer
        ids = np.where((np.abs(ds_obs.time.values - itime) < 2.0 * time_buffer))[0]

        # extract lat,lon,values
        # print("Obs grid:")
        # print(ds_obs[variable].values.shape,
        #     ds_obs.longitude.values.shape, ds_obs.latitude.values.shape)
        values = np.ravel(ds_obs[variable].values[ids])
        lons = np.ravel(ds_obs.longitude.values[ids])
        lats = np.ravel(ds_obs.latitude.values[ids])

        # print("Obs grid (Subsampled):")
        # print(ids.shape, values.shape, lons.shape, lats.shape)

        # mask all nans
        msk = np.isfinite(values)

        # print(msk.shape)

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


def bin_observations_xr(
    ds_obs: xr.Dataset, ds_ref: xr.Dataset, variable: str, time_buffer: np.timedelta64
) -> xr.Dataset:

    return bin_observations_coords(
        ds_obs=ds_obs,
        variable=variable,
        lon_coords=ds_ref.longitude.values,
        lat_coords=ds_ref.latitude.values,
        time_coords=ds_ref.time.values,
        time_buffer=time_buffer,
    )

    # # create binning object
    # binning = pyinterp.Binning2D(
    #     pyinterp.Axis(ds_ref.longitude.values), pyinterp.Axis(ds_ref.latitude.values)
    # )
    #
    # # initialize datasets
    # ds_obs_binned = []
    #
    # for t in tqdm(ds_ref.time):
    #     binning.clear()
    #
    #     # get all indices within timestamp + buffer
    #     ids = np.where((np.abs(ds_obs.time.values - t.values) < 2.0 * time_buffer))[0]
    #
    #     # extract lat,lon,values
    #     values = np.ravel(ds_obs[variable].values[ids])
    #     lons = np.ravel(ds_obs.longitude.values[ids])
    #     lats = np.ravel(ds_obs.latitude.values[ids])
    #
    #     # mask all nans
    #     msk = np.isfinite(values)
    #
    #     binning.push(lons[msk], lats[msk], values[msk])
    #
    #     gridded = (
    #         ("time", "latitude", "longitude"),
    #         binning.variable("mean").T[None, ...],
    #     )
    #
    #     # create gridded dataset
    #     ds_obs_binned.append(
    #         xr.Dataset(
    #             {variable: gridded},
    #             {
    #                 "time": [t.values],
    #                 "latitude": np.array(binning.y),
    #                 "longitude": np.array(binning.x),
    #             },
    #         ).astype("float32", casting="same_kind")
    #     )
    #
    # # concatenate final dataset
    # ds_obs_binned = xr.concat(ds_obs_binned, dim="time")
    # return ds_obs_binned
