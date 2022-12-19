import pyinterp
import pyinterp.fill
import pyinterp.backends.xarray
from loguru import logger
import numpy as np
import xarray as xr
from einops import rearrange
from inr4ssh._src.preprocess.coords import Grid2DT, Coordinates2DT


def create_pyinterp_grid_2dt(da: xr.DataArray, is_circle: bool = False):

    # define source axis (and values)
    lon_source_axis = pyinterp.Axis(da.longitude.values, is_circle=is_circle)
    lat_source_axis = pyinterp.Axis(da.latitude.values, is_circle=False)
    time_source_axis = pyinterp.TemporalAxis(da.time.values)

    values_source = da.transpose("longitude", "latitude", "time").values

    # create source grid
    grid_source = pyinterp.Grid3D(
        lon_source_axis, lat_source_axis, time_source_axis, values_source
    )

    return grid_source


def regrid_2dt_from_grid(
    da: xr.DataArray, grid_target: Grid2DT, is_circle: bool = False
) -> xr.DataArray:

    # create spatial temporal pyinterp grid
    grid_source = create_pyinterp_grid_2dt(da, is_circle=is_circle)

    # # TODO: try to integrate pyinterp.backends.xarray
    # grid_source = pyinterp.backends.xarray.Grid3D(da)

    safe_cast = lambda x: pyinterp.TemporalAxis(da.time.values).safe_cast(x)

    # spatial-temporal interpolation
    data_interp = pyinterp.trivariate(
        grid_source,
        grid_target.lon_grid.flatten(),
        grid_target.lat_grid.flatten(),
        safe_cast(grid_target.time_grid.flatten()),
        bounds_error=False,
    )

    # get coordinates
    coords = grid_target.create_coords()

    # reshape
    data_interp = rearrange(
        data_interp,
        "(Lon Lat Time) -> Time Lat Lon",
        Lon=coords.lon_coords.shape[0],
        Lat=coords.lat_coords.shape[0],
        Time=coords.time_coords.shape[0],
    )

    # create data array
    da_interp = xr.DataArray(
        data_interp,
        dims=[
            "time",
            "latitude",
            "longitude",
        ],
        coords={
            "longitude": coords.lon_coords,
            "latitude": coords.lat_coords,
            "time": coords.time_coords,
        },
    )

    return da_interp


def regrid_2dt_from_da(
    da: xr.DataArray, da_ref: xr.DataArray, is_circle: bool = False
) -> xr.DataArray:

    grid_target = Coordinates2DT(
        lon_coords=da_ref.longitude.values,
        lat_coords=da_ref.latitude.values,
        time_coords=da_ref.time.values,
    ).create_grid()

    return regrid_2dt_from_grid(da=da, grid_target=grid_target, is_circle=is_circle)


def oi_regrid(da_source: xr.DataArray, da_target: xr.DataArray) -> xr.DataArray:

    logger.info("     Regridding...")

    # Define source grid
    x_source_axis = pyinterp.Axis(da_source["longitude"][:].values, is_circle=False)
    y_source_axis = pyinterp.Axis(da_source["latitude"][:].values)
    z_source_axis = pyinterp.TemporalAxis(da_source["time"][:].values)
    ssh_source = da_source[:].T
    grid_source = pyinterp.Grid3D(
        x_source_axis, y_source_axis, z_source_axis, ssh_source.data
    )

    # Define target grid
    mx_target, my_target, mz_target = np.meshgrid(
        da_target["longitude"].values,
        da_target["latitude"].values,
        z_source_axis.safe_cast(da_target["time"].values),
        indexing="ij",
    )
    # Spatio-temporal Interpolation
    ssh_interp = (
        pyinterp.trivariate(
            grid_source,
            mx_target.flatten(),
            my_target.flatten(),
            mz_target.flatten(),
            bounds_error=False,
        )
        .reshape(mx_target.shape)
        .T
    )

    # MB add extrapolation in NaN values if needed
    if np.isnan(ssh_interp).any():
        logger.info("     NaN found in ssh_interp, starting extrapolation...")
        x_source_axis = pyinterp.Axis(da_target["longitude"].values, is_circle=False)
        y_source_axis = pyinterp.Axis(da_target["latitude"].values)
        z_source_axis = pyinterp.TemporalAxis(da_target["time"][:].values)
        grid = pyinterp.Grid3D(
            x_source_axis, y_source_axis, z_source_axis, ssh_interp.T
        )
        has_converged, filled = pyinterp.fill.gauss_seidel(grid)
    else:
        filled = ssh_interp.T

    # Save to dataset
    da_interp = xr.DataArray(
        filled.T,
        dims=["time", "latitude", "longitude"],
        coords={
            "time": da_target["time"].values,
            "longitude": da_target["longitude"].values,
            "latitude": da_target["latitude"].values,
        },
    )

    return da_interp
