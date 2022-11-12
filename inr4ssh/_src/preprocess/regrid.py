import pyinterp
import pyinterp.fill
from loguru import logger
import numpy as np
import xarray as xr


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
