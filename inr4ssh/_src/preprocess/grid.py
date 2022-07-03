
import xarray as xr
import numpy as np
from .coords import create_spatiotemporal_coords


def get_meshgrid(res: float, nx: int, ny: int):
    dx = res
    dy = res
    x = np.linspace(-1, 1, int(nx)) * (nx - 1) * dx / 2
    y = np.linspace(-1, 1, int(ny)) * (ny - 1) * dy / 2
    return np.meshgrid(x, y)


def create_spatiotemporal_grid(
        lon_min, lon_max, lon_dx,
        lat_min, lat_max, lat_dy,
        time_min, time_max, time_dt):
    glon, glat, gtime = create_spatiotemporal_coords(
        lon_min, lon_max, lon_dx,
        lat_min, lat_max, lat_dy,
        time_min, time_max, time_dt
    )  # output OI time grid

    # Make 3D grid
    glon2, glat2, gtime2 = np.meshgrid(glon, glat, gtime)
    lon_coords = glon2.flatten()
    lat_coords = glat2.flatten()
    time_coords = gtime2.flatten()

    return lon_coords, lat_coords, time_coords


def create_grids(ds: xr.DataArray, variable: str, is_circle=True):
    import pyinterp
    x_axis = pyinterp.Axis(ds["longitude"][:] % 360., is_circle=is_circle)
    y_axis = pyinterp.Axis(ds["latitude"][:])
    z_axis = pyinterp.TemporalAxis(ds["time"][:].values)
    var = ds[variable][:]
    var = var.transpose("longitude", "latitude", "time")

    try:
        var[var.mask] = np.nan
    except AttributeError:
        pass

    grid = pyinterp.Grid3D(x_axis, y_axis, z_axis, var.values)

    del ds

    return x_axis, y_axis, z_axis, grid