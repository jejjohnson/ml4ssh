import numpy as np
import pyinterp
import xarray as xr


def create_spatiotemporal_coords(
        lon_min, lon_max, lon_dx,
        lat_min, lat_max, lat_dy,
        time_min, time_max, time_dt):
    # create all coordinates
    glon = np.arange(lon_min, lon_max + lon_dx, lon_dx)  # output OI longitude grid
    glat = np.arange(lat_min, lat_max + lat_dy, lat_dy)  # output OI latitude grid
    gtime = np.arange(time_min, time_max + time_dt, time_dt)  # output OI time grid

    return glon, glat, gtime


def extract_gridded_coords(
        ds: xr.Dataset,
        lon_min=0.,
        lon_max=360.,
        lat_min=-90,
        lat_max=90.,
        time_min='1900-10-01',
        time_max='2100-01-01',
        variable="ssh",
        is_circle=True):
    ds = ds.sel(time=slice(time_min, time_max), drop=True)
    ds = ds.where((ds["longitude"] % 360. >= lon_min) & (ds["longitude"] % 360. <= lon_max), drop=True)
    ds = ds.where((ds["latitude"] >= lat_min) & (ds["latitude"] <= lat_max), drop=True)

    x_axis = pyinterp.Axis(ds["longitude"][:].values % 360., is_circle=is_circle)
    y_axis = pyinterp.Axis(ds["latitude"][:].values)
    z_axis = pyinterp.TemporalAxis(ds["time"][:].values)

    var = ds[variable][:]
    var = var.transpose('longitude', 'latitude', 'time')

    # The undefined values must be set to nan.
    try:
        var[var.mask] = float("nan")
    except AttributeError:
        pass

    grid = pyinterp.Grid3D(x_axis, y_axis, z_axis, var.data)

    del ds

    return x_axis, y_axis, z_axis, grid