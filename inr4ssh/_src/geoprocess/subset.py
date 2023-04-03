import xarray as xr
import numpy as np


def subset_time(ds: xr.Dataset, time_min: str, time_max: str, **kwargs) -> xr.Dataset:
    time_min = np.datetime64(time_min)
    time_max = np.datetime64(time_max)
    return ds.sel(time=slice(time_min, time_max), **kwargs)


def subset_longitude(
    ds: xr.Dataset, lon_min: float, lon_max: float, **kwargs
) -> xr.Dataset:
    return ds.where((ds["lon"] >= lon_min) & (ds["lon"] <= lon_max), **kwargs)


def subset_latitude(
    ds: xr.Dataset, lat_min: float, lat_max: float, **kwargs
) -> xr.Dataset:
    return ds.where((ds["lat"] >= lat_min) & (ds["lat"] <= lat_max), **kwargs)
