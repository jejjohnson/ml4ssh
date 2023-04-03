import xarray as xr


def sortby_time(ds: xr.Dataset, dims: str) -> xr.Dataset:
    return ds.sortby(dims)
