import xarray as xr


def transform_360_to_180(ds: xr.Dataset) -> xr.Dataset:
    # ds["lon"] = (ds["lon"] % 360) - 180
    ds["lon"] = ((ds["lon"] + 180) % 360) - 180
    return ds
