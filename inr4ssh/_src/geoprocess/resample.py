import xarray as xr
import typing as tp


def coarsen_mean(ds: xr.Dataset, dim: tp.Mapping) -> xr.Dataset:
    return ds.coarsen(dim=dim).mean()


def coarsen_median(ds: xr.Dataset, dim: tp.Mapping) -> xr.Dataset:
    return ds.coarsen(dim=dim).median()


def resample_mean(ds: xr.Dataset, time: str) -> xr.Dataset:
    return ds.resample(time=time).mean()


def resample_median(ds: xr.Dataset, time: str) -> xr.Dataset:
    return ds.resample(time=time).median()
