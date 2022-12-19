from typing import Union
import numpy as np
from dataclasses import dataclass
import xarray as xr
import ml_collections
import pyinterp

from inr4ssh._src.preprocess.spatial import convert_lon_360_180


@dataclass
class Bounds2DT:
    lon_min: np.ndarray
    lon_max: np.ndarray
    dlon: Union[float, np.ndarray]
    lat_min: np.ndarray
    lat_max: np.ndarray
    dlat: Union[float, np.ndarray]
    time_min: np.ndarray
    time_max: np.ndarray
    dtime: Union[np.ndarray, str, np.timedelta64]

    def __init__(
        self,
        lon_min: np.ndarray,
        lon_max: np.ndarray,
        dlon: Union[float, np.ndarray],
        lat_min: np.ndarray,
        lat_max: np.ndarray,
        dlat: Union[float, np.ndarray],
        time_min: np.ndarray,
        time_max: np.ndarray,
        dtime: Union[np.ndarray, str, np.timedelta64],
    ):
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.dlon = dlon
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.dlat = dlat
        self.time_min = time_min
        self.time_max = time_max
        if isinstance(dtime, str):
            dtime = np.timedelta64(*dtime.split("_"))
        elif isinstance(dtime, np.timedelta64) | isinstance(dtime, np.ndarray):
            pass
        else:
            raise ValueError(f"Unrecognized type for {dtime} - {type(dtime)}")

        self.dtime = dtime

    @staticmethod
    def init_from_config(config: ml_collections.ConfigDict):

        return Bounds2DT(
            lon_min=config.lon_min,
            lon_max=config.lon_max,
            dlon=config.dlon,
            lat_min=config.lat_min,
            lat_max=config.lat_max,
            dlat=config.dlat,
            time_min=config.time_min,
            time_max=config.time_max,
            dtime=config.dtime,
        )

    def create_coordinates(self):

        lon_coords, lat_coords, time_coords = create_spatiotemporal_coords(
            lon_min=self.lon_min,
            lon_max=self.lon_max,
            lon_dx=self.dlon,
            lat_min=self.lat_min,
            lat_max=self.lat_max,
            lat_dy=self.dlat,
            time_min=self.time_min,
            time_max=self.time_max,
            time_dt=self.dtime,
        )

        return Coordinates2DT(
            lon_coords=lon_coords, lat_coords=lat_coords, time_coords=time_coords
        )


@dataclass
class Coordinates2DT:
    lon_coords: np.ndarray
    lat_coords: np.ndarray
    time_coords: np.ndarray

    def create_pyinterp_axis(self):
        raise NotImplementedError()

    def create_grid(self):
        lon_grid, lat_grid, time_grid = np.meshgrid(
            self.lon_coords, self.lat_coords, self.time_coords, indexing="ij"
        )

        return Grid2DT(lon_grid=lon_grid, lat_grid=lat_grid, time_grid=time_grid)

    def create_grid_xr(self):
        raise NotImplementedError()

    @staticmethod
    def create_from_xr(da):
        return Coordinates2DT(
            lon_coords=da.longitude.values,
            lat_coords=da.latitude.values,
            time_coords=da.time.values,
        )

    def create_bounds(self):
        # get easy bounds
        lon_min = self.lon_coords.min()
        lon_max = self.lon_coords.max()
        lat_min = self.lat_coords.min()
        lat_max = self.lat_coords.max()
        time_min = self.time_coords.min()
        time_max = self.time_coords.max()

        # get differences
        dlon = np.diff(self.lon_coords)[1:-1].mean()
        dlat = np.diff(self.lat_coords)[1:-1].mean()
        dtime = np.diff(self.time_coords)[1:-1].mean()
        return Bounds2DT(
            lon_min=lon_min,
            lon_max=lon_max,
            dlon=dlon,
            lat_min=lat_min,
            lat_max=lat_max,
            dlat=dlat,
            time_min=time_min,
            time_max=time_max,
            dtime=dtime,
        )


@dataclass
class Grid2DT:
    lon_grid: np.ndarray
    lat_grid: np.ndarray
    time_grid: np.ndarray

    def create_coords(self):
        return Coordinates2DT(
            lon_coords=self.lon_grid[:, 0, 0],
            lat_coords=self.lat_grid[0, :, 0],
            time_coords=self.time_grid[0, 0, :],
        )


def correct_longitude_domain(ds):
    if ds["longitude"].min() < 0:
        ds["longitude"] = xr.where(
            ds["longitude"] >= 180.0, ds["longitude"] - 360.0, ds["longitude"]
        )
    return ds


def correct_coordinate_labels(ds):
    try:
        ds = ds.rename({"lat": "latitude"})
    except ValueError:
        pass

    try:
        ds = ds.rename({"lon": "longitude"})
    except ValueError:
        pass

    return ds


def create_spatiotemporal_coords(
    lon_min, lon_max, lon_dx, lat_min, lat_max, lat_dy, time_min, time_max, time_dt
):
    # create all coordinates
    glon = np.arange(lon_min, lon_max + lon_dx, lon_dx)  # output OI longitude grid
    glat = np.arange(lat_min, lat_max + lat_dy, lat_dy)  # output OI latitude grid
    gtime = np.arange(time_min, time_max + time_dt, time_dt)  # output OI time grid

    return glon, glat, gtime


def extract_gridded_coords(
    ds: xr.Dataset,
    lon_min=0.0,
    lon_max=360.0,
    lat_min=-90,
    lat_max=90.0,
    time_min="1900-10-01",
    time_max="2100-01-01",
    variable="ssh",
    is_circle=True,
):
    import pyinterp

    ds = ds.sel(time=slice(time_min, time_max), drop=True)
    try:
        ds = ds.where(
            (ds["longitude"] >= lon_min) & (ds["longitude"] <= lon_max), drop=True
        )
        ds = ds.where(
            (ds["latitude"] >= lat_min) & (ds["latitude"] <= lat_max), drop=True
        )
    except ValueError:
        ds = ds.sel(longitude=slice(lon_min, lon_max))
        ds = ds.sel(latitude=slice(lat_min, lat_max))

    x_axis = pyinterp.Axis(ds["longitude"][:].values, is_circle=is_circle)
    y_axis = pyinterp.Axis(ds["latitude"][:].values)
    z_axis = pyinterp.TemporalAxis(ds["time"][:].values)

    var = ds[variable][:]
    var = var.transpose("longitude", "latitude", "time")

    # The undefined values must be set to nan.
    try:
        var[var.mask] = float("nan")
    except AttributeError:
        pass

    grid = pyinterp.Grid3D(x_axis, y_axis, z_axis, var.data)

    del ds

    return x_axis, y_axis, z_axis, grid
