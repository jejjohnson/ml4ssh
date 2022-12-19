from typing import Tuple
from einops import repeat, rearrange
import numpy as np
import xarray as xr
from inr4ssh._src.operators.boundaries import bc_conditions


def coriolis_parameter(
    lat: np.ndarray, omega: float = 7.2921159e-5, min_val: float = 1.0e-8
) -> np.ndarray:
    """The Coriolis parameter
    Args:
    -----
    lat (np.ndarray): the latitude (degrees)
    omega (float): the angular velocity of the Earth [rad/s]
        (default=7.2921159e-5)
    min_val (float): the minimum value for the coriolis param near
        the equation (defualt=1.0e-8)

    Returns:
    --------
    fc (np.ndarray): the Coriolis parameter.
    """
    # angular velocity of the Earth [rad/s]

    # calculate constant
    fc = 2 * omega * np.sin(np.deg2rad(lat))

    # avoid zero near equator, bound fc by min val as
    fc = np.sign(fc) * np.maximum(np.abs(fc), min_val)

    return fc


def lonlat2dxdy(
    lon: np.ndarray, lat: np.ndarray, deg2m: float = 111.0e3, bc: str = "neumann"
) -> Tuple[np.ndarray, np.ndarray]:
    dlon = np.gradient(lon)
    dlat = np.gradient(lat)

    dx = np.sqrt(
        (dlon[1] * deg2m * np.cos(np.deg2rad(lat))) ** 2 + (dlat[1] * deg2m) ** 2
    )
    dy = np.sqrt(
        (dlon[0] * deg2m * np.cos(np.deg2rad(lat))) ** 2 + (dlat[0] * deg2m) ** 2
    )

    # pad the boundaries
    dx = bc_conditions(dx[1:-1, 1:-1, None], bc=bc).squeeze()
    dy = bc_conditions(dy[1:-1, 1:-1, None], bc=bc).squeeze()

    return dx, dy


def velocity(data, dx, dy, bc: str = "neumann", loc: str = "center"):
    """Computes the velocity given a field

    u = -d psi/dy
    v = -d psi/dx

    Args:
    ----
    data (np.ndarray): the data array for the integration
        shape=(Dx,Dy,...)
    dx (np.ndarray): the derivatives for the x component
        shape=(Dx,Dy)
    dy (np.ndarray): the derivatives for the y components
        shape=(Dx,Dy)

    Returns:
    --------
    u (np.ndarray): the u-component of the data array
        shape=(Dx,Dy,...)
    v (np.ndarray): the v-component of the data array
        shape=(Dx,Dy,...)
    """

    data = bc_conditions(data, bc=bc)

    # add a dimension for broadcasting
    if data.ndim > 2:
        dx = repeat(dx, "dlon dlat -> dlon dlat 1")
        dy = repeat(dy, "dlon dlat -> dlon dlat 1")

    if loc.lower() == "center":
        u = (data[2:, 1:-1] - data[:-2, 1:-1]) / (2 * dy)
        v = (data[1:-1, 2:] - data[1:-1, :-2]) / (2 * dx)
    elif loc.lower() == "faces":
        u = (data[1:, 1:-1] - data[:-1, 1:-1]) / (2 * dy)
        v = (data[1:-1, 1:] - data[1:-1, :-1]) / (2 * dx)
    else:
        raise ValueError(f"Unrecognized location: {loc}")

    return u, v


def laplacian_2d(data, dx, dy, bc: str = "neumann"):

    data = bc_conditions(data, bc=bc)

    # add a dimension for broadcasting
    if data.ndim > 2:
        dx = repeat(dx, "dlon dlat -> dlon dlat 1")
        dy = repeat(dy, "dlon dlat -> dlon dlat 1")

    return (data[2:, 1:-1] + data[:-2, 1:-1] - 2 * data[1:-1, 1:-1]) / dy**2 + (
        data[1:-1, 2:] + data[1:-1, :-2] - 2 * data[1:-1, 1:-1]
    ) / dx**2


def kinetic_energy(u, v):
    return np.sqrt(v**2 + u**2)


def enstropy(pv):
    return 0.5 * pv**2


def ssh_2_uv(
    data: np.ndarray,
    longitude: np.ndarray,
    latitude: np.ndarray,
    g: float = 9.81,
    deg2m: float = 111.0e3,
    bc: str = "neumann",
    loc: str = "center",
):

    if longitude.ndim == 1:
        # create meshgrid from lon,lat coordinates
        longitude, latitude = np.meshgrid(longitude, latitude, indexing="ij")

    # calculate the coriolis parameter
    f = coriolis_parameter(latitude)

    # get gradients for longitude/latitude coords
    dlon, dlat = lonlat2dxdy(longitude, latitude, deg2m=deg2m, bc=bc)

    # add dimension (broadcast temporal dim)
    f = repeat(f, "lon lat -> lon lat 1")

    # calculate velocity
    u, v = (g / f) * velocity(data, dlon, dlat, bc=bc, loc=loc)

    return u, v


def ssh_2_rv(
    data: np.ndarray,
    longitude: np.ndarray,
    latitude: np.ndarray,
    g: float = 9.81,
    deg2m: float = 111.0e3,
    bc: str = "neumann",
    loc: str = "center",
):

    if longitude.ndim == 1:
        # create meshgrid from lon,lat coordinates
        longitude, latitude = np.meshgrid(longitude, latitude, indexing="ij")

    # calculate the coriolis parameter
    f = coriolis_parameter(latitude)

    # get gradients for longitude/latitude coords
    dlon, dlat = lonlat2dxdy(longitude, latitude, deg2m=deg2m, bc=bc)

    # add dimension (broadcast temporal dim)
    f = repeat(f, "lon lat -> lon lat 1")

    # calculate relative vorticity
    data = (g / f) * laplacian_2d(data, dlon, dlat, bc=bc)

    return data


def ssh2rv_da_2dt(
    da: xr.DataArray, g: float = 9.81, bc: str = "neumann", loc: str = "center"
):

    da = da.transpose("longitude", "latitude", "time")
    rv = ssh_2_rv(
        da.values, da.longitude.values, da.latitude.values, g=g, bc=bc, loc=loc
    )

    # re-permute axis
    rv = rearrange(rv, "Lon Lat Time -> Time Lat Lon")

    return rv


def ssh2rv_ds_2dt(
    ds: xr.Dataset,
    variable: str = "ssh",
    g: float = 9.81,
    bc: str = "neumann",
    loc: str = "center",
):

    # calculate the uv components
    rv = ssh2rv_da_2dt(da=ds[variable], g=g, bc=bc, loc=loc)

    # create variables within dataset
    ds["rv"] = (("time", "latitude", "longitude"), rv)

    return ds


def ssh2uv_da_2dt(
    da: xr.DataArray, g: float = 9.81, bc: str = "neumann", loc: str = "center"
):

    da = da.transpose("longitude", "latitude", "time")
    u, v = ssh_2_uv(
        da.values, da.longitude.values, da.latitude.values, g=g, bc=bc, loc=loc
    )

    # re-permute axis
    u = rearrange(u, "Lon Lat Time -> Time Lat Lon")
    v = rearrange(v, "Lon Lat Time -> Time Lat Lon")

    return u, v


def ssh2uv_ds_2dt(
    ds: xr.Dataset,
    variable: str = "ssh",
    g: float = 9.81,
    bc: str = "neumann",
    loc: str = "center",
):

    # calculate the uv components
    u, v = ssh2uv_da_2dt(da=ds[variable], g=g, bc=bc, loc=loc)

    # create variables within dataset
    ds["u"] = (("time", "latitude", "longitude"), u)
    ds["v"] = (("time", "latitude", "longitude"), v)

    return ds
