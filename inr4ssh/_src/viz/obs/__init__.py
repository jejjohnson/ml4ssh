import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.reset_defaults()
sns.set_context(context="talk", font_scale=0.7)


def plot_obs_demo(
    ds: xr.Dataset,
    central_date: np.datetime64,
    delta_t: np.timedelta64,
    variable: str,
    verbose: bool = True,
):

    tmin = central_date - delta_t
    tmax = central_date + delta_t

    ds = ds.sel(time=slice(tmin, tmax))

    ds = ds.drop_duplicates(dim="time")
    if verbose:
        print(ds[variable].shape)

    vmin = ds[variable].min().values
    vmax = ds[variable].max().values

    fig, ax = plt.subplots()
    pts = ax.scatter(
        ds.longitude.values,
        ds.latitude.values,
        c=ds[variable].values,
        s=20,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("Longitude", fontweight="bold")
    ax.set_ylabel("Latitude", fontweight="bold")

    plt.colorbar(pts, extend="both")

    return fig, ax
