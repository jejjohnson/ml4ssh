import numpy as np
import xarray as xr


# def subset_spatial(ds, config):
#     time_min = config.time_min
#     time_max = config.time_max

#     ds = ds.sel(
#         time=slice(time_min - np.timedelta64(int(2 * time_buffer), time_buffer_order),
#                    time_max + np.timedelta64(int(2 * time_buffer), time_buffer_order)),
#         drop=True
#     )
#     return ds


def temporal_subset(
    ds, time_min, time_max, time_buffer: int = 7.0, time_buffer_order: str = "D"
):
    ds = ds.sel(
        time=slice(
            time_min - np.timedelta64(int(2 * time_buffer), time_buffer_order),
            time_max + np.timedelta64(int(2 * time_buffer), time_buffer_order),
        ),
        drop=True,
    )
    return ds


def spatial_subset(ds, lon_min, lon_max, lon_buffer, lat_min, lat_max, lat_buffer):

    ds = ds.where(
        (ds["longitude"] >= lon_min - lon_buffer)
        & (ds["longitude"] <= lon_max + lon_buffer)
        & (ds["latitude"] >= lat_min - lat_buffer)
        & (ds["latitude"] <= lat_max + lat_buffer),
        drop=True,
    )

    return ds
