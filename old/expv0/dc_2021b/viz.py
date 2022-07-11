import sys, os
from pyprojroot import here
root = here(project_files=[".root"])
sys.path.append(str(root))

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from inr4ssh._src.utils import temporal_subset, spatial_subset, create_spatiotemporal_grid



def add_viz_args(parser):
    # longitude subset
    parser.add_argument('--viz-lon-min', type=float, default=295.0)
    parser.add_argument('--viz-lon-max', type=float, default=305.0)
    parser.add_argument('--viz-dlon', type=float, default=0.1)
    
    # latitude subset
    parser.add_argument('--viz-lat-min', type=float, default=33.0)
    parser.add_argument('--viz-lat-max', type=float, default=43.0)
    parser.add_argument('--viz-dlat', type=float, default=0.1)
    
    # temporal subset
    parser.add_argument('--viz-time-min', type=str, default="2017-01-01")
    parser.add_argument('--viz-time-max', type=str, default="2017-12-31")
    parser.add_argument('--viz-dtime', type=str, default="1_D")
    
    # OI params
    parser.add_argument('--viz-lon-buffer', type=float, default=1.0)
    parser.add_argument('--viz-lat-buffer', type=float, default=1.0)
    parser.add_argument('--viz-time-buffer', type=float, default=7.0)

    return parser


def generate_plot_data(args):

    # create spatiotemporal grid
    lon_coords, lat_coords, time_coords = create_spatiotemporal_grid(
        lon_min=args.viz_lon_min,
        lon_max=args.viz_lon_max,
        lon_dx=args.viz_dlon if not args.smoke_test else 2,
        lat_min=args.viz_lat_min,
        lat_max=args.viz_lat_max,
        lat_dy=args.viz_dlat if not args.smoke_test else 2,
        time_min=np.datetime64(args.viz_time_min),
        time_max=np.datetime64(args.viz_time_max),
        time_dt=np.timedelta64(*args.viz_dtime.split("_")) if not args.smoke_test else np.timedelta64(5, "D"),
    )

    # temporal subset
    df_grid = pd.DataFrame({
        "longitude": lon_coords,
        "latitude": lat_coords,
        "time": time_coords,
    })
    
    # add vtime 
    # NOTE: THIS IS THE GLOBAL MINIMUM TIME FOR THE DATASET
    dtime = np.timedelta64(*args.viz_dtime.split("_"))
    df_grid["vtime"] = (df_grid['time'].values - np.datetime64("2016-12-01")) / dtime
    
    # add column attributes
    df_grid.attrs["input_cols"] = ["longitude", "latitude", "vtime"]
    df_grid.attrs["output_cols"] = ["sla_unfiltered"]

    
    return df_grid