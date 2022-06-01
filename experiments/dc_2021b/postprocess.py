import sys, os
from pyprojroot import here
root = here(project_files=[".root"])
sys.path.append(str(root))

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from ml4ssh._src.utils import temporal_subset, spatial_subset, create_spatiotemporal_grid


def add_postprocess_args(parser):
    # longitude subset
    parser.add_argument('--eval-lon-min', type=float, default=295.0)
    parser.add_argument('--eval-lon-max', type=float, default=305.0)
    parser.add_argument('--eval-dlon', type=float, default=0.2)
    
    # latitude subset
    parser.add_argument('--eval-lat-min', type=float, default=33.0)
    parser.add_argument('--eval-lat-max', type=float, default=43.0)
    parser.add_argument('--eval-dlat', type=float, default=0.2)
    
    # temporal subset
    parser.add_argument('--eval-time-min', type=str, default="2017-01-01")
    parser.add_argument('--eval-time-max', type=str, default="2017-12-31")
    parser.add_argument('--eval-dtime', type=str, default="1_D")
    
    # OI params
    parser.add_argument('--eval-lon-buffer', type=float, default=2.0)
    parser.add_argument('--eval-lat-buffer', type=float, default=2.0)
    parser.add_argument('--eval-time-buffer', type=float, default=7.0)

    return parser


def postprocess_data(df, args):

    # convert to xarray
    ds_oi = df.reset_index().set_index(["latitude", "longitude", "time"]).to_xarray()
    
    # load correction dataset
    ref_dir = Path(args.ref_data_dir).joinpath("mdt.nc")
    ds_correct = xr.open_dataset(ref_dir)
    
    # rename values
    ds_correct = ds_correct.rename({"lat": "latitude", "lon": "longitude"})
    
    # interpolate the points to evaluation grid
    ds_correct = ds_correct.interp(longitude=ds_oi.longitude, latitude=ds_oi.latitude)
    

    # add correction
    ds_oi["ssh"] = ds_oi["pred"] + ds_correct["mdt"]

    return ds_oi


def generate_eval_data(args):

    # create spatiotemporal grid
    lon_coords, lat_coords, time_coords = create_spatiotemporal_grid(
        lon_min=args.eval_lon_min,
        lon_max=args.eval_lon_max,
        lon_dx=args.eval_dlon if not args.smoke_test else 2,
        lat_min=args.eval_lat_min,
        lat_max=args.eval_lat_max,
        lat_dy=args.eval_dlat if not args.smoke_test else 2,
        time_min=np.datetime64(args.eval_time_min),
        time_max=np.datetime64(args.eval_time_max),
        time_dt=np.timedelta64(*args.eval_dtime.split("_")) if not args.smoke_test else np.timedelta64(5, "D"),
    )

    # temporal subset
    df_grid = pd.DataFrame({
        "longitude": lon_coords,
        "latitude": lat_coords,
        "time": time_coords,
    })
    
    # add vtime 
    # NOTE: THIS IS THE GLOBAL MINIMUM TIME FOR THE DATASET
    dtime = np.timedelta64(*args.eval_dtime.split("_"))
    df_grid["vtime"] = (df_grid['time'].values - np.datetime64("2016-12-01")) / dtime
    
    # add column attributes
    df_grid.attrs["input_cols"] = ["longitude", "latitude", "vtime"]
    df_grid.attrs["output_cols"] = ["sla_unfiltered"]

    
    return df_grid