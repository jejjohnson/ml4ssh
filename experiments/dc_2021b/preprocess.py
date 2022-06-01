import sys, os
from pyprojroot import here
root = here(project_files=[".root"])
sys.path.append(str(root))

import numpy as np
from ml4ssh._src.utils import temporal_subset, spatial_subset

def add_preprocess_args(parser):
    # longitude subset
    parser.add_argument('--lon-min', type=float, default=285.0)
    parser.add_argument('--lon-max', type=float, default=315.0)
    parser.add_argument('--dlon', type=float, default=0.2)
    
    # latitude subset
    parser.add_argument('--lat-min', type=float, default=23.0)
    parser.add_argument('--lat-max', type=float, default=53.0)
    parser.add_argument('--dlat', type=float, default=0.2)
    
    # temporal subset
    parser.add_argument('--time-min', type=str, default="2016-12-01")
    parser.add_argument('--time-max', type=str, default="2018-01-31")
    parser.add_argument('--dtime', type=str, default="1_D")
    
    # Buffer Params
    parser.add_argument('--lon-buffer', type=float, default=1.0)
    parser.add_argument('--lat-buffer', type=float, default=1.0)
    parser.add_argument('--time-buffer', type=float, default=7.0)

    return parser


def preprocess_data(ds_obs, args):
    
    time_min = np.datetime64(args.time_min)
    time_max = np.datetime64(args.time_max)
    dtime = np.timedelta64(*args.dtime.split("_"))
    # temporal subset
    ds_obs = temporal_subset(
        ds_obs,
        time_min=time_min,
        time_max=time_max,
        time_buffer=args.time_buffer)


    # convert to dataframe
    data = ds_obs.to_dataframe().reset_index().dropna()
    
    # add vtime 
    # NOTE: THIS IS THE GLOBAL MINIMUM TIME FOR THE DATASET
    data["vtime"] = (data['time'].values - np.datetime64("2016-12-01")) / dtime
    
    # add column attributes
    data.attrs["input_cols"] = ["longitude", "latitude", "vtime"]
    data.attrs["output_cols"] = ["sla_unfiltered"]
    
    return data