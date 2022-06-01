import sys, os
from pyprojroot import here
root = here(project_files=[".root"])
sys.path.append(str(root))

from typing import NamedTuple
from pathlib import Path
import numpy as np
import xarray as xr
from ml4ssh._src.utils import get_gridded_data, create_grids
from ml4ssh._src.interp import interp_on_alongtrack
from ml4ssh._src.psd import select_track_segments, find_wavelength_crossing, compute_psd_scores
from ml4ssh._src.stats import calculate_nrmse, compute_ts_stats

def add_eval_args(parser):
    parser.add_argument("--eval-batch-size", type=int, default=10_000)

    # binning for the along track
    parser.add_argument("--eval-bin-lat-step", type=float, default=1.0)
    parser.add_argument("--eval-bin-lon-step", type=float, default=1.0)
    parser.add_argument("--eval-bin-time-step", type=str, default="1D")
    parser.add_argument("--eval-min-obs", type=int, default=10)

    # power spectrum
    parser.add_argument("--eval-psd-delta-t", type=float, default=0.9434)
    parser.add_argument("--eval-psd-velocity", type=float, default=6.77)
    parser.add_argument("--eval-psd-jitter", type=float, default=1e-4)

    return parser


def get_rmse_metrics(ds_oi, args):

    test_data_dir = Path(args.test_data_dir).joinpath("dt_gulfstream_c2_phy_l3_20161201-20180131_285-315_23-53.nc")
    ds_alongtrack = xr.open_dataset(test_data_dir)
    ds_alongtrack = ds_alongtrack.sortby("time")

    # Get All Along Tracks
    tracks = interp_on_alongtrack(
        ds_oi,
        ds_alongtrack,
        lon_min=args.eval_lon_min,
        lon_max=args.eval_lon_max,
        lat_max=args.eval_lat_max,
        lat_min=args.eval_lat_min,
        time_min=np.datetime64(args.eval_time_min),
        time_max=np.datetime64(args.eval_time_max),
        is_circle=True
    )

    time_alongtrack = tracks[0]
    lat_alongtrack = tracks[1]
    lon_alongtrack = tracks[2]
    ssh_alongtrack = tracks[3]
    ssh_map_interp = tracks[4]

    # =====================
    # STATISTICAL METRICS
    # ====================

    # calculate the (normalized) RMSE
    ts_stats = calculate_nrmse(
        ssh_alongtrack, 
        ssh_map_interp, 
        time_alongtrack, 
        dt_freq=args.eval_bin_time_step, 
        min_obs=args.eval_min_obs
    )

    # ==================
    # PSD
    # =================


    return RMSEStats(ts_stats[0], ts_stats[1], ts_stats[2], ts_stats[3])


def get_psd_metrics(ds_oi, args):

    test_data_dir = Path(args.test_data_dir).joinpath("dt_gulfstream_c2_phy_l3_20161201-20180131_285-315_23-53.nc")
    ds_alongtrack = xr.open_dataset(test_data_dir)
    ds_alongtrack = ds_alongtrack.sortby("time")

    # Get All Along Tracks
    tracks = interp_on_alongtrack(
        ds_oi,
        ds_alongtrack,
        lon_min=args.eval_lon_min,
        lon_max=args.eval_lon_max,
        lat_max=args.eval_lat_max,
        lat_min=args.eval_lat_min,
        time_min=np.datetime64(args.eval_time_min),
        time_max=np.datetime64(args.eval_time_max),
        is_circle=True
    )

    time_alongtrack = tracks[0]
    lat_alongtrack = tracks[1]
    lon_alongtrack = tracks[2]
    ssh_alongtrack = tracks[3]
    ssh_map_interp = tracks[4]

    # =====================
    # PSD METRICS
    # ====================

    delta_x = args.eval_psd_velocity * args.eval_psd_delta_t

    # compute along track segments
    tracks = select_track_segments(
        time_alongtrack,
        lat_alongtrack,
        lon_alongtrack,
        ssh_alongtrack,
        ssh_map_interp,
    )

    # compute scores
    wavenumber, psd_ref, psd_study, psd_diff, resolved_scale = compute_psd_scores(
        ssh_true=tracks[2],
        ssh_pred=tracks[3],
        delta_x=delta_x,
        npt=tracks[-1],
        scaling="density", 
        noverlap=0
    )


    return PSDStats(wavenumber=wavenumber, psd_ref=psd_ref, psd_study=psd_study, psd_diff=psd_diff, resolved_scale=resolved_scale)

class RMSEStats(NamedTuple):
    rmse_mean : float
    rmse_std : float
    nrmse_mean : float
    nrmse_std : float
    
    def __str__(self):
        return (f"RMSE (Mean): {self.rmse_mean:.3f}"
              f"\nRMSE (Std): {self.rmse_std:.3f}"
              f"\nNRMSE (Mean): {self.nrmse_mean:.3f}"
              f"\nNRMSE (Std): {self.nrmse_std:.3f}")

class PSDStats(NamedTuple):
    wavenumber : float
    resolved_scale : float
    psd_ref : np.ndarray
    psd_study : np.ndarray
    psd_diff : np.ndarray
    
    def __str__(self):
        return (f"Resolved Scale: {self.resolved_scale:.3f} (km)")

    
