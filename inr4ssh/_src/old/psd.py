from scipy.interpolate import interp1d
from scipy import signal
import numpy as np
from typing import List
import warnings

def find_wavelength_crossing(
    psd_ref: np.ndarray, 
    psd_diff: np.ndarray, 
    wavenumber: np.ndarray, 
    query: float=0.5, 
    jitter: float=1e-15
) -> float:
    
    wavenumber += jitter
    
    y = 1. / wavenumber
    
    x = (1. - psd_diff / psd_ref)
    
    f = interp1d(x, y)
    
    try:
        ynew = f(query)
    except ValueError:
        warnings.warn(f"The interpolated value is outside the range.")
        ynew = 1e10
        
    return ynew


def compute_psd_scores(
    ssh_true: List[np.ndarray], 
    ssh_pred: List[np.ndarray],
    delta_x: float,
    npt: int,
    **kwargs
):
    
    
    # power spectrum density reference field
    wavenumber, global_psd_ref = signal.welch(
        np.asarray(ssh_true).flatten(),
        fs=1.0 / delta_x,
        nperseg=npt,
        **kwargs
    )
    
    # power spectrum density study field
    _, global_psd_study = signal.welch(
        np.asarray(ssh_pred).flatten(),
        fs=1.0 / delta_x,
        nperseg=npt,
        **kwargs
    )
    
    # power spectrum density difference field
    _, global_psd_diff = signal.welch(
        np.asarray(ssh_pred).flatten() - np.asarray(ssh_true).flatten(),
        fs=1.0 / delta_x,
        nperseg=npt,
        scaling="density",
        noverlap=0
    )
    resolved_scale = find_wavelength_crossing(
        global_psd_ref,
        global_psd_diff,
        wavenumber
    )
    return wavenumber, global_psd_ref, global_psd_study, global_psd_diff, resolved_scale


def select_track_segments(
    time_alongtrack: np.ndarray,
    lat_alongtrack: np.ndarray,
    lon_alongtrack: np.ndarray,
    ssh_alongtrack: np.ndarray,
    ssh_map_interp: np.ndarray,
    delta_x: float=None,
    delta_t: float=0.9434,
    velocity: float=6.77,
    length_scale: float=1_000,
    segment_overlapping: float=0.25,
):
    """
    Parameters:
    -----------
    delta_t : float, default=0.9434
        the number of seconds
    velocity : float, default=6.77
        the velocity (km/s)
    length_scale: float, default=1_000
        the segment length cale in km
    segment_overlapping: float=0.25
        the amount of segment overlapping allowed
    """
    if delta_x is None:
        delta_x = velocity * delta_t
    # max delta t of 4 seconds to cut tracks
    max_delta_t_gap = 4 * np.timedelta64(1, "s")
    # get number of points to consider for resolution = lengthscale in km
    delta_t_jd = delta_t / (3600 * 24)
    npt = int(length_scale / delta_x) 

    # cut tracks when diff time longer than 4 delta t
    indx = np.where((np.diff(time_alongtrack) > max_delta_t_gap))[0]
    track_segment_length = np.insert(np.diff(indx), [0], indx[0])
    
    list_lat_segment = []
    list_lon_segment = []
    list_ssh_alongtrack_segment = []
    list_ssh_map_interp_segment = []


    # Long track >= npt
    selected_track_segment = np.where(track_segment_length >= npt)[0]

    if selected_track_segment.size > 0:

        for track in selected_track_segment:

            if track-1 >= 0:
                index_start_selected_track = indx[track-1]
                index_end_selected_track = indx[track]
            else:
                index_start_selected_track = 0
                index_end_selected_track = indx[track]

            start_point = index_start_selected_track
            end_point = index_end_selected_track

            for sub_segment_point in range(start_point, end_point - npt, int(npt*segment_overlapping)):

                # Near Greenwhich case
                if ((lon_alongtrack[sub_segment_point + npt - 1] < 50.)
                    and (lon_alongtrack[sub_segment_point] > 320.)) \
                        or ((lon_alongtrack[sub_segment_point + npt - 1] > 320.)
                            and (lon_alongtrack[sub_segment_point] < 50.)):

                    tmp_lon = np.where(lon_alongtrack[sub_segment_point:sub_segment_point + npt] > 180,
                                       lon_alongtrack[sub_segment_point:sub_segment_point + npt] - 360,
                                       lon_alongtrack[sub_segment_point:sub_segment_point + npt])
                    mean_lon_sub_segment = np.median(tmp_lon)

                    if mean_lon_sub_segment < 0:
                        mean_lon_sub_segment = mean_lon_sub_segment + 360.
                else:

                    mean_lon_sub_segment = np.median(lon_alongtrack[sub_segment_point:sub_segment_point + npt])

                mean_lat_sub_segment = np.median(lat_alongtrack[sub_segment_point:sub_segment_point + npt])

                ssh_alongtrack_segment = np.ma.masked_invalid(ssh_alongtrack[sub_segment_point:sub_segment_point + npt])

                ssh_map_interp_segment = []
                ssh_map_interp_segment = np.ma.masked_invalid(ssh_map_interp[sub_segment_point:sub_segment_point + npt])
                if np.ma.is_masked(ssh_map_interp_segment):
                    ssh_alongtrack_segment = np.ma.compressed(np.ma.masked_where(np.ma.is_masked(ssh_map_interp_segment), ssh_alongtrack_segment))
                    ssh_map_interp_segment = np.ma.compressed(ssh_map_interp_segment)

                if ssh_alongtrack_segment.size > 0:
                    list_ssh_alongtrack_segment.append(ssh_alongtrack_segment)
                    list_lon_segment.append(mean_lon_sub_segment)
                    list_lat_segment.append(mean_lat_sub_segment)
                    list_ssh_map_interp_segment.append(ssh_map_interp_segment)
    
    return list_lon_segment, list_lat_segment, list_ssh_alongtrack_segment, list_ssh_map_interp_segment, npt 