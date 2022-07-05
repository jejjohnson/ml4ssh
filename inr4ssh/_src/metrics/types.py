from typing import NamedTuple, List
import numpy as np

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


class AlongTrackData(NamedTuple):
    time : np.ndarray
    lat : np.ndarray
    lon : np.ndarray
    ssh_alongtrack : np.ndarray
    ssh_map : np.ndarray
    
    
class ListAlongTrackSegments(NamedTuple):
    lon : List[np.ndarray]
    lat : List[np.ndarray]
    ssh_alongtrack : List[np.ndarray]
    ssh_map : List[np.ndarray]
    npt : int