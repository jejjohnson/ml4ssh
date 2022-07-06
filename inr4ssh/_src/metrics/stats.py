import numpy as np
import xarray as xr
from .types import RMSEStats, PSDStats


def calculate_nrmse(true, pred, time_vector: np.ndarray, dt_freq: str="1D", min_obs: int=10):
    """
    Parameters:
    -----------
    true : np.ndarray,
        the predictions
    pred : np.ndarray
        the predictions for the vector
    time : np.ndarray
        the time stampes
    dt_freq : str, default="1D"
        the binning when computing the statistics
    min_obs : int, default=10
        the minimum number of values in the binning process
    
    Returns:
    --------
    rmse_mean : float,
        the mean rmse for the predictions along the same track
    rmse_std : float,
        the stdev rmse for the predictions along the same track
    nrmse_mean : float,
        the normalized mean rmse for the predictions along the same track
    nrmse_std : float,
        the normalized stdev rmse for the predictions along the same track
    """
    
    # DATA RMSE
    da_diff = xr.DataArray(true - pred, coords=[time_vector], dims="time")
    
    # diff rmse
    rmse_pred = np.sqrt(np.square(da_diff).resample(time=dt_freq).mean())
    
    # mask score if num obs < min_
    vcount = da_diff.resample(time=dt_freq).count().values
    
    rmse_pred_masked = np.ma.masked_where(vcount < min_obs, rmse_pred)
    
    # calculate mean, std for all values
    rmse_mean = np.ma.mean(np.ma.masked_invalid(rmse_pred_masked))
    rmse_std = np.ma.std(np.ma.masked_invalid(rmse_pred_masked))
    
    # NORMALIZED SCORE
    # convert to xarray
    da_true = xr.DataArray(true, coords=[time_vector], dims="time")
    
    # true rmse
    rmse_true = np.sqrt(np.square(da_true).resample(time=dt_freq).mean())
    
    # normalize the score
    nrmse_score = 1. - rmse_pred / rmse_true
    
    vcount = da_true.resample(time=dt_freq).count().values
    nrmse_score_masked = np.ma.masked_where(vcount < min_obs, nrmse_score)
    
    nrmse_mean = np.ma.mean(np.ma.masked_invalid(nrmse_score_masked))
    nrmse_std = np.ma.std(np.ma.masked_invalid(nrmse_score_masked))
    
    return RMSEStats(
        rmse_mean, rmse_std, nrmse_mean, nrmse_std
    )


def compute_ts_stats(data: np.ndarray, time_vector: np.ndarray, dt_freq: str="1D", stat: str="diff"):
    
    # convert to xarray
    da = xr.DataArray(data, coords=[time_vector], dims="time")
    
    # resample
    da_rs = da.resample(time=dt_freq)
    
    # compute statistics
    ds = xr.Dataset(
        {
            "mean": (("time"), da_rs.mean().values),
            "min": (("time"), da_rs.min().values),
            "max": (("time"), da_rs.max().values),
            "count": (("time"), da_rs.count().values),
            "variance": (("time"), da_rs.var().values),
            "median": (("time"), da_rs.median().values),
            "mae": (("time"), np.abs(da).resample(time=dt_freq).mean()),
            "mse": (("time"), np.square(da).resample(time=dt_freq).mean()),
            "rmse": (("time"), np.sqrt(np.square(da).resample(time=dt_freq).mean())),
                       
        },
        {"time": da_rs.mean()["time"]},
    )
    
    return ds


def calculate_rmse_elementwise(y_true, y_pred):
    
    # calculate difference
    diff = y_true.squeeze() - y_pred.squeeze()
    
    # calculate mean and variance
    rmse_pred = np.sqrt(np.square(diff))
    rmse_mean = rmse_pred.mean()
    rmse_std = rmse_pred.std()
    
    return rmse_mean, rmse_std

def calculate_nrmse_elementwise(y_true, y_pred, normalization: str="custom", mask_invalid: bool=True):
    
    # calculate difference
    diff = y_true.squeeze() - y_pred.squeeze()
    
    # calculate prediction rmse
    rmse_pred = np.sqrt(np.square(diff))
    rmse_true = rmse_normalization(y_true, normalization=normalization)
    
    nrmse_score = 1. - rmse_pred / rmse_true
    
    if mask_invalid:
        nrmse_score = np.ma.masked_invalid(nrmse_score)
        
        nrmse_mean = np.ma.mean(nrmse_score)
        nrmse_std = np.ma.std(nrmse_score)
        
    else:
        nrmse_mean = nrmse_score.mean()
        nrmse_std = nrmse_score.std()
    
    return nrmse_mean, nrmse_std

def rmse_normalization(y_true, normalization: str="custom"):
    
    if normalization == "custom":
        return np.sqrt(np.square(y_true).mean()) 
    elif normalization == "std":
        return np.std(y_true)
    elif normalization == "mean":
        return np.mean(y_true)
    elif normalization == "minmax":
        return np.max(y_true) - np.min(y_true)
    elif normalization == "iqr":
        return np.subtract(*np.percentile(y_true, [75, 25]))
    else:
        raise ValueError(f"Unrecognized normalization: {normalization}") 