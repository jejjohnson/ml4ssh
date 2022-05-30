import numpy as np
import xarray as xr

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
    
    return rmse_mean, rmse_std, nrmse_mean, nrmse_std


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