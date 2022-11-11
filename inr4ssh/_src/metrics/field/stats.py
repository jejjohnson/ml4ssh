def rmse_spacetime(da, da_ref):

    rmse_xyt = ((da - da_ref) ** 2).mean()

    return rmse_xyt


def nrmse_spacetime(da, da_ref):
    rmse_xyt = rmse_spacetime(da, da_ref)
    std = (da_ref**2).mean()
    nrmse_xyt = 1.0 - (rmse_xyt / std)
    return nrmse_xyt


def rmse_time(da, da_ref):

    rmse_t = ((da - da_ref) ** 2).mean(dim=["longitude", "latitude"]) ** 0.5
    return rmse_t


def nrmse_time(da, da_ref):

    rmse_t = rmse_time(da, da_ref)

    std = (da_ref**2).mean(dim=["longitude", "latitude"]) ** 0.5

    nrmse_t = 1.0 - (rmse_t / std)

    return nrmse_t


def rmse_space(da, da_ref):

    rmse_xy = ((da - da_ref) ** 2).mean(dim=["time"]) ** 0.5
    return rmse_xy


def nrmse_space(da, da_ref):

    rmse_xy = rmse_space(da, da_ref)

    std = (da_ref**2).mean(dim=["time"]) ** 0.5

    nrmse_xy = 1.0 - (rmse_xy / std)

    return nrmse_xy
