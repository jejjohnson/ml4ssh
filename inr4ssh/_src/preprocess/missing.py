from operator import mul
from functools import reduce
import numpy as np


def generate_random_missing_data(
    data: np.ndarray,
    missing_data_rate: float = 0.75,
    return_mask: bool = False,
    seed: int = 123,
) -> np.ndarray:
    rng = np.random.RandomState(seed)
    # get total dims of data
    dims = data.shape
    total_dims = reduce(mul, dims)
    
    if float(missing_data_rate) == 0.0:
        return data

    # subset data
    size_subset = int(missing_data_rate * total_dims)

    # subset integers
    idx = np.arange(0, total_dims)
    idx_rand = rng.choice(idx, replace=False, size=(size_subset,))

    # get a subset of indices
    data_nans = data.copy().ravel()
    data_nans[idx_rand] = np.nan

    if return_mask:
        # get mask of where there are nans
        data_nans = np.isnan(data_nans)

    data_nans = data_nans.reshape(dims)

    assert data_nans.shape == data.shape

    return data_nans


def generate_skipped_missing_data(
    data: np.ndarray,
    step: int=5,
    dim: int=0
):
    
    steps = np.arange(0, data.shape[dim], step=step)
    missing = np.zeros(data.shape, dtype=bool)

    if dim == 0:
        missing[steps, ...] = 1.0
    elif dim == 1:
        missing[:, steps, :] = 1.0

    elif dim == 2:
        missing[..., steps] = 1.0
    else:
        raise ValueError(f"Incompatible dims")

    data[~missing] = np.nan
    
    return data


