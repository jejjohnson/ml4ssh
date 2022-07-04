import pandas as pd
import numpy as np


def array_2_da(coords, data, name="full_pred"):
    return pd.DataFrame(np.hstack([coords, data]), columns=["Nx", "Ny", "Nt", name]).set_index(
        ["Nx", "Ny", "Nt"]).to_xarray()


def create_mask(coords, name="train", factor=1):
    return array_2_da(coords, factor * np.ones((coords.shape[0], 1)), name=name)

def df_2_xr(coords: pd.DataFrame):
    
    return coords.reset_index().set_index(["latitude", "longitude", "time"]).to_xarray()
    