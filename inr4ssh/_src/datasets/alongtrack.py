import torch
import pandas as pd
import xarray as xr
import numpy as np


class AlongTrackDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        ds,
        spatial_columns,
        temporal_columns,
        output_columns=None,
        transform=None,
    ):
        if isinstance(ds, xr.Dataset):
            df = ds.to_dataframe().reset_index().dropna()
        else:
            df = ds
        self.x = df[spatial_columns].values
        self.t = df[temporal_columns].values
        if output_columns is not None:
            self.y = df[output_columns].values
        else:
            self.y = None
        self.transform = transform
        self.output_columns = output_columns
        self.spatial_columns = spatial_columns
        self.temporal_columns = temporal_columns

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):

        # get appropriate columns
        samples = {"spatial": self.x[item], "temporal": self.t[item]}
        if self.y is not None:
            samples["output"] = self.y[item]

        # do transform if necessary
        if self.transform:
            samples = self.transform(samples)

        return samples

    def create_predict_df(self, outputs: np.ndarray) -> pd.DataFrame:
        if self.y is not None:
            assert outputs.shape == self.y.shape
        df = pd.DataFrame()
        df[self.spatial_columns] = self.x
        df[self.temporal_columns] = self.t
        if self.y is not None:
            df[self.output_columns] = self.y
            names = list(map(lambda x: x + "_predict", self.output_columns))
        else:
            print(len(outputs.shape[1:]))
            names = ["predict"] * len(outputs.shape[1:])
        df[names] = outputs
        return df

    def create_predict_ds(self, outputs: np.ndarray) -> xr.Dataset:
        ds = (
            self.create_predict_df(outputs).set_index(self.temporal_columns).to_xarray()
        )
        return ds


class AlongTrackEvalDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        ds,
        spatial_columns,
        temporal_columns,
        output_columns,
        transform=None,
    ):
        if isinstance(ds, xr.Dataset):
            df = ds.to_dataframe().reset_index().dropna()
        else:
            df = ds
        self.x = df[spatial_columns].values
        self.t = df[temporal_columns].values
        self.transform = transform
        self.output_columns = output_columns
        self.spatial_columns = spatial_columns
        self.temporal_columns = temporal_columns

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):

        # get appropriate columns
        samples = {
            "spatial": self.x[item],
            "temporal": self.t[item],
            "output": self.y[item],
        }

        # do transform if necessary
        if self.transform:
            samples = self.transform(samples)

        return samples

    def create_predict_df(self, outputs: np.ndarray) -> pd.DataFrame:
        assert outputs.shape == self.y.shape
        df = pd.DataFrame()
        df[self.spatial_columns] = self.x
        df[self.temporal_columns] = self.t
        names = list(map(lambda x: x + "_predict", self.output_columns))
        df[names] = outputs
        return df

    def create_predict_ds(self, outputs: np.ndarray) -> xr.Dataset:
        ds = (
            self.create_predict_df(outputs).set_index(self.temporal_columns).to_xarray()
        )
        return ds
