import xarray as xr
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader
import torch
import numpy as np
import pandas as pd
from features import InputScalingTransform


class QGSimulation(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):

        # load data
        data = xr.open_dataset(self.config.data.data_dir, engine="netcdf4")

        data_df = data.to_dataframe().reset_index()

        # subset variables of interest
        x_df = data_df[["Nx", "Ny", "steps"]]
        y_df = data_df[["p"]]

        # get spatial/temporal min/max limits
        x_min = x_df.min(axis=0)
        x_max = x_df.max(axis=0)

        # create invertible transformation
        transform = InputScalingTransform(x_min.values, x_max.values)
        self.transform = transform

        # create prediction dataset (everything)
        predict_ds = TensorDataset(
            torch.FloatTensor(x_df.values), torch.FloatTensor(y_df.values)
        )

        # create train/val/test datasets
        n_datapoints = len(predict_ds)
        train_split = int(self.config.split.train_prct * n_datapoints)
        valid_split = n_datapoints - train_split

        # random split
        train_ds, valid_ds = torch.utils.data.random_split(
            predict_ds, (train_split, valid_split)
        )

        self.ds_train = train_ds
        self.ds_valid = valid_ds
        self.ds_predict = predict_ds

    def create_predictions_df(self):

        # load data
        data = xr.open_dataset(self.config.data.data_dir, engine="netcdf4")

        return data.to_dataframe().reset_index()

    def create_predictions_ds(self, predictions):

        # load data
        data = xr.open_dataset(self.config.data.data_dir, engine="netcdf4")

        data_df = data.to_dataframe().reset_index()

        # subset variables of interest
        col_coords = ["Nx", "Ny", "steps"]
        x_df = data_df[col_coords]
        y_df = data_df[["p"]]

        data = np.concatenate(
            [
                x_df.values,
                y_df.values,
                predictions.numpy(),
            ],
            axis=1,
        )

        data = (
            pd.DataFrame(data, columns=col_coords + ["true"] + ["pred"])
            .set_index(col_coords)
            .to_xarray()
        )
        return data

    def train_dataloader(self):
        return DataLoader(
            self.ds_train,
            batch_size=self.config.dl.batchsize_train,
            shuffle=True,
            num_workers=self.config.dl.num_workers,
            pin_memory=self.config.dl.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_valid,
            batch_size=self.config.dl.batchsize_val,
            shuffle=False,
            num_workers=self.config.dl.num_workers,
            pin_memory=self.config.dl.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.ds_predict,
            batch_size=self.config.dl.batchsize_test,
            shuffle=False,
            num_workers=self.config.dl.num_workers,
            pin_memory=self.config.dl.pin_memory,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.ds_predict,
            batch_size=self.config.dl.batchsize_predict,
            shuffle=False,
            num_workers=self.config.dl.num_workers,
            pin_memory=self.config.dl.pin_memory,
        )
