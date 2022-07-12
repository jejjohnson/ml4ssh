import pandas as pd
import pytorch_lightning as pl
from loguru import logger
from ..data.qg import load_qg_data
from ..features.temporal import MinMaxFixedScaler
from ..preprocess.obs import add_noise
from ..preprocess.missing import generate_random_missing_data, generate_skipped_missing_data
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


class QGSimulation(pl.LightningModule):
    def __init__(self,
                 data, preprocess,
                 traintest,
                 features,
                 dataloader,
                 eval):
        super().__init__()
        self.data = data
        self.preprocess = preprocess
        self.traintest = traintest
        self.features = features
        self.dataloader = dataloader
        self.eval = eval

    def setup(self, stage=None):
        logger.info("Getting data...")

        logger.info("loading data...")
        ds = load_qg_data(self.data.train_data_dir)

        data = self.get_train_data(ds["p"].values, self.preprocess)

        logger.info("converting to dataframe...")
        ds["obs"] = (ds[self.features.variable].dims, data)
        df = ds["obs"].to_dataframe()

        logger.info("getting feature scaler...")
        scaler = get_feature_scaler(self.features)

        X_test = df[["time", "Nx", "Ny"]]
        y_test = df[[self.features.variable]]

        df = df.dropna()

        xtrain, ytrain = df[["time", "Nx", "Ny"]], df[["p"]]

        xtrain, xvalid, ytrain, yvalid = train_test_split(
            xtrain, ytrain,
            train_size=self.traintest.train_size,
            random_state=self.traintest.seed_split
        )

        logger.info("Creating dataloaders...")
        self.ds_train = TensorDataset(
            torch.FloatTensor(xtrain),
            torch.FloatTensor(ytrain)
        )
        self.ds_valid = TensorDataset(
            torch.FloatTensor(xvalid),
            torch.FloatTensor(yvalid)
        )
        self.ds_test = TensorDataset(
            torch.FloatTensor(X_test),
            torch.FloatTensor(y_test)
        )

        self.dim_in = xtrain.shape[-1]
        self.dim_out = ytrain.shape[-1]


    @staticmethod
    def get_obs_data(data, traintest):

        logger.info("sampling spatiotemporal res...")
        data = generate_skipped_missing_data(
            data,
            step=traintest.coarsen_Nx, dim=1
        )
        data = generate_skipped_missing_data(
            data,
            step=traintest.coarsen_Ny, dim=2
        )
        data = generate_skipped_missing_data(
            data,
            step=traintest.coarsen_time, dim=0
        )

        logger.info("random subset...")
        data = generate_random_missing_data(
            data,
            missing_data_rate=traintest.missing_data,
            return_mask=False,
            seed=traintest.seed_missing_data
        )

        logger.info("adding noise...")
        data = add_noise(data, noise=traintest.noise, seed=traintest.seed_noise)

        return data

    def train_dataloader(self):
        return DataLoader(
            self.ds_train,
            batch_size=self.dataloader.batch_size,
            shuffle=self.dataloader.train_shuffle,
            num_workers=self.dataloader.num_workers,
            pin_memory=self.dataloader.pin_memory
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_valid,
            batch_size=self.dataloader.batch_size,
            shuffle=False,
            num_workers=self.dataloader.num_workers,
            pin_memory=self.dataloader.pin_memory
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_test,
            batch_size=self.dataloader.batch_size_eval,
            shuffle=False,
            num_workers=self.dataloader.num_workers,
            pin_memory=self.dataloader.pin_memory
        )

def get_feature_scaler(config):

    spatial_features = ["Nx", "Ny"]
    temporal_features = ["time"]

    # SPATIAL TRANFORMATIONS
    # spatial transform
    spatial_transforms = []

    if config.minmaxfixed_spatial:
        spatial_transforms.append(
            ("minmaxfixed", MinMaxFixedScaler(
                np.asarray(config.min_spatial),
                np.asarray(config.max_spatial))
             )
        )

    if config.minmax_spatial:
        spatial_transforms.append(
            ("minmax", MinMaxScaler(feature_range=(-1, 1)))
        )

    spatial_transform = Pipeline(spatial_transforms)

    # TEMPORAL TRANSFORMATIONS
    # temporal transform
    temporal_transforms = []

    if config.minmaxfixed_temporal:
        temporal_transforms.append(
            ("minmaxfixed", MinMaxFixedScaler(
                np.asarray(config.min_temporal),
                np.asarray(config.max_temporal))
             )
        )

    if config.minmax_temporal:
        temporal_transforms.append(
            ("minmax", MinMaxScaler(feature_range=(-1, 1)))
        )

    temporal_transform = Pipeline(temporal_transforms)

    scaler = ColumnTransformer(
        transformers=[
            ("space", spatial_transform, spatial_features),
            ("time", temporal_transform, temporal_features),
        ],
        remainder="drop",
    )

    return scaler


