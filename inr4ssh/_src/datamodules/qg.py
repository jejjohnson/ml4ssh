import pandas as pd
import pytorch_lightning as pl
from loguru import logger
from ..data.qg import load_qg_data
from ..features.temporal import MinMaxFixedScaler
from ..preprocess.obs import add_noise
from ..preprocess.missing import (
    generate_random_missing_data,
    generate_skipped_missing_data,
)
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


class QGSimulation(pl.LightningModule):
    def __init__(
        self,
        data,
        preprocess,
        traintest,
        features,
        dataloader,
        # eval
    ):
        super().__init__()
        self.data = data
        self.preprocess = preprocess
        self.traintest = traintest
        self.features = features
        self.dataloader = dataloader
        # self.eval = eval
        self.cols_coords = ["time", "Nx", "Ny"]
        self.cols_features = [self.features.variable]
        self.cols_obs = ["obs"]

    def setup(self, stage=None):
        logger.info("Getting data...")

        logger.info("loading data...")
        ds = load_qg_data(self.data.data_dir)

        ds = self.preprocess_data(ds, self.preprocess)

        if self.cols_features[0] == "ssh":

            ds_obs = ds["p"].values.copy()

        else:
            ds_obs = ds[self.cols_features[0]].values.copy()

        data_obs = self.get_obs_data(ds_obs, self.traintest)

        logger.info("converting to dataframe...")
        ds["obs"] = (ds[self.cols_features].dims, data_obs)
        # print(ds[self.cols_features])

        df = ds[self.cols_obs + self.cols_features].to_dataframe().reset_index()

        logger.info("getting feature scaler...")
        scaler = get_feature_scaler(self.features)

        scaler.fit(df[self.cols_coords])

        # PREDICTION COLUMNS
        # all datapoints in the dataset
        X_predict = df[self.cols_coords]
        y_predict = df[self.cols_features]
        assert np.sum(np.isnan(y_predict.values)) == 0
        self.X_predict_coords = X_predict[self.cols_coords]

        X_predict = scaler.transform(X_predict)

        # TRAINING COLUMNS
        # all datapoints that are observed
        df_obs = df.dropna()

        xtrain, ytrain = df_obs[self.cols_coords], df_obs[self.cols_obs]

        xtrain, xvalid, ytrain, yvalid = train_test_split(
            xtrain,
            ytrain,
            train_size=self.traintest.train_size,
            random_state=self.traintest.seed_split,
        )

        self.X_train_coords = xtrain[self.cols_coords]
        self.X_valid_coords = xvalid[self.cols_coords]

        xtrain = scaler.transform(xtrain)
        xvalid = scaler.transform(xvalid)

        assert np.sum(np.isnan(yvalid.values)) == 0
        assert np.sum(np.isnan(ytrain.values)) == 0

        # TESTING DATA
        # all data points that are not observed
        df_unobs = df.drop(df_obs.index)
        X_test = df_unobs[self.cols_coords]
        y_test = df_unobs[self.cols_features]
        assert np.sum(np.isnan(y_test.values)) == 0

        self.X_test_coords = X_test[self.cols_coords]
        X_test = scaler.transform(X_test)

        logger.info("Creating dataloaders...")
        self.ds_train = TensorDataset(
            torch.FloatTensor(xtrain), torch.FloatTensor(ytrain.values)
        )
        self.ds_valid = TensorDataset(
            torch.FloatTensor(xvalid), torch.FloatTensor(yvalid.values)
        )
        self.ds_test = TensorDataset(
            torch.FloatTensor(X_test), torch.FloatTensor(y_test.values)
        )
        self.ds_predict = TensorDataset(
            torch.FloatTensor(X_predict), torch.FloatTensor(y_predict.values)
        )

        self.dim_in = xtrain.shape[-1]
        self.dim_out = ytrain.shape[-1]
        self.scaler = scaler

    @staticmethod
    def preprocess_data(data, preprocess):

        logger.info("preprocessing data...")

        # rename the dimensions
        try:
            data = data.rename({"steps": "time"})
        except ValueError:
            pass

        # reshape data
        data = data.transpose("time", "Nx", "Ny")

        # subset data
        if preprocess.subset_Nx:
            data = data.isel(Nx=slice(preprocess.Nx_min, preprocess.Nx_max))
        if preprocess.subset_Ny:
            data = data.isel(Ny=slice(preprocess.Ny_min, preprocess.Ny_max))
        if preprocess.subset_time:
            data = data.isel(time=slice(preprocess.time_min, preprocess.time_max))

        # Coarsen X direction
        if preprocess.coarsen_Nx:
            logger.info("coarsening data (Nx)...")
            data = data.coarsen(
                dim={"Nx": preprocess.coarsen_Nx},
                boundary=preprocess.boundary_spatial,
            ).mean()
        if preprocess.coarsen_Ny:
            logger.info("coarsening data (Ny)...")
            data = data.coarsen(
                dim={"Ny": preprocess.coarsen_Ny},
                boundary=preprocess.boundary_spatial,
            ).mean()
        if preprocess.coarsen_time:
            logger.info("coarsening data (time)...")
            data = data.coarsen(
                dim={"time": preprocess.coarsen_time},
                boundary=preprocess.boundary_time,
            ).mean()

        return data

    @staticmethod
    def get_obs_data(data, traintest):

        logger.info("sampling spatiotemporal res...")
        data = generate_skipped_missing_data(data, step=traintest.step_Nx, dim=1)
        data = generate_skipped_missing_data(data, step=traintest.step_Ny, dim=2)
        data = generate_skipped_missing_data(data, step=traintest.step_time, dim=0)

        logger.info("random subset...")
        data = generate_random_missing_data(
            data,
            missing_data_rate=traintest.missing_data,
            return_mask=False,
            seed=traintest.seed_missing_data,
        )

        logger.info("adding noise...")
        if traintest.noise is not None:
            data = add_noise(data, noise=traintest.noise, seed=traintest.seed_noise)

        return data

    def create_xr_dataset(self, split: str = "train"):
        if split == "train":
            data = np.concatenate(
                [self.X_train_coords, self.ds_train[:][1].numpy()], axis=1
            )
        elif split == "test":
            data = np.concatenate(
                [self.X_test_coords, self.ds_test[:][1].numpy()], axis=1
            )
        elif split == "valid":
            data = np.concatenate(
                [self.X_valid_coords, self.ds_valid[:][1].numpy()], axis=1
            )
        elif split == "predict":
            data = np.concatenate(
                [self.X_predict_coords, self.ds_predict[:][1].numpy()], axis=1
            )
        else:
            raise ValueError(f"Unrecognized split: {split}")
        data = (
            pd.DataFrame(data, columns=self.cols_coords + [split])
            .set_index(self.cols_coords)
            .to_xarray()
        )
        return data

    def create_predictions_ds(self, predictions):

        data = np.concatenate(
            [
                self.X_predict_coords,
                self.ds_predict[:][-1].numpy(),
                predictions.numpy(),
            ],
            axis=1,
        )

        data = (
            pd.DataFrame(data, columns=self.cols_coords + ["true"] + ["pred"])
            .set_index(self.cols_coords)
            .to_xarray()
        )
        return data

    def train_dataloader(self):
        return DataLoader(
            self.ds_train,
            batch_size=self.dataloader.batch_size,
            shuffle=self.dataloader.train_shuffle,
            num_workers=self.dataloader.num_workers,
            pin_memory=self.dataloader.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_valid,
            batch_size=self.dataloader.batch_size,
            shuffle=False,
            num_workers=self.dataloader.num_workers,
            pin_memory=self.dataloader.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.ds_test,
            batch_size=self.dataloader.batch_size,
            shuffle=False,
            num_workers=self.dataloader.num_workers,
            pin_memory=self.dataloader.pin_memory,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.ds_predict,
            batch_size=self.dataloader.batch_size_eval,
            shuffle=False,
            num_workers=self.dataloader.num_workers,
            pin_memory=self.dataloader.pin_memory,
        )


def get_feature_scaler(config):

    spatial_features = ["Nx", "Ny"]
    temporal_features = ["time"]

    # SPATIAL TRANFORMATIONS
    # spatial transform
    spatial_transforms = []

    if config.minmax_fixed_spatial:
        spatial_transforms.append(
            (
                "minmaxfixed",
                MinMaxFixedScaler(
                    np.asarray(config.min_spatial), np.asarray(config.max_spatial)
                ),
            )
        )

    if config.minmax_spatial:
        spatial_transforms.append(("minmax", MinMaxScaler(feature_range=(-1, 1))))

    spatial_transform = Pipeline(spatial_transforms)

    # TEMPORAL TRANSFORMATIONS
    # temporal transform
    temporal_transforms = []

    if config.minmax_fixed_temporal:
        temporal_transforms.append(
            (
                "minmaxfixed",
                MinMaxFixedScaler(
                    np.asarray(config.min_temporal), np.asarray(config.max_temporal)
                ),
            )
        )

    if config.minmax_temporal:
        temporal_transforms.append(("minmax", MinMaxScaler(feature_range=(-1, 1))))

    temporal_transform = Pipeline(temporal_transforms)

    scaler = ColumnTransformer(
        transformers=[
            ("space", spatial_transform, spatial_features),
            ("time", temporal_transform, temporal_features),
        ],
        remainder="drop",
    )

    return scaler
