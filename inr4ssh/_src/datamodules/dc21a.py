import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from loguru import logger
from torch.utils.data import TensorDataset, DataLoader
from inr4ssh._src.features.spatial import Spherical2Cartesian3D
from inr4ssh._src.features.temporal import TimeMinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from inr4ssh._src.preprocess.grid import create_spatiotemporal_grid
from inr4ssh._src.preprocess.subset import temporal_subset
import pandas as pd
import numpy as np
from inr4ssh._src.data.ssh_obs import load_ssh_altimetry_data_train
import torch

import torch
import numpy as np
from pathlib import Path
from loguru import logger
from inr4ssh._src.datasets.utils import get_num_training
from inr4ssh._src.preprocess.coords import correct_coordinate_labels
from inr4ssh._src.preprocess.grid import create_spatiotemporal_grid
import pandas as pd
from inr4ssh._src.datasets.alongtrack import AlongTrackEvalDataset
from torchvision.transforms import Compose
from ml_collections import config_dict
import pytorch_lightning as pl
import xarray as xr
from inr4ssh._src.transforms.dataset import transform_factory
from inr4ssh._src.datasets.alongtrack import AlongTrackDataset
from inr4ssh._src.preprocess.spatial import convert_lon_360_180


class DC21AlongTrackDM(pl.LightningDataModule):
    def __init__(
        self, config=None, root: config_dict.ConfigDict = None, download=False
    ):
        super().__init__()
        self.root = root
        if config is None:
            config = get_demo_config()
        self.config = config
        self.download = download

    def prepare_download(self):
        # download
        pass

    def setup(self, stage=None):

        # check

        self.ds_train, self.ds_valid, self.ds_test, self.ds_predict = self._setup()

    def _open_xr_ds(self, dataset: str):
        # TODO: check if root directory exists
        # TODO: download dataset if option

        # open xarray dataset
        logger.info("Opening xarray dataset...")
        logger.info(f"Dataset: {dataset}")
        logger.info(f"Dataset dir: {self.config.datadir.obs_dir}")

        return None

    def _get_dataset_train(self):

        ds_path = Path(self.config.datadir.staging.staging_dir).joinpath("train.nc")

        logger.debug(f"{ds_path}")
        ds = xr.open_dataset(ds_path)

        # correct the labels
        logger.info("Correcting labels...")
        ds = correct_coordinate_labels(ds)

        logger.info("Sorting array by time...")
        ds = ds.sortby("time")

        # temporal subset
        if self.config.preprocess.subset_time.subset_time:
            logger.info("Subsetting temporal...")
            time_min = self.config.preprocess.subset_time.time_min
            time_max = self.config.preprocess.subset_time.time_max
            logger.debug(f"Time Min: {time_min} | Time Max: {time_max}...")
            ds = ds.sel(time=slice(time_min, time_max), drop=True)

        # spatial subset
        if self.config.preprocess.subset_spatial.subset_spatial:
            logger.info("Subseting spatial...")
            lon_min = self.config.preprocess.subset_spatial.lon_min
            lon_max = self.config.preprocess.subset_spatial.lon_max
            lat_min = self.config.preprocess.subset_spatial.lat_min
            lat_max = self.config.preprocess.subset_spatial.lat_max
            logger.debug(f"Lon Min: {lon_min} | Lon Max: {lon_max}...")
            logger.debug(f"Lat Min: {lat_min} | Lat Max: {lat_max}...")
            ds = ds.where(
                (ds["longitude"] >= lon_min)
                & (ds["longitude"] <= lon_max)
                & (ds["latitude"] >= lat_min)
                & (ds["latitude"] <= lat_max),
                drop=True,
            )

        # get dataloader transformations
        transforms = Compose(transform_factory(self.config.transform))

        # create dataset
        logger.info("Creating dataset...")
        ds = AlongTrackDataset(
            ds=ds,
            spatial_columns=["longitude", "latitude"],
            temporal_columns=["time"],
            output_columns=["sla_unfiltered"],
            transform=transforms,
        )

        return ds

    def get_data_test(self):

        ds_path = Path(self.config.datadir.staging.staging_dir).joinpath("test.nc")

        logger.debug(f"{ds_path}")
        ds = xr.open_dataset(ds_path)

        # correct the labels
        logger.info("Correcting labels...")
        ds = correct_coordinate_labels(ds)

        logger.info("Sorting array by time...")
        ds = ds.sortby("time")

        # temporal subset
        logger.info("Subsetting temporal...")
        time_min = self.config.evaluation.time_min
        time_max = self.config.evaluation.time_max
        logger.debug(f"Time Min: {time_min} | Time Max: {time_max}...")
        ds = ds.sel(time=slice(time_min, time_max), drop=True)

        # spatial subset
        logger.info("Subseting spatial...")
        lon_min = self.config.evaluation.lon_min
        lon_max = self.config.evaluation.lon_max
        lat_min = self.config.evaluation.lat_min
        lat_max = self.config.evaluation.lat_max
        logger.debug(f"Lon Min: {lon_min} | Lon Max: {lon_max}...")
        logger.debug(f"Lat Min: {lat_min} | Lat Max: {lat_max}...")
        ds = ds.where(
            (ds["longitude"] >= lon_min)
            & (ds["longitude"] <= lon_max)
            & (ds["latitude"] >= lat_min)
            & (ds["latitude"] <= lat_max),
            drop=True,
        )
        return ds

    def _get_dataset_test(self):

        ds = self.get_data_test()

        # get dataloader transformations
        transforms = Compose(transform_factory(self.config.transform))

        # create dataset
        logger.info("Creating dataset...")
        ds = AlongTrackDataset(
            ds=ds,
            spatial_columns=["longitude", "latitude"],
            temporal_columns=["time"],
            output_columns=["sla_unfiltered"],
            transform=transforms,
        )

        return ds

    def _dataset_split(self, ds):

        logger.info(f"Train/Valid Split ({self.config.traintest.train_prct*100}%)...")
        num_train, num_valid = get_num_training(
            len(ds), train_prct=self.config.traintest.train_prct
        )
        logger.info(f"Creating train/valid datasets...")
        ds_train, ds_valid = torch.utils.data.random_split(
            ds,
            [num_train, num_valid],
            generator=torch.Generator().manual_seed(self.config.traintest.seed),
        )

        if self.config.traintest.subset_random > 0.0:
            num_subset, dummy = get_num_training(
                len(ds_train), train_prct=self.config.traintest.subset_random
            )
            logger.info(f"Creating random subset...")
            ds_train, _ = torch.utils.data.random_split(
                ds_train,
                [num_subset, dummy],
                generator=torch.Generator().manual_seed(
                    self.config.traintest.subset_seed
                ),
            )
        # train/valid dataset
        logger.info(f"{len(ds)}={len(ds_train):,}/{len(ds_valid):,} pts")

        return ds_train, ds_valid

    def _get_dataset_eval(self):

        # TODO: create grid-coordinates
        logger.info("Creating spatial-temporal grid...")
        lon_coords, lat_coords, time_coords = create_spatiotemporal_grid(
            lon_min=self.config.evaluation.lon_min,
            lon_max=self.config.evaluation.lon_max,
            lon_dx=self.config.evaluation.dlon,
            lat_min=self.config.evaluation.lat_min,
            lat_max=self.config.evaluation.lat_max,
            lat_dy=self.config.evaluation.dlat,
            time_min=np.datetime64(self.config.evaluation.time_min),
            time_max=np.datetime64(self.config.evaluation.time_max),
            time_dt=np.timedelta64(
                self.config.evaluation.dt_freq, self.config.evaluation.dt_unit
            ),
        )

        # get dataloader transformations
        transforms = Compose(transform_factory(self.config.transform))

        coords = pd.DataFrame(
            {
                "longitude": lon_coords,
                "latitude": lat_coords,
                "time": time_coords,
            }
        )
        # TODO: create dataset from grid coordinates
        logger.info("Creating prediction dataset...")
        ds = AlongTrackDataset(
            ds=coords,
            spatial_columns=["longitude", "latitude"],
            temporal_columns=["time"],
            output_columns=None,
            transform=transforms,
        )

        return ds

    def get_dataset_correction(self):

        # get directory
        ref_dir = Path(self.config.datadir.raw.correction_dir).joinpath("mdt.nc")

        # load dataset
        ds_correct = xr.open_dataset(ref_dir)

        # rename values
        # correct the labels
        logger.info("Correcting labels...")
        ds_correct = correct_coordinate_labels(ds_correct)

        # correct longitude dimensions
        ds_correct["longitude"] = convert_lon_360_180(ds_correct["longitude"])

        # subset region
        ds_correct = ds_correct.where(
            (ds_correct["longitude"] >= self.config.evaluation.lon_min)
            & (ds_correct["longitude"] <= self.config.evaluation.lon_max)
            & (ds_correct["latitude"] >= self.config.evaluation.lat_min)
            & (ds_correct["latitude"] <= self.config.evaluation.lat_max),
            drop=True,
        )

        return ds_correct

    def correct_ssh(self, da):

        ds_correct = self.get_dataset_correction()

        ds_correct = ds_correct.interp(longitude=da.longitude, latitude=da.latitude)

        return da + ds_correct["mdt"]

    def _setup(self):
        # TODO: check if root directory exists
        # TODO: download dataset if option

        # ============================
        # DATASET (TRAINING)
        # ============================
        # open xarray dataset
        logger.info("Creating Dataset (Training)...")
        ds = self._get_dataset_train()

        # ============================
        # DATASET (TRAINING/VALIDATION)
        # ============================
        ds_train, ds_valid = self._dataset_split(ds)

        # =============================
        # DATASET (TESTING)
        # =============================
        logger.info("Creating Dataset (Test)...")
        ds_test = self._get_dataset_test()

        # =============================
        # DATASET (EVALUATION)
        # =============================
        logger.info("Creating Dataset (Evalulation)...")
        ds_predict = self._get_dataset_eval()

        return ds_train, ds_valid, ds_test, ds_predict

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.ds_train,
            batch_size=self.config.dataloader.batchsize_train,
            shuffle=self.config.dataloader.shuffle_train,
            num_workers=self.config.dataloader.num_workers_train,
            pin_memory=self.config.dataloader.pin_memory_train,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.ds_valid,
            batch_size=self.config.dataloader.batchsize_valid,
            shuffle=self.config.dataloader.shuffle_valid,
            num_workers=self.config.dataloader.num_workers_valid,
            pin_memory=self.config.dataloader.pin_memory_valid,
        )

    def test_dataloader(self):
        # raise NotImplementedError()
        return torch.utils.data.DataLoader(
            self.ds_test,
            batch_size=self.config.dataloader.batchsize_test,
            shuffle=self.config.dataloader.shuffle_test,
            num_workers=self.config.dataloader.num_workers_test,
            pin_memory=self.config.dataloader.pin_memory_test,
        )

    def predict_dataloader(self):
        # raise NotImplementedError()
        return torch.utils.data.DataLoader(
            self.ds_predict,
            batch_size=self.config.dataloader.batchsize_predict,
            shuffle=self.config.dataloader.shuffle_predict,
            num_workers=self.config.dataloader.num_workers_predict,
            pin_memory=self.config.dataloader.pin_memory_predict,
        )


def get_demo_config():
    config = config_dict.ConfigDict()

    # data directory
    config.data = data = config_dict.ConfigDict()
    data.dataset_dir = "/Volumes/EMANS_HDD/data/dc20a_osse/test/ml/nadir1.nc"
    data.ref_dir = (
        "/Volumes/EMANS_HDD/data/dc20a_osse/raw/dc_ref/NATL60-CJM165_GULFSTREAM*"
    )

    # preprocessing
    config.preprocess = config_dict.ConfigDict()
    config.preprocess.subset_time = subset_time = config_dict.ConfigDict()
    subset_time.subset_time = True
    subset_time.time_min = "2012-10-22"
    subset_time.time_max = "2012-12-02"

    config.preprocess.subset_spatial = subset_spatial = config_dict.ConfigDict()
    subset_spatial.subset_spatial = True
    subset_spatial.lon_min = -65.0
    subset_spatial.lon_max = -55.0
    subset_spatial.lat_min = 33.0
    subset_spatial.lat_max = 43.0

    # transformations
    config.preprocess.transform = transform = config_dict.ConfigDict()
    transform.time_transform = "minmax"
    transform.time_min = "2011-01-01"
    transform.time_max = "2013-12-12"

    # train/valid arguments
    config.traintest = traintest = config_dict.ConfigDict()
    traintest.train_prct = 0.9
    traintest.seed = 42

    # dataloader
    config.dataloader = dataloader = config_dict.ConfigDict()
    # train dataloader
    dataloader.batchsize_train = 32
    dataloader.num_workers_train = 1
    dataloader.shuffle_train = True
    dataloader.pin_memory_train = False
    # valid dataloader
    dataloader.batchsize_valid = 32
    dataloader.num_workers_valid = 1
    dataloader.shuffle_valid = False
    dataloader.pin_memory_valid = False
    # test dataloader
    dataloader.batchsize_test = 32
    dataloader.num_workers_test = 1
    dataloader.shuffle_test = False
    dataloader.pin_memory_test = False
    # predict dataloader
    dataloader.batchsize_predict = 32
    dataloader.num_workers_predict = 1
    dataloader.shuffle_predict = False
    dataloader.pin_memory_predict = False

    # EVALUATION
    config.evaluation = evaluation = config_dict.ConfigDict()
    evaluation.lon_min = -65.0
    evaluation.lon_max = -55.0
    evaluation.dlon = 0.1
    evaluation.lat_min = 33.0
    evaluation.lat_max = 43.0
    evaluation.dlat = 0.1

    evaluation.time_min = "2012-10-22"
    evaluation.time_max = "2012-12-02"
    evaluation.dt_freq = 1
    evaluation.dt_unit = "D"

    evaluation.time_resample = "1D"

    return config


class SSHAltimetry(pl.LightningDataModule):
    def __init__(self, data, preprocess, traintest, features, dataloader, eval):
        super().__init__()
        self.data = data
        self.preprocess = preprocess
        self.traintest = traintest
        self.features = features
        self.dataloader = dataloader
        self.eval = eval

    def setup(self, stage=None):

        logger.info("Getting training data...")
        X, y, scaler = self.get_train_data(self.data, self.preprocess, self.features)

        self.scaler = scaler
        logger.info("Train/Validation Split...")
        xtrain, ytrain, xvalid, yvalid = self.split_train_data(X, y, self.traintest)

        logger.info("Getting evalulation data...")
        X = self.get_eval_data(self.eval)

        self.X_pred_index = X[["latitude", "longitude", "time"]]

        logger.info("scaling evaluation data...")
        X = scaler.transform(X)

        self.scaler = scaler

        logger.info("Creating dataloaders...")
        self.ds_train = TensorDataset(
            torch.FloatTensor(xtrain), torch.FloatTensor(ytrain)
        )
        self.ds_valid = TensorDataset(
            torch.FloatTensor(xvalid), torch.FloatTensor(yvalid)
        )
        self.ds_predict = TensorDataset(torch.FloatTensor(X))
        self.dim_in = xtrain.shape[-1]
        self.dim_out = ytrain.shape[-1]

    @staticmethod
    def get_train_data(data, preprocess, features):
        # load data from train dir
        logger.info("loading data...")
        ds = load_ssh_altimetry_data_train(data.train_data_dir)

        logger.info("subsetting data...")
        ds = preprocess_data(ds, preprocess)

        logger.info("getting feature scaler...")
        scaler = get_feature_scaler(features)

        logger.info("feature scaling...")
        X = scaler.fit_transform(ds)
        y = ds[["sla_unfiltered"]].values

        return X, y, scaler

    @staticmethod
    def get_eval_data(config):
        logger.info("loading evaluation data...")
        test_df = get_evalulation_data(config)

        return test_df

    @staticmethod
    def split_train_data(X, y, config):

        xtrain, xvalid, ytrain, yvalid = train_test_split(
            X, y, train_size=config.train_size, random_state=config.seed_split
        )

        return xtrain, ytrain, xvalid, yvalid

    def train_dataloader(self):
        return DataLoader(
            self.ds_train,
            batch_size=self.dataloader.batchsize_train,
            shuffle=self.dataloader.train_shuffle,
            num_workers=self.dataloader.num_workers,
            pin_memory=self.dataloader.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_valid,
            batch_size=self.dataloader.batchsize_valid,
            shuffle=False,
            num_workers=self.dataloader.num_workers,
            pin_memory=self.dataloader.pin_memory,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.ds_predict,
            batch_size=self.dataloader.batchsize_predict,
            shuffle=False,
            num_workers=self.dataloader.num_workers,
            pin_memory=self.dataloader.pin_memory,
        )


def preprocess_data(ds_obs, config):

    time_min = np.datetime64(config.time_min)
    time_max = np.datetime64(config.time_max)
    dtime = np.timedelta64(*config.dtime.split("_"))
    # temporal subset
    ds_obs = temporal_subset(
        ds_obs, time_min=time_min, time_max=time_max, time_buffer=config.time_buffer
    )

    # convert to dataframe
    data = ds_obs.to_dataframe().reset_index().dropna()

    # add vtime
    # NOTE: THIS IS THE GLOBAL MINIMUM TIME FOR THE DATASET
    data["vtime"] = (data["time"].values - np.datetime64("2016-12-01")) / dtime

    # add column attributes
    data.attrs["input_cols"] = ["longitude", "latitude", "time"]
    data.attrs["output_cols"] = ["sla_unfiltered"]

    return data


def get_evalulation_data(config):

    # create spatiotemporal grid
    lon_coords, lat_coords, time_coords = create_spatiotemporal_grid(
        lon_min=config.lon_min,
        lon_max=config.lon_max,
        lon_dx=config.dlon,
        lat_min=config.lat_min,
        lat_max=config.lat_max,
        lat_dy=config.dlat,
        time_min=np.datetime64(config.time_min),
        time_max=np.datetime64(config.time_max),
        time_dt=np.timedelta64(config.dtime_freq, config.dtime_unit),
    )

    df_grid = pd.DataFrame(
        {
            "longitude": lon_coords,
            "latitude": lat_coords,
            "time": time_coords,
        }
    )

    return df_grid


def get_feature_scaler(config):

    spatial_features = ["longitude", "latitude"]
    temporal_features = ["time"]

    # SPATIAL TRANFORMATIONS
    # spatial transform
    spatial_features = ["longitude", "latitude"]
    spatial_transforms = []

    if config.cartesian:
        spatial_transforms.append(
            ("cartesian3d", Spherical2Cartesian3D(radius=config.spherical_radius))
        )

    if config.minmax_spatial:
        spatial_transforms.append(("minmax", MinMaxScaler(feature_range=(-1, 1))))

    spatial_transform = Pipeline(spatial_transforms)

    # TEMPORAL TRANSFORMATIONS
    # temporal transform
    temporal_features = ["time"]
    temporal_transforms = []

    if config.abs_time:
        temporal_transforms.append(
            (
                "timestd",
                TimeMinMaxScaler(
                    julian_date=config.julian_time,
                    time_min=config.abs_time_min,
                    time_max=config.abs_time_max,
                ),
            )
        )

    if config.minmax_temporal:
        temporal_transforms.append(("minmax", MinMaxScaler(feature_range=(-1, 1))))

    temporal_transform = Pipeline(temporal_transforms)

    scaler = ColumnTransformer(
        transformers=[
            ("cartesian3d", spatial_transform, spatial_features),
            ("timeminmax", temporal_transform, temporal_features),
        ],
        remainder="drop",
    )

    return scaler
