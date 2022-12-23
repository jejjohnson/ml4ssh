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


class AlongTrackDataModule(pl.LightningDataModule):
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

    def get_alongtrack_dataset(self, subset_time, subset_spatial):
        # open xarray dataset
        logger.info("Opening xarray dataset...")
        logger.info(f"Dataset: {self.config.experiment}")

        ds_path = Path(self.config.datadir.staging.staging_dir).joinpath(
            f"{self.config.experiment}.nc"
        )

        ds = xr.open_dataset(ds_path)

        # correct the labels
        logger.info("Correcting labels...")
        ds = correct_coordinate_labels(ds)

        logger.info("Sorting array by time...")
        ds = ds.sortby("time")

        # temporal subset
        if subset_time.subset_time:
            logger.info("Subsetting temporal...")
            time_min = subset_time.time_min
            time_max = subset_time.time_max
            logger.debug(f"Time Min: {time_min} | Time Max: {time_max}...")
            ds = ds.sel(time=slice(time_min, time_max), drop=True)

        # spatial subset
        if subset_spatial.subset_spatial:
            logger.info("Subseting spatial...")
            lon_min = subset_spatial.lon_min
            lon_max = subset_spatial.lon_max
            lat_min = subset_spatial.lat_min
            lat_max = subset_spatial.lat_max
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
            output_columns=["ssh_model"],
            transform=transforms,
        )
        return ds

    def get_natl60_dataset(self, subset_time, subset_spatial, resample):

        logger.info("Opening xarray dataset...")
        logger.info(f"Dataset: {self.config.datadir.clean.ref_dir}")
        ds_filenames = Path(self.config.datadir.clean.ref_dir).joinpath(
            "NATL60-CJM165_GULFSTREAM_y*"
        )
        logger.info(f"Dataset: {ds_filenames}")
        ds = xr.open_mfdataset(str(ds_filenames), engine="netcdf4")

        # temporal subset
        if subset_time.subset_time:
            logger.info("Subsetting temporal...")
            time_min = subset_time.time_min
            time_max = subset_time.time_max
            logger.debug(f"Time Min: {time_min} | Time Max: {time_max}...")
            ds = ds.sel(time=slice(time_min, time_max), drop=True)

        # spatial subset
        if subset_spatial.subset_spatial:
            logger.info("Subseting spatial...")
            lon_min = subset_spatial.lon_min
            lon_max = subset_spatial.lon_max
            lat_min = subset_spatial.lat_min
            lat_max = subset_spatial.lat_max
            logger.debug(f"Lon Min: {lon_min} | Lon Max: {lon_max}...")
            logger.debug(f"Lat Min: {lat_min} | Lat Max: {lat_max}...")
            ds = ds.where(
                (ds["longitude"] >= lon_min)
                & (ds["longitude"] <= lon_max)
                & (ds["latitude"] >= lat_min)
                & (ds["latitude"] <= lat_max),
                drop=True,
            )

        if resample.time_resample is not None:
            ds = ds.resample(time=resample.time_resample).mean()

        if self.config.preprocess.resample.coarsen_lon > 0:
            ds = ds.coarsen(
                dim={"lon": self.config.preprocess.resample.coarsen_lon}
            ).mean()

        if self.config.preprocess.resample.coarsen_lat > 0:
            ds = ds.coarsen(
                dim={"lat": self.config.preprocess.resample.coarsen_lat}
            ).mean()

        ds = correct_coordinate_labels(ds)

        logger.info("Creating coordinates...")
        x, y, z = np.meshgrid(
            ds.coords["time"].data,
            ds.coords["latitude"].data,
            ds.coords["longitude"].data,
        )

        ds_ref_coords = pd.DataFrame(
            {
                "longitude": z.flatten(),
                "latitude": y.flatten(),
                "time": x.flatten(),
                "ssh_model": ds["sossheig"].data.flatten(),
            }
        )

        # get dataloader transformations
        transforms = Compose(transform_factory(self.config.transform))

        logger.info("Creating natl60 dataset...")
        ds_natl60 = AlongTrackDataset(
            ds=ds_ref_coords,
            spatial_columns=["longitude", "latitude"],
            temporal_columns=["time"],
            output_columns=["ssh_model"],
            transform=transforms,
        )

        return ds_natl60

    def _setup(self):
        # TODO: check if root directory exists
        # TODO: download dataset if option

        if self.config.preprocess.dataset == "alongtrack":
            ds_train = self.get_alongtrack_dataset(
                subset_time=self.config.preprocess.subset_time,
                subset_spatial=self.config.preprocess.subset_spatial,
            )
        elif self.config.preprocess.dataset == "natl60":
            ds_train = self.get_natl60_dataset(
                subset_time=self.config.preprocess.subset_time,
                subset_spatial=self.config.preprocess.subset_spatial,
                resample=self.config.preprocess.resample,
            )
        else:
            raise ValueError(f"Unrecognized")

        # open xarray dataset
        logger.info("Opening xarray dataset...")
        logger.info(f"Dataset: {self.config.experiment}")

        ds_path = Path(self.config.datadir.staging.staging_dir).joinpath(
            f"{self.config.experiment}.nc"
        )

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
            output_columns=["ssh_model"],
            transform=transforms,
        )

        # train/test split
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

        # train/valid dataset
        logger.info(f"{len(ds_train):,}/{len(ds_valid):,} pts")

        # TEST
        logger.info("Opening xarray dataset...")
        logger.info(f"Dataset: {self.config.datadir.clean.ref_dir}")
        ds_filenames = Path(self.config.datadir.clean.ref_dir).joinpath(
            "NATL60-CJM165_GULFSTREAM_y*"
        )
        logger.info(f"Dataset: {ds_filenames}")
        ds = xr.open_mfdataset(str(ds_filenames), engine="netcdf4")

        logger.info("Subsetting data...")
        ds = ds.sel(
            time=slice(
                self.config.evaluation.time_min, self.config.evaluation.time_max
            ),
            lon=slice(self.config.evaluation.lon_min, self.config.evaluation.lon_max),
            lat=slice(self.config.evaluation.lat_min, self.config.evaluation.lat_max),
            drop=True,
        )

        if self.config.evaluation.time_resample is not None:
            ds = ds.resample(time=self.config.evaluation.time_resample).mean()

        if self.config.evaluation.lon_coarsen > 0:
            ds = ds.coarsen(dim={"lon": self.config.evaluation.lon_coarsen}).mean()

        if self.config.evaluation.lat_coarsen > 0:
            ds = ds.coarsen(dim={"lat": self.config.evaluation.lat_coarsen}).mean()

        ds = correct_coordinate_labels(ds)

        logger.info("Creating coordinates...")
        x, y, z = np.meshgrid(
            ds.coords["time"].data,
            ds.coords["latitude"].data,
            ds.coords["longitude"].data,
        )

        ds_ref_coords = pd.DataFrame(
            {
                "longitude": z.flatten(),
                "latitude": y.flatten(),
                "time": x.flatten(),
                "ssh_model": ds["sossheig"].data.flatten(),
            }
        )

        logger.info("Creating test dataset...")
        ds_test = AlongTrackDataset(
            ds=ds_ref_coords,
            spatial_columns=["longitude", "latitude"],
            temporal_columns=["time"],
            output_columns=["ssh_model"],
            transform=transforms,
        )
        # TODO: predict dataset
        logger.info(f"{len(ds_test):,} pts")

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

        coords = pd.DataFrame(
            {
                "longitude": lon_coords,
                "latitude": lat_coords,
                "time": time_coords,
            }
        )
        # TODO: create dataset from grid coordinates
        logger.info("Creating prediction dataset...")
        ds_predict = AlongTrackDataset(
            ds=coords,
            spatial_columns=["longitude", "latitude"],
            temporal_columns=["time"],
            output_columns=None,
            transform=transforms,
        )
        # TODO: predict dataset
        logger.info(f"{len(ds_predict):,} pts")
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


class NATL60DataModule(pl.LightningDataModule):
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

    def get_alongtrack_dataset(self, subset_time, subset_spatial):
        # open xarray dataset
        logger.info("Opening xarray dataset...")
        logger.info(f"Dataset: {self.config.experiment}")

        ds_path = Path(self.config.datadir.staging.staging_dir).joinpath(
            f"{self.config.experiment}.nc"
        )

        ds = xr.open_dataset(ds_path)

        # correct the labels
        logger.info("Correcting labels...")
        ds = correct_coordinate_labels(ds)

        logger.info("Sorting array by time...")
        ds = ds.sortby("time")

        # temporal subset
        if subset_time.subset_time:
            logger.info("Subsetting temporal...")
            time_min = subset_time.time_min
            time_max = subset_time.time_max
            logger.debug(f"Time Min: {time_min} | Time Max: {time_max}...")
            ds = ds.sel(time=slice(time_min, time_max), drop=True)

        # spatial subset
        if subset_spatial.subset_spatial:
            logger.info("Subseting spatial...")
            lon_min = subset_spatial.lon_min
            lon_max = subset_spatial.lon_max
            lat_min = subset_spatial.lat_min
            lat_max = subset_spatial.lat_max
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
            output_columns=["ssh_model"],
            transform=transforms,
        )
        return ds

    def get_natl60_dataset(self, subset_time, subset_spatial, resample):

        logger.info("Opening xarray dataset...")
        logger.info(f"Dataset: {self.config.datadir.clean.ref_dir}")
        ds_filenames = Path(self.config.datadir.clean.ref_dir).joinpath(
            "NATL60-CJM165_GULFSTREAM_y*"
        )
        logger.info(f"Dataset: {ds_filenames}")
        ds = xr.open_mfdataset(str(ds_filenames), engine="netcdf4")

        # temporal subset
        if subset_time.subset_time:
            logger.info("Subsetting temporal...")
            time_min = subset_time.time_min
            time_max = subset_time.time_max
            logger.debug(f"Time Min: {time_min} | Time Max: {time_max}...")
            ds = ds.sel(time=slice(time_min, time_max), drop=True)

        # spatial subset
        if subset_spatial.subset_spatial:
            logger.info("Subseting spatial...")
            lon_min = subset_spatial.lon_min
            lon_max = subset_spatial.lon_max
            lat_min = subset_spatial.lat_min
            lat_max = subset_spatial.lat_max
            logger.debug(f"Lon Min: {lon_min} | Lon Max: {lon_max}...")
            logger.debug(f"Lat Min: {lat_min} | Lat Max: {lat_max}...")
            ds = ds.where(
                (ds["longitude"] >= lon_min)
                & (ds["longitude"] <= lon_max)
                & (ds["latitude"] >= lat_min)
                & (ds["latitude"] <= lat_max),
                drop=True,
            )

        if resample.time_resample is not None:
            ds = ds.resample(time=resample.time_resample).mean()

        if resample.coarsen_lon > 0:
            ds = ds.coarsen(dim={"lon": resample.coarsen_lon}).mean()

        if resample.coarsen_lat > 0:
            ds = ds.coarsen(dim={"lat": resample.coarsen_lat}).mean()

        ds = correct_coordinate_labels(ds)

        # transform to get coordinates
        ds = ds.rename({"sossheig": "ssh_model"})
        ds = ds.transpose("time", "latitude", "longitude")
        #
        # logger.info("Creating coordinates...")
        # x, y, z = np.meshgrid(
        #     ds.coords["time"].data,
        #     ds.coords["latitude"].data,
        #     ds.coords["longitude"].data,
        # )
        #
        # ds_ref_coords = pd.DataFrame(
        #     {
        #         "longitude": z.flatten(),
        #         "latitude": y.flatten(),
        #         "time": x.flatten(),
        #         "ssh_model": ds["sossheig"].data.flatten(),
        #     }
        # )

        ds_ref_coords = ds.to_dataframe().reset_index()

        # get dataloader transformations
        transforms = Compose(transform_factory(self.config.transform))

        logger.info("Creating natl60 dataset...")
        ds_natl60 = AlongTrackDataset(
            ds=ds_ref_coords[["longitude", "latitude", "time", "ssh_model"]],
            spatial_columns=["longitude", "latitude"],
            temporal_columns=["time"],
            output_columns=["ssh_model"],
            transform=transforms,
        )

        return ds_natl60

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

    def _setup(self):
        # TODO: check if root directory exists
        # TODO: download dataset if option

        if self.config.preprocess.dataset == "alongtrack":
            ds = self.get_alongtrack_dataset(
                subset_time=self.config.preprocess.subset_time,
                subset_spatial=self.config.preprocess.subset_spatial,
            )
        elif self.config.preprocess.dataset == "natl60":
            ds = self.get_natl60_dataset(
                subset_time=self.config.preprocess.subset_time,
                subset_spatial=self.config.preprocess.subset_spatial,
                resample=self.config.preprocess.resample,
            )
        else:
            raise ValueError(f"Unrecognized dataset: {self.config.preprocess.dataset}")

        # # open xarray dataset
        # logger.info("Opening xarray dataset...")
        # logger.info(f"Dataset: {self.config.experiment}")
        #
        # ds_path = Path(self.config.datadir.staging.staging_dir).joinpath(
        #     f"{self.config.experiment}.nc"
        # )
        #
        # ds = xr.open_dataset(ds_path)
        #
        # # correct the labels
        # logger.info("Correcting labels...")
        # ds = correct_coordinate_labels(ds)
        #
        # logger.info("Sorting array by time...")
        # ds = ds.sortby("time")
        #
        # # temporal subset
        # if self.config.preprocess.subset_time.subset_time:
        #     logger.info("Subsetting temporal...")
        #     time_min = self.config.preprocess.subset_time.time_min
        #     time_max = self.config.preprocess.subset_time.time_max
        #     logger.debug(f"Time Min: {time_min} | Time Max: {time_max}...")
        #     ds = ds.sel(time=slice(time_min, time_max), drop=True)
        #
        # # spatial subset
        # if self.config.preprocess.subset_spatial.subset_spatial:
        #     logger.info("Subseting spatial...")
        #     lon_min = self.config.preprocess.subset_spatial.lon_min
        #     lon_max = self.config.preprocess.subset_spatial.lon_max
        #     lat_min = self.config.preprocess.subset_spatial.lat_min
        #     lat_max = self.config.preprocess.subset_spatial.lat_max
        #     logger.debug(f"Lon Min: {lon_min} | Lon Max: {lon_max}...")
        #     logger.debug(f"Lat Min: {lat_min} | Lat Max: {lat_max}...")
        #     ds = ds.where(
        #         (ds["longitude"] >= lon_min)
        #         & (ds["longitude"] <= lon_max)
        #         & (ds["latitude"] >= lat_min)
        #         & (ds["latitude"] <= lat_max),
        #         drop=True,
        #     )
        #
        # # get dataloader transformations
        # transforms = Compose(transform_factory(self.config.transform))
        #
        # # create dataset
        # logger.info("Creating dataset...")
        # ds = AlongTrackDataset(
        #     ds=ds,
        #     spatial_columns=["longitude", "latitude"],
        #     temporal_columns=["time"],
        #     output_columns=["ssh_model"],
        #     transform=transforms,
        # )

        # train/test split
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

        # train/valid dataset
        logger.info(f"{len(ds_train):,}/{len(ds_valid):,} pts")

        # TEST

        # logger.info("Opening xarray dataset...")
        # logger.info(f"Dataset: {self.config.datadir.clean.ref_dir}")
        # ds_filenames = Path(self.config.datadir.clean.ref_dir).joinpath(
        #     "NATL60-CJM165_GULFSTREAM_y*"
        # )
        # logger.info(f"Dataset: {ds_filenames}")
        # ds = xr.open_mfdataset(str(ds_filenames), engine="netcdf4")
        #
        # logger.info("Subsetting data...")
        # ds = ds.sel(
        #     time=slice(
        #         self.config.evaluation.time_min, self.config.evaluation.time_max
        #     ),
        #     lon=slice(self.config.evaluation.lon_min, self.config.evaluation.lon_max),
        #     lat=slice(self.config.evaluation.lat_min, self.config.evaluation.lat_max),
        #     drop=True,
        # )
        #
        # if self.config.evaluation.time_resample is not None:
        #     ds = ds.resample(time=self.config.evaluation.time_resample).mean()
        #
        # if self.config.evaluation.lon_coarsen > 0:
        #     ds = ds.coarsen(dim={"lon": self.config.evaluation.lon_coarsen}).mean()
        #
        # if self.config.evaluation.lat_coarsen > 0:
        #     ds = ds.coarsen(dim={"lat": self.config.evaluation.lat_coarsen}).mean()
        #
        # ds = correct_coordinate_labels(ds)
        #
        # logger.info("Creating coordinates...")
        # x, y, z = np.meshgrid(
        #     ds.coords["time"].data,
        #     ds.coords["latitude"].data,
        #     ds.coords["longitude"].data,
        # )
        #
        # ds_ref_coords = pd.DataFrame(
        #     {
        #         "longitude": z.flatten(),
        #         "latitude": y.flatten(),
        #         "time": x.flatten(),
        #         "ssh_model": ds["sossheig"].data.flatten(),
        #     }
        # )
        #
        # logger.info("Creating test dataset...")
        # ds_test = AlongTrackDataset(
        #     ds=ds_ref_coords,
        #     spatial_columns=["longitude", "latitude"],
        #     temporal_columns=["time"],
        #     output_columns=["ssh_model"],
        #     transform=transforms,
        # )
        #

        logger.info("Creating TEST Dataset...")
        if self.config.preprocess.dataset == "alongtrack":
            ds_test = self.get_alongtrack_dataset(
                subset_time=self.config.evaluation,
                subset_spatial=self.config.evaluation,
            )
        elif self.config.preprocess.dataset == "natl60":
            ds_test = self.get_natl60_dataset(
                subset_time=self.config.evaluation,
                subset_spatial=self.config.evaluation,
                resample=self.config.evaluation,
            )
        else:
            raise ValueError(f"Unrecognized dataset: {self.config.preprocess.dataset}")
        # TODO: predict dataset
        logger.info(f"{len(ds_test):,} pts")

        # # TODO: create grid-coordinates
        # logger.info("Creating spatial-temporal grid...")
        # lon_coords, lat_coords, time_coords = create_spatiotemporal_grid(
        #     lon_min=self.config.evaluation.lon_min,
        #     lon_max=self.config.evaluation.lon_max,
        #     lon_dx=self.config.evaluation.dlon,
        #     lat_min=self.config.evaluation.lat_min,
        #     lat_max=self.config.evaluation.lat_max,
        #     lat_dy=self.config.evaluation.dlat,
        #     time_min=np.datetime64(self.config.evaluation.time_min),
        #     time_max=np.datetime64(self.config.evaluation.time_max),
        #     time_dt=np.timedelta64(
        #         self.config.evaluation.dt_freq, self.config.evaluation.dt_unit
        #     ),
        # )
        #
        # coords = pd.DataFrame(
        #     {
        #         "longitude": lon_coords,
        #         "latitude": lat_coords,
        #         "time": time_coords,
        #     }
        # )
        # # TODO: create dataset from grid coordinates
        # logger.info("Creating prediction dataset...")
        # ds_predict = AlongTrackDataset(
        #     ds=coords,
        #     spatial_columns=["longitude", "latitude"],
        #     temporal_columns=["time"],
        #     output_columns=None,
        #     transform=transforms,
        # )
        # # TODO: predict dataset

        ds_predict = self._get_dataset_eval()

        logger.info(f"{len(ds_predict):,} pts")
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
