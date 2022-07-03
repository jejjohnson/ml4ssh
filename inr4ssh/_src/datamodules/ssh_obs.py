import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
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

class SSHAltimetry(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
    def setup(self, stage=None):
        
        logger.info("Getting training data...")
        X, y, scaler = self.get_train_data(self.config)
        
        self.scaler = scaler
        logger.info("Train/Validation Split...")
        xtrain, ytrain, xvalid, yvalid = self.split_train_data(X, y, self.config)
        
        logger.info("Getting evalulation data...")
        X = self.get_eval_data(self.config)
        
        logger.info("scaling evaluation data...")
        X = scaler.transform(X)
        
        logger.info("Creating dataloaders...")
        self.ds_train = TensorDataset(
            torch.FloatTensor(xtrain), 
            torch.FloatTensor(ytrain)
        )
        self.ds_valid = TensorDataset(
            torch.FloatTensor(xvalid),
            torch.FloatTensor(yvalid)
        )
        self.ds_predict = TensorDataset(
            torch.FloatTensor(X)
        )
        
        return self
    
    @staticmethod
    def get_train_data(config):
        # load data from train dir
        logger.info("loading data...")
        ds = load_ssh_altimetry_data_train(config.train_data_dir)
        
        logger.info("subsetting data...")
        ds = preprocess_data(ds, config)
        
        logger.info("getting feature scaler...")
        scaler = get_feature_scaler(config)
        
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
            X, y, 
            train_size=config.train_size, 
            random_state=config.seed_split
        )
        
        return xtrain, ytrain, xvalid, yvalid
        
        
    
    def train_dataloader(self):
        return DataLoader(
            self.ds_train, 
            batch_size=self.config.batch_size, 
            shuffle=self.config.train_shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
    def val_dataloader(self):
        return DataLoader(
            self.ds_valid, 
            batch_size=self.config.batch_size, 
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
    def predict_dataloader(self):
        return DataLoader(
            self.ds_predict, 
            batch_size=self.config.batch_size, 
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=False
        )
    
    


def preprocess_data(ds_obs, config):
    
    time_min = np.datetime64(config.time_min)
    time_max = np.datetime64(config.time_max)
    dtime = np.timedelta64(*config.dtime.split("_"))
    # temporal subset
    ds_obs = temporal_subset(
        ds_obs,
        time_min=time_min,
        time_max=time_max,
        time_buffer=config.time_buffer)


    # convert to dataframe
    data = ds_obs.to_dataframe().reset_index().dropna()
    
    # add vtime 
    # NOTE: THIS IS THE GLOBAL MINIMUM TIME FOR THE DATASET
    data["vtime"] = (data['time'].values - np.datetime64("2016-12-01")) / dtime
    
    # add column attributes
    data.attrs["input_cols"] = ["longitude", "latitude", "time"]
    data.attrs["output_cols"] = ["sla_unfiltered"]
    
    return data


def get_evalulation_data(config):
    
    # create spatiotemporal grid
    lon_coords, lat_coords, time_coords = create_spatiotemporal_grid(
        lon_min=config.eval_lon_min,
        lon_max=config.eval_lon_max,
        lon_dx=config.eval_dlon,
        lat_min=config.eval_lat_min,
        lat_max=config.eval_lat_max,
        lat_dy=config.eval_dlat,
        time_min=np.datetime64(config.eval_time_min),
        time_max=np.datetime64(config.eval_time_max),
        time_dt=np.timedelta64(*config.eval_dtime.split("_"))
    )
    
    df_grid = pd.DataFrame({
        "longitude": lon_coords,
        "latitude": lat_coords,
        "time": time_coords,
    })
    
    return df_grid

def get_feature_scaler(config):
    
    spatial_features = ["longitude", "latitude"]
    temporal_features = ["time"]
    
    # spatial transform
    spatial_transform = Pipeline([
        ("cartesian3d", Spherical2Cartesian3D(radius=config.spherical_radius))
    ])

    spatial_features = ["longitude", "latitude"]

    # temporal transform
    temporal_transform = Pipeline([
        ("timestd", TimeMinMaxScaler(julian_date=config.julian_time)),
    ])

    temporal_features = ["time"]
    
    scaler = ColumnTransformer(
        transformers=[
            ("cartesian3d", spatial_transform, spatial_features),
            ("timeminmax", temporal_transform, temporal_features),
        ],
        remainder="drop",
    )

    return scaler