import datetime
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import torch
from einops import rearrange


class Spherical2Cartesian3D(BaseEstimator, TransformerMixin):
    def __init__(self, radius: float=6371.010, units: str="degrees"):
        self.radius = radius
        self.units = units
        
    def fit(self, X: pd.DataFrame(), y=None):
        return self

    
    def transform(self, X, y=None):
        
        
        lon = X["longitude"]
        lat = X["latitude"]
        
        if self.units == "degrees":
            lon = np.deg2rad(lon)
            lat = np.deg2rad(lat)
            
                        
        x, y, z = spherical_to_cartesian_3d(
            lon=lon, lat=lat, radius=self.radius
        )
        
        X = np.stack([x,y,z], axis=1)
        
        
        return X
    
    def inverse_transform(self, X: pd.DataFrame(), y=None):
        
        
        lon, lat, _ = cartesian_to_spherical_3d(
            x=X["x"], y=X["y"], z=X["z"]
        )
        X = np.stack([lon, lat], axis=1)
        
        return X


class Cartesian3D2Spherical(Spherical2Cartesian3D):
    def __init__(self, radius: float=6371.010):
        super().__init__(radius=radius)
    

    
    def transform(self, X: pd.DataFrame(), y=None):
        
        X = super().inverse_transform(X=X, y=y)
        
        return X
    
    def inverse_transform(self, X: pd.DataFrame(), y=None):
        
        X = super().transform(X=X, y=y)
        
        return X



    
def spherical_to_cartesian_3d(lon, lat, radius: float=6371.010):
    
    x = radius * np.cos(lat) * np.cos(lon)
    y = radius * np.cos(lat) * np.sin(lon)
    z = radius * np.sin(lat)
    
    return x, y, z


def cartesian_to_spherical_3d(x, y, z):
    
    radius = np.sqrt(x**2 + y**2 + z**2)
    lon = np.arctan2(y, x)
    lat = np.arcsin( z / radius)
    
    return lon, lat, radius