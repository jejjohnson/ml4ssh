import datetime
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import torch
from einops import rearrange

class JulianDateTransform(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.fmt ='%Y-%m-%d'
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        # convert to julian values
        X_jdate = pd.DatetimeIndex(X).to_julian_date().values
        
        # return 2D array
        return X_jdate[:, None]

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
        
        
class TimeMinMaxScaler(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        time_min: str=np.datetime64("2005-01-01"),
        time_max: str=np.datetime64("2022-01-01"),
        julian_date: bool=True,
    ):
        self.time_min = time_min
        self.time_max = time_max
        self.julian_date = julian_date
        
        return None
        
    def fit(self, X: pd.DataFrame(), y=None):
        return self

    
    def transform(self, X, y=None):
        
        X = X["time"]
        
        if not self.julian_date:
            
            time_min, time_max = self.time_min, self.time_max
            
        else:
            X = pd.DatetimeIndex(X).to_julian_date().copy()
            
            time_min = pd.DatetimeIndex([self.time_min]).to_julian_date()
            time_max = pd.DatetimeIndex([self.time_max]).to_julian_date()
            
        
        X = (X - time_min) / (time_max - time_min)
        

        return X.values[:, None]
    
    
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


def get_image_coordinates(image: torch.Tensor, min_val: int=-1, max_val: int=1):
    # get image size
    image_height, image_width, _ = image.shape

    # get all coordinates
    coordinates = [
        torch.linspace(min_val, max_val, steps=image_height),
        torch.linspace(min_val, max_val, steps=image_width)
    ]

    # create meshgrid of pairwise coordinates
    coordinates = torch.meshgrid(*coordinates, indexing="ij")

    # stack tensors together
    coordinates = torch.stack(coordinates, dim=-1)

    # rearrange to coordinate vector
    coordinates = rearrange(coordinates, "h w c -> (h w) c")
    pixel_values = rearrange(image, "h w c -> (h w) c")

    return coordinates, pixel_values


# def split_image_data(seed: Optional[int]=123):
#
#     return x_train, y_train, x_valid, y_valid, x_test, y_test