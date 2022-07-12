from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


class MinMaxFixedScaler(BaseEstimator, TransformerMixin):
    def __init__(self, min_val: np.ndarray, max_val: np.ndarray):
        self.min_val = np.asarray(min_val)
        self.max_val = np.asarray(max_val)
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        
        return (X - self.min_val) / (self.max_val - self.min_val)
    
    def inverse_transform(self, X, y=None):
        
        return X * (self.max_val - self.min_val) + self.min_val
    

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
    
    
class TimeMinMaxScaler(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        time_min: str="2005-01-01",
        time_max: str="2022-01-01",
        julian_date: bool=True,
    ):
        
        self.time_min = time_min
        self.time_max = time_max
        self.julian_date = julian_date
        

    def fit(self, X: pd.DataFrame(), y=None):
        return self

    
    def transform(self, X, y=None):
        
        X = X["time"]

        time_min, time_max = np.datetime64(self.time_min), np.datetime64(self.time_max)
        
            
        if self.julian_date:
            X = pd.DatetimeIndex(X).to_julian_date().copy()
            
            time_min = pd.DatetimeIndex([time_min]).to_julian_date()
            time_max = pd.DatetimeIndex([time_max]).to_julian_date()
            
        
        X = (X - time_min) / (time_max - time_min)
        

        return X.values[:, None]