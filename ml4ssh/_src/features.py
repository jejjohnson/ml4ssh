import datetime
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

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