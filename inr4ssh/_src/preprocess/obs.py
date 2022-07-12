import numpy as np


def add_noise(ds_var, sigma: float=0.01, noise: str="gauss", seed: int=123, df: int=3):
    
    rng = np.random.RandomState(seed)
    
    if noise == "gauss":
        return ds_var + sigma * rng.standard_normal(size=ds_var.shape)
    elif noise == "cauchy":
        return ds_var + sigma * rng.standard_cauchy(size=ds_var.shape)
        
    elif noise == "tstudent":
        return ds_var + sigma * rng.standard_t(df=df, size=ds_var.shape)
    
    elif noise == "exp":
        return ds_var + sigma * rng.standard_exponential(size=ds_var.shape)
    else:
        raise ValueError(f"Unrecognized noise: {noise}")