import numpy as np


def add_noise(data, sigma: float=0.01, noise: str="gauss", seed: int=123, df: int=3):
    
    rng = np.random.RandomState(seed)
    
    if noise == "gauss":
        return data + sigma * rng.standard_normal(size=data.shape)
    elif noise == "cauchy":
        return data + sigma * rng.standard_cauchy(size=data.shape)
        
    elif noise == "tstudent":
        return data + sigma * rng.standard_t(df=df, size=data.shape)
    
    elif noise == "exp":
        return data + sigma * rng.standard_exponential(size=data.shape)
    else:
        raise ValueError(f"Unrecognized noise: {noise}")