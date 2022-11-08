import torch
import numpy as np

import pandas as pd


class TimeJulian:
    def __init__(self):
        pass

    def __call__(self, sample):
        x = sample["temporal"]
        shape = x.shape
        x = pd.to_datetime(x.flatten()).to_julian_date()
        x = np.asarray(x)
        x = x.reshape(*shape)
        sample["temporal"] = x
        return sample


class TimeJulianMinMax:
    def __init__(self, time_min: str = "2005-01-10", time_max: str = "2022-01-01"):
        self.time_min = time_min
        self.time_max = time_max

    def __call__(self, sample):
        x = sample["temporal"]
        shape = x.shape
        x = pd.to_datetime(x.flatten()).to_julian_date()
        time_min = pd.to_datetime(np.datetime64(self.time_min)).to_julian_date()
        time_max = pd.to_datetime(np.datetime64(self.time_max)).to_julian_date()
        x = (np.asarray(x) - np.asarray(time_min)) / (
            np.asarray(time_max) - np.asarray(time_min)
        )
        x = x.reshape(*shape)
        sample["temporal"] = x
        return sample


class TimeMinMax:
    def __init__(
        self,
        time_min: str = "2005-01-10",
        time_max: str = "2022-01-01",
        julian_time: bool = False,
    ):
        self.time_min = time_min
        self.time_max = time_max
        self.julian_time = julian_time

    def __call__(self, sample):
        x = sample["temporal"]
        time_min, time_max = np.datetime64(self.time_min), np.datetime64(self.time_max)
        x = (x - time_min) / (time_max - time_min)
        sample["temporal"] = x
        return sample


class ToTensor:
    def __init__(self):
        pass

    def __call__(self, sample):
        sample["spatial"] = torch.FloatTensor(sample["spatial"])
        sample["temporal"] = torch.FloatTensor(sample["temporal"])
        try:
            sample["output"] = torch.FloatTensor(sample["output"])
        except:
            pass
        return sample


def get_num_training(num_data: int, train_prct: float = 0.9):

    num_train = int(np.floor(num_data * train_prct))
    num_valid = int(np.ceil(num_data * (1 - train_prct)))

    assert num_train + num_valid == num_data

    return num_train, num_valid
