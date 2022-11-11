from ml_collections import config_dict
import math
from typing import Optional
from inr4ssh._src.transforms.spatial import SpatialDegree2Rads
from inr4ssh._src.transforms import Identity


def default_transforms_spatial_config():
    config = config_dict.ConfigDict()
    config.transform = "deg2rad"
    config.scaler = [1.0 / math.pi, 1 / (math.pi / 2.0)]

    return config


def default_transforms_temporal_config():
    config = config_dict.ConfigDict()
    config.transform = "identity"

    return config


def spatial_transform_factory(config: Optional[config_dict.ConfigDict] = None):
    if config is None:
        config = default_transforms_spatial_config()

    if config.transform == "deg2rad":
        return SpatialDegree2Rads(scaler=config.scaler)
    elif config.transform == "identity":
        return Identity()
    else:
        raise ValueError(f"Unrecognized transformation")


def temporal_transform_factory(config: Optional[config_dict.ConfigDict] = None):
    if config is None:
        config = default_transforms_temporal_config()

    if config.transform == "identity":
        return Identity()
    else:
        raise ValueError(f"Unrecognized transformation")
