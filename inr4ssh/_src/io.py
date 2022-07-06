from typing import Callable, Optional, Dict, Any, List
import pickle
import tqdm
import xarray as xr
from dataclasses import asdict


def load_multiple_nc_files(files: List[str]) -> List:
    list_of_datasets = []

    for ifile in tqdm.tqdm(files):
        ids = xr.open_dataset(ifile)
        list_of_datasets.append(ids)

    return list_of_datasets

def save_object(model, path):
    with open(path, 'wb') as file:
        pickle.dump(model, file)
    return None

def load_object(path):
    with open(path, "rb") as file:
        return pickle.load(file)


def transform_dict(config_dict: Dict, expand: bool = True):
    """
    General function to transform any dictionary into wandb config acceptable format
    (This is mostly due to datatypes that are not able to fit into YAML format which makes wandb angry)
    The expand argument is used to expand iterables into dictionaries so that these configs can be used when compare across runs
    """
    ret: Dict[str, Any] = {}
    for k, v in config_dict.items():
        if v is None or isinstance(v, (int, float, str)):
            ret[k] = v
        elif isinstance(v, (list, tuple, set)):
            # Need to check if item in iterable is YAML-friendly
            t = transform_dict(dict(enumerate(v)), expand)
            # Transform back to iterable if expand is False
            ret[k] = t if expand else [t[i] for i in range(len(v))]
        elif isinstance(v, dict):
            ret[k] = transform_dict(v, expand)
        else:
            # Transform to YAML-friendly (str) format
            # Need to handle both Classes, Callables, Object Instances
            # Custom Classes might not have great __repr__ so __name__ might be better in these cases
            vname = v.__name__ if hasattr(v, '__name__') else v.__class__.__name__
            ret[k] = f"{v.__module__}:{vname}"
    return ret

def simpleargs_2_ndict(args):

    params_dict = {}
    for i in vars(args):
        params_dict[i] = asdict(getattr(args, i))
    return params_dict
