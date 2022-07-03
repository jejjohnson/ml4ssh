from typing import List
import pickle
import tqdm
import xarray as xr


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