from pathlib import Path
from inr4ssh._src.io import check_if_file, check_if_directory, runcmd
from inr4ssh._src.data.dc21a import download_obs, download_correction, download_results
import yaml
from loguru import logger


def download(datadir: str, creds_file: str, dataset: str = "obs") -> None:
    """The script to download the datasets.

    Args:
        datadir (str): the directory to store the dataset
        dataset (str):
            options = {"obs", "correction", "results"}
    """

    # check if directory
    dataset = dataset.lower()

    logger.debug(f"Data directory: {datadir}")
    check_if_directory(datadir)

    # check if credentials file exists
    logger.debug(f"Credentials file: {creds_file}")
    check_if_file(creds_file)

    datadir = Path(datadir).joinpath("raw")
    datadir.mkdir(parents=True, exist_ok=True)

    # extract credentials from file
    logger.info("Loading yaml file...")
    with open(creds_file, "r") as file:
        creds = yaml.safe_load(file)
        username = creds["username"]
        password = creds["password"]

    if dataset == "obs":
        # create obs directory
        datadir = Path(datadir).joinpath("obs")
        datadir.mkdir(parents=True, exist_ok=True)

        # download the data
        logger.info("Downloading altimetry tracks...")
        download_obs(str(datadir), username=username, password=password)

    elif dataset == "correction":
        # create obs directory
        datadir = Path(datadir).joinpath("correction")
        datadir.mkdir(parents=True, exist_ok=True)

        # download the data
        logger.info("Downloading correction dataset...")
        download_correction(str(datadir), username=username, password=password)

    elif dataset == "evaluation":
        # check directory
        # make reference directory
        # download the data
        raise NotImplementedError()

    elif dataset == "results":
        # create obs directory
        datadir = Path(datadir).joinpath("results")
        datadir.mkdir(parents=True, exist_ok=True)

        # download the data
        logger.info("Downloading results dataset...")
        download_results(str(datadir), username=username, password=password)

    else:
        raise ValueError(f"Unrecognized dataset: {dataset}")
