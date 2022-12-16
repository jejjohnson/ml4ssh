from typing import List
from pathlib import Path
from inr4ssh._src.io import runcmd
from ml_collections import config_dict
from inr4ssh._src.files import (
    get_subset_elements,
    check_list_equal_elem,
    list_all_files,
    check_if_file,
)
import json
from inr4ssh._src.paths import get_root_path

URL_OBS = "https://ige-meom-opendap.univ-grenoble-alpes.fr/thredds/fileServer/meomopendap/extract/ocean-data-challenges/dc_data1/dc_obs.tar.gz"
URL_REF = "https://ige-meom-opendap.univ-grenoble-alpes.fr/thredds/fileServer/meomopendap/extract/ocean-data-challenges/dc_data1/dc_ref.tar.gz"


def download_obs(datadir: str) -> None:

    runcmd(f"wget --directory-prefix={datadir} {URL_OBS}")

    runcmd(f"tar -xvf {datadir}/dc_obs.tar.gz --directory={datadir}")

    runcmd(f"rm -f {datadir}/dc_obs.tar.gz")

    return None


def download_ref(datadir: str) -> None:

    runcmd(f"wget --directory-prefix={datadir} {URL_REF}")

    runcmd(f"tar -xvf {datadir}/dc_ref.tar.gz --directory={datadir}")

    runcmd(f"rm -f {datadir}/dc_ref.tar.gz")

    return None


def get_osse_2020a_setup() -> config_dict.ConfigDict:
    config = config_dict.ConfigDict()

    # SUBSET Arguments
    config.raw_nadir = [
        "2020a_SSH_mapping_NATL60_topex-poseidon_interleaved.nc",
        "2020a_SSH_mapping_NATL60_envisat.nc",
        "2020a_SSH_mapping_NATL60_geosat2.nc",
        "2020a_SSH_mapping_NATL60_jason1.nc",
    ]

    config.raw_swot = [
        "2020a_SSH_mapping_NATL60_karin_swot.nc",
    ]
    config.raw_swotnadir = [
        "2020a_SSH_mapping_NATL60_nadir_swot.nc",
    ]

    # SUBSET Arguments
    config.nadir4 = [
        "2020a_SSH_mapping_NATL60_topex-poseidon_interleaved.nc",
        "2020a_SSH_mapping_NATL60_envisat.nc",
        "2020a_SSH_mapping_NATL60_geosat2.nc",
        "2020a_SSH_mapping_NATL60_jason1.nc",
    ]

    config.nadir1 = ["2020a_SSH_mapping_NATL60_jason1.nc"]

    config.swot1 = [
        "2020a_SSH_mapping_NATL60_karin_swot.nc",
        "2020a_SSH_mapping_NATL60_nadir_swot.nc",
    ]
    config.swot1nadir1 = [
        "2020a_SSH_mapping_NATL60_karin_swot.nc",
        "2020a_SSH_mapping_NATL60_nadir_swot.nc",
    ]
    config.swot1nadir5 = [
        "2020a_SSH_mapping_NATL60_karin_swot.nc",
        "2020a_SSH_mapping_NATL60_nadir_swot.nc",
        "2020a_SSH_mapping_NATL60_topex-poseidon_interleaved.nc",
        "2020a_SSH_mapping_NATL60_envisat.nc",
        "2020a_SSH_mapping_NATL60_geosat2.nc",
        "2020a_SSH_mapping_NATL60_jason1.nc",
    ]

    return config


def get_swot_obs_setup_files(files: List[str], setup: str = "nadir1"):

    # initialize setup config
    setup_config = get_osse_2020a_setup()

    # get specific scenario
    setup_filenames = setup_config[setup]

    setup_files = get_subset_elements(setup_filenames, files)

    assert len(setup_files) == len(setup_filenames)

    return setup_files


def check_dc20a_files(
    directory: str, json_file: dict = None, dataset: str = "obs"
) -> bool:

    if json_file is None:
        json_file = get_root_path().joinpath("inr4ssh/_src/data/dc20a.json")

    check_if_file(json_file)

    # load json directory
    with open(json_file, "r") as f:
        json_file_list = json.load(f)

    # get files in directory
    obs_files = list_all_files(directory, ext="*.nc", full_path=False)

    # check if files are the same
    return check_list_equal_elem(obs_files, json_file_list[dataset])


def get_raw_altimetry_files(obs_dir: str = None, dataset: str = "nadir") -> List[str]:
    """
    Experiments:
    * "nadir"
    * "swot"
    * "swotnadir"
    """

    data_config = get_osse_2020a_setup()

    if obs_dir is None:
        obs_dir = ""

    dataset = "raw_" + dataset
    obs_files = list(map(lambda x: Path(obs_dir).joinpath(x), data_config[dataset]))

    return obs_files
