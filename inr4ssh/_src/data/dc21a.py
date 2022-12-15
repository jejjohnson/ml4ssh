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
from tqdm import tqdm
import json
from inr4ssh._src.paths import get_root_path

URL_OBS_BASE = "https://tds.aviso.altimetry.fr/thredds/fileServer/2021a-SSH-mapping-OSE-along-track-data/"
URL_OBS_ALG = "dt_gulfstream_alg_phy_l3_20161201-20180131_285-315_23-53.nc"
URL_OBS_C2 = "dt_gulfstream_c2_phy_l3_20161201-20180131_285-315_23-53.nc"
URL_OBS_H2G = "dt_gulfstream_h2g_phy_l3_20161201-20180131_285-315_23-53.nc"
URL_OBS_J2G = "dt_gulfstream_j2g_phy_l3_20161201-20180131_285-315_23-53.nc"
URL_OBS_J2N = "dt_gulfstream_j2n_phy_l3_20161201-20180131_285-315_23-53.nc"
URL_OBS_J3 = "dt_gulfstream_j3_phy_l3_20161201-20180131_285-315_23-53.nc"
URL_OBS_S3A = "dt_gulfstream_s3a_phy_l3_20161201-20180131_285-315_23-53.nc"

URL_MAP_BASE = (
    "https://tds.aviso.altimetry.fr/thredds/fileServer/2021a-SSH-mapping-OSE-grid-data/"
)
URL_MDT = "mdt.nc"

URL_RESULTS = {
    "duacs": "OSE_ssh_mapping_DUACS.nc",
    "oi": "OSE_ssh_mapping_BASELINE.nc",
    "bfn": "OSE_ssh_mapping_BFN.nc",
    "miost": "OSE_ssh_mapping_MIOST.nc",
    "mdt": "mdt.nc",
    "dymost": "OSE_ssh_mapping_DYMOST.nc",
    "4dvar": "OSE_ssh_mapping_4dvarNet.nc",
    "4dvar_2022": "OSE_ssh_mapping_4dvarNet_2022.nc",
}


def download_obs(datadir: str, username: str, password: str):
    download_fn = lambda url: runcmd(
        f"wget --user={username} --password={password} --directory-prefix={datadir} {URL_OBS_BASE}{url}",
        verbose=False,
    )

    URLS = [
        URL_OBS_ALG,
        URL_OBS_C2,
        URL_OBS_H2G,
        URL_OBS_J2G,
        URL_OBS_J2N,
        URL_OBS_J3,
        URL_OBS_S3A,
    ]
    with tqdm(URLS) as pbar:
        for iurl in pbar:

            pbar.set_description(f"Downloading: {iurl}")
            download_fn(iurl)

    return None


def download_correction(datadir: str, username: str, password: str):

    runcmd(
        f"wget --user={username} --password={password} --directory-prefix={datadir} {URL_MAP_BASE}{URL_MDT}",
        verbose=False,
    )

    return None


def download_results(datadir: str, username: str, password: str, dataset: str = "all"):

    if not dataset == "all":
        url = URL_RESULTS[dataset]
        runcmd(
            f"wget --user={username} --password={password} --directory-prefix={datadir} {URL_MAP_BASE}{url}",
            verbose=False,
        )
    else:
        with tqdm(URL_RESULTS.values()) as pbar:
            for iurl in pbar:
                pbar.set_description(f"Download: {iurl}")
                runcmd(
                    f"wget --user={username} --password={password} --directory-prefix={datadir} {URL_MAP_BASE}{iurl}",
                    verbose=False,
                )

    return None
