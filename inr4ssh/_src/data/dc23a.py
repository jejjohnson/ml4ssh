from typing import List, Optional
from ml_collections import config_dict
from dataclasses import dataclass
from inr4ssh._src.io import runcmd
from inr4ssh._src.files import list_all_files, get_subset_elements, get_subset_files_str
from tqdm import tqdm
import itertools

URL_OBS = "https://ige-meom-opendap.univ-grenoble-alpes.fr/thredds/fileServer/meomopendap/extract/JEJOHNSON/dc23_ose/raw/data_emmanuel.tar.gz"


@dataclass
class DC23aDataFiles:
    path: str
    altimeters = config_dict.ConfigDict()
    altimeters.dependent = [
        "c2",
        "h2a",
        "h2ag",
        "h2b",
        "j2",
        "j2g",
        "j2n",
        "j3",
        "s3a",
    ]
    altimeters.independent = [
        "sentinel3b",
        "altika",
    ]
    altimeters.grid = [
        "duacs",
    ]
    altimeters.all = altimeters.independent + altimeters.dependent
    years = [
        "2015",
        "2016",
        "2017",
        "2018",
        "2019",
    ]

    def get_altimeters(self, stage: str = "train"):
        if stage == "train":
            return self.altimeters.dependent
        elif stage == "evaluation":
            return self.altimeters.independent
        elif stage == "grid":
            return self.altimeters.grid
        elif stage == "all":
            return self.altimeters.dependent + self.altimeters.independent
        else:
            raise ValueError(f"Unrecognized stage: {stage}")

    def files_all(self):
        return list_all_files(self.path)

    def files_from_str(self, altimeter: str, year: Optional[str] = None):
        # TODO: ext=f"{altimeter}/{year}/**/*"

        files = list_all_files(self.path, ext=f"**/{altimeter}/**/*.nc")

        if year is not None:
            files = get_subset_files_str(files, year)

        return files

    def files_from_list(self, altimeters: List[str], years: Optional[List[str]] = None):
        files = []

        if years is not None:

            for ialtimeter, iyear in tqdm(list(itertools.product(altimeters, years))):
                files += self.files_from_str(altimeter=ialtimeter, year=iyear)
        else:
            for ialtimeter in tqdm(altimeters):
                files += self.files_from_str(altimeter=ialtimeter, year=None)

        return files

    def train_files_all(self):
        return self.files_from_list(altimeters=self.altimeters.dependent)

    def valid_files_all(self):
        return self.files_from_list(altimeters=self.altimeters.independent)

    def grid_files_all(self):
        return self.files_from_list(altimeters=self.altimeters.grid)


def download_obs(datadir: str):

    runcmd(f"wget --directory-prefix={datadir} {URL_OBS}")

    runcmd(f"tar -xvf {datadir}/data_raw.tar.gz --directory={datadir}")

    runcmd(f"rm -f {datadir}/data_raw.tar.gz")
