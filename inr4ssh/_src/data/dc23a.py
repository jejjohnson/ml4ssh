from inr4ssh._src.io import runcmd

URL_OBS = "https://ige-meom-opendap.univ-grenoble-alpes.fr/thredds/fileServer/meomopendap/extract/JEJOHNSON/dc23_ose/raw/data_emmanuel.tar.gz"


def download_obs(datadir: str):

    runcmd(f"wget --directory-prefix={datadir} {URL_OBS}")

    runcmd(f"tar -xvf {datadir}/data_raw.tar.gz --directory={datadir}")

    runcmd(f"rm -f {datadir}/data_raw.tar.gz")
