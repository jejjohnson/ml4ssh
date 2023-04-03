import pytorch_lightning as pl
from loguru import logger


class ATDataModule(pl.LightningDataModule):
    def __init__(self, subset_time: str) -> None:
        self.data_config = data_config

    def prepare_download(self):
        pass

    def setup(self) -> None:
        self._setup()

    def _setup(self) -> None:
        logger.info("Opening xarray dataset...")
