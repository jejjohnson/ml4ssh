import pytorch_lightning as pl
import torch

from ..data.images import load_fox, load_cameraman
from ..features.coords import get_image_coordinates
from torch.utils.data import random_split, DataLoader, TensorDataset
from einops import rearrange


class Image(pl.LightningDataModule):
    def __init__(self, batch_size: int=32, shuffle: bool=False, split_style: str="regular"):
        super().__init__()
        self.batch_size = batch_size
        self.split_style = split_style
        self.shuffle = shuffle

    def setup(self, stage=None):
        img = self.load_image()
        coords, pixel_vals = self.image_2_coordinates(img)
        coords, pixel_vals = torch.FloatTensor(coords), torch.FloatTensor(pixel_vals)

        coords_train, pixels_train = coords[::2], pixel_vals[::2]

        self.ds_train = TensorDataset(coords_train, pixels_train)

        coords_valid, pixels_valid = coords[1::2], pixel_vals[1::2]

        self.ds_valid = TensorDataset(coords_valid, pixels_valid)

        self.ds_test = TensorDataset(coords, pixel_vals)

        return self

    def load_image(self):
        raise NotImplementedError

    def coordinates_2_image(self, coords):
        return rearrange(coords, "(h w) c -> h w c", h=self.image_height, w=self.image_width)

    def image_2_coordinates(self, image):
        return get_image_coordinates(image)

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        return DataLoader(self.ds_valid, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.batch_size)


class ImageFox(Image):
    image_height = 512
    image_width = 512

    def load_image(self):
        return load_fox()

class ImageCameraman(Image):
    image_height = 512
    image_width = 512
    def load_image(self):
        return load_cameraman()

