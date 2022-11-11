import pytorch_lightning as pl
from typing import Dict, Any, cast
import torch.nn as nn
import torch
from inr4ssh._src.trainers.optimizers import optimizer_factory
from inr4ssh._src.trainers.lr_schedulers import lr_scheduler_factory
from inr4ssh._src.losses import loss_factory


class INRModel(pl.LightningModule):
    def __init__(
        self,
        model,
        spatial_transform=None,
        temporal_transform=None,
        optimizer_config=None,
        lr_scheduler_config=None,
        loss_config=None,
        **kwargs,
    ):
        super().__init__()

        # self.save_hyperparameters()
        self.model = model
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.optimizer_config = optimizer_config
        self.lr_scheduler_config = lr_scheduler_config
        self.hyperparams = cast(Dict[str, Any], self.hparams)
        self.loss_fn = loss_factory(loss_config)

    def forward(self, x):
        return self.model(x)

    def _data_loss(self, batch):
        x, y = self._extract_spacetime(batch=batch, outputs=True)

        pred = self.forward(x)

        # data loss function
        loss = self.loss_fn(y, pred)

        return loss

    def _extract_spacetime(self, batch, outputs=False):

        x_space, x_time = batch["spatial"], batch["temporal"]

        if self.spatial_transform is not None:
            x_space = self.spatial_transform(x_space)

        if self.temporal_transform is not None:
            x_time = self.temporal_transform(x_time)

        x = torch.cat([x_space, x_time], dim=1)

        if outputs:
            return x, batch["output"]
        else:
            return x

    def training_step(self, batch, batch_idx):

        # loss function
        loss = self._data_loss(batch)

        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):

        # loss function
        loss = self._data_loss(batch)

        self.log("val_loss", loss, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):

        # loss function
        loss = self._data_loss(batch)

        self.log("test_loss", loss, prog_bar=True)

        return loss

    def predict_step(self, batch, batch_idx):
        # output
        x = self._extract_spacetime(batch=batch, outputs=False)

        pred = self.forward(x)

        return pred

    def configure_optimizers(self):

        # configure optimizer
        optimizer = optimizer_factory(self.optimizer_config)(self.model.parameters())
        # configure scheduler
        scheduler = lr_scheduler_factory(self.lr_scheduler_config)(optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
