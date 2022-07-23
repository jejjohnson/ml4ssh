from typing import Dict, Any, cast
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl


class ImageModel(pl.LightningModule):
    def __init__(self, model, **kwargs):
        super().__init__()

        self.save_hyperparameters()
        self.model = model
        self.hyperparams = cast(Dict[str, Any], self.hparams)
        self.loss = nn.MSELoss(reduction="mean")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # output
        x, y = batch

        pred = self.forward(x)

        # loss function
        loss = self.loss(pred, y)

        self.log("train_loss", loss)
        self.log("psnr", -10 * torch.log10(2.0 * loss))

        return loss

    def validation_step(self, batch, batch_idx):
        # output
        x, y = batch

        pred = self.forward(x)

        # loss function
        loss = self.loss(pred, y)

        self.log("val_loss", loss)

        return loss

    def test_step(self, batch, batch_idx):
        # output
        x, y = batch

        pred = self.forward(x)

        # loss function
        loss = self.loss(pred, y)

        self.log("test_loss", loss)

        return loss

    def predict_step(self, batch, batch_idx):
        # output
        x, y = batch

        pred = self.forward(x)

        return pred

    def configure_optimizers(self):

        optimizer = Adam(self.model.parameters(), lr=self.hyperparams.get("lr", 1e-4))
        scheduler = ReduceLROnPlateau(
            optimizer, patience=self.hyperparams.get("lr_schedule_patience", 100)
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
