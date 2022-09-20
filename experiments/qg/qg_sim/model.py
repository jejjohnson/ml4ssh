import torch
from torch.optim import Adam
from typing import Dict, Any, cast
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import torch.nn as nn
import pytorch_lightning as pl


class INRModel(pl.LightningModule):
    def __init__(
        self,
        model,
        loss_data,
        reg_pde,
        qg: bool = True,
        lr: float = 1e-4,
        warmup: int = 50,
        num_epochs: int = 300,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["model", "reg_pde", "loss_data"])
        self.model = model
        self.hyperparams = cast(Dict[str, Any], self.hparams)
        self.loss_data = loss_data
        self.reg_pde = reg_pde

    def forward(self, x):
        return self.model(x)

    def _data_loss(self, batch):
        x, y = batch

        pred = self.forward(x)

        # parse inputs
        x, y = batch

        # data loss function
        loss = self.loss_data(y, pred)

        return loss

    def _qg_loss(self, batch):

        x, y = batch

        loss = self.reg_pde.forward(x, self.model)

        return loss

    def training_step(self, batch, batch_idx):

        # loss function
        loss_data = self._data_loss(batch)

        if self.hyperparams.get("qg", False):
            # x_var = torch.autograd.Variable(x, requires_grad=True)
            # out = self.forward(x_var)
            # reg = qg_loss(out, x_var, 1.0, 1.0, 1.0, "mean")
            loss_reg = self._qg_loss(batch)

            loss = loss_data + loss_reg

            self.log("train_reg", loss_reg, prog_bar=True)
            self.log("train_data", loss_data, prog_bar=True)
        else:
            loss = loss_data

        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):

        # loss function
        loss_data = self._data_loss(batch)

        if self.hyperparams.get("qg", False):
            # x_var = torch.autograd.Variable(x, requires_grad=True)
            # out = self.forward(x_var)
            # reg = qg_loss(out, x_var, 1.0, 1.0, 1.0, "mean")
            loss_reg = self._qg_loss(batch)

            loss = loss_data + loss_reg

            self.log("val_reg", loss_reg, prog_bar=True)
            self.log("val_data", loss_data, prog_bar=True)
        else:
            loss = loss_data

        self.log("val_loss", loss, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        with torch.set_grad_enabled(True):

            # loss function
            loss_data = self._data_loss(batch)

            if self.hyperparams.get("qg", False):
                # x_var = torch.autograd.Variable(x, requires_grad=True)
                # out = self.forward(x_var)
                # reg = qg_loss(out, x_var, 1.0, 1.0, 1.0, "mean")
                loss_reg = self._qg_loss(batch)

                loss = loss_data + loss_reg

                self.log("test_reg", loss_reg, prog_bar=True)
                self.log("test_data", loss_data, prog_bar=True)
            else:
                loss = loss_data

            self.log("test_loss", loss, prog_bar=True)

        return loss

    def predict_step(self, batch, batch_idx):
        # output
        x, y = batch

        pred = self.forward(x)

        return pred

    def configure_optimizers(self):

        # configure optimizer
        optimizer = Adam(self.model.parameters(), lr=self.hyperparams.get("lr", 1e-4))

        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.hyperparams.get("warmup", 10),
            max_epochs=self.hyperparams.get("num_epochs", 100),
            warmup_start_lr=self.hyperparams.get("warmup_start_lr", 1e-6),
            eta_min=self.hyperparams.get("eta_min", 1e-6),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
