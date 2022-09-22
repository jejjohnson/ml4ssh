from inr4ssh._src.models.activations import get_activation
from inr4ssh._src.models.siren import SirenNet, ModulatedSirenNet
from inr4ssh._src.models.mfn import FourierNet, GaborNet
from optimizers import optimizer_factory, lr_scheduler_factory
from losses import loss_factory
import pytorch_lightning as pl
import torch.nn as nn


def model_factory(model, dim_in, dim_out, config):

    if model == "siren":
        return SirenNet(
            dim_in=dim_in,
            dim_hidden=config.hidden_dim,
            dim_out=dim_out,
            num_layers=config.num_layers,
            w0=config.w0,
            w0_initial=config.w0_initial,
            use_bias=config.use_bias,
            c=config.c,
            final_activation=get_activation(config.final_activation),
        )
    elif model == "modsiren":
        return ModulatedSirenNet(
            dim_in=dim_in,
            dim_hidden=config.hidden_dim,
            dim_out=dim_out,
            num_layers=config.num_layers,
            w0=config.w0,
            w0_initial=config.w0_initial,
            c=config.c,
            final_activation=get_activation(config.final_activation),
            latent_dim=config.latent_dim,
            num_layers_latent=config.num_layers,
            operation=config.operation,
        )

    elif model == "fouriernet":
        return FourierNet(
            dim_in=dim_in,
            dim_out=dim_out,
            dim_hidden=config.hidden_dim,
            num_layers=config.num_layers,
            input_scale=config.input_scale,
            weight_scale=config.weight_scale,
            use_bias=config.use_bias,
            final_activation=get_activation(config.final_activation),
        )
    elif model == "gabornet":
        return GaborNet(
            dim_in=dim_in,
            dim_out=dim_out,
            dim_hidden=config.hidden_dim,
            num_layers=config.num_layers,
            input_scale=config.input_scale,
            weight_scale=config.weight_scale,
            alpha=config.alpha,
            beta=config.beta,
            use_bias=config.use_bias,
            final_activation=get_activation(config.final_activation),
        )
    else:
        raise ValueError(f"Unrecognized model: {model}")


class CoordinatesLearner(pl.LightningModule):
    def __init__(self, model: nn.Module, params_loss, params_optim, params_lr):
        super().__init__()
        self.model = model
        self.loss = loss_factory(params_loss)
        self.params_optim = params_optim
        self.params_lr = params_lr

    def forward(self, x):
        return self.model(x)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):

        (x,) = batch

        pred = self.forward(x)

        return pred

    def training_step(self, batch, batch_idx):
        x, y = batch
        # loss function
        pred = self.forward(x)
        loss = self.loss(pred, y)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        # loss function
        pred = self.forward(x)
        loss = self.loss(pred, y)

        self.log("valid_loss", loss)

        return loss

    def configure_optimizers(self):

        optimizer = optimizer_factory(self.params_optim)(params=self.model.parameters())

        scheduler = lr_scheduler_factory(self.params_lr)(optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "valid_loss",
        }
