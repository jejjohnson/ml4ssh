import sys, os


from pyprojroot import here

# spyder up to find the root
root = here(project_files=[".root"])

# append to path
sys.path.append(str(root))

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pathlib import Path
import wandb
import numpy as np
import pandas as pd
from data import QGSimulation
from losses import RegQG, initialize_data_loss
from model import INRModel
from inr4ssh._src.models.siren import SirenNet
import ml_collections
from figures import plot_maps
import tqdm
from loguru import logger

seed_everything(123)


def initialize_siren_model(config, x_init, y_init):

    net = SirenNet(
        dim_in=x_init.shape[1],
        dim_out=y_init.shape[1],
        dim_hidden=config.dim_hidden,
        num_layers=config.num_layers,
        w0=config.w0,
        w0_initial=config.w0_initial,
        c=config.c,
        final_activation=config.final_activation,
    )

    return net


def initialize_callbacks(config, save_dir):
    # model checkpoints
    model_cb = ModelCheckpoint(
        dirpath=str(Path(save_dir).joinpath("checkpoints")),
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    callbacks = [
        model_cb,
    ]

    return callbacks


def train(config: ml_collections.ConfigDict, workdir: str, savedir: str):

    # initialize logger
    logger.info("Initializaing Logger...")
    wandb_logger = WandbLogger(
        config=config.to_dict(),
        mode=config.log.mode,
        project=config.log.project,
        entity=config.log.entity,
        dir=config.log.log_dir,
        resume=False,
    )

    # initialize dataloader
    logger.info("Initializing data module...")
    dm = QGSimulation(config)
    dm.setup()

    x_init, y_init = dm.ds_train[:10]

    # initialize model
    net = initialize_siren_model(config.model, x_init, y_init)

    # initialize regularization
    reg_loss = RegQG(config.loss.alpha)

    # initialize dataloss
    data_loss = initialize_data_loss(config.loss)

    # initialize learner
    learn = INRModel(
        net,
        reg_pde=reg_loss,
        loss_data=data_loss,
        learning_rate=config.optim.learning_rate,
        warmup=config.optim.warmup,
        num_epochs=config.optim.num_epochs,
        alpha=config.loss.alpha,
        qg=config.loss.qg,
    )

    # initialize callbacks
    callbacks = initialize_callbacks(config, wandb_logger.experiment.dir)

    # initialize trainer
    trainer = Trainer(
        min_epochs=1,
        max_epochs=config.optim.num_epochs,
        accelerator=config.trainer.accelerator,
        devices=config.trainer.devices,
        enable_progress_bar=True,
        logger=wandb_logger,
        callbacks=callbacks,
        accumulate_grad_batches=config.trainer.grad_batches,
    )

    # train
    trainer.fit(
        learn,
        datamodule=dm,
    )

    # res = trainer.test(learn, datamodule=dm)
    # t0 = time.time()
    predictions = trainer.predict(learn, datamodule=dm, return_predictions=True)
    predictions = torch.cat(predictions)

    ds_pred = dm.create_predictions_ds(predictions)

    from inr4ssh._src.operators import differential_simp as diffops_simp
    from inr4ssh._src.operators import differential as diffops

    learn.model.eval()
    coords, truths, preds, grads, qs = [], [], [], [], []
    for ix, iy in tqdm.tqdm(dm.predict_dataloader()):
        with torch.set_grad_enabled(True):
            # prediction
            ix = torch.autograd.Variable(ix.clone(), requires_grad=True)
            p_pred = learn.model(ix)

            # p_pred = p_pred.clone()
            # p_pred.require_grad_ = True

            # gradient
            p_grad = diffops_simp.gradient(p_pred, ix)
            # p_grad = diffops.grad(p_pred, ix)
            # q
            q = diffops_simp.divergence(p_grad, ix)
            # q = diffops.div(p_grad, ix)

        # collect
        truths.append(iy)
        coords.append(ix)
        preds.append(p_pred)
        grads.append(p_grad)
        qs.append(q)

    coords = torch.cat(coords).detach().numpy()
    preds = torch.cat(preds).detach().numpy()
    truths = torch.cat(truths).detach().numpy()
    grads = torch.cat(grads).detach().numpy()
    qs = torch.cat(qs).detach().numpy()

    df_data = dm.create_predictions_df()

    np.testing.assert_array_almost_equal(coords, df_data[["Nx", "Ny", "steps"]])
    np.testing.assert_array_almost_equal(truths, df_data[["p"]])

    df_data["p_pred"] = preds
    df_data["u_pred"] = grads[:, 0]
    df_data["v_pred"] = grads[:, 1]
    df_data["q_pred"] = qs

    xr_data = df_data.set_index(["Nx", "Ny", "steps"]).to_xarray()

    print(xr_data)

    # stream function
    plot_maps(
        xr_data.p_pred,
        name="p_pred",
        wandb_fn=wandb_logger.experiment.log,
        cmap="viridis",
    )
    plot_maps(xr_data.p, name="p", wandb_fn=wandb_logger.experiment.log, cmap="viridis")
    plot_maps(
        np.abs(xr_data.p - xr_data.p_pred),
        name="p_abs",
        wandb_fn=wandb_logger.experiment.log,
        cmap="Reds",
    )

    # potential vorticity
    plot_maps(
        xr_data.q_pred,
        name="q_pred",
        wandb_fn=wandb_logger.experiment.log,
        cmap="RdBu_r",
    )
    plot_maps(xr_data.q, name="q", wandb_fn=wandb_logger.experiment.log, cmap="RdBu_r")
    plot_maps(
        np.abs(xr_data.q - xr_data.q_pred),
        name="q_abs",
        wandb_fn=wandb_logger.experiment.log,
        cmap="Reds",
    )

    # U velocity
    plot_maps(
        xr_data.u_pred,
        name="u_pred",
        wandb_fn=wandb_logger.experiment.log,
        cmap="GnBu_r",
    )
    plot_maps(xr_data.u, name="u", wandb_fn=wandb_logger.experiment.log, cmap="GnBu_r")
    plot_maps(
        np.abs(xr_data.u - xr_data.u_pred),
        name="u_abs",
        wandb_fn=wandb_logger.experiment.log,
        cmap="Reds",
    )

    # V velocity
    plot_maps(
        xr_data.v_pred,
        name="v_pred",
        wandb_fn=wandb_logger.experiment.log,
        cmap="GnBu_r",
    )
    plot_maps(xr_data.v, name="v", wandb_fn=wandb_logger.experiment.log, cmap="GnBu_r")
    plot_maps(
        np.abs(xr_data.v - xr_data.v_pred),
        name="v_abs",
        wandb_fn=wandb_logger.experiment.log,
        cmap="Reds",
    )

    wandb.finish()
