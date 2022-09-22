import sys, os


from pyprojroot import here

# spyder up to find the root
root = here(project_files=[".root"])

# append to path
sys.path.append(str(root))

import ml_collections
import copy
import tqdm
import wandb
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


from losses import RegQG, initialize_data_loss
from model import INRModel
from figures import plot_maps

from inr4ssh._src.io import get_wandb_config, get_wandb_model
from inr4ssh._src.datamodules.qg_sim import QGSimulation
from inr4ssh._src.models.siren import SirenNet


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


def load_model_from_cpkt(config):

    # load previous model
    logger.info(f"Loading previous wandb model...")
    best_model = get_wandb_model(config.run_path, config.model_path)
    best_model.download(replace=True)

    return best_model.name


def initialize_callbacks(config, save_dir, name: str = None):
    # model checkpoints
    file_path = "checkpoints"
    if name is not None:
        file_path = "checkpoints" + "_" + name
    model_cb = ModelCheckpoint(
        dirpath=str(Path(save_dir).joinpath(file_path)),
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    callbacks = [
        model_cb,
    ]

    return callbacks


def get_physical_arrays(model, dm):

    from inr4ssh._src.operators import differential_simp as diffops_simp

    model.eval()
    coords, truths, preds, grads, qs = [], [], [], [], []
    for ix, iy in tqdm.tqdm(dm.predict_dataloader()):
        with torch.set_grad_enabled(True):
            # prediction
            ix = torch.autograd.Variable(ix.clone(), requires_grad=True)
            p_pred = model(ix)

            # p_pred = p_pred.clone()
            # p_pred.require_grad_ = True

            # gradient
            p_grad = diffops_simp.gradient(p_pred, ix)
            # p_grad = diffops.grad(p_pred, ix)
            # q
            q = diffops_simp.divergence(p_grad, ix)
            # q = diffops.div(p_grad, ix)

        # collect
        truths.append(iy.detach().cpu())
        coords.append(ix.detach().cpu())
        preds.append(p_pred.detach().cpu())
        grads.append(p_grad.detach().cpu())
        qs.append(q.detach().cpu())

    coords = torch.cat(coords).numpy()
    preds = torch.cat(preds).numpy()
    truths = torch.cat(truths).numpy()
    grads = torch.cat(grads).numpy()
    qs = torch.cat(qs).numpy()

    df_data = dm.create_predictions_df()

    np.testing.assert_array_almost_equal(coords, df_data[["Nx", "Ny", "steps"]])
    np.testing.assert_array_almost_equal(truths, df_data[["p"]])

    df_data["p_pred"] = preds
    df_data["u_pred"] = -grads[:, 0]
    df_data["v_pred"] = grads[:, 1]
    df_data["q_pred"] = qs

    xr_data = df_data.set_index(["Nx", "Ny", "steps"]).to_xarray()

    return xr_data


def plot_physical_quantities(xr_data, wandb_logger, name: str = "img"):

    # stream function
    plot_maps(
        xr_data.p_pred,
        name=f"p_pred_{name}",
        wandb_fn=wandb_logger.experiment.log,
        cmap="viridis",
    )
    plot_maps(xr_data.p, name="p", wandb_fn=wandb_logger.experiment.log, cmap="viridis")
    plot_maps(
        np.abs(xr_data.p - xr_data.p_pred),
        name=f"p_abs_{name}",
        wandb_fn=wandb_logger.experiment.log,
        cmap="Reds",
    )

    # potential vorticity
    plot_maps(
        xr_data.q_pred,
        name=f"q_pred_{name}",
        wandb_fn=wandb_logger.experiment.log,
        cmap="RdBu_r",
    )
    plot_maps(xr_data.q, name="q", wandb_fn=wandb_logger.experiment.log, cmap="RdBu_r")
    plot_maps(
        np.abs(xr_data.q - xr_data.q_pred),
        name=f"q_abs_{name}",
        wandb_fn=wandb_logger.experiment.log,
        cmap="Reds",
    )

    # U velocity
    plot_maps(
        xr_data.u_pred,
        name=f"u_pred_{name}",
        wandb_fn=wandb_logger.experiment.log,
        cmap="GnBu_r",
    )
    plot_maps(xr_data.u, name="u", wandb_fn=wandb_logger.experiment.log, cmap="GnBu_r")
    plot_maps(
        np.abs(xr_data.u - xr_data.u_pred),
        name=f"u_abs_{name}",
        wandb_fn=wandb_logger.experiment.log,
        cmap="Reds",
    )

    # V velocity
    plot_maps(
        xr_data.v_pred,
        name=f"v_pred_{name}",
        wandb_fn=wandb_logger.experiment.log,
        cmap="GnBu_r",
    )
    plot_maps(xr_data.v, name="v", wandb_fn=wandb_logger.experiment.log, cmap="GnBu_r")
    plot_maps(
        np.abs(xr_data.v - xr_data.v_pred),
        name=f"v_abs_{name}",
        wandb_fn=wandb_logger.experiment.log,
        cmap="Reds",
    )

    return None


def train(config: ml_collections.ConfigDict, workdir: str, savedir: str):

    # ==================================
    # EXPERIMENT I: QG IMAGE
    # ==================================

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
    logger.info(f"Number of Data Points: {len(dm.ds_predict)}...")
    logger.info(f"Number of Training: {len(dm.ds_train)}...")
    logger.info(f"Number of Validation: {len(dm.ds_valid)}...")
    # ==================================
    # EXPERIMENT Ia: SIREN + NO QG LOSS
    # ==================================

    x_init, y_init = dm.ds_train[:10]

    # initialize model
    logger.info("Initializing SIREN Model...")
    net = initialize_siren_model(config.model, x_init, y_init)

    # initialize dataloss
    logger.info("Initializing Loss...")
    data_loss = initialize_data_loss(config.loss)

    logger.info("Initializing PL Model...")
    learn = INRModel(
        model=net,
        loss_data=data_loss,
        reg_pde=None,
        learning_rate=config.optim.learning_rate,
        warmup=config.optim.warmup,
        warmup_start_lr=config.optim.warmup_start_lr,
        eta_min=config.optim.eta_min,
        num_epochs=config.optim.num_epochs,
        alpha=0.0,
        qg=False,
    )

    # initialize callbacks
    logger.info("Initializing Callbacks...")
    callbacks = initialize_callbacks(config, wandb_logger.experiment.dir)

    # initialize trainer
    logger.info("Initializing PL Trainer...")
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
    logger.info("Initializing PL Trainer...")
    trainer.fit(
        learn,
        datamodule=dm,
    )

    logger.info("Getting physical quantities...")
    xr_data = get_physical_arrays(learn.model, dm)

    logger.info("Plotting physical quantities...")
    plot_physical_quantities(xr_data, wandb_logger, name="img")

    # ==================================
    # EXPERIMENT Ib: PRE-SIREN + QG LOSS
    # ==================================

    # initialize regularization
    logger.info("Initializing QG PDE REG...")
    reg_loss = RegQG(config.loss.alpha)

    # initialize dataloss
    logger.info("Initializing Loss...")
    data_loss = initialize_data_loss(config.loss)

    logger.info("Initializing PL Model (with Pretrained)...")
    learn_qg = INRModel(
        model=copy.deepcopy(learn.model),
        loss_data=data_loss,
        reg_pde=reg_loss,
        learning_rate=config.optim_qg.learning_rate,
        warmup=config.optim_qg.warmup,
        warmup_start_lr=config.optim_qg.warmup_start_lr,
        eta_min=config.optim_qg.eta_min,
        num_epochs=config.optim.num_epochs,
        alpha=config.loss.alpha,
        qg=True,
    )

    # initialize callbacks
    logger.info("Initializing Callbacks...")
    callbacks = initialize_callbacks(config, wandb_logger.experiment.dir, "qg")

    # initialize trainer
    logger.info("Initializing PL Trainer...")
    trainer = Trainer(
        min_epochs=1,
        max_epochs=config.optim_qg.num_epochs,
        accelerator=config.trainer_qg.accelerator,
        devices=config.trainer_qg.devices,
        enable_progress_bar=True,
        logger=wandb_logger,
        callbacks=callbacks,
        accumulate_grad_batches=config.trainer.grad_batches,
    )

    # train
    logger.info("Starting training (QG)...")
    trainer.fit(
        learn_qg,
        datamodule=dm,
    )

    logger.info("Getting physical quantities...")
    xr_data = get_physical_arrays(learn_qg.model, dm)

    logger.info("Plotting physical quantities...")
    plot_physical_quantities(xr_data, wandb_logger, name="img_qg")

    logger.info("Finishing...")
    wandb.finish()
