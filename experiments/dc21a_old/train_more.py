import sys, os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from pyprojroot import here

# spyder up to find the root
root = here(project_files=[".root"])

# append to path
sys.path.append(str(root))

import config
from simple_parsing import ArgumentParser
import time
from loguru import logger

import torch
from ml_collections import config_dict

from inr4ssh._src.io import get_wandb_config, get_wandb_model
from inr4ssh._src.io import save_object
from inr4ssh._src.datamodules.dc21a import SSHAltimetry
from inr4ssh._src.metrics.psd import compute_psd_scores
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader

import torch.nn as nn
import pytorch_lightning as pl

from models import model_factory
from optimizers import optimizer_factory, lr_scheduler_factory
from losses import loss_factory, regularization_factory

from utils import (
    get_interpolation_alongtrack_prediction_ds,
    get_alongtrack_prediction_ds,
)
from utils import (
    plot_psd_figs,
    get_grid_stats,
    postprocess_predictions,
    get_alongtrack_stats,
)


import wandb
from inr4ssh._src.io import simpleargs_2_ndict

seed_everything(123)


def main(args):

    # INITIALIZE LOGGER
    logger.info("Initializing logger...")

    log_options = args.logging

    # if log_options.wandb_mode in ["offline", "disabled"]:
    #     # TODO: download all files
    #
    # elif log_options.wandb_mode == "online":
    #     # TODO: download only the model and data
    #
    # else:
    #     raise ValueError(f"Unrecognized wandb_mode: {log_options.wandb_mode}")

    # load old wandb config
    logger.info(f"Loading old wandb config...")
    old_config = get_wandb_config(log_options.run_path)
    old_args = config_dict.ConfigDict(old_config)

    logger.info(f"Converting args to dict...")
    params_dict = simpleargs_2_ndict(args)

    logger.info(f"Initializing wandb logger...")
    wandb_logger = WandbLogger(
        config=params_dict,
        mode=log_options.mode,
        project=log_options.project,
        entity=log_options.entity,
        dir=log_options.log_dir,
        resume=False,
    )

    # DATA MODULE
    logger.info("Initializing data module...")
    dm = SSHAltimetry(
        data=args.data,
        preprocess=old_args.preprocess,
        traintest=old_args.traintest,
        features=old_args.features,
        dataloader=old_args.dataloader,
        eval=old_args.eval,
    )

    dm.setup()

    # objects
    logger.info("saving scaler transform...")
    path_scaler = "./scaler.pickle"

    # models to save
    save_object(dm.scaler, path_scaler)

    # save with wandb
    logger.info("logging scaler transform...")
    wandb_logger.experiment.save(str(path_scaler))

    logger.info("extracting train and test...")
    x_train, y_train = dm.ds_train[:]

    logger.info(f"Creating {old_args.model.model} neural network...")

    dim_in = x_train.shape[1]
    dim_out = y_train.shape[1]

    # update params
    params_dict["model"].update({"dim_in": dim_in, "dim_out": dim_out})
    wandb_logger.experiment.config.update(params_dict, allow_val_change=True)

    net = model_factory(
        model=old_args.model.model, dim_in=dim_in, dim_out=dim_out, config=old_args
    )

    # logger.info(f"Adding {args.optimizer.optimizer} optimizer...")
    # # optimizer = get_optimizer(args.optimizer)
    #
    # logger.info(f"Adding {args.lr_scheduler.lr_scheduler} optimizer...")

    logger.info("Initializing callbacks...")

    callbacks = [
        ModelCheckpoint(
            dirpath=f"{wandb_logger.experiment.dir}/checkpoints",
            monitor="valid_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        ),
        EarlyStopping(
            monitor="valid_loss", mode="min", patience=args.callbacks.patience
        ),
    ]

    # update params
    params_dict["callbacks"].update({"patience": args.callbacks.patience})
    # callbacks = list()
    #
    # callbacks += [("lr_scheduler", get_lr_scheduler(args.lr_scheduler))]
    #
    # callbacks += get_callbacks(args.callbacks, wandb_run)

    # ============================
    # PYTORCH LIGHTNING CLASS
    # ============================

    logger.info("Initializing trainer class...")

    class CoordinatesLearner(pl.LightningModule):
        def __init__(self, model: nn.Module, params):
            super().__init__()
            self.model = model
            self.loss = loss_factory(params)
            self.params = params

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

            optimizer = optimizer_factory(self.params)(params=self.model.parameters())

            scheduler = lr_scheduler_factory(self.params)(optimizer=optimizer)

            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "valid_loss",
            }

    # load previous model
    logger.info(f"Loading previous wandb model...")
    best_model = get_wandb_model(log_options.run_path, log_options.model_path)
    best_model.download(replace=True)

    learn = CoordinatesLearner.load_from_checkpoint(
        checkpoint_path=best_model.name, model=net, params=args
    )

    # start trainer
    logger.info("Initializing trainer...")
    trainer = Trainer(
        min_epochs=args.optimizer.min_epochs,
        max_epochs=args.optimizer.num_epochs,
        accelerator="mps" if args.optimizer.device == "mps" else None,
        devices=1 if args.optimizer.device == "mps" else None,
        gpus=args.optimizer.gpus,
        enable_progress_bar=True,
        logger=wandb_logger,
        callbacks=callbacks,
    )

    logger.info("Training...")
    trainer.fit(learn, datamodule=dm)

    # ==============================
    # GRID PREDICTIONS
    # ==============================

    logger.info("GRID STATS...")

    # TESTING
    logger.info("Making predictions (grid)...")
    t0 = time.time()
    with torch.inference_mode():
        predictions = trainer.predict(learn, datamodule=dm, return_predictions=True)
        predictions = torch.cat(predictions)
        predictions = predictions.numpy()
    t1 = time.time() - t0

    logger.info(f"Time Taken for {dm.ds_predict[:][0].shape[0]} points: {t1:.4f} secs")
    wandb_logger.log_metrics(
        {
            "time_predict_grid": t1,
        }
    )

    ds_oi = postprocess_predictions(predictions, dm, args, logger)

    alongtracks, tracks = get_interpolation_alongtrack_prediction_ds(
        ds_oi, args, logger
    )

    logger.info("Getting RMSE Metrics (GRID)...")
    rmse_metrics = get_grid_stats(
        alongtracks, args.metrics, None, wandb_logger.log_metrics
    )

    logger.info(f"Grid Stats: {rmse_metrics}")

    # compute scores
    logger.info("Computing PSD Scores (Grid)...")
    psd_metrics = compute_psd_scores(
        ssh_true=tracks.ssh_alongtrack,
        ssh_pred=tracks.ssh_map,
        delta_x=args.metrics.velocity * args.metrics.delta_t,
        npt=tracks.npt,
        scaling="density",
        noverlap=0,
    )

    logger.info(f"Grid PSD: {psd_metrics}")

    logger.info(f"Resolved scale (grid): {psd_metrics.resolved_scale:.2f}")
    wandb_logger.log_metrics(
        {
            "resolved_scale_grid": psd_metrics.resolved_scale,
        }
    )
    #
    logger.info(f"Plotting PSD Score and Spectrum (Grid)...")
    plot_psd_figs(psd_metrics, logger, wandb_logger.experiment.log, method="grid")
    logger.info("Finished GRID Script...!")

    # ==============================
    # ALONGTRACK PREDICTIONS
    # ==============================

    logger.info("ALONGTRACK STATS...")

    X_test, y_test = get_alongtrack_prediction_ds(dm, args, logger)

    # initialize dataset
    ds_test = TensorDataset(
        torch.FloatTensor(X_test)
        # torch.Tensor(y_test)
    )
    # initialize dataloader
    dl_test = DataLoader(
        ds_test,
        batch_size=args.dataloader.batch_size_eval,
        shuffle=False,
        num_workers=args.dataloader.num_workers,
        pin_memory=args.dataloader.pin_memory,
    )

    logger.info(f"Predicting alongtrack data...")
    t0 = time.time()
    with torch.inference_mode():
        predictions = trainer.predict(
            learn, dataloaders=dl_test, return_predictions=True
        )
        predictions = torch.cat(predictions)
        predictions = predictions.numpy()
    t1 = time.time() - t0

    wandb_logger.log_metrics(
        {
            "time_predict_alongtrack": t1,
        }
    )

    logger.info("Calculating stats (alongtrack)...")
    get_alongtrack_stats(y_test, predictions, logger, wandb_logger.log_metrics)

    # PSD
    logger.info(f"Getting PSD Scores (alongtrack)...")
    psd_metrics = compute_psd_scores(
        ssh_true=y_test.squeeze(),
        ssh_pred=predictions.squeeze(),
        delta_x=args.metrics.velocity * args.metrics.delta_t,
        npt=None,
        scaling="density",
        noverlap=0,
    )

    logger.info(f"Resolved scale (alongtrack): {psd_metrics.resolved_scale:.2}")
    wandb_logger.log_metrics(
        {
            "resolved_scale_alongtrack": psd_metrics.resolved_scale,
        }
    )

    logger.info(f"Plotting PSD Score and Spectrum (AlongTrack)...")
    plot_psd_figs(psd_metrics, logger, wandb_logger.experiment.log, method="alongtrack")

    wandb.finish()


if __name__ == "__main__":
    # initialize argparse
    parser = ArgumentParser()

    # add all experiment arguments
    parser.add_arguments(config.Logging, dest="logging")
    parser.add_arguments(config.DataDir, dest="data")
    parser.add_arguments(config.PreProcess, dest="preprocess")
    parser.add_arguments(config.Features, dest="features")
    parser.add_arguments(config.TrainTestSplit, dest="traintest")
    parser.add_arguments(config.DataLoader, dest="dataloader")
    parser.add_arguments(config.Model, dest="model")
    parser.add_arguments(config.Siren, dest="siren")
    parser.add_arguments(config.ModulatedSiren, dest="modsiren")
    parser.add_arguments(config.MFN, dest="mfn")
    parser.add_arguments(config.Losses, dest="losses")
    parser.add_arguments(config.Optimizer, dest="optimizer")
    parser.add_arguments(config.LRScheduler, dest="lr_scheduler")
    parser.add_arguments(config.Callbacks, dest="callbacks")
    parser.add_arguments(config.EvalData, dest="eval")
    parser.add_arguments(config.Metrics, dest="metrics")
    parser.add_arguments(config.Viz, dest="viz")

    # parse args
    args = parser.parse_args()

    main(args)
