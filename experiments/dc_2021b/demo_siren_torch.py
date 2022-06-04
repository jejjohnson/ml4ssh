#!/usr/bin/env python
# coding: utf-8


import sys, os
from pyprojroot import here


# spyder up to find the root
root = here(project_files=[".root"])
# append to path
sys.path.append(str(root))
# current file
filepath = os.path.dirname(__file__)

from pathlib import Path
import argparse
import wandb
import time
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
sns.reset_defaults()
sns.set_context(context="talk", font_scale=0.7)

import numpy as np
from ml4ssh._src.io import load_object, save_object
from ml4ssh._src.viz import create_movie, plot_psd_spectrum, plot_psd_score
from ml4ssh._src.utils import get_meshgrid, calculate_gradient, calculate_laplacian



# import parsers
from data import get_data_args, load_data, make_mini_batcher
from preprocess import add_preprocess_args, preprocess_data
from features import add_feature_args, feature_transform
from split import add_split_args, split_data
from model import add_model_args
from loss import add_loss_args
from logger import add_logger_args
from optimizer import add_optimizer_args
from postprocess import add_postprocess_args, postprocess_data, generate_eval_data
from evaluation import add_eval_args, get_rmse_metrics, get_psd_metrics
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.utilities.argparse import add_argparse_args
from pytorch_lightning.loggers import WandbLogger
seed_everything(123)

from loguru import logger


class PointsDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        X = self.X[index]
        y = self.y[index]
        
        return X, y


def main(args):
    args.framework = "pytorch"
    # ============
    # Logger
    # ============
    wandb_logger = WandbLogger(
        config=args,
        mode=args.wandb_mode,
        project=args.project,
        entity=args.entity,
        dir=args.log_dir,
        resume=args.wandb_resume
    )


    # load data
    data = load_data(args)

    # preprocess data
    data = preprocess_data(data, args)

    # feature transformation
    data, scaler = feature_transform(data, args)

    # split data
    xtrain, ytrain, xvalid, yvalid = split_data(data, args)    

    args.in_dim = xtrain.shape[-1]
    args.n_train = xtrain.shape[0]
    args.n_valid = xvalid.shape[0]

    wandb_logger.log_hyperparams(
        {
            "in_dim": args.in_dim,
            "n_train": args.n_train,
            "n_valid": args.n_valid,
        }
    )

    
    # initialize dataset
    logger.info("Initializing datasets...")
    ds_train = PointsDataset(torch.Tensor(xtrain), torch.Tensor(ytrain))
    ds_valid = PointsDataset(torch.Tensor(xvalid), torch.Tensor(yvalid))
    
    # initialize dataloader
    logger.info("Initializing dataloaders...")
    dl_train = DataLoader(
        ds_train, 
        batch_size=args.batch_size, 
        shuffle=True, 
        pin_memory=True,
        num_workers=args.num_workers
    )
    dl_valid = DataLoader(
        ds_valid, 
        batch_size=args.batch_size, 
        shuffle=False, 
        pin_memory=True,
        num_workers=args.num_workers,
    )
    
    # ==============
    # MODEL
    # ==============
    logger.info("Initializing siren model...")
    model = SirenNet(
        dim_in=3,
        dim_hidden=args.hidden_dim,
        dim_out=ytrain.shape[1], 
        num_layers=args.n_hidden, 
        w0=args.w0,
        w0_initial=args.w0_initial,
        use_bias=True,
        final_activation=None
    )
   
    
    # ==============
    # TRAINER
    # ==============
    logger.info("Initializing trainer class...")
    class Learner2DPlane(pl.LightningModule):
        def __init__(self, model:nn.Module):
            super().__init__()
            self.model = model
            self.loss = nn.MSELoss(reduction="mean")

        def forward(self, x):
            return self.model(x)

        def predict_step(self, batch, batch_idx, dataloader_idx=0):


            x, y = batch
            pred = self.model(x)

            return pred

        def training_step(self, batch, batch_idx):
            x, y = batch
            # loss function
            pred = self.model(x)
            loss = self.loss(pred, y)

            self.log("train_loss", loss)

            return loss

        def validation_step(self, batch, batch_idx):
            x, y = batch
            # loss function
            pred = self.model(x)
            loss = self.loss(pred, y)

            self.log("valid_loss", loss)

            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.model.parameters(), lr=args.learning_rate)

    learn = Learner2DPlane(model)
    
    # CALLBACKS
    logger.info("Initializing callbacks...")
    callbacks = [
        ModelCheckpoint(
            dirpath = "checkpoints",
        ),
        EarlyStopping(monitor="valid_loss", mode="min", patience=3),

    ]
    # ============
    # Training
    # ============
    
    # start trainer
    logger.info("Initializing trainer...")
    trainer = Trainer(
        min_epochs=args.min_epochs, 
        max_epochs=args.n_epochs, 
        gpus=args.gpus, 
        enable_progress_bar=True, 
        logger=wandb_logger,
        callbacks=callbacks,
    )
    
    logger.info("Training...")
    trainer.fit(learn, train_dataloaders=dl_train, val_dataloaders=dl_valid)


    # SAVING MODELS

    # objects
    logger.info("saving scaler value...")
    path_scaler = "scaler.pickle"

    # models to save
    save_object(scaler, path_scaler)

    # save with wandb
    wandb_logger.experiment.save(str(path_scaler))


    # POST PROCESSING

    df_grid = generate_eval_data(args)

    df_pred = feature_transform(df_grid.copy(), args, scaler=scaler)
    
    # set input columns
    df_pred = df_grid[df_grid.attrs["input_cols"]].values
    
    # initialize dataset
    ds_test = PointsDataset(torch.Tensor(df_pred), torch.Tensor(df_pred))
    # initialize dataloader
    dl_test = DataLoader(
        ds_test, 
        batch_size=args.eval_batch_size, 
        shuffle=False, 
        pin_memory=True
    )
    
    t0 = time.time()
    predictions = trainer.predict(learn, dataloaders=dl_test)
    t1 = time.time() - t0
    
    wandb_logger.log_metrics(
        {
            "time_predict_batches": t1,
        }
    )
    
    
    df_grid["pred"] = predictions[0].numpy()
    
    logger.info("Creating Final OI Product...")

    ds_oi = postprocess_data(df_grid, args)


    logger.info("Getting Metrics...")
    rmse_metrics = get_rmse_metrics(ds_oi, args)

    wandb_logger.log_metrics(
        {
            "model_rmse_mean": rmse_metrics[0],
            "model_rmse_std": rmse_metrics[1],
            "model_nrmse_mean": rmse_metrics[2],
            "model_nrmse_std": rmse_metrics[3],
        }
    )

    psd_metrics = get_psd_metrics(ds_oi, args)

    wandb_logger.log_metrics(
        {
            "resolved_scale": psd_metrics.resolved_scale,
        }
    )

    # FIGURES

    logger.info("Creating Figures...")
    fig, ax = plot_psd_spectrum(
        psd_metrics.psd_study, 
        psd_metrics.psd_ref, 
        psd_metrics.wavenumber
    )


    wandb_logger.experiment.log(
        {
            "model_psd_spectrum": wandb.Image(fig),
        }
    )


    fig, ax = plot_psd_score(
        psd_metrics.psd_diff, 
        psd_metrics.psd_ref, 
        psd_metrics.wavenumber, 
        psd_metrics.resolved_scale)

    wandb_logger.experiment.log(
        {
            "model_psd_score": wandb.Image(fig),
        }
    )
    logger.info("Done...!")


#     # MOVIES

#     save_path = wandb.run.dir #Path(root).joinpath("experiments/dc_2021b")
#     if args.smoke_test:
#         create_movie(ds_oi.ssh.isel(time=slice(50,60)), f"pred", "time", cmap="viridis", file_path=save_path)
#     else:
#         create_movie(ds_oi.ssh, f"pred", "time", cmap="viridis", file_path=save_path)

    
#     wandb.log(
#         {
#             "predictions_gif": wandb.Image(f"{save_path}/movie_pred.gif"),
#         }
#     )


#     # GRADIENTS


#     ds_oi["ssh_grad"] = calculate_gradient(ds_oi["ssh"], "longitude", "latitude")




#     if args.smoke_test:
#         create_movie(ds_oi.ssh_grad.isel(time=slice(50,60)), f"pred_grad", "time", cmap="Spectral_r", file_path=save_path)
#     else:
#         create_movie(ds_oi.ssh_grad, f"pred_grad", "time", cmap="Spectral_r", file_path=save_path)


#     wandb.log(
#         {
#             "predictions_grad_gif": wandb.Image(f"{save_path}/movie_pred_grad.gif"),
#         }
#     )


#     # LAPLACIAN


#     ds_oi["ssh_lap"] = calculate_laplacian(ds_oi["ssh"], "longitude", "latitude")


#     if args.smoke_test:
#         create_movie(ds_oi.ssh_lap.isel(time=slice(50,60)), f"pred_lap", "time", cmap="RdBu_r", file_path=save_path)
#     else:
#         create_movie(ds_oi.ssh_lap, f"pred_lap", "time", cmap="RdBu_r", file_path=save_path)

    

#     wandb.log(
#         {
#             "predictions_laplacian_gif": wandb.Image(f"{save_path}/movie_pred_lap.gif"),
#         }
#     )

            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser = add_logger_args(parser)

    # data
    parser = get_data_args(parser)

    # preprocessing, feature transform, split
    parser = add_preprocess_args(parser)
    parser = add_feature_args(parser)
    parser = add_split_args(parser)

    # model, optimizer, loss
    parser = add_model_args(parser)
    parser = add_optimizer_args(parser)
    parser = add_loss_args(parser)
    parser = add_argparse_args(Trainer, parser)

    # postprocessing, metrics
    parser = add_postprocess_args(parser)
    parser = add_eval_args(parser)

    args = parser.parse_args()

    main(args)
