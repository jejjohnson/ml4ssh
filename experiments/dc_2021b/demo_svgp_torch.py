#!/usr/bin/env python
# coding: utf-8


import sys, os
from pyprojroot import here


# spyder up to find the root
root = here(project_files=[".root"])
# local = here(project_files=[".local"])
# append to path
sys.path.append(str(root))
# sys.path.append(str(local))
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
from ml4ssh._src.models_torch.siren import SirenNet
from ml4ssh._src.viz import create_movie, plot_psd_spectrum, plot_psd_score
from ml4ssh._src.utils import get_meshgrid, calculate_gradient, calculate_laplacian



# import parsers
from data import get_data_args, load_data
from preprocess import add_preprocess_args, preprocess_data
from features import add_feature_args, feature_transform
from split import add_split_args, split_data
from models.gp_torch import (
    add_model_args, 
    get_inducing_points, 
    get_kernel, 
    get_likelihood, 
    get_variational_dist
)
from losses.gp_torch import add_loss_args, get_loss_fn
from logger import add_logger_args
from optimizers.gp_torch import add_optimizer_args
from postprocess import add_postprocess_args, postprocess_data, generate_eval_data
from evaluation import add_eval_args, get_rmse_metrics, get_psd_metrics


import torch
import gpytorch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.utilities.argparse import add_argparse_args

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
    wandb.init(
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
    xtrain = data[data.attrs["input_cols"]].values
    ytrain = data[data.attrs["output_cols"]].values.squeeze()

    args.in_dim = xtrain.shape[-1]
    args.n_train = xtrain.shape[0]

    if args.smoke_test:

        rng = np.random.RandomState(args.split_seed)
        idx = rng.choice(np.arange(args.n_train), size=2_000)
        xtrain = xtrain[idx]
        ytrain = ytrain[idx]

    wandb.config.update(
        {
            "in_dim": args.in_dim,
            "n_train": args.n_train,
        }
    )

    
    # initialize dataset
    xtrain_tensor = torch.Tensor(xtrain)
    ytrain_tensor = torch.Tensor(ytrain)
    if torch.cuda.is_available():
        xtrain_tensor, ytrain_tensor = xtrain_tensor.cuda(), ytrain_tensor.cuda()

    logger.info("Initializing dataset...")
    ds_train = TensorDataset(xtrain_tensor, ytrain_tensor)
    
    # initialize dataloader
    logger.info("Initializing dataloaders...")
    dl_train = DataLoader(
        ds_train, 
        batch_size=args.batch_size, 
        shuffle=True, 
        pin_memory=False,
        num_workers=args.num_workers
    )
    
    # ==============
    # MODEL
    # ==============
    logger.info("Initializing siren model...")
    class SVGPModel(gpytorch.models.ApproximateGP):
        def __init__(self, kernel, inducing_points, variational_dist):
            variational_strategy = gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_dist, learn_inducing_locations=args.learn_inducing
            )
            super().__init__(variational_strategy)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = kernel

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
   
    # get inducing points
    inducing_points = get_inducing_points(xtrain, args)

    # get kernel
    kernel = get_kernel(args)

    # get variational dist
    variational_dist = get_variational_dist(torch.Tensor(inducing_points), args)

    # initialize model
    model = SVGPModel(
        kernel=kernel,
        variational_dist=variational_dist,
        inducing_points=torch.Tensor(inducing_points)
    )

    # initialize likelihood
    likelihood = get_likelihood(args)
    
    # ==============
    # TRAINER
    # ==============
    logger.info("Initializing training...")
    
    variational_ngd_optimizer = gpytorch.optim.NGD(
        model.variational_parameters(), 
        num_data=args.n_train, 
        lr=args.learning_rate_ng
    )

    hyperparameter_optimizer = torch.optim.Adam([
        {'params': model.hyperparameters()},
        {'params': likelihood.parameters()},
    ], lr=args.learning_rate)

    if torch.cuda.is_available() and args.gpus > 0:
        model = model.cuda()
        likelihood = likelihood.cuda()
        
        
    model.train()
    likelihood.train()
    mll = get_loss_fn(likelihood, model, args.n_train, args=args)
        
    # ============
    # Training
    # ============

    epochs_iter = tqdm.tqdm(range(100), desc="Epoch")

    for i in epochs_iter:
        minibatch_iter = tqdm.tqdm(dl_train, desc="Minibatch", leave=False)

        for j, (x_batch, y_batch) in enumerate(minibatch_iter):
            ### Perform NGD step to optimize variational parameters
            variational_ngd_optimizer.zero_grad()
            hyperparameter_optimizer.zero_grad()
            output = model(x_batch)
            loss = -mll(output, y_batch)
            minibatch_iter.set_postfix(loss=loss.item())
            loss.backward()
            variational_ngd_optimizer.step()
            hyperparameter_optimizer.step()
            wandb.log({"nll_loss": loss.item(), "batch":j , "epoch":i})


    # ============
    # SAVING
    # ============
    
    # objects
    path_scaler = "scaler.pickle"
    path_model = "model.pickle"

    # models to save
    torch.save(model.state_dict(), path_model)
    save_object(scaler, path_scaler)

    # save with wandb
    wandb.save(str(path_scaler))
    wandb.save(str(path_model))


    # POST PROCESSING

    df_grid = generate_eval_data(args)

    df_pred = feature_transform(df_grid.copy(), args, scaler=scaler)
    
    # set input columns
    xtest = df_pred[df_pred.attrs["input_cols"]].values
    
    # initialize dataset
    xtest = torch.Tensor(xtest)
    if torch.cuda.is_available():
        xtest = xtest.cuda()
        
    ds_test = TensorDataset(xtest)
    # initialize dataloader
    dl_test = DataLoader(
        ds_test, 
        batch_size=args.eval_batch_size, 
        shuffle=False, 
        pin_memory=False
    )
    
    model.eval()
    likelihood.eval()
    means = torch.tensor([])
    variances = torch.tensor([])
    
    t0 = time.time()

    with torch.no_grad():
        for x_batch in tqdm.tqdm(dl_test):
            preds = model(x_batch[0])
            means = torch.cat([means, preds.mean.cpu()])
            variances = torch.cat([variances, preds.variance.cpu()])
            
    t1 = time.time() - t0
    
    wandb.log(
        {
            "time_predict_batches": t1,
        }
    )
    
    
    df_grid["pred"] = means.numpy()
    df_grid["var"] = variances.numpy()
    
    logger.info("Creating Final OI Product...")

    ds_oi = postprocess_data(df_grid, args)


    logger.info("Getting Metrics...")
    rmse_metrics = get_rmse_metrics(ds_oi, args)

    wandb.log(
        {
            "model_rmse_mean": rmse_metrics[0],
            "model_rmse_std": rmse_metrics[1],
            "model_nrmse_mean": rmse_metrics[2],
            "model_nrmse_std": rmse_metrics[3],
        }
    )

    psd_metrics = get_psd_metrics(ds_oi, args)

    wandb.log(
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


    wandb.log(
        {
            "model_psd_spectrum": wandb.Image(fig),
        }
    )


    fig, ax = plot_psd_score(
        psd_metrics.psd_diff, 
        psd_metrics.psd_ref, 
        psd_metrics.wavenumber, 
        psd_metrics.resolved_scale)

    wandb.log(
        {
            "model_psd_score": wandb.Image(fig),
        }
    )
    logger.info("Done...!")

            

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
