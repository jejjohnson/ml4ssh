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
from inr4ssh._src.io import load_object, save_object
from inr4ssh._src.models.siren import SirenNet
from inr4ssh._src.viz import create_movie, plot_psd_spectrum, plot_psd_score
from inr4ssh._src.utils import get_meshgrid, calculate_gradient, calculate_laplacian



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
from optimizers.LBFGS import FullBatchLBFGS
from postprocess import add_postprocess_args, postprocess_data, generate_eval_data
from evaluation import add_eval_args, get_rmse_metrics, get_psd_metrics

import gc
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
        idx = rng.choice(np.arange(args.n_train), size=args.subsample)
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
    
    xtrain_tensor = xtrain_tensor.contiguous()
    ytrain_tensor = ytrain_tensor.contiguous()
    
    if torch.cuda.is_available():
        n_devices = torch.cuda.device_count()
        logger.info(f" {n_devices} found...!")
        output_device = torch.device("cuda:0")
        xtrain_tensor, ytrain_tensor = xtrain_tensor.cuda(output_device), ytrain_tensor.cuda(output_device)
        
        
    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood, n_devices):
            super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            base_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=args.in_dim))

            self.covar_module = gpytorch.kernels.MultiDeviceKernel(
                base_covar_module, device_ids=range(n_devices),
                output_device=output_device
            )

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        
        
    def train(train_x,
              train_y,
              n_devices,
              output_device,
              checkpoint_size,
              preconditioner_size,
              n_training_iter,
              wandb_logger=False
    ):
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(output_device)
        model = ExactGPModel(train_x, train_y, likelihood, n_devices).to(output_device)
        model.train()
        likelihood.train()

        # optimizer = FullBatchLBFGS(model.parameters(), lr=args.learning_rate)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)


        with gpytorch.beta_features.checkpoint_kernel(checkpoint_size), \
             gpytorch.settings.max_preconditioner_size(preconditioner_size):

            # def closure():
            #     optimizer.zero_grad()
            #     output = model(train_x)
            #     loss = -mll(output, train_y)
            #     return loss

            # loss = closure()
            # loss.backward()
            with tqdm.trange(n_training_iter) as pbar:
                for i in pbar:
                    # options = {'closure': closure, 'current_loss': loss, 'max_ls': 10}
                    # loss, _, _, _, _, _, _, fail = optimizer.step(options)
                    optimizer.zero_grad()
                    output = model(train_x)
                    loss = - mll(output, train_y)
                    loss.backward()
    
                    if wandb_logger:
                        wandb.log({"nll_loss": loss.item(), "epoch": i})
                        
                    pbar.set_description(f"nll_loss: {loss.item():.5f}")

                    # if fail:
                    #     logger.info('Convergence reached!')
                    #     break
                    optimizer.step()

        print(f"Finished training on {args.n_train} data points using {n_devices} GPUs.")
        return model, likelihood
        
    def find_best_gpu_setting(train_x,
                              train_y,
                              n_devices,
                              output_device,
                              preconditioner_size
    ):
        N = train_x.size(0)

        # Find the optimum partition/checkpoint size by decreasing in powers of 2
        # Start with no partitioning (size = 0)
        settings = [0] + [int(n) for n in np.ceil(N / 2**np.arange(1, np.floor(np.log2(N))))]

        for checkpoint_size in settings:
            logger.info('Number of devices: {} -- Kernel partition size: {}'.format(n_devices, checkpoint_size))
            try:
                # Try a full forward and backward pass with this setting to check memory usage
                model, likelihood = train(train_x, train_y,
                             n_devices=n_devices, output_device=output_device,
                             checkpoint_size=checkpoint_size,
                             preconditioner_size=preconditioner_size, n_training_iter=args.n_epochs, wandb_logger=True)

                # when successful, break out of for-loop and jump to finally block
                break
            except RuntimeError as e:
                logger.info('RuntimeError: {}'.format(e))
            except AttributeError as e:
                logger.info('AttributeError: {}'.format(e))
            finally:
                # handle CUDA OOM error
                gc.collect()
                torch.cuda.empty_cache()
            
        return model, likelihood, checkpoint_size
    
    
    # Set a large enough preconditioner size to reduce the number of CG iterations run
    logger.info("Training with the best GPU settings!")
    preconditioner_size = 100
    
    model, likelihood, checkpoint_size = find_best_gpu_setting(
        train_x=xtrain_tensor, 
        train_y=ytrain_tensor,
        n_devices=n_devices,
        output_device=output_device,
        preconditioner_size=preconditioner_size,
    )
    
    # logger.info("Done with finding best GPU settings!")
    # logger.info(f"Kernel Partition Size: {checkpoint_size}")
    # logger.info("Starting real training...")
    # model, likelihood = train(
    #     xtrain_tensor,
    #     ytrain_tensor,
    #     n_devices,
    #     output_device,
    #     checkpoint_size,
    #     preconditioner_size,
    #     args.n_epochs,
    #     wandb_logger=True
    # )
    # objects
    logger.info("Saving Model...")
    path_scaler = "scaler.pickle"
    path_model = "model.pickle"
    path_likelihood = "likelihood.pickle"

    # models to save
    torch.save(model.state_dict(), path_model)
    torch.save(likelihood.state_dict(), path_likelihood)
    save_object(scaler, path_scaler)

    # save with wandb
    wandb.save(str(path_scaler))
    wandb.save(str(path_model))
    wandb.save(str(path_likelihood))
    
    # =================
    # POST PROCESSING
    # =================
    logger.info("Setting Up Predictions...")
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
    
    logger.info("Doing Caching...")
    t0 = time.time()
    with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.beta_features.checkpoint_kernel(checkpoint_size):
        preds = model(xtest[:2])
        del preds
    t1 = time.time() - t0
    
    wandb.log(
        {
            "time_predict_caching": t1,
        }
    )
    
    logger.info("Doing Predictions...")
    t0 = time.time()
    with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.beta_features.checkpoint_kernel(checkpoint_size):
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
