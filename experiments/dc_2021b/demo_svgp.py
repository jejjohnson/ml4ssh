#!/usr/bin/env python
# coding: utf-8


import sys, os
from pyprojroot import here


# spyder up to find the root
root = here(project_files=[".root"])
# append to path
sys.path.append(str(root))
from pathlib import Path
import time
import argparse
import wandb
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.reset_defaults()
sns.set_context(context="talk", font_scale=0.7)
# import hvplot.xarray

import tensorflow as tf
import tensorflow_datasets as tfd
import gpflow
from gpflow import set_trainable
from gpflow.optimizers import NaturalGradient
import numpy as np
from gpflow.utilities import print_summary
from tqdm.notebook import trange
# # Ensure TF does not see GPU and grab all GPU memory.
import tensorflow as tf
from ml4ssh._src.io import load_object, save_object
from ml4ssh._src.viz import create_movie, plot_psd_spectrum, plot_psd_score
from ml4ssh._src.utils import get_meshgrid, calculate_gradient, calculate_laplacian

# import parsers
from data import get_data_args, load_data
from preprocess import add_preprocess_args, preprocess_data
from features import add_feature_args, feature_transform
from split import add_split_args, split_data
from models.gp_tf import add_model_args, get_likelihood, get_kernel, get_inducing_points
from loss import add_loss_args, get_loss_fn
from logger import add_logger_args
from optimizer import add_optimizer_gpflow_args
from postprocess import add_postprocess_args, postprocess_data, generate_eval_data
from evaluation import add_eval_args, get_rmse_metrics, get_psd_metrics
from smoke_test import add_winter_smoke_test_args, add_january_smoke_test_args

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def main(args):
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
    
    # extract data
    xtrain = data[data.attrs["input_cols"]].values
    ytrain = data[data.attrs["output_cols"]].values

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
    print(f"N Training: {xtrain.shape[0]:_}")
    
    # DATASET
    def make_ds(shuffle=True):
        ds = tf.data.Dataset.from_tensor_slices(
            (xtrain.astype(np.float64), ytrain.astype(np.float64))
        )


        ds = ds.prefetch(args.prefetch_buffer)
        ds = ds.repeat()
        if shuffle:
            ds = ds.shuffle(buffer_size=10 * args.batch_size)
        ds = ds.batch(args.batch_size)


        return iter(ds)
    
    ds_train = make_ds()
    # MODEL
    
    # get kernel
    kernel = get_kernel(args)
    # get likelihood
    likelihood = get_likelihood(args)
    # get inducing points
    Z = get_inducing_points(xtrain, args)


    # don't train the inducing inputs
    model = gpflow.models.SVGP(kernel, likelihood, Z.astype(np.float64), num_data=xtrain.shape[0])

    # train the inducing inputs
    gpflow.set_trainable(model.inducing_variable, True)
    
    print_summary(model)
    
    # OPTIMIZER

    # Create an Adam Optimizer
    ordinary_adam_opt = tf.optimizers.Adam(args.learning_rate)


    # NatGrads and Adam for SVGP
    # Stop Adam from optimizing the variational parameters
    set_trainable(model.q_mu, False)
    set_trainable(model.q_sqrt, False)

    # Create the optimize_tensors for SVGP
    adam_opt = tf.optimizers.Adam(args.learning_rate)

    natgrad_opt = NaturalGradient(gamma=args.learning_rate_ng)
    variational_params = [(model.q_mu, model.q_sqrt)]


    # make training loss
    training_loss = model.training_loss_closure(ds_train, compile=True)
    
    
    # STEPS
    n_steps_per_epoch = args.n_train / args.batch_size
    steps = int(n_steps_per_epoch * args.n_epochs) if not args.smoke_test else 500


    wandb.config.update(
        {
            "steps": steps,
            "n_steps_per_epoch": n_steps_per_epoch,
        }
    )
    
    
    # TRAINING
    

    @tf.function
    def optimization_step():
        adam_opt.minimize(training_loss, var_list=model.trainable_variables)

    @tf.function
    def natgrad_optimization_step():
        natgrad_opt.minimize(training_loss, var_list=variational_params)

    with trange(steps) as pbar:
        for step in pbar:
            optimization_step()
            natgrad_optimization_step()
            elbo = -training_loss().numpy()

            wandb.log({"elbo": elbo}, step=step)

            if step % 10 == 0:
                pbar.set_description(f"Loss (ELBO): {elbo:.4e}")
                
                
    print_summary(model)
    
    # SAVING
    # objects
    path_model = Path(wandb.run.dir).joinpath("params.pickle")
    path_scaler = Path(wandb.run.dir).joinpath("scaler.pickle")

    # models to save
    save_object(gpflow.utilities.parameter_dict(gpflow.utilities.parameter_dict(model)), path_model)
    save_object(scaler, path_scaler)

    # save with wandb
    wandb.save(str(path_model), policy="now")
    wandb.save(str(path_scaler), policy="now")
    
    wandb.log({
        "time_scale": model.kernel.lengthscales[0].numpy(),
        "lon_scale": model.kernel.lengthscales[1].numpy(),
        "lat_scale": model.kernel.lengthscales[2].numpy(),
        "variance": model.kernel.variance.numpy(),
        "noise": model.likelihood.variance.numpy(),
    })
    
    
    

    def predict_grid(gp_model, n_batches:int=5_000):
        # generate grid
        df_grid = generate_eval_data(args)

        # set input columns
        df_pred = df_grid[df_grid.attrs["input_cols"]].values

        # create dataloader
        ds_test = tf.data.Dataset.from_tensor_slices(df_pred).batch(n_batches)
        n_iters = len(ds_test)
        means, variances = [], []
        ds_test = iter(ds_test)
        with trange(n_iters) as pbar:
            for i in pbar:
                ix = next(ds_test)
                # predict using GP
                imean, ivar = gp_model.predict_f(ix)

                # add stuff
                means.append(imean)
                variances.append(ivar)

        mean = np.vstack(means)
        variance = np.vstack(variances)

        df_grid["pred"] = mean
        df_grid["variance"] = variance

        return df_grid


    t0 = time.time()
    # make predictions
    df_grid = predict_grid(model)
    t1 = time.time() - t0

    # create OI
    ds_oi = postprocess_data(df_grid, args)
    
    wandb.config.update(
        {
            "time_predict_batches": t1,
        }
    )
    rmse_metrics = get_rmse_metrics(ds_oi, args)
    print(rmse_metrics)

    wandb.log(
        {
            "model_rmse_mean": rmse_metrics[0],
            "model_rmse_std": rmse_metrics[1],
            "model_nrmse_mean": rmse_metrics[2],
            "model_nrmse_std": rmse_metrics[3],
        }
    )
    
    psd_metrics = get_psd_metrics(ds_oi, args)
    print(psd_metrics)
    

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
    parser = add_optimizer_gpflow_args(parser)
    parser = add_loss_args(parser)

    # postprocessing, metrics
    parser = add_postprocess_args(parser)
    parser = add_eval_args(parser)

    args = parser.parse_args()

    main(args)
