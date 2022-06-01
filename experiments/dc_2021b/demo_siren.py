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

#import tensorflow as tf
# Ensure TF does not see GPU and grab all GPU memory.
#tf.config.set_visible_devices([], device_type='GPU')
# ENSURE JAX SEES GPU
os.environ["JAX_PLATFORM_NAME"] = "GPU"
# ENSURE JAX DOESNT PREALLOCATE
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = str(False)


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
import jax
import jax.random as jrandom
import jax.numpy as jnp
import equinox as eqx
from ml4ssh._src.data import make_mini_batcher
from ml4ssh._src.io import load_object, save_object
from ml4ssh._src.viz import create_movie, plot_psd_spectrum, plot_psd_score
from ml4ssh._src.utils import get_meshgrid, calculate_gradient, calculate_laplacian



# import parsers
from data import get_data_args, load_data
from preprocess import add_preprocess_args, preprocess_data
from features import add_feature_args, feature_transform
from split import add_split_args, split_data
from model import add_model_args, get_model
from loss import add_loss_args, get_loss_fn
from logger import add_logger_args
from optimizer import add_optimizer_args, get_optimizer
from postprocess import add_postprocess_args, postprocess_data, generate_eval_data
from evaluation import add_eval_args, get_rmse_metrics, get_psd_metrics

import tensorflow as tf
# Ensure TF does not see GPU and grab all GPU memory.
tf.config.set_visible_devices([], device_type='GPU')


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

    # split data
    xtrain, ytrain, xvalid, yvalid = split_data(data, args)

    args.in_dim = xtrain.shape[-1]
    args.n_train = xtrain.shape[0]
    args.n_valid = xvalid.shape[0]

    wandb.config.update(
        {
            "in_dim": args.in_dim,
            "n_train": args.n_train,
            "n_valid": args.n_valid,
        }
    )

    # model
    model = get_model(args)

    # optimizer
    optimizer = get_optimizer(args)

    # loss
    make_step, val_step = get_loss_fn(args)

    # init model
    opt_state = optimizer.init(model)

    n_steps_per_epoch = args.n_train / args.batch_size
    steps = int(n_steps_per_epoch * args.n_epochs) if not args.smoke_test else 100


    wandb.config.update(
        {
            "steps": steps,
            "n_steps_per_epoch": n_steps_per_epoch,
        }
    )


    # ============
    # Training
    # ============

    train_ds = make_mini_batcher(xtrain, ytrain, args.batch_size, 1, shuffle=True)
    valid_ds = make_mini_batcher(xvalid, yvalid, args.batch_size, 1, shuffle=False)


    losses = {} 
    losses["train"] = []
    losses["valid"] = []


    with tqdm.trange(steps) as pbar:
        for step in pbar:
            
            ix, iy = next(train_ds)
            loss, grads = make_step(
                model, 
                jnp.asarray(ix.astype(np.float32)), 
                jnp.asarray(iy.astype(np.float32))
            )
            
            updates, opt_state = optimizer.update(grads, opt_state)
            model = eqx.apply_updates(model, updates)
            
            losses["train"].append(loss)
            wandb.log({"train_loss": loss}, step=step)
            ix, iy = next(valid_ds)
            # validation step
            vloss = val_step(
                model, 
                jnp.asarray(ix.astype(np.float32)), 
                jnp.asarray(iy.astype(np.float32))
            )
            losses["valid"].append(vloss)
            
            
            wandb.log({"val_loss": vloss}, step=step)
            
            if step % 10 == 0:
                pbar.set_description(f"Step: {step:_} | Train Loss: {loss:.3e} | Valid Loss: {vloss:.3e}")
                


    # SAVING MODELS

    # objects
    path_model = Path(wandb.run.dir).joinpath("model.pickle")
    path_scaler = Path(wandb.run.dir).joinpath("scaler.pickle")

    # models to save
    save_object(model, path_model)
    save_object(scaler, path_scaler)

    # save with wandb
    wandb.save(str(path_model), policy="now")
    wandb.save(str(path_scaler), policy="now")


    # POST PROCESSING

    df_grid = generate_eval_data(args)

    df_grid.describe()


    df_pred = feature_transform(df_grid.copy(), args, scaler=scaler)
    df_pred.describe(), df_grid.describe()


    # PREDICTIONS

    @jax.jit
    def pred_step(model, data):
        return jax.vmap(model)(data)


    from ml4ssh._src.model_utils import batch_predict
    from functools import partial



    df_pred = jnp.asarray(df_pred[df_pred.columns.difference(["time"])].values)

    fn = partial(pred_step, model)

    # time predictions
    t0 = time.time()

    df_grid["pred"] = batch_predict(df_pred, fn, args.eval_batch_size)

    t1 = time.time() - t0

    wandb.config.update(
        {
            "time_predict_batches": t1,
        }
    )



    ds_oi = postprocess_data(df_grid, args)



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


    # FIGURES


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


    # MOVIES

    save_path = wandb.run.dir #Path(root).joinpath("experiments/dc_2021b")
    if args.smoke_test:
        create_movie(ds_oi.ssh.isel(time=slice(50,60)), f"pred", "time", cmap="viridis", file_path=save_path)
    else:
        create_movie(ds_oi.ssh, f"pred", "time", cmap="viridis", file_path=save_path)

    
    wandb.log(
        {
            "predictions_gif": wandb.Image(f"{save_path}/movie_pred.gif"),
        }
    )


    # GRADIENTS


    ds_oi["ssh_grad"] = calculate_gradient(ds_oi["ssh"], "longitude", "latitude")




    if args.smoke_test:
        create_movie(ds_oi.ssh_grad.isel(time=slice(50,60)), f"pred_grad", "time", cmap="Spectral_r", file_path=save_path)
    else:
        create_movie(ds_oi.ssh_grad, f"pred_grad", "time", cmap="Spectral_r", file_path=save_path)


    wandb.log(
        {
            "predictions_grad_gif": wandb.Image(f"{save_path}/movie_pred_grad.gif"),
        }
    )


    # LAPLACIAN


    ds_oi["ssh_lap"] = calculate_laplacian(ds_oi["ssh"], "longitude", "latitude")


    if args.smoke_test:
        create_movie(ds_oi.ssh_lap.isel(time=slice(50,60)), f"pred_lap", "time", cmap="RdBu_r", file_path=save_path)
    else:
        create_movie(ds_oi.ssh_lap, f"pred_lap", "time", cmap="RdBu_r", file_path=save_path)

    

    wandb.log(
        {
            "predictions_laplacian_gif": wandb.Image(f"{save_path}/movie_pred_lap.gif"),
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
    parser = add_optimizer_args(parser)
    parser = add_loss_args(parser)

    # postprocessing, metrics
    parser = add_postprocess_args(parser)
    parser = add_eval_args(parser)

    args = parser.parse_args()

    main(args)
