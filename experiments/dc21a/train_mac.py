import sys, os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from pyprojroot import here

# spyder up to find the root
root = here(project_files=[".root"])

# append to path
sys.path.append(str(root))


import time
import argparse
import imageio
from pathlib import Path
import numpy as np
import pytorch_lightning as pl
import torch
import wandb

from inr4ssh._src.io import save_object
from inr4ssh._src.data.ssh_obs import (
    load_ssh_altimetry_data_test,
    load_ssh_altimetry_data_train,
    load_ssh_correction,
)
from inr4ssh._src.datamodules.ssh_obs import SSHAltimetry
from inr4ssh._src.features.data_struct import df_2_xr
from inr4ssh._src.interp import interp_on_alongtrack
from inr4ssh._src.metrics.psd import compute_psd_scores, select_track_segments
from inr4ssh._src.metrics.stats import (
    calculate_nrmse,
    calculate_nrmse_elementwise,
    calculate_rmse_elementwise,
)
from inr4ssh._src.models.activations import get_activation
from inr4ssh._src.models.siren import ModulatedSirenNet, Modulator, Siren, SirenNet
from inr4ssh._src.postprocess.ssh_obs import postprocess
from inr4ssh._src.preprocess.coords import (
    correct_coordinate_labels,
    correct_longitude_domain,
)
from inr4ssh._src.preprocess.subset import spatial_subset, temporal_subset
from inr4ssh._src.viz.psd import plot_psd_score, plot_psd_spectrum
from loguru import logger
from skorch import NeuralNetRegressor
from skorch.callbacks import EarlyStopping, LRScheduler, WandbLogger, Checkpoint
from skorch.dataset import ValidSplit
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.notebook import tqdm as tqdm

pl.seed_everything(123)

import matplotlib.pyplot as plt
import seaborn as sns
from inr4ssh._src.viz.movie import create_movie

import config

sns.reset_defaults()
sns.set_context(context="talk", font_scale=0.7)


import wandb

def main(args):

    # modify args
    args.train_data_dir = "/Users/eman/.CMVolumes/cal1_workdir/data/dc_2021/raw/train"
    args.ref_data_dir = "/Users/eman/.CMVolumes/cal1_workdir/data/dc_2021/raw/ref"
    args.test_data_dir = "/Users/eman/.CMVolumes/cal1_workdir/data/dc_2021/raw/test"
    args.wandb_log_dir = "/Users/eman/code_projects/logs"
    # #
    args.time_min = "2017-01-01"
    args.time_max = "2017-02-01"
    args.eval_time_min = "2017-01-01"
    args.eval_time_max = "2017-02-01"
    args.eval_dtime = "12_h"

    
    # INITIALIZE LOGGER
    logger.info("Initializing logger...")
    wandb_run = wandb.init(
            config=args,
            mode=args.wandb_mode,
            project=args.wandb_project,
            entity=args.wandb_entity,
            dir=args.wandb_log_dir,
            resume=args.wandb_resume
        )
    
    # DATA MODULE
    logger.info("Initializing data module...")
    dm = SSHAltimetry(args)
    dm.setup()

    # objects
    logger.info("saving scaler transform...")
    path_scaler = "./scaler.pickle"

    # models to save
    save_object(dm.scaler, path_scaler)

    # save with wandb
    logger.info("logging scaler transform...")
    wandb_run.save(str(path_scaler))

    x_train, y_train = dm.ds_train[:]
    x_valid, y_valid = dm.ds_valid[:]
    X_test, = dm.ds_predict[:]
    x_train = torch.cat([x_train, x_valid])
    y_train = torch.cat([y_train, y_valid])

    logger.info("Creating neural network...")
    dim_in = x_train.shape[1]
    dim_hidden = args.hidden_dim
    dim_out = y_train.shape[1]
    num_layers = args.n_hidden
    w0 = args.siren_w0
    w0_initial = args.siren_w0_initial
    c = args.siren_c
    final_activation = get_activation(args.final_activation)

    siren_net = SirenNet(
        dim_in=dim_in,
        dim_hidden=dim_hidden,
        dim_out=dim_out,
        num_layers=num_layers,
        w0=w0,
        w0_initial=w0_initial,
        c=c,
        final_activation=final_activation
    )

    logger.info(f"Setting Device: {args.device}")
    logger.info("Setting callbacks...")
    lr_scheduler = LRScheduler(
        policy="ReduceLROnPlateau",
        monitor="valid_loss",
        mode="min",
        factor=getattr(args, "lr_schedule_factor", 0.1),
        patience=getattr(args, "lr_schedule_patience", 10),
    )

    # early stopping
    estop_callback = EarlyStopping(
        monitor="valid_loss",
        patience=getattr(args, "lr_estopping_patience", 20),
    )
    
    # wandb logger
    wandb_callback = WandbLogger(wandb_run, save_model=True)
    
    # checkpoint
    save_path = wandb.run.dir
    
    cp_callback = Checkpoint(
        dirname=f"{Path(save_path).joinpath('checkpoints')}",
        monitor="valid_loss_best",
        f_params="params.ckpt",
        f_optimizer="params.ckpt",
        f_criterion="criterion.ckpt",
        f_history="history.json",
        f_pickle="model.ckpt"
        
    )

    callbacks = [
        ("earlystopping", estop_callback),
        ("lrscheduler", lr_scheduler),
        ("wandb_logger", wandb_callback),
        ("checkpoint_logger", cp_callback),
    ]

    # train split percentage
    train_split = ValidSplit(1.0 - args.train_size, stratified=False)

    skorch_net = NeuralNetRegressor(
        module=siren_net,
        max_epochs=args.num_epochs,
        lr=args.learning_rate,
        batch_size=args.batch_size,
        device=args.device,
        optimizer=torch.optim.Adam,
        train_split=train_split,
        callbacks=callbacks

    )
    logger.info("Training...")
    skorch_net.fit(x_train, y_train)

    # TESTING
    logger.info("Making predictions...")
    t0 = time.time()
    predictions = skorch_net.predict(X_test)
    t1 = time.time() - t0

    logger.info(f"Time Taken for {X_test.shape[0]} points: {t1:.4f} secs")
    wandb_run.log(
            {
                "time_predict": t1,
            }
        )
    # ==================================
    # PREDICTIONS - ALONGTRACK
    # ==================================

    from utils import get_alongtrack_prediction_ds

    X_test, y_test = get_alongtrack_prediction_ds(dm, args, logger)
    
    logger.info(f"Predicting alongtrack data...")
    t0 = time.time()
    predictions = skorch_net.predict(torch.Tensor(X_test))
    t1 = time.time() - t0


    from utils import get_alongtrack_stats

    logger.info("Calculating alongtrack stats...")
    get_alongtrack_stats(y_test, predictions, logger, wandb_run)

    # PSD
    logger.info(f"Getting PSD Scores...")
    psd_metrics = compute_psd_scores(
        ssh_true=y_test.squeeze(),
        ssh_pred=predictions.squeeze(),
        delta_x=args.eval_psd_velocity * args.eval_psd_delta_t,
        npt=None,
        scaling="density",
        noverlap=0,
    )

    from utils import plot_psd_figs

    plot_psd_figs(psd_metrics, logger, wandb_run, method="grid")
    
    wandb_run.finish()

    return None


if __name__ == '__main__':
    # initialize argparse
    parser = argparse.ArgumentParser()

    # add all experiment arguments
    parser = config.add_logging_args(parser)
    parser = config.add_data_dir_args(parser)
    parser = config.add_data_preprocess_args(parser)
    parser = config.add_feature_transform_args(parser)
    parser = config.add_train_split_args(parser)
    parser = config.add_dataloader_args(parser)
    parser = config.add_model_args(parser)
    parser = config.add_loss_args(parser)
    parser = config.add_optimizer_args(parser)
    parser = config.add_eval_data_args(parser)
    parser = config.add_eval_metrics_args(parser)
    parser = config.add_viz_data_args(parser)

    # parse args
    args = parser.parse_args()

    main(args)