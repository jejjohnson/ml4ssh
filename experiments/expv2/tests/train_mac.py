import sys, os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from pyprojroot import here

# spyder up to find the root
root = here(project_files=[".root"])
local = here(project_files=[".local"])

# append to path
sys.path.append(str(root))
sys.path.append(str(local))

import config
import argparse
import time
from loguru import logger

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from skorch import NeuralNetRegressor
from skorch.dataset import ValidSplit
from skorch.callbacks import EarlyStopping, LRScheduler, WandbLogger

from inr4ssh._src.data.ssh_obs import load_ssh_correction
from inr4ssh._src.datamodules.ssh_obs import SSHAltimetry
from inr4ssh._src.features.data_struct import df_2_xr
from inr4ssh._src.models.activations import get_activation
from inr4ssh._src.models.siren import SirenNet
from inr4ssh._src.models.utils import get_torch_device
from inr4ssh._src.postprocess.ssh_obs import postprocess


import wandb

def main(args):

    # modify args
    args.train_data_dir = "/Users/eman/.CMVolumes/cal1_workdir/data/dc_2021/raw/train"
    args.ref_data_dir = "/Users/eman/.CMVolumes/cal1_workdir/data/dc_2021/raw/ref"
    args.test_data_dir = "/Users/eman/.CMVolumes/cal1_workdir/data/dc_2021/raw/test"
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

    dm = dm.setup()

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

    callbacks = [
        ("earlystopping", estop_callback),
        ("lrscheduler", lr_scheduler),
        ("wandb_logger", wandb_callback),
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
    # POSTPROCESS
    # convert to da
    logger.info("Convert data to xarray ds...")
    ds_oi = dm.X_pred_index
    ds_oi["ssh"] = predictions
    ds_oi = df_2_xr(ds_oi)

    # open correction dataset
    logger.info("Loading SSH corrections...")
    ds_correct = load_ssh_correction(args.ref_data_dir)

    # correct predictions
    logger.info("Correcting SSH predictions...")
    ds_oi = postprocess(ds_oi, ds_correct)
    
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