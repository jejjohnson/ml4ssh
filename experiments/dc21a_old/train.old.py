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
from torch.optim.lr_scheduler import CosineAnnealingLR

from skorch import NeuralNetRegressor
from skorch.dataset import ValidSplit
from skorch.callbacks import EarlyStopping, LRScheduler, WandbLogger

from inr4ssh._src.io import save_object
from inr4ssh._src.interp import interp_on_alongtrack
from inr4ssh._src.data.ssh_obs import load_ssh_correction, load_ssh_altimetry_data_test
from inr4ssh._src.datamodules.dc21a import SSHAltimetry
from inr4ssh._src.features.data_struct import df_2_xr
from inr4ssh._src.models.activations import get_activation
from inr4ssh._src.models.siren import SirenNet
from inr4ssh._src.models.utils import get_torch_device
from inr4ssh._src.postprocess.ssh_obs import postprocess
from inr4ssh._src.metrics.psd import compute_psd_scores

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


def main(args):

    # INITIALIZE LOGGER
    logger.info("Initializing logger...")

    log_options = args.logging

    # print(wandb_vars)
    params_dict = simpleargs_2_ndict(args)
    wandb_run = wandb.init(
        config=params_dict,
        mode=log_options.wandb_mode,
        project=log_options.wandb_project,
        entity=log_options.wandb_entity,
        dir=log_options.wandb_log_dir,
        resume=log_options.wandb_resume,
    )

    # DATA MODULE
    logger.info("Initializing data module...")
    dm = SSHAltimetry(
        data=args.data,
        preprocess=args.preprocess,
        traintest=args.traintest,
        features=args.features,
        dataloader=args.dataloader,
        eval=args.eval,
    )

    dm.setup()

    # objects
    logger.info("saving scaler transform...")
    path_scaler = "./scaler.pickle"

    # models to save
    save_object(dm.scaler, path_scaler)

    # save with wandb
    logger.info("logging scaler transform...")
    wandb_run.save(str(path_scaler))

    logger.info("extracting train and test...")
    x_train, y_train = dm.ds_train[:]
    x_valid, y_valid = dm.ds_valid[:]
    (X_test,) = dm.ds_predict[:]
    x_train = torch.cat([x_train, x_valid])
    y_train = torch.cat([y_train, y_valid])

    logger.info(f"Creating {args.model.model} neural network...")

    dim_in = x_train.shape[1]
    dim_out = y_train.shape[1]

    # update params
    params_dict["model"].update({"dim_in": dim_in, "dim_out": dim_out})
    wandb_run.config.update(params_dict, allow_val_change=True)

    from models import model_factory

    net = model_factory(
        model=args.model.model, dim_in=dim_in, dim_out=dim_out, config=args
    )

    from optimizers import get_optimizer, get_lr_scheduler
    from callbacks import get_callbacks

    logger.info(f"Adding {args.optimizer.optimizer} optimizer...")
    optimizer = get_optimizer(args.optimizer)

    logger.info(f"Adding {args.lr_scheduler.lr_scheduler} optimizer...")

    callbacks = list()

    callbacks += [("lr_scheduler", get_lr_scheduler(args.lr_scheduler))]

    callbacks += get_callbacks(args.callbacks, wandb_run)

    #
    # logger.info(f"Setting Device: {args.device}")
    # logger.info("Setting callbacks...")
    # lr_scheduler = LRScheduler(
    #     policy="ReduceLROnPlateau",
    #     monitor="valid_loss",
    #     mode="min",
    #     factor=getattr(args, "lr_schedule_factor", 0.05),
    #     patience=getattr(args, "lr_schedule_patience", 10),
    # )
    #
    # # early stopping
    # estop_callback = EarlyStopping(
    #     monitor="valid_loss",
    #     patience=getattr(args, "lr_estopping_patience", 20),
    # )
    #
    # # wandb logger
    # wandb_callback = WandbLogger(wandb_run, save_model=True)
    #
    # callbacks = [
    #     ("earlystopping", estop_callback),
    #     ("lrscheduler", lr_scheduler),
    #     ("wandb_logger", wandb_callback),
    # ]
    #
    # # train split percentage
    train_split = ValidSplit(1.0 - args.traintest.train_size, stratified=False)
    #
    skorch_net = NeuralNetRegressor(
        module=net,
        max_epochs=args.optimizer.num_epochs,
        lr=args.optimizer.learning_rate,
        batch_size=args.dataloader.batch_size,
        device=args.optimizer.device,
        optimizer=optimizer,
        train_split=train_split,
        callbacks=callbacks,
    )
    logger.info("Training...")
    skorch_net.fit(x_train, y_train)

    # ==============================
    # GRID PREDICTIONS
    # ==============================

    logger.info("GRID STATS...")

    # TESTING
    logger.info("Making predictions (grid)...")
    t0 = time.time()
    predictions = skorch_net.predict(X_test)
    t1 = time.time() - t0

    logger.info(f"Time Taken for {X_test.shape[0]} points: {t1:.4f} secs")
    wandb_run.log(
        {
            "time_predict_grid": t1,
        }
    )

    ds_oi = postprocess_predictions(predictions, dm, args, logger)

    alongtracks, tracks = get_interpolation_alongtrack_prediction_ds(
        ds_oi, args, logger
    )

    logger.info("Getting RMSE Metrics (GRID)...")
    rmse_metrics = get_grid_stats(alongtracks, args.metrics, logger, wandb_run)
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

    wandb_run.log(
        {
            "resolved_scale": psd_metrics.resolved_scale,
        }
    )
    #
    logger.info(f"Plotting PSD Score and Spectrum (Grid)...")
    plot_psd_figs(psd_metrics, logger, wandb_run, method="grid")
    logger.info("Finished GRID Script...!")

    # ==============================
    # ALONGTRACK PREDICTIONS
    # ==============================

    logger.info("ALONGTRACK STATS...")

    X_test, y_test = get_alongtrack_prediction_ds(dm, args, logger)

    logger.info(f"Predicting alongtrack data...")
    t0 = time.time()
    predictions = skorch_net.predict(torch.Tensor(X_test))
    t1 = time.time() - t0

    wandb_run.log(
        {
            "time_predict_alongtrack": t1,
        }
    )

    logger.info("Calculating stats (alongtrack)...")
    get_alongtrack_stats(y_test, predictions, logger, wandb_run)

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

    logger.info(f"Plotting PSD Score and Spectrum (AlongTrack)...")
    plot_psd_figs(psd_metrics, logger, wandb_run, method="alongtrack")

    wandb_run.finish()


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
