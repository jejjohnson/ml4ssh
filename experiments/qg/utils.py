from pathlib import Path
import torch
import tqdm
import numpy as np
from loguru import logger
from pytorch_lightning.callbacks import ModelCheckpoint
from inr4ssh._src.models.siren import SirenNet
from inr4ssh._src.io import get_wandb_config, get_wandb_model
from figures import plot_maps


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


def load_model_from_cpkt(config):

    # load previous model
    logger.info(f"Loading previous wandb model...")
    best_model = get_wandb_model(config.run_path, config.model_path)
    best_model.download(replace=True)

    return best_model.name


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
            q = diffops_simp.divergence(p_grad, ix, (0, 1))
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
