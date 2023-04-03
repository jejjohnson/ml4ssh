from pathlib import Path
import hydra
from omegaconf import OmegaConf
import torch
import loguru
import pyrootutils

pyrootutils.setup_root(__file__, indicator=".root", pythonpath=True)

root = pyrootutils.find_root(search_from=__file__, indicator=".root")


def download(cfg):
    # ===============================
    # logger
    # ===============================
    hydra.utils.instantiate(cfg.data)
    # # ===============================
    # # DATA MODULE
    # # ===============================
    # # data module
    # preprocessing = hydra.utils.call(cfg.preprocessing)
    # postprocessing = hydra.utils.call(cfg.postprocessing)
    # data = hydra.utils.call(cfg.data)
    # dm = hydra.utils.call(cfg.datamodule)

    # # ===============================
    # # MODEL
    # # ===============================
    # # transforms, model, optimizers
    # optimizer = hydra.utils.call(cfg.optimizer)
    # model = hydra.utils.call(cfg.model)
    # lr_scheduler = hydra.utils.call(cfg.lr_scheduler)
    # callbacks = hydra.utils.call(cfg.callbacks)

    # # ===============================
    # # LEARNER
    # # ===============================
    # # learner
    # learner
    # # trainer = hydra.utils.call(cfg.trainer, model, lr_scheduler, optimizer)
    # trainer = hydra.utils.call(cfg.trainer, learner=le, logger=logger)

    # # ===============================
    # # FIT
    # # ===============================
    # # fit
    # trainer.fit(...)

    # # ===============================
    # # PREDICTIONS
    # # ===============================
    # predictions = hydra.utils.call(cfg.predictions, trainer)

    # # ===============================
    # # DIAGNOSTICS
    # # ===============================
    # diagnostics = hydra.utils.call(cfg.diagnostics, predictions)
    pass


if __name__ == "__main__":
    main()
