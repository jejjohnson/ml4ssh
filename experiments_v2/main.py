from pathlib import Path
import hydra
from omegaconf import OmegaConf
import torch
import loguru
import download


@hydra.main(config_path="../config", config_name="main", version_base="1.2")
def main(cfg):
    # ===============================
    # logger
    # ===============================

    stage = cfg.get("stage")
    loguru.logger.info(f"Initializing Stage: {stage}")

    if stage == "download":
        loguru.logger.info("Starting Download Script...")
        hydra.utils.instantiate(cfg.data)
    elif stage == "preprocess":
        hydra.utils.instantiate(cfg.preprocess)
    elif stage == "geoprocess":
        raise NotImplementedError()
    elif stage == "ml_ready":
        raise NotImplementedError()
    elif stage == "train":
        raise NotImplementedError()
    elif stage == "train_more":
        raise NotImplementedError()
    elif stage == "evaluation":
        raise NotImplementedError()
    elif stage == "viz":
        raise NotImplementedError()
    else:
        raise ValueError(f"Unrecognized stage: {stage}")
    # loguru.logger.info("Initializaing Logger...")
    # logger = hydra.utils.call(cfg.logger)
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
