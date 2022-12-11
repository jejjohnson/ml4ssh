import sys, os

import ml_collections

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from pyprojroot import here

# spyder up to find the root
root = here(project_files=[".root"])

# append to path
sys.path.append(str(root))

from loguru import logger
from pathlib import Path
from inr4ssh._src.logging.wandb import load_wandb_checkpoint, load_wandb_run_config
from ml_collections import config_dict


def update_config_pretrain(config):

    if config.pretrain:

        # load previous config
        logger.info(f"Loading previous wandb config...")
        logger.info(
            f"wandb run: {config.pretrain_entity}/{config.pretrain_project}/{config.pretrain_id}"
        )
        prev_config = load_wandb_run_config(
            entity=config.pretrain_entity,
            project=config.pretrain_project,
            id=config.pretrain_id,
        )
        # print(prev_config)
        # prev_config = config_dict.ConfigDict(prev_config)

        # load previous checkpoint
        logger.info(f"Downloading prev run checkpoint...")
        logger.info(f"Prev Run: {config.pretrain_reference}")
        checkpoint_dir = load_wandb_checkpoint(
            entity=config.pretrain_entity,
            project=config.pretrain_project,
            reference=config.pretrain_reference,
            mode="online",
        )

        checkpoint_file = Path(checkpoint_dir).joinpath(config.pretrain_checkpoint)
        logger.info(f"Checkpoint file: {checkpoint_file}")

        # TODO: fix hack for pretraining config params
        logger.info(f"Hack: copying prev config pretrain params...")
        pretrain = True
        pretrain_id = config.pretrain_id
        pretrain_checkpoint = config.pretrain_checkpoint
        pretrain_reference = config.pretrain_reference

        # overwrite config
        logger.info(f"Overwriting previous config...")

        config = config_dict.ConfigDict(prev_config["model"])
        config.pretrain = pretrain
        config.pretrain_id = pretrain_id
        config.pretrain_checkpoint = pretrain_checkpoint
        config.pretrain_reference = pretrain_reference
        config.pretrain_checkpoint_file = checkpoint_file

    return config
