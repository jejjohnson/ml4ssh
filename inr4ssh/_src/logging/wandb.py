import wandb


def load_wandb_run_config(entity: str, project: str, id: str):

    api = wandb.Api()
    reference = str(f"{entity}/{project}/{id}")
    prev_run = api.run(reference)
    # prev_run = api.run("ige/inr4ssh/pbi50xfu")
    prev_cfg = prev_run.config

    return prev_cfg


def load_wandb_checkpoint(
    entity: str,
    project: str,
    reference: str,
    mode: str = "offline",
    root=None,
):
    """Loads a checkpoint given a reference

    Args:
        reference (str): _description_
        mode (str, optional): _description_. Defaults to "disabled".

    Returns:
        _type_: _description_
    """
    api = wandb.Api()
    reference = f"{entity}/{project}/{reference}"
    artifact = api.artifact(reference)
    artifact_dir = artifact.download()

    return artifact_dir
    # # TODO: add root for artifact download
    # run = wandb.init(entity=entity, project=project, mode=mode, resume=False)
    # reference = f"{entity}/{project}/{reference}"
    # artifact = run.use_artifact(reference, type="checkpoints")
    # artifact_dir = artifact.download()
    # wandb.finish()

    # return artifact_dir
