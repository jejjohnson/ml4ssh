from typing import Optional, List
from simple_parsing import ArgumentParser
from simple_parsing.helpers import Serializable
from dataclasses import dataclass, field
from simple_parsing.helpers import list_field

# ======================
# LOGGING
# ======================
@dataclass
class Logging(Serializable):
    project: str = "inr4ssh"
    entity: str = "ige"
    log_dir: str = "/mnt/meom/workdir/johnsonj/logs"
    resume: str = "allow"
    mode: str = "offline"
    smoke_test: str = "store_true"
    id: Optional[str] = None
    run_path: Optional[str] = None
    model_path: Optional[str] = None

# ======================
# Data Directories
# ======================
@dataclass
class DataDir(Serializable):
    train_data_dir: str = "/home/johnsonj/data/dc_2021/raw/train"
    ref_data_dir: str = "/home/johnsonj/data/dc_2021/raw/ref/"
    test_data_dir: str = "/home/johnsonj/data/dc_2021/raw/test/"

# ======================
# DATA FEATURES
# ======================
@dataclass
class Features(Serializable):
    variable: str = "p"
    # spatial subset
    minmax_spatial: bool = True
    minmax_fixed_spatial: bool = True
    min_spatial: float = -3.14
    max_spatial: float = 3.14
    # temporal subset
    minmax_temporal: bool = True
    minmax_fixed_temporal: bool = True
    min_temporal: float = 0
    max_temporal: float = 124

# ======================
# TRAIN/VAL SPLIT
# ======================
@dataclass
class TrainTestSplit(Serializable):
    # spatial subset
    coarsen_Nx: int = 2
    coarsen_Ny: int = 2
    # temporal subset
    coarsen_time: int = 4
    # noise
    noise: str = "gauss"
    sigma: float = 0.01
    seed_noise: float = 666
    # LABELs GENERATION
    split_method: Optional[str] = "random"  # random, temporal, spatial
    missing_data: float = 0.90
    seed_missing_data: int = 777
    # train val split
    train_size: float = 0.9
    seed_split : int = 999

# ======================
# DATALOADER
# ======================
@dataclass
class DataLoader(Serializable):
    # dataloader
    train_shuffle: bool = True
    pin_memory: bool = False
    num_workers: int = 0
    batch_size: int = 4096
    batch_size_eval: int = 10_000

# ======================
# MODEL
# ======================
@dataclass
class Model(Serializable):
    model: str = "siren"
    # encoder specific
    encoder: Optional[str] = None


@dataclass
class Siren(Serializable):
    num_layers: int = 5
    hidden_dim: int = 512
    model_seed: int = 42
    use_bias: bool = True
    final_activation: str = "identity"
    w0_initial: float = 30.0
    w0: float = 1.0
    final_scale: float = 1.0
    c: float = 6.0


@dataclass
class ModulatedSiren(Siren):
    latent_dim: int = 256
    num_layers_latent: int = 3
    operation: str = "sum"

@dataclass
class MFN(Serializable):
    num_layers: int = 5
    hidden_dim: int = 512
    use_bias: bool = True
    input_scale: float = 256.0
    weight_scale: float = 1.0
    alpha: float = 6.0
    beta: float = 1.0
    final_activation: str = "identity"

# ======================
# LOSSES
# ======================
@dataclass
class Losses(Serializable):
    loss: str = "mse" # Options: "mse", "nll", "kld"
    reduction: str = "mean"

    # QG PINN Loss Args
    qg: bool = False
    qg_reg: str = 0.1

# ======================
# OPTIMIZER
# ======================
@dataclass
class Optimizer(Serializable):
    optimizer: str = "adam" # "adamw" # "adamax"
    learning_rate: float = 1e-4
    num_epochs: int = 300
    min_epochs: int = 1
    device: str = "cpu"
    gpus: int = 0 # the number of GPUS (pytorch-lightning)

@dataclass
class LRScheduler(Serializable):
    # LR Scheduler
    lr_scheduler: str = "reduce" # Options: "cosine", "onecyle", "step", "multistep"
    patience: int = 10
    factor: float = 0.1
    steps: int = 250
    gamma: float = 0.1
    min_learning_rate: float = 1e-5
    milestones: List[int] = list_field(500, 1000, 1500, 2000, 2500)

@dataclass
class Callbacks(Serializable):
    # wandb logging
    wandb: bool = True
    model_checkpoint: bool = True

    # early stopping
    early_stopping: bool = True
    patience: int = 20