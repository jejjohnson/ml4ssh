from typing import Optional, List
from simple_parsing import ArgumentParser
from simple_parsing.helpers import Serializable
from dataclasses import dataclass
# ======================
# LOGGING
# ======================
@dataclass
class Logging(Serializable):
    wandb_project: str = "inr4ssh"
    wandb_entity: str = "ige"
    wandb_log_dir: str = "/mnt/meom/workdir/johnsonj/logs"
    wandb_resume: str = "allow"
    wandb_mode: str = "offline"
    smoke_test: str = "store_true"
    wandb_id: Optional[str] = None

# ======================
# Data Directories
# ======================
@dataclass
class DataDir(Serializable):
    train_data_dir: str = "/home/johnsonj/data/dc_2021/raw/train"
    ref_data_dir: str = "/home/johnsonj/data/dc_2021/raw/ref/"
    test_data_dir: str = "/home/johnsonj/data/dc_2021/raw/test/"

# ======================
# DATA PREPROCESS
# ======================
@dataclass
class PreProcess(Serializable):
    # longitude subset
    lon_min: float = 285.0
    lon_max: float = 315.0
    dlon: float = 0.2
    lon_buffer: float = 1.0
    # latitude subset
    lat_min: float = 23.0
    lat_max: float = 53.0
    dlat: float = 0.2
    lat_buffer: float = 1.0
    # temporal subset
    time_min: str = "2016-12-01"
    time_max: str = "2018-01-31"
    dtime: str = "1_D"
    time_buffer: float = 7.0


# ======================
# FEATURES
# ======================
@dataclass
class Features(Serializable):
    # spatial
    julian_time: bool = True
    abs_time: bool = True
    abs_time_min: str = "2005-01-01"
    abs_time_max: str = "2022-01-01"
    feature_scaler: str = "minmax"

    # temporal
    cartesian: bool = True
    minmax_spatial: bool = True
    minmax_temporal: bool = True
    spherical_radius: float = 1.0
    min_time_scale: float = -1.0
    max_time_scale: float = 1.0

# ======================
# TRAIN/VAL SPLIT
# ======================
@dataclass
class TrainTestSplit(Serializable):
    train_size: float = 0.9
    split_method: Optional[str] = "random" # random, temporal, spatial
    seed_split: int = 666
    seed_shuffle: int = 321
    split_time_freq: Optional[str] = None # "1_D"
    split_spatial: str = "random" # "regular" # "upper" # "lower" # "altimetry"

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
    loss: str = "mse"
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
    lr_scheduler: str = "reduce" # "cosine" # "onecyle" #
    patience: int = 10
    factor: float = 0.1

@dataclass
class Callbacks(Serializable):
    # wandb logging
    wandb: bool = True
    save_model: bool = True

    # early stopping
    early_stopping: bool = True
    patience: int = 20

# ======================
# CALLBACKS
# ======================

# ======================
# Evaluation DATA
# ======================
@dataclass
class EvalData(Serializable):
    lon_min: float = 295.0
    lon_max: float = 305.0
    dlon: float = 0.2
    lat_min: float = 33.0
    lat_max: float = 43.0
    dlat: float = 0.2
    time_min: str = "2017-01-01"
    time_max: str = "2017-12-31"
    dtime_freq: int = 1
    dtime_unit: str = "D"
    lon_buffer: float = 2.0
    lat_buffer: float = 2.0
    time_buffer: float = 7.0


# ======================
# Evaluation METRICS
# ======================
@dataclass
class Metrics(Serializable):
    # binning along track
    bin_lat_step: float = 1.0
    bin_lon_step: float = 1.0
    bin_time_step: str = "1D"
    min_obs: int = 10

    # power spectrum
    delta_t: float = 0.9434
    velocity: float = 6.77
    jitter: float = 1e-4

# ======================
# VIZ
# ======================
@dataclass
class Viz(Serializable):
    lon_min: float = 295.0
    lon_max: float = 305.0
    dlon: float = 0.1
    lon_buffer: float = 1.0
    # latitude subset
    lat_min: float = 33.0
    lat_max: float = 43.0
    dlat: float = 0.1
    lat_buffer: float = 1.0
    # temporal subset
    time_min: str = "2017-01-01"
    time_max: str = "2017-12-31"
    dtime_freq: int = 1
    dtime_unit: str = "D"
    time_buffer: float = 7.0
