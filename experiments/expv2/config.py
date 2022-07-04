import argparse
# ======================
# LOGGING
# ======================
wandb_project = "inr4ssh"
wandb_entity = "ige"
wandb_log_dir = "./"
wandb_resume = "allow"
wandb_mode = "offline"
smoke_test = "store_true"
wandb_id = None

def add_logging_args(parser):
    parser.add_argument('--wandb-project', type=str, default=wandb_project)
    parser.add_argument('--wandb-entity', type=str, default=wandb_entity)
    parser.add_argument('--wandb-log-dir', type=str, default=wandb_log_dir)
    parser.add_argument('--wandb-resume', type=str, default=wandb_resume)
    parser.add_argument('--wandb-mode', type=str, default=wandb_mode)
    parser.add_argument('--smoke-test', action="store_true")
    parser.add_argument('--wandb-id', type=str, default=wandb_id)
    return parser

# ======================
# Data Directories
# ======================
train_data_dir = "/home/johnsonj/data/dc_2021/raw/train"
ref_data_dir = "/home/johnsonj/data/dc_2021/raw/ref/"
test_data_dir = "/home/johnsonj/data/dc_2021/raw/test/"

def add_data_dir_args(parser):
    parser.add_argument('--train-data-dir', type=str, default=train_data_dir)
    parser.add_argument('--ref-data-dir', type=str, default=ref_data_dir)
    parser.add_argument('--test-data-dir', type=str, default=test_data_dir)
    return parser

# ======================
# DATA PREPROCESS
# ======================
# longitude subset
lon_min = 285.0
lon_max = 315.0
dlon = 0.2
lon_buffer = 1.0
# latitude subset
lat_min = 23.0
lat_max = 53.0
dlat = 0.2
lat_buffer = 1.0
# temporal subset
time_min = "2016-12-01"
time_max = "2018-01-31"
dtime = "1_D"
time_buffer = 7.0

def add_data_preprocess_args(parser):
    # longitude subset
    parser.add_argument('--lon-min', type=float, default=lon_min)
    parser.add_argument('--lon-max', type=float, default=lon_max)
    parser.add_argument('--dlon', type=float, default=dlon)
    
    # latitude subset
    parser.add_argument('--lat-min', type=float, default=lat_min)
    parser.add_argument('--lat-max', type=float, default=lat_max)
    parser.add_argument('--dlat', type=float, default=dlat)
    
    # temporal subset
    parser.add_argument('--time-min', type=str, default=time_min)
    parser.add_argument('--time-max', type=str, default=time_max)
    parser.add_argument('--dtime', type=str, default=dtime)
    
    # Buffer Params
    parser.add_argument('--lon-buffer', type=float, default=lon_buffer)
    parser.add_argument('--lat-buffer', type=float, default=lat_buffer)
    parser.add_argument('--time-buffer', type=float, default=time_buffer)
    return parser

# ======================
# FEATURES
# ======================
julian_time = True
abs_time_min = "2005-01-01"
abs_time_max = "2022-01-01"
feature_scaler = "minmax"
spherical_radius = 1.0
min_time_scale = -1.0
max_time_scale = 1.0

def add_feature_transform_args(parser):
    # temporal coordinates transform
    parser.add_argument("--julian-time", type=bool, default=julian_time)
    parser.add_argument("--abs-time-min", type=str, default=abs_time_min)
    parser.add_argument("--abs-time-max", type=str, default=abs_time_max)
    parser.add_argument("--feature-scaler", type=str, default=feature_scaler)
    parser.add_argument("--min-time-scale", type=float, default=min_time_scale)
    parser.add_argument("--max-time-scale", type=float, default=max_time_scale)
    # spatial coordinates transform
    parser.add_argument("--spherical-radius", type=float, default=spherical_radius)
    return parser

# ======================
# TRAIN/VAL SPLIT
# ======================
train_size = 0.9
train_split_method = "random" # random, temporal, spatial
train_seed_split = 666
train_seed_shuffle = 321
train_split_time_freq = None # "1_D"
train_split_spatial = "random" # "regular" # "upper" # "lower" # "altimetry"


def add_train_split_args(parser):
    parser.add_argument("--train-size", type=float, default=train_size)
    parser.add_argument("--train-seed-split", type=int, default=train_seed_split)
    parser.add_argument("--train-seed-shuffle", type=int, default=train_seed_shuffle)
    parser.add_argument("--train-split-method", type=str, default=train_split_method)
    # temporal split options
    parser.add_argument("--train-split-time-freq", type=int, default=train_split_time_freq)
    # temporal split options
    parser.add_argument("--train-split-spatial", type=str, default=train_split_spatial)
    return parser

# ======================
# DATALOADER
# ======================
# dataloader
dl_train_shuffle = True
dl_pin_memory = False
dl_num_workers = 0
batch_size = 4096
batch_size_eval = 10_000

def add_dataloader_args(parser):
    parser.add_argument("--dl-train-shuffle", type=bool, default=dl_train_shuffle)
    parser.add_argument("--dl-pin-memory", type=bool, default=dl_pin_memory)
    parser.add_argument("--dl-num-workers", type=int, default=dl_num_workers)
    parser.add_argument('--batch-size', type=int, default=batch_size)
    parser.add_argument('--batch-size-eval', type=int, default=batch_size_eval)
    return parser
# ======================
# MODEL
# ======================
model = "siren"

# encoder specific
encoder = None

# model specific
hidden_dim = 512
n_hidden = 6
model_seed = 42
activation = "swish"
final_activation = "identity"

# siren specific
siren_w0_initial = 30.0
siren_w0 = 1.0
siren_final_scale = 1.0
siren_c = 6.0

def add_model_args(parser):
    parser.add_argument('--model', type=str, default=model)
    parser.add_argument('--encoder', type=str, default=encoder)
    # NEURAL NETWORK SPECIFIC
    parser.add_argument('--hidden-dim', type=int, default=hidden_dim)
    parser.add_argument('--n-hidden', type=int, default=n_hidden)
    parser.add_argument('--model-seed', type=str, default=model_seed)
    parser.add_argument('--activation', type=str, default=activation)
    parser.add_argument('--final-activation', type=str, default=final_activation)

    # SIREN SPECIFIC
    parser.add_argument('--siren-w0-initial', type=float, default=siren_w0_initial)
    parser.add_argument('--siren-w0', type=float, default=siren_w0)
    parser.add_argument('--siren-c', type=float, default=siren_c)
    parser.add_argument('--siren-final-scale', type=float, default=siren_final_scale)
    return parser

# ======================
# LOSSES
# ======================
loss = "mse"
loss_reduction = "mean"

# QG PINN Loss Args
loss_qg = False
loss_qg_reg = 0.1

def add_loss_args(parser):
    parser.add_argument('--loss', type=str, default=loss)
    parser.add_argument('--loss-reduction', type=str, default=loss_reduction)
    parser.add_argument('--loss-qg', type=bool, default=loss_qg)
    parser.add_argument('--loss-qg-reg', type=float, default=loss_qg_reg)
    return parser

# ======================
# OPTIMIZER
# ======================
optimizer = "adam" # "adamw" # "adamax"
learning_rate = 1e-4
num_epochs = 300


# LR Scheduler
lr_scheduler = "reduce" # "cosine" # "onecyle" #
patience = 100

def add_optimizer_args(parser):
    # optimizer args
    parser.add_argument('--optimizer', type=str, default=optimizer)
    parser.add_argument('--learning-rate', type=float, default=learning_rate)
    parser.add_argument('--num-epochs', type=int, default=num_epochs)

    # learning rate scheduler
    parser.add_argument('--lr-scheduler', type=str, default=lr_scheduler)
    parser.add_argument('--patience', type=int, default=patience)
    return parser

# ======================
# Evaluation DATA
# ======================
eval_lon_min = 295.0
eval_lon_max = 305.0
eval_dlon = 0.2
eval_lat_min = 33.0
eval_lat_max = 43.0
eval_dlat = 0.2
eval_time_min = "2017-01-01"
eval_time_max = "2017-12-31"
eval_dtime = "1_D"
eval_lon_buffer = 2.0
eval_lat_buffer = 2.0
eval_time_buffer = 7.0

def add_eval_data_args(parser):
    # longitude subset
    parser.add_argument('--eval-lon-min', type=float, default=eval_lon_min)
    parser.add_argument('--eval-lon-max', type=float, default=eval_lon_max)
    parser.add_argument('--eval-dlon', type=float, default=eval_dlon)
    
    # latitude subset
    parser.add_argument('--eval-lat-min', type=float, default=eval_lat_min)
    parser.add_argument('--eval-lat-max', type=float, default=eval_lat_max)
    parser.add_argument('--eval-dlat', type=float, default=eval_dlat)
    
    # temporal subset
    parser.add_argument('--eval-time-min', type=str, default=eval_time_min)
    parser.add_argument('--eval-time-max', type=str, default=eval_time_max)
    parser.add_argument('--eval-dtime', type=str, default=eval_dtime)
    
    # OI params
    parser.add_argument('--eval-lon-buffer', type=float, default=eval_lon_buffer)
    parser.add_argument('--eval-lat-buffer', type=float, default=eval_lat_buffer)
    parser.add_argument('--eval-time-buffer', type=float, default=eval_time_buffer)
    
    return parser

# ======================
# Evaluation METRICS
# ======================

# binning along track
eval_bin_lat_step = 1.0
eval_bin_lon_step = 1.0
eval_bin_time_step = "1D"
eval_min_obs = 10

# power spectrum
eval_psd_delta_t = 0.9434
eval_psd_velocity = 6.77
eval_psd_jitter = 1e-4

def add_eval_metrics_args(parser):
    # binning along track
    parser.add_argument('--eval-bin-lat-step', type=float, default=eval_bin_lat_step)
    parser.add_argument('--eval-bin-lon-step', type=float, default=eval_bin_lon_step)
    parser.add_argument('--eval-bin-time-step', type=str, default=eval_bin_time_step)
    parser.add_argument('--eval-min-obs', type=int, default=eval_min_obs)
    # power spectrum
    parser.add_argument('--eval-psd-delta-t', type=float, default=eval_psd_delta_t)
    parser.add_argument('--eval-psd-velocity', type=float, default=eval_psd_velocity)
    parser.add_argument('--eval-psd-jitter', type=float, default=eval_psd_jitter)

    return parser

# ======================
# VIZ
# ======================
viz_lon_min = 295.0
viz_lon_max = 305.0
viz_dlon = 0.1
viz_lon_buffer = 1.0
# latitude subset
viz_lat_min = 33.0
viz_lat_max = 43.0
viz_dlat = 0.1
viz_lat_buffer = 1.0
# temporal subset
viz_time_min = "2017-01-01"
viz_time_max = "2017-12-31"
viz_dtime = "1_D"
viz_time_buffer = 7.0

def add_viz_data_args(parser):
    # binning along track
    parser.add_argument('--viz-lon-min', type=float, default=viz_lon_min)
    parser.add_argument('--viz-lon-max', type=float, default=viz_lon_max)
    parser.add_argument('--viz-dlon', type=float, default=viz_dlon)
    parser.add_argument('--viz-lon-buffer', type=float, default=viz_lon_buffer)
    # power spectrum
    parser.add_argument('--viz-lat-min', type=float, default=viz_lat_min)
    parser.add_argument('--viz-lat-max', type=float, default=viz_lat_max)
    parser.add_argument('--viz-dlat', type=float, default=viz_dlat)
    parser.add_argument('--viz-lat-buffer', type=float, default=viz_lat_buffer)

    parser.add_argument('--viz-time-min', type=str, default=viz_time_min)
    parser.add_argument('--viz-time-max', type=str, default=viz_time_max)
    parser.add_argument('--viz-dtime', type=str, default=viz_dtime)
    parser.add_argument('--viz-time-buffer', type=float, default=viz_time_buffer)
    return parser