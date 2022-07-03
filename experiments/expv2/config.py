# ======================
# LOGGING
# ======================
project = "inr4ssh"
entity = "ige"
log_dir = "./"
wandb_resume = "allow"
wandb_mode = "offline"
smoke_test = "store_true"
wandb_id = None

# ======================
# Data Directories
# ======================
train_data_dir = "/home/johnsonj/data/dc_2021/raw/train"
ref_data_dir = "/home/johnsonj/data/dc_2021/raw/ref/"
test_data_dir = "/home/johnsonj/data/dc_2021/raw/test/"

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

# ======================
# FEATURES
# ======================
julian_time = True
feature_scaler = "minmax"
min_scale = -1.0
max_scale = 1.0

# ======================
# TRAIN/VAL SPLIT
# ======================
split = "random" # random, temporal, spatial
seed_split = 666
seed_shuffle = 321
split_time_freq = "1_D"

# ======================
# MODEL
# ======================
model = "siren"


# encoder specific
encoder = None

# model specific
hidden_dim = 512
n_hidden = 6
seed_model = 42
activation = "swish"

# siren specific
w0_initial = 30.0
w0 = 1.0
final_scale = 1.0

# ======================
# LOSSES
# ======================
loss = "mse"

# ======================
# OPTIMIZER
# ======================
optimizer = "adam"
learning_rate = 1e-4
num_epochs = 300
batch_size = 4096
num_workers = 0

# LR Scheduler
patience = 100

# ======================
# Evaluation
# ======================
eval_batch_size = 10_000

# binning along track
eval_bin_lat_step = 1.0
eval_bin_lon_step = 1.0
eval_bin_time_step = "1D"
eval_min_obs = 10

# power spectrum
eval_psd_delta_t = 0.9434
eval_psd_velocity = 6.77
eval_psd_jitter = 1e-4

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
