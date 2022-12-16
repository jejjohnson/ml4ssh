from ml_collections import config_dict
import math


def get_wandb_config() -> config_dict.ConfigDict:
    config = config_dict.ConfigDict()

    config.mode = "disabled"
    config.project = "inr4ssh"
    config.entity = "ige"
    config.log_dir = "/Users/eman/code_projects/logs"
    config.resume = False
    config.id = config_dict.placeholder(str)
    return config


def get_datadir_raw():
    config = config_dict.ConfigDict()

    config.correction_dir = "/Volumes/EMANS_HDD/data/dc21b_ose/test_2/raw/correction/"
    config.obs_dir = "/Volumes/EMANS_HDD/data/dc21b_ose/test_2/raw/obs/"

    return config


def get_datadir_staging():
    config = config_dict.ConfigDict()

    config.staging_dir = "/Volumes/EMANS_HDD/data/dc21b_ose/test_2/ml_ready"

    return config


def get_preprocess_config():
    # preprocessing
    config = preprocess = config_dict.ConfigDict()
    preprocess.subset_time = subset_time = config_dict.ConfigDict()
    subset_time.subset_time = True
    subset_time.time_min = "2016-12-01"
    subset_time.time_max = "2018-01-31"

    config.subset_spatial = subset_spatial = config_dict.ConfigDict()
    subset_spatial.subset_spatial = True
    # subset_spatial.lon_min = -65.0
    # subset_spatial.lon_max = -55.0
    # subset_spatial.lat_min = 33.0
    # subset_spatial.lat_max = 43.0
    subset_spatial.lon_min = -75.0  # 285.0
    subset_spatial.lon_max = -45.0  # 315.0
    subset_spatial.lat_min = 23.0
    subset_spatial.lat_max = 53.0

    return config


def get_transformations_config():
    config = transform = config_dict.ConfigDict()
    transform.time_transform = "minmax"
    transform.time_min = "2016-06-01"
    transform.time_max = "2019-01-06"

    return config


def get_dataloader_config():
    config = dataloader = config_dict.ConfigDict()
    # train dataloader
    dataloader.batchsize_train = 32
    dataloader.num_workers_train = 0
    dataloader.shuffle_train = True
    dataloader.pin_memory_train = False
    # valid dataloader
    dataloader.batchsize_valid = 4096
    dataloader.num_workers_valid = 0
    dataloader.shuffle_valid = False
    dataloader.pin_memory_valid = False
    # test dataloader
    dataloader.batchsize_test = 1_000
    dataloader.num_workers_test = 0
    dataloader.shuffle_test = False
    dataloader.pin_memory_test = False
    # predict dataloader
    dataloader.batchsize_predict = 1_000
    dataloader.num_workers_predict = 0
    dataloader.shuffle_predict = False
    dataloader.pin_memory_predict = False

    return config


def get_traintest_config():
    config = traintest = config_dict.ConfigDict()

    traintest.train_prct = 0.9
    traintest.seed = 42

    return config


def get_spatial_encoders_config():
    # SPATIAL_TEMPORAL ENCODERS
    config = transform_spatial = config_dict.ConfigDict()
    transform_spatial.transform = "deg2rad"
    transform_spatial.scaler = [1.0 / math.pi, 1.0 / (math.pi / 2.0)]
    return config


def get_temporal_encoders_config():

    config = transform_temporal = config_dict.ConfigDict()
    transform_temporal.transform = "identity"

    return config


def get_model_config():
    config = model = config_dict.ConfigDict()

    model.model = "siren"
    # encoder specific
    model.encoder = config_dict.placeholder(str)
    # generalized
    model.num_layers = 5
    model.hidden_dim = 256
    model.use_bias = True
    model.final_activation = "identity"
    # SIREN SPECIFIC
    model.model_seed = 42
    model.w0_initial = 30.0
    model.w0 = 1.0
    model.final_scale = 1.0
    model.c = 6.0

    # # MODULATED SIREN SPECIFIC
    # model.latent_dim = 256
    # model.num_layers_latent = 3
    # model.operation = "sum"
    #
    # # MFN SPECIFIC
    # model.input_scale = 256.0
    # model.weight_scale = 1.0
    # model.alpha = 6.0
    # model.beta = 1.0

    # pretraining params
    model.pretrain = True
    model.pretrain_reference = "experiment-ckpts:v17"
    model.pretrain_checkpoint = "last.ckpt"
    model.pretrain_id = "299njfhp"  # ige/inr4ssh/299njfhp
    model.pretrain_entity = "ige"
    model.pretrain_project = "inr4ssh"

    return config


def get_optimizer_config():
    # OPTIMIZER
    config = optimizer = config_dict.ConfigDict()
    optimizer.optimizer = "adam"
    optimizer.learning_rate = 1e-4
    return config


def get_trainer_config():
    config = trainer = config_dict.ConfigDict()
    trainer.num_epochs = 10
    trainer.accelerator = "mps"  # "cpu", "gpu"
    trainer.devices = 1
    trainer.strategy = config_dict.placeholder(str)
    trainer.num_nodes = 1
    trainer.grad_batches = 10
    trainer.dev_run = False
    trainer.deterministic = True

    return config


def get_lr_scheduler_config():
    # LEARNING RATE WARMUP
    config = lr_scheduler = config_dict.ConfigDict()
    lr_scheduler.lr_scheduler = "warmcosine"
    lr_scheduler.warmup_epochs = 5
    lr_scheduler.max_epochs = 20
    lr_scheduler.warmup_lr = 0.0
    lr_scheduler.eta_min = 0.0
    return config


def get_callbacks_config():
    # CALLBACKS
    config = callbacks = config_dict.ConfigDict()
    # wandb logging
    callbacks.wandb = True
    callbacks.model_checkpoint = True
    # wandb artifacts
    callbacks.wandb_artifact = True
    # early stopping
    callbacks.early_stopping = False
    callbacks.patience = 20
    # model watch
    callbacks.watch_model = False
    # tqdm
    callbacks.tqdm = True
    callbacks.tqdm_refresh = 10

    return config


def get_loss_config():
    # LOSSES
    config = loss = config_dict.ConfigDict()
    loss.loss = "mse"
    loss.reduction = "mean"
    return config


def get_evaluation_config():
    # EVALUATION
    config = evaluation = config_dict.ConfigDict()
    evaluation.lon_min = -65.0  # 295.0
    evaluation.lon_max = -55.0  # 305.0
    evaluation.dlon = 0.1
    evaluation.lat_min = 33.0
    evaluation.lat_max = 43.0
    evaluation.dlat = 0.2
    # evaluation.lon_min = -65.0
    # evaluation.lon_max = -55.0
    # evaluation.dlon = 0.1
    # evaluation.lon_coarsen = 0
    # evaluation.lat_min = 33.0
    # evaluation.lat_max = 43.0
    # evaluation.dlat = 0.1
    # evaluation.lat_coarsen = 0

    evaluation.time_min = "2017-01-01"
    evaluation.time_max = "2017-12-31"
    evaluation.dt_freq = 1
    evaluation.dt_unit = "D"
    evaluation.time_resample = "1D"
    return config


def get_metrics_config():
    config = metrics = config_dict.ConfigDict()

    # binning alongtrack
    metrics.bin_lat_step = 1.0
    metrics.bin_lon_step = 1.0
    metrics.bin_time_step = "1D"
    metrics.min_obs = 10
    # power spectrum
    metrics.delta_t = 0.9434
    metrics.velocity = 6.77
    metrics.jitter = 1e-4

    return config


# ======================
# LOGGING
# ======================


def get_config():
    config = config_dict.ConfigDict()

    # LOGGING
    config.experiment = "swot1nadir5"
    config.log = get_wandb_config()

    # DATA DIRECTORIES
    config.datadir = config_dict.ConfigDict()

    # raw data
    config.datadir.raw = get_datadir_raw()
    config.datadir.staging = get_datadir_staging()

    # TODO: results

    config.dataloader = get_dataloader_config()

    config.traintest = get_traintest_config()
    config.preprocess = get_preprocess_config()
    config.transform = get_transformations_config()

    config.trainer = get_trainer_config()
    config.optimizer = get_optimizer_config()
    config.lr_scheduler = get_lr_scheduler_config()
    config.lr_scheduler.max_epochs = config.trainer.num_epochs
    config.callbacks = get_callbacks_config()
    config.encoder_spatial = get_spatial_encoders_config()
    config.encoder_temporal = get_temporal_encoders_config()
    config.model = get_model_config()
    config.loss = get_loss_config()
    config.metrics = get_metrics_config()
    config.evaluation = get_evaluation_config()

    return config
