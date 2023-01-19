from ml_collections import config_dict
import math


def get_wandb_config() -> config_dict.ConfigDict:
    config = config_dict.ConfigDict()

    config.mode = "offline"
    config.project = "inr4ssh"
    config.entity = "ige"
    config.log_dir = "/gpfsscratch/rech/cli/uvo53rl/"
    config.resume = False
    config.id = config_dict.placeholder(str)
    return config


def get_datadir_raw():
    config = config_dict.ConfigDict()

    config.ref_dir = "/gpfswork/rech/yrf/commun/data_challenges/dc20a_osse/raw/dc_ref"
    config.obs_dir = "/gpfswork/rech/yrf/commun/data_challenges/dc20a_osse/raw/dc_obs/"

    return config


def get_datadir_clean():
    config = config_dict.ConfigDict()

    config.ref_dir = "/gpfswork/rech/yrf/commun/data_challenges/dc20a_osse/raw/dc_ref/"
    config.obs_dir = (
        "/gpfswork/rech/yrf/commun/data_challenges/dc20a_osse/work_eman/clean/"
    )

    return config


def get_datadir_staging():

    config = config_dict.ConfigDict()

    config.staging_dir = (
        "/gpfswork/rech/yrf/commun/data_challenges/dc20a_osse/work_eman/ml_ready"
    )

    return config


def get_osse_2020a_setup() -> config_dict.ConfigDict:
    config = config_dict.ConfigDict()

    # SUBSET Arguments
    config.nadir4 = [
        "2020a_SSH_mapping_NATL60_topex-poseidon_interleaved.nc",
        "2020a_SSH_mapping_NATL60_envisat.nc",
        "2020a_SSH_mapping_NATL60_geosat2.nc",
        "2020a_SSH_mapping_NATL60_jason1.nc",
    ]

    config.nadir1 = ["2020a_SSH_mapping_NATL60_jason1.nc"]

    config.swot1 = [
        "2020a_SSH_mapping_NATL60_karin_swot.nc",
        "2020a_SSH_mapping_NATL60_nadir_swot.nc",
    ]
    config.swot1nadir1 = [
        "2020a_SSH_mapping_NATL60_karin_swot.nc",
        "2020a_SSH_mapping_NATL60_nadir_swot.nc",
    ]
    config.swot1nadir5 = [
        "2020a_SSH_mapping_NATL60_karin_swot.nc",
        "2020a_SSH_mapping_NATL60_nadir_swot.nc",
        "2020a_SSH_mapping_NATL60_topex-poseidon_interleaved.nc",
        "2020a_SSH_mapping_NATL60_envisat.nc",
        "2020a_SSH_mapping_NATL60_geosat2.nc",
        "2020a_SSH_mapping_NATL60_jason1.nc",
    ]

    return config


def get_eval_period() -> config_dict.ConfigDict:
    """This period goes over 42 days:
    - 2012-10-01
    - 2012-12-02
    This is the equivalent for 2 SWOT cycles period for the new SWOT satellite.
    We allow for a spin up period of:
    - 2012-10-01
    - 2012-10-22
    however, we can include this within the training/evaluation scripts.
    """

    config = config_dict.ConfigDict()

    config.time_min = "2012-10-01"
    config.time_max = "2012-12-02"

    return config


def get_train_period() -> config_dict.ConfigDict:
    """
    This period goes over 42 days:
        - 2012-10-01
        - 2012-12-02
        This is the equivalent for a period for the SWOT data.
        We allow for a spin up period of:
        - 2012-10-01
        - 2012-10-22
        however, we can include this within the training/evaluation scripts.
    """
    config = config_dict.ConfigDict()

    config.time_min = "2013-01-01"
    config.time_max = "2013-09-30"

    return config


def get_transformations_config():
    config = transform = config_dict.ConfigDict()
    transform.time_transform = "minmax"
    transform.time_min = "2012-01-01"
    transform.time_max = "2013-01-01"

    return config


def get_dataloader_config():
    config = dataloader = config_dict.ConfigDict()
    # train dataloader
    dataloader.batchsize_train = 4096
    dataloader.num_workers_train = 16
    dataloader.shuffle_train = True
    dataloader.pin_memory_train = True
    # valid dataloader
    dataloader.batchsize_valid = 4096
    dataloader.num_workers_valid = 16
    dataloader.shuffle_valid = False
    dataloader.pin_memory_valid = True
    # test dataloader
    dataloader.batchsize_test = 4096
    dataloader.num_workers_test = 16
    dataloader.shuffle_test = False
    dataloader.pin_memory_test = True
    # predict dataloader
    dataloader.batchsize_predict = 10000
    dataloader.num_workers_predict = 16
    dataloader.shuffle_predict = False
    dataloader.pin_memory_predict = True

    return config


def get_traintest_config():
    config = traintest = config_dict.ConfigDict()

    traintest.train_prct = 0.9
    traintest.seed = 42

    return config


def get_optimizer_config():
    # OPTIMIZER
    config = optimizer = config_dict.ConfigDict()
    optimizer.optimizer = "adamw"
    optimizer.learning_rate = 1e-4
    return config


def get_trainer_config():
    config = trainer = config_dict.ConfigDict()
    trainer.num_epochs = 10
    trainer.accelerator = "gpu"  # "cpu", "gpu"
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
    lr_scheduler.patience = 100
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

    # pretraining params
    model.pretrain = True
    model.pretrain_reference = "experiment-ckpts:v17"
    model.pretrain_checkpoint = "last.ckpt"
    model.pretrain_id = "299njfhp"  # ige/inr4ssh/299njfhp
    model.pretrain_entity = "ige"
    model.pretrain_project = "inr4ssh"

    return config


def get_evaluation_config():
    # EVALUATION
    config = evaluation = config_dict.ConfigDict()
    evaluation.dataset = "natl60"

    evaluation.subset_spatial = True
    evaluation.lon_min = -65.0
    evaluation.lon_max = -55.0
    evaluation.dlon = 0.1
    evaluation.lon_coarsen = 0
    evaluation.lat_min = 33.0
    evaluation.lat_max = 43.0
    evaluation.dlat = 0.1
    evaluation.lat_coarsen = 0

    evaluation.subset_time = True
    evaluation.time_min = "2012-10-22"
    evaluation.time_max = "2012-12-02"
    evaluation.dt_freq = 1
    evaluation.dt_unit = "D"
    evaluation.time_resample = "1D"
    return config


def get_preprocess_config():
    # preprocessing
    config = preprocess = config_dict.ConfigDict()
    preprocess.dataset = "alongtrack"
    preprocess.subset_time = subset_time = config_dict.ConfigDict()
    subset_time.subset_time = True
    subset_time.time_min = "2012-10-02"
    subset_time.time_max = "2012-12-02"

    preprocess.subset_spatial = subset_spatial = config_dict.ConfigDict()
    subset_spatial.subset_spatial = True
    subset_spatial.lon_min = -65.0
    subset_spatial.lon_max = -55.0
    subset_spatial.lat_min = 33.0
    subset_spatial.lat_max = 43.0

    preprocess.resample = resample = config_dict.ConfigDict()
    resample.time_resample = config_dict.placeholder(str)  # "12h"
    resample.coarsen_lon = 0
    resample.coarsen_lat = 0

    return config


def get_config():
    config = config_dict.ConfigDict()

    # LOGGING
    config.experiment = "swot1nadir5"
    config.log = get_wandb_config()

    # DATA DIRECTORIES
    config.datadir = config_dict.ConfigDict()

    # raw data
    config.datadir.raw = get_datadir_raw()
    config.datadir.clean = get_datadir_clean()
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
    config.evaluation = get_evaluation_config()
    return config
