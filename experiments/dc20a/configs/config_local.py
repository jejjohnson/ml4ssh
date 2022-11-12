from ml_collections import config_dict
import math


def get_config():

    config = config_dict.ConfigDict()

    # LOGGING
    config.log = config_dict.ConfigDict()
    config.log.mode = "disabled"  # "online" #
    config.log.project = "inr4ssh"
    config.log.entity = "ige"
    config.log.log_dir = "/Users/eman/code_projects/logs/"
    config.log.resume = False

    # data directory
    config.data = data = config_dict.ConfigDict()
    data.dataset_dir = "/Volumes/EMANS_HDD/data/dc20a_osse/test/ml/nadir4.nc"
    data.ref_dir = (
        "/Volumes/EMANS_HDD/data/dc20a_osse/raw/dc_ref/NATL60-CJM165_GULFSTREAM*"
    )
    # preprocessing
    config.preprocess = config_dict.ConfigDict()
    config.preprocess.subset_time = subset_time = config_dict.ConfigDict()
    subset_time.subset_time = True
    subset_time.time_min = "2012-10-22"
    subset_time.time_max = "2012-12-02"

    config.preprocess.subset_spatial = subset_spatial = config_dict.ConfigDict()
    subset_spatial.subset_spatial = True
    subset_spatial.lon_min = -65.0
    subset_spatial.lon_max = -55.0
    subset_spatial.lat_min = 33.0
    subset_spatial.lat_max = 43.0

    # transformations
    config.preprocess.transform = transform = config_dict.ConfigDict()
    transform.time_transform = "minmax"
    transform.time_min = "2012-01-01"
    transform.time_max = "2013-01-01"

    # DATALOADER
    # dataloader
    config.dataloader = dataloader = config_dict.ConfigDict()
    # train dataloader
    dataloader.batchsize_train = 32
    dataloader.num_workers_train = 0
    dataloader.shuffle_train = True
    dataloader.pin_memory_train = False
    # valid dataloader
    dataloader.batchsize_valid = 32
    dataloader.num_workers_valid = 0
    dataloader.shuffle_valid = False
    dataloader.pin_memory_valid = False
    # test dataloader
    dataloader.batchsize_test = 32
    dataloader.num_workers_test = 0
    dataloader.shuffle_test = False
    dataloader.pin_memory_test = False
    # predict dataloader
    dataloader.batchsize_predict = 32
    dataloader.num_workers_predict = 0
    dataloader.shuffle_predict = False
    dataloader.pin_memory_predict = False

    # train/valid arguments
    config.traintest = traintest = config_dict.ConfigDict()
    traintest.train_prct = 0.9
    traintest.seed = 42

    # EVALUATION
    config.evaluation = evaluation = config_dict.ConfigDict()
    evaluation.lon_min = -65.0
    evaluation.lon_max = -55.0
    evaluation.dlon = 0.1
    evaluation.lat_min = 33.0
    evaluation.lat_max = 43.0
    evaluation.dlat = 0.1

    evaluation.time_min = "2012-10-22"
    evaluation.time_max = "2012-12-02"
    evaluation.dt_freq = 1
    evaluation.dt_unit = "D"

    evaluation.time_resample = "1D"
    # , get_demo_config

    # config = get_demo_config()

    config.preprocess.subset_spatial.subset_spatial = True
    config.preprocess.subset_time.subset_time = True

    # MODEL
    config.model = model = config_dict.ConfigDict()

    model.model = "siren"
    # encoder specific
    model.encoder = config_dict.placeholder(str)
    # generalized
    model.num_layers = 5
    model.hidden_dim = 64  # 256
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

    # SPATIAL_TEMPORAL ENCODERS
    config.transform_spatial = config_dict.ConfigDict()
    config.transform_spatial.transform = "deg2rad"
    config.transform_spatial.scaler = [1.0 / math.pi, 1.0 / (math.pi / 2.0)]

    config.transform_temporal = config_dict.ConfigDict()
    config.transform_temporal.transform = "identity"

    # LOSSES
    config.loss = config_dict.ConfigDict()
    config.loss.loss = "mse"
    config.loss.reduction = "mean"

    # OPTIMIZER
    config.optimizer = config_dict.ConfigDict()
    config.optimizer.optimizer = "adam"
    config.optimizer.learning_rate = 1e-4

    # TRAINER
    config.trainer = config_dict.ConfigDict()
    config.trainer.num_epochs = 10
    config.trainer.accelerator = "mps"  # "cpu", "gpu"
    config.trainer.devices = 1
    config.trainer.strategy = config_dict.placeholder(str)
    config.trainer.num_nodes = 1
    config.trainer.grad_batches = 10
    config.trainer.dev_run = False
    config.trainer.deterministic = True

    # LEARNING RATE WARMUP
    config.lr_scheduler = config_dict.ConfigDict()
    config.lr_scheduler.lr_scheduler = "warmcosine"
    config.lr_scheduler.warmup_epochs = 5
    config.lr_scheduler.max_epochs = config.trainer.num_epochs
    config.lr_scheduler.warmup_lr = 0.0
    config.lr_scheduler.eta_min = 0.0

    # CALLBACKS
    config.callbacks = config_dict.ConfigDict()
    # wandb logging
    config.callbacks.wandb = True
    config.callbacks.model_checkpoint = True
    # early stopping
    config.callbacks.early_stopping = False
    config.callbacks.patience = 20
    # model watch
    config.callbacks.watch_model = False
    # tqdm
    config.callbacks.tqdm = True
    config.callbacks.tqdm_refresh = 10

    return config
