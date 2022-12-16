from ml_collections import config_dict


# ======================
# LOGGING
# ======================
def get_config():
    config = config_dict.ConfigDict()

    # ===============================
    # LOGGING ARGS
    # ===============================
    config.log = log = config_dict.ConfigDict()
    log.mode = "offline"
    log.project = "inr4ssh"
    log.entity = "ige"
    log.log_dir = "/gpfsscratch/rech/cli/uvo53rl"
    log.resume = False
    # log.id = None
    # log.run_path = None
    # log.model_path = None

    # ===============================
    # DATA ARGS
    # ===============================
    config.data = data = config_dict.ConfigDict()
    data.data = "dc21a"
    data.train_data_dir = "/gpfsdswork/projects/rech/cli/uvo53rl/data/data_challenges/ssh_mapping_2021/train"
    data.ref_data_dir = "/gpfsdswork/projects/rech/cli/uvo53rl/data/data_challenges/ssh_mapping_2021/ref"
    data.test_data_dir = "/gpfsdswork/projects/rech/cli/uvo53rl/data/data_challenges/ssh_mapping_2021/test"

    # ===============================
    # PREPROCESSING ARGS
    # ===============================
    config.preprocess = preprocess = config_dict.ConfigDict()
    # longitude subset
    preprocess.lon_min = 285.0
    preprocess.lon_max = 315.0
    preprocess.dlon = 0.2
    preprocess.lon_buffer = 1.0
    # latitude subset
    preprocess.lat_min = 23.0
    preprocess.lat_max = 53.0
    preprocess.dlat = 0.2
    preprocess.lat_buffer = 1.0
    # temporal subset
    preprocess.time_min = "2016-12-01"
    preprocess.time_max = "2018-01-31"
    preprocess.dtime = "1_D"
    preprocess.time_buffer = 7.0

    # ===============================
    # FEATURES ARGS
    # ===============================
    config.features = features = config_dict.ConfigDict()
    # spatial
    features.julian_time = True
    features.abs_time = True
    features.abs_time_min = "2005-01-01"
    features.abs_time_max = "2022-01-01"
    features.feature_scaler = "minmax"
    # temporal
    features.cartesian = True
    features.minmax_spatial = True
    features.minmax_temporal = True
    features.spherical_radius = 1.0
    features.min_time_scale = -1.0
    features.max_time_scale = 1.0

    # ===============================
    # TRAIN/VAL SPLIT ARGS
    # ===============================
    config.split = split = config_dict.ConfigDict()
    split.train_size = 0.9
    split.split_method = "random"  # random, temporal, spatial
    split.seed_split = 666
    split.seed_shuffle = 321
    split.split_time_freq = config_dict.placeholder(str)  # "1_D"
    split.split_spatial = "random"  # "regular" # "upper" # "lower" # "altimetry"

    # ===============================
    # DATALOADER ARGS
    # ===============================
    config.dataloader = dataloader = config_dict.ConfigDict()
    # dataloader
    dataloader.train_shuffle = True
    dataloader.pin_memory = True
    dataloader.num_workers = 10
    dataloader.batchsize_train = 128
    dataloader.batchsize_valid = 4096
    dataloader.batchsize_test = 1_000
    dataloader.batchsize_predict = 10_000

    # ===============================
    # MODEL ARGS
    # ===============================
    config.model = model = config_dict.ConfigDict()
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
    # MODULATED SIREN
    model.latent_dim = 256
    model.num_layers_latent = 3
    model.operation = "sum"
    # MULTIPLICATIVE FILTER NETWORKS
    model.input_scale = 256.0
    model.weight_scale = 1.0
    model.alpha = 6.0
    model.beta = 1.0

    # ===============================
    # LOSSES ARGS
    # ===============================
    config.loss = loss = config_dict.ConfigDict()
    loss.loss = "mse"  # Options: "mse", "nll", "kld"
    loss.reduction = "mean"
    # QG PINN Loss Args
    loss.qg = False
    loss.qg_reg = 0.1

    # ===============================
    # LOSSES ARGS
    # ===============================
    config.optimizer = optimizer = config_dict.ConfigDict()
    optimizer.optimizer = "adam"  # "adamw" # "adamax"
    optimizer.learning_rate = 1e-4
    optimizer.num_epochs = 2_000
    optimizer.min_epochs = 1

    # ===============================
    # TRAINER ARGS
    # ===============================
    config.trainer = config_dict.ConfigDict()
    config.trainer.accelerator = "gpu"  # "cpu", "mps"
    config.trainer.devices = 1
    # "ddp"
    config.trainer.strategy = config_dict.placeholder(str)
    # 4
    config.trainer.num_nodes = 1
    config.trainer.grad_batches = 10
    config.trainer.dev_run = False

    # ===============================
    # LR SCHEDULER ARGS
    # ===============================
    config.lr_scheduler = lr_scheduler = config_dict.ConfigDict()
    lr_scheduler.lr_scheduler = (
        "warmcosine"  # Options: "cosine", "onecyle", "step", "multistep"
    )

    # warmup cosine annealing specific
    lr_scheduler.warmup_epochs = 100
    lr_scheduler.max_epochs = optimizer.num_epochs
    lr_scheduler.eta_min = 1e-6
    lr_scheduler.warmup_lr = 1e-6

    # Early Stopping Specific
    lr_scheduler.patience = 10
    lr_scheduler.factor = 0.1
    lr_scheduler.steps = 250
    lr_scheduler.gamma = 0.1
    lr_scheduler.min_learning_rate = 1e-5
    lr_scheduler.milestones = [500, 1000, 1500, 2000, 2500]

    # ===============================
    # CALLBACKS ARGS
    # ===============================
    config.callbacks = callbacks = config_dict.ConfigDict()
    # wandb logging
    callbacks.wandb = True
    callbacks.model_checkpoint = True
    # early stopping
    callbacks.early_stopping = False
    callbacks.patience = 20
    callbacks.watch_model = False

    # ===============================
    # EVALULATION DATA ARGS
    # ===============================
    config.eval_data = eval_data = config_dict.ConfigDict()
    eval_data.lon_min = 295.0
    eval_data.lon_max = 305.0
    eval_data.dlon = 0.2
    eval_data.lat_min = 33.0
    eval_data.lat_max = 43.0
    eval_data.dlat = 0.2
    eval_data.time_min = "2017-01-01"
    eval_data.time_max = "2017-12-31"
    eval_data.dtime_freq = 1
    eval_data.dtime_unit = "D"
    eval_data.lon_buffer = 2.0
    eval_data.lat_buffer = 2.0
    eval_data.time_buffer = 7.0

    # ===============================
    # EVALULATION DATA ARGS
    # ===============================
    config.metrics = metrics = config_dict.ConfigDict()
    # binning along track
    metrics.bin_lat_step = 1.0
    metrics.bin_lon_step = 1.0
    metrics.bin_time_step = "1D"
    metrics.min_obs = 10
    # power spectrum
    metrics.delta_t = 0.9434
    metrics.velocity = 6.77
    metrics.jitter = 1e-4

    # ===============================
    # VIZ ARGS
    # ===============================
    config.viz = viz = config_dict.ConfigDict()
    viz.lon_min = 295.0
    viz.lon_max = 305.0
    viz.dlon = 0.1
    viz.lon_buffer = 1.0
    # latitude subset
    viz.lat_min = 33.0
    viz.lat_max = 43.0
    viz.dlat = 0.1
    viz.lat_buffer = 1.0
    # temporal subset
    viz.time_min = "2017-01-01"
    viz.time_max = "2017-12-31"
    viz.dtime_freq = 1
    viz.dtime_unit = "D"
    viz.time_buffer = 7.0

    return config
