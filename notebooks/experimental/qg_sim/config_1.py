from ml_collections import config_dict


def get_config():
    cfg = config_dict.ConfigDict()

    # logging args
    cfg.log = config_dict.ConfigDict()
    cfg.log.mode = "online"  # "disabled"
    cfg.log.project = "inr4ssh"
    cfg.log.entity = "ige"
    cfg.log.log_dir = "/Users/eman/code_projects/logs/"
    cfg.log.resume = False

    # data args
    cfg.data = config_dict.ConfigDict()
    cfg.data.data_dir = (
        f"/Users/eman/code_projects/torchqg/data/qgsim_simple_128x128.nc"
    )

    # preprocessing args
    cfg.pre = config_dict.ConfigDict()
    cfg.pre.noise = 0.01
    cfg.pre.dt = 1.0
    cfg.pre.time_min = 500
    cfg.pre.time_max = 511
    cfg.pre.seed = 123

    # train/test args
    cfg.split = config_dict.ConfigDict()
    cfg.split.train_prct = 0.9

    # dataloader args
    cfg.dl = config_dict.ConfigDict()
    cfg.dl.batchsize_train = 2048
    cfg.dl.batchsize_val = 1_000
    cfg.dl.batchsize_test = 5_000
    cfg.dl.batchsize_predict = 10_000
    cfg.dl.num_workers = 0
    cfg.dl.pin_memory = False

    # loss arguments
    cfg.loss = config_dict.ConfigDict()
    cfg.loss.qg = True
    cfg.loss.alpha = 1e-4

    # optimizer args
    cfg.optim = config_dict.ConfigDict()
    cfg.optim.warmup = 10
    cfg.optim.num_epochs = 100
    cfg.optim.learning_rate = 1e-4

    # trainer args
    cfg.trainer = config_dict.ConfigDict()
    cfg.trainer.accelerator = None
    cfg.trainer.devices = 1
    cfg.trainer.grad_batches = 1
    return cfg
