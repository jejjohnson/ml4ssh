from ml_collections import config_dict
from typing import Optional


def get_config():
    config = config_dict.ConfigDict()

    # logging args
    config.log = config_dict.ConfigDict()
    config.log.mode = "disabled"  # "online" #
    config.log.project = "inr4ssh"
    config.log.entity = "ige"
    config.log.log_dir = "/Users/eman/code_projects/logs/"
    config.log.resume = False

    # data args
    config.data = config_dict.ConfigDict()
    config.data.data_dir = (
        f"/Users/eman/code_projects/torchqg/data/qgsim_simple_128x128.nc"
    )

    # checkpoint args
    config.pretrained = config_dict.ConfigDict()
    config.pretrained.checkpoint = False
    config.pretrained.run_path = "ige/inr4ssh/2z8tsrfn"
    config.pretrained.model_path = "checkpoints/epoch=836-step=329778.ckpt"

    # preprocessing args
    config.pre = config_dict.ConfigDict()
    config.pre.noise = 0.01
    config.pre.dt = 1.0
    config.pre.time_subset = True
    config.pre.time_min = 0
    config.pre.time_max = 1
    config.pre.seed = 123

    # train/test args
    config.split = config_dict.ConfigDict()
    config.split.train_prct = 0.9

    # dataloader args
    config.dl = config_dict.ConfigDict()
    config.dl.batchsize_train = 2048
    config.dl.batchsize_val = 2048
    config.dl.batchsize_test = 4096
    config.dl.batchsize_predict = 64
    config.dl.num_workers = 0
    config.dl.pin_memory = False

    # model arguments
    config.model = model = config_dict.ConfigDict()
    model.dim_hidden = 256
    model.num_layers = 4
    model.w0 = 1.0
    model.w0_initial = 30.0
    model.c = 6.0
    model.final_activation = None

    # loss arguments
    config.loss = loss = config_dict.ConfigDict()
    loss.loss = "mse"
    loss.reduction = "mean"
    loss.qg = False
    loss.alpha = 1e-4

    # optimizer args
    config.optim = config_dict.ConfigDict()
    config.optim.warmup = 100
    config.optim.num_epochs = 10_000
    config.optim.learning_rate = 1e-3
    config.optim.warmup_start_lr = 1e-5
    config.optim.eta_min = 1e-5

    # trainer args
    config.trainer = config_dict.ConfigDict()
    config.trainer.accelerator = "cpu"
    config.trainer.devices = 1
    config.trainer.grad_batches = 1

    return config
