from ml_collections import config_dict


def get_config():
    config = config_dict.ConfigDict()

    # data directory
    config.data = data = config_dict.ConfigDict()
    data.dataset_dir = "/Volumes/EMANS_HDD/data/dc20a_osse/test/ml/nadir1.nc"

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
    transform.time_min = "2011-01-01"
    transform.time_max = "2013-12-12"

    # train/valid arguments
    config.traintest = traintest = config_dict.ConfigDict()
    traintest.train_prct = 0.9
    traintest.seed = 42

    # dataloader
    config.dataloader = dataloader = config_dict.ConfigDict()
    # train dataloader
    dataloader.batchsize_train = 32
    dataloader.num_workers_train = 1
    dataloader.shuffle_train = True
    dataloader.pin_memory_train = False
    # valid dataloader
    dataloader.batchsize_valid = 32
    dataloader.num_workers_valid = 1
    dataloader.shuffle_valid = False
    dataloader.pin_memory_valid = False
    # predict dataloader
    dataloader.batchsize_predict = 32
    dataloader.num_workers_predict = 1
    dataloader.shuffle_predict = False
    dataloader.pin_memory_predict = False

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

    return config
