import numpy as np


def get_num_training(num_data: int, train_prct: float = 0.9):

    num_train = int(np.floor(num_data * train_prct))
    num_valid = int(np.ceil(num_data * (1 - train_prct)))

    assert num_train + num_valid == num_data

    return num_train, num_valid
