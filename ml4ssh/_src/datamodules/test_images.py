from .images import ImageFox


def test_fox_shapes():
    dm = ImageFox().setup()

    ds_train = dm.ds_train

    assert ds_train.tensors[0].shape[1] == 2
    assert ds_train.tensors[1].shape[1] == 3

    ds_test = dm.ds_test

    assert ds_test.tensors[0].shape[1] == 2
    assert ds_test.tensors[1].shape[1] == 3

