import torch

def exists(val):
    return val is not None

def cast_tuple(val, repeat = 1):
    return val if isinstance(val, tuple) else ((val,) * repeat)


def get_torch_device():
    if torch.has_mps:
        return "mps"
    elif torch.has_cuda:
        return "cuda"
    else:
        return "cpu"
