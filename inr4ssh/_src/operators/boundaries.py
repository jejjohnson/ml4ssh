import numpy as np


def bc_conditions(data, bc: str = "neumann"):

    if bc.lower() == "neumann":
        return bc_neumann(data)
    elif bc.lower() == "dirichlet":
        return bc_dirichlet(data)
    elif bc.lower() == "dirichlet_face":
        return bc_dirichlet_face(data)
    elif bc.lower() == "periodic":
        return bc_periodic(data)


def bc_dirichlet(data):

    # pad the boundaries
    data = np.pad(data, ((1, 1), (1, 1), (0, 0)), "constant")

    data[0, ...] = -data[1, ...]
    data[-1, ...] = -data[-2, ...]
    data[:, 0, ...] = -data[:, 0, ...]
    data[:, -1, ...] = -data[:, -2, ...]

    # do the corners
    data[0, 0, ...] = -data[0, 1, ...] - data[1, 0, ...] - data[1, 1, ...]
    data[-1, 0, ...] = -data[-1, 1, ...] - data[-2, 0, ...] - data[-2, 1, ...]
    data[0, -1, ...] = -data[1, -1, ...] - data[0, -2, ...] - data[1, -2, ...]
    data[-1, -1, ...] = -data[-1, -2, ...] - data[-2, -2, ...] - data[-2, -1, ...]

    return data


def bc_dirichlet_face(data):

    # pad the boundaries
    data = np.pad(data, ((1, 1), (1, 1), (0, 0)), "constant")

    data[0, ...] = 0
    data[-1, ...] = 0
    data[:, 0, ...] = 0
    data[:, -1, ...] = 0

    # do the corners
    data[0, 0, ...] = 0
    data[-1, 0, ...] = 0
    data[0, -1, ...] = 0
    data[-1, -1, ...] = 0

    return data


def bc_neumann(data):

    # pad the boundaries
    data = np.pad(data, ((1, 1), (1, 1), (0, 0)), "constant")

    data[0, ...] = data[1, ...]
    data[-1, ...] = data[-2, ...]
    data[:, 0, ...] = data[:, 1, ...]
    data[:, -1, ...] = data[:, -2, ...]

    # do the corners
    data[0, 0, ...] = data[1, 1, ...]
    data[-1, 0, ...] = data[-2, 1, ...]
    data[0, -1, ...] = data[1, -2, ...]
    data[-1, -1, ...] = data[-2, -2, ...]

    return data


def bc_periodic(data):

    # pad the boundaries
    data = np.pad(data, ((1, 1), (1, 1), (0, 0)), "constant")

    data[0, ...] = data[-2, ...]
    data[-1, ...] = data[1, ...]
    data[:, 0, ...] = data[:, -2, ...]
    data[:, -1, ...] = data[:, 1, ...]

    return data
