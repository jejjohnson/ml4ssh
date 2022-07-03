
def calculate_gradient(da, x_coord: str = "Nx", y_coord: str = "Ny", edge_order=2):
    # first marginal derivative
    dx = da.differentiate(coord=x_coord, edge_order=2)
    dy = da.differentiate(coord=y_coord, edge_order=2)

    return 0.5 * (dx ** 2 + dy ** 2)


def calculate_laplacian(da, x_coord: str = "Nx", y_coord: str = "Ny", edge_order=2):
    # second marginal derivative
    dx2 = da.differentiate(coord=x_coord, edge_order=2).differentiate(coord=x_coord, edge_order=2)
    dy2 = da.differentiate(coord=x_coord, edge_order=2).differentiate(coord=x_coord, edge_order=2)

    return 0.5 * (dx2 ** 2 + dy2 ** 2)