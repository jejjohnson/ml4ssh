def calculate_gradient(da, x_coord: str = "Nx", y_coord: str = "Ny", edge_order=2):
    # first marginal derivative
    dx = da.differentiate(coord=x_coord, edge_order=2)
    dy = da.differentiate(coord=y_coord, edge_order=2)

    return (dx**2 + dy**2) ** 0.5


def calculate_laplacian(da, x_coord: str = "Nx", y_coord: str = "Ny", edge_order=2):
    # second marginal derivative
    dx2 = da.differentiate(coord=x_coord, edge_order=edge_order).differentiate(
        coord=x_coord, edge_order=edge_order
    )
    dy2 = da.differentiate(coord=y_coord, edge_order=edge_order).differentiate(
        coord=y_coord, edge_order=edge_order
    )

    return dx2 + dy2
