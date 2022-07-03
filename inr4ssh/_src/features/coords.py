import numpy as np
from einops import rearrange

def get_image_coordinates(image: torch.Tensor, min_val: int=-1, max_val: int=1):
    # get image size
    image_height, image_width, _ = image.shape

    # get all coordinates
    coordinates = [
        np.linspace(min_val, max_val, steps=image_height),
        np.linspace(min_val, max_val, steps=image_width)
    ]

    # create meshgrid of pairwise coordinates
    coordinates = np.meshgrid(*coordinates, indexing="ij")

    # stack tensors together
    coordinates = np.stack(coordinates, dim=-1)

    # rearrange to coordinate vector
    coordinates = rearrange(coordinates, "h w c -> (h w) c")
    pixel_values = rearrange(image, "h w c -> (h w) c")

    return coordinates, pixel_values