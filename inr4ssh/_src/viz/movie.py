from pathlib import Path
from xmovie import Movie
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.reset_defaults()
sns.set_context(context="talk", font_scale=0.7)


def create_movie(
    var, name, framedim: str = "steps", cmap: str = "RdBu_r", file_path=None, **kwargs
):

    if file_path is not None:
        file_name = Path(file_path).joinpath(f"movie_{name}.gif")
    else:
        file_name = Path(f"./movie_{name}.gif")

    # var = var.transpose(("time", "latitude", "longitude"))

    mov = Movie(var, framedim=framedim, cmap=cmap, **kwargs)
    mov.save(
        file_name,
        remove_movie=False,
        progress=True,
        framerate=5,
        gif_framerate=5,
        overwrite_existing=True,
    )

    return None
