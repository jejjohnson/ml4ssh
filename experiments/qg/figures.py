import wandb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.reset_defaults()
sns.set_context(context="talk", font_scale=0.7)


def plot_maps(ds, name: str = "predictions", wandb_fn=None, **kwargs):

    map = ds.thin(steps=1).plot.imshow(col="steps", robust=True, col_wrap=3, **kwargs)
    # plt.tight_layout()
    if wandb_fn is not None:
        wandb_fn(
            {
                f"facet_maps_{name}": wandb.Image(map.fig),
            }
        )
    plt.close(map.fig)
