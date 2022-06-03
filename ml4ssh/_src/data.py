from typing import Optional, Iterator
import tensorflow.data as tfd
import jax.numpy as jnp
import pandas as pd
import tensorflow_datasets as tfds


def make_mini_batcher(
    
    X, y,
    batch_size: Optional[int] = 32,
    prefetch_buffer: Optional[int] = 5,
    shuffle: Optional[bool] = True,
    buffer_size: Optional[int]=None,
    seed: Optional[int]=None,
 ) -> Iterator:

    n = X.shape[0]

    batch_size = min(batch_size, n)

    # Make dataloader, set batch size and prefetch buffer:
    ds = tfd.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=10 * batch_size, seed=seed)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(prefetch_buffer)
    ds = ds.repeat()

    # # ALTERNATIVE (BUT SLOW :())
    # # Make dataloader, set batch size and prefetch buffer:
    # if buffer_size is None:
    #     buffer_size = batch_size
    # ds = tfd.Dataset.from_tensor_slices((X, y))
    # ds = ds.cache()
    # ds = ds.repeat()
    # if shuffle:
    #     ds = ds.shuffle(buffer_size=buffer_size)
    # ds = ds.batch(batch_size)
    # ds = ds.prefetch(prefetch_buffer)

    # Make iterator:
    ds = iter(tfds.as_numpy(ds))
    return ds
