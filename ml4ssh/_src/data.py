from typing import Optional, Iterator
import tensorflow.data as tfd
import jax.numpy as jnp
import pandas as pd
import tensorflow_datasets as tfds


  # ds = ds.shuffle(buffer_size=10 * batch_size)
  # ds = ds.batch(batch_size)
  # ds = ds.prefetch(buffer_size=5)
  # ds = ds.repeat()

def make_mini_batcher(
    
    X, y,
    batch_size: Optional[int] = 32,
    prefetch_buffer: Optional[int] = 5,
    shuffle: Optional[bool] = True
 ) -> Iterator:

    n = X.shape[0]

    batch_size = min(batch_size, n)

    # Make dataloader, set batch size and prefetch buffer:
    ds = tfd.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=batch_size)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(prefetch_buffer)
    ds = ds.repeat()

    # Make iterator:
    ds = iter(tfds.as_numpy(ds))
    return ds
