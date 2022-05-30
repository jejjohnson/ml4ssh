from typing import Optional, Iterator
import tensorflow.data as tfd
import jax.numpy as jnp
import pandas as pd

def make_mini_batcher(
    
    X, y,
    batch_size: Optional[int] = 32,
    prefetch_buffer: Optional[int] = 1,
    shuffle: Optional[bool] = True
 ) -> Iterator:

    n = X.shape[0]

    batch_size = min(batch_size, n)

    # Make dataloader, set batch size and prefetch buffer:
    ds = tfd.Dataset.from_tensor_slices((X, y))
    ds = ds.cache()
    ds = ds.repeat()
    if shuffle:
        ds = ds.shuffle(buffer_size=10 * batch_size)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(prefetch_buffer)

    # Make iterator:
    train_iter = iter(ds)

    # Batch loader:
    def next_batch():
        x_batch, y_batch = train_iter.next()
        return jnp.asarray(x_batch.numpy()), jnp.asarray(y_batch.numpy())

    return next_batch
