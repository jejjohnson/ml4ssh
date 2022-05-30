from sklearn.utils import gen_batches
import tqdm
import numpy as np

def batch_predict(data, fn, batch_size):
    
    n_vals = data.shape[0]
    predictions = []
    
    for idx in tqdm.tqdm(gen_batches(n_vals, batch_size=batch_size)):
        predictions.append(fn(data[idx]))
    
    predictions = np.vstack(predictions)
    return predictions
    