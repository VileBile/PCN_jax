import tensorflow_datasets as tfds
import jax.numpy as jnp
from jax import vmap

def genMNISTset(ols):
    start, stop, step = ols.indices(10000)
    onehot = jnp.zeros(stop-start).reshape(stop-start,1)
    ds = tfds.load("mnist",split="train",shuffle_files=True)
    TS  = {}
    for d in ds:
        lab = d["label"].numpy()
        im = jnp.asarray(d["image"]/255).reshape(28*28,1) 
        x = jnp.concatenate((im,onehot.at[lab].set(1)))
        idx = jnp.r_[(slice(0,28*28),ols)]
        idx = idx.reshape(len(idx),1)
        temo = (x,idx)
        if lab not in TS.keys():
            TS[lab] = [temo]
        else:
            TS[lab].append(temo)

    return TS        


    
