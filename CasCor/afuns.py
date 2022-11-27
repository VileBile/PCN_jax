import jax.numpy as jnp

def hardtanh(x):
    x = x.at[x>1].set(1)
    x = x.at[x<-1].set(-1)
    return x

def hsig(x):
    fx = x/6 + 1/2
    fx = fx.at[x>=3].set(1)
    fx = fx.at[x<-3].set(0)
    return fx

def hswish(x):
    fx = x*(x+3)/6
    fx = fx.at[x>=3].set(x[x>=3])
    fx = fx.at[x<-3].set(0)
    return fx



