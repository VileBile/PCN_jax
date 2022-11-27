import jax.numpy as jnp
import jax.random as jr

from jax.scipy.linalg import block_diag
from jax import grad, value_and_grad, jit, vmap

def cons_f(fbs):
    def f(x):
        fx = x
        for fi, i in fbs:
            fx.at[i].set(fi(x[i]))
        return fx
    return f
    


def energy(x,W,P,f):
    return (x-W*f(x)).T * P * (x-W*f(x)) -log(det(P)) # + constraints (in terms of lap multipliers)


def asim(d,W,P,f,h,T):
    s, i = d
    x = jnp.zeros(W.shape[1])
    x = x.at[i].set(s)
    fx = jnp.zeros(x.shape)
    for _ in range(1,T):
       x = x - h*grad(energy,0)(x,W,P,f)
       x.at[i].set(s)
    return x

def acom(eps,D,W,P,f,l,h,T):
    for _ in range(1,eps):
        for d in D:
            # optimize X
            x = asim(d,W,P,f,h,T)
            
            #compute err and gradients
            e, dW, dP = value_and_grad(energy,(1,2))(x,W,P,f)

            # update W, P 
            W = W.at[Wmask].add(-l*dW[Wmask])
            P = P.at[Pmask].add(-l*dP[Pmask])
            
            # resimetrize P (bcos rounding errs)
            P = (P+P.T)/2  
    return W, P 










