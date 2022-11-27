import jax.numpy as jnp
import jax.random as jr

from jax.scipy.linalg import block_diag

def consPmask(n):
    Pm = jnp.ones((n[0],n[0]))
    for ni in n[1:]:
        Pm = block_diag(Pm,jnp.ones((ni,ni)))
    return Pm.astype(int) 

def consWmask(neighs):
    if type(neighs) == int:
        print("is dense")
        return jnp.ones((neighs,neighs)).astype(int) 

    if type(neighs[1]) == int:
        print("is a ff-dense net right?")
        nneur = sum(neighs)
        Wm = jnp.zeros((nneur,nneur))
        for i in range(0,len(neighs)-1):
            print(neighs[i])
            Wm = Wm.at[sum(neighs[0:i+1]):sum(neighs[0:i+2]),sum(neighs[0:i]):sum(neighs[0:i+1])].set(1)

        return Wm.astype(int)

    nneur = max(max(neighs))    # [(m,r)(M,R)] -> [(M,R)] -> M/R 
    Wm = jnp.zeros((nneur,nneur))
    Wm = Wm.at[neighs].set(1)
    return Wm.astype(int)