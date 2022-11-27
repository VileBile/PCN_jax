

from colabcode import ColabCode
ColabCode()



import sys, os, importlib.util
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import jax.random as jr
import pickle
from jax.scipy.linalg import block_diag
from jax import grad, value_and_grad, jit, vmap

import aesara
aesara.config.floatX="float32"

import nvidia_smi

nvidia_smi.nvmlInit()
handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

print("Total memory:", info.total)

nvidia_smi.nvmlShutdown()


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="2.0"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"


spec1 = importlib.util.spec_from_file_location("modelcons","/home/marko/Documents/Python/CasCor/modelcons.py")
modelcons = importlib.util.module_from_spec(spec1)
sys.modules["modelcons"] = modelcons
spec1.loader.exec_module(modelcons)

spec2 = importlib.util.spec_from_file_location("afuns","/home/marko/Documents/Python/CasCor/afuns.py")
af = importlib.util.module_from_spec(spec2)
sys.modules["afuns"] = af
spec2.loader.exec_module(af)


tsf = open("/home/marko/Documents/Python/CasCor/TS.pickle","rb")
TS = pickle.load(tsf)
tsf.close()
ts = TS[0][0:1] #+ TS[1][0:2] +  TS[9][0:2]

def cons_f(fbs):
    def f(x):
        for fi, i in fbs:
            x = x.at[i].set(fi(x[i]))
        return x
    return f
    


def energy(x,W,P,f):
    return jnp.dot(jnp.dot((x-jnp.dot(W,f(x))).T,P),(x-jnp.dot(W,f(x)))) - jnp.log(jnp.linalg.det(P)) # + constraints (in terms of lap multipliers)


def asim(d,W,P,f,h,T):
    s, i = d
    x = jnp.zeros(W.shape[1])
    x = x.at[i].set(s)
    for _ in range(1,T):
       x = x - h*grad(energy,0)(x,W,P,f)
       x.at[i].set(s)
    return x

def acom(eps,D,W,Wmask,P,Pmask,f,l,h,T):
    for ep in range(1,eps):
        for d in D:
            mue = 0
            mudP = 0
            mudW = 0
            c = 0
            # optimize X
            x = asim(d,W,P,f,h,T)
            
            #compute err and gradients
            e, gradtemp= value_and_grad(energy,(1,2))(x,W,P,f)
            dW, dP = gradtemp
            mue = mue*c + e 
            mudW = mudW*c + jnp.linalg.norm(dW) 
            mudP = mudP*c + jnp.linalg.norm(dP) 
            c += 1
            mue /= c
            mudP /= c
            mudW /= c


            # update W, P 
            W = W.at[Wmask].add(-l*dW[Wmask])
            P = P.at[Pmask].add(-l*dP[Pmask])
            
            # resimetrize P (bcos rounding errs)
            P = (P+P.T)/2
            print("'")  
        print("\nep: ", ep)
        print("\err: ", mue)
        print("\dP: ", mudP)
        print("\ndW: ", mudW)
    return W, P 


n = [794]
nneur = sum(n)
ols = slice(nneur-10,nneur)

Wmask = modelcons.consWmask(nneur)
Pmask = modelcons.consPmask(n)

P = jnp.zeros_like(Pmask) + jnp.eye(nneur)
W = (2*jr.uniform(jr.PRNGKey(0),shape = Wmask.shape) - 1)*10**-2




fbs = [(af.hardtanh,slice(0,nneur))]
f = cons_f(fbs)

h = 10**-1
l = 10**-3
eps = 20
T = 100

W, P = acom(eps,ts,W,Wmask,P,Pmask,f,l,h,T)









