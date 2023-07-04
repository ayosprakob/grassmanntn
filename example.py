

#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
#                                    README                                   #
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
#                                                                             #
#  --- How to use ---                                                         #
#  run the script in console with the command                                 #
#  > python3 example.py                                                       #
#  You can modify parameters by, e.g.                                         #
#  > python3 example.py --mass=0.5 --beta=2                                   #
#                                                                             #
#  The followings are the parameters as well as their initialized values:     #
#  --beta=1.0 (gauge inverse-coupling)                                        #
#  --mass=1.0 (fermion mass; dimensionful)                                    #
#  --charge=1.0 (fermion charge)                                              #
#  --spacing=1.0 (lattice spacing; dimensionful)                              #
#  --mu=0.0 (chemical potential; dimensionful)                                #
#  --Nf=1 (flavor number)                                                     #
#  --K=2 (Z_K gauge theory)                                                   #
#  --cgsteps=5 (the number of ATRG iterations, starting from the x-direction) #
#  --Dcutz=32 (flavor coarse-graining Dcut)                                   #
#  --Dcutxy=32 (space-time coarse-graining Dcut)                              #
#  --boundary_conditions=anti-periodic (b.c. in the imaginary-time direction) #
#  --cls or --clear_screen (add this if you want to clear the screen)         #
#  --show_progress (show the progress bars)                                   #
#                                                                             #
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#


import os
# Clearing the Screen
#os.system('clear')

import sparse as sp
import numpy as np
import argparse
import random
import time
import math
import sys
import gc
import tracemalloc

tracemalloc.start()

cI = complex(0,1)

import grassmanntn as gtn
from grassmanntn import param
from grassmanntn import gauge2d as gauge
from grassmanntn import sparse as sparse
from grassmanntn import dense as dense

#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
#                               Argument Parser                               #
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

parse = argparse.ArgumentParser()
parse.add_argument('--beta',    default=1.0, type=float)
parse.add_argument('--mass',    default=1.0, type=float)
parse.add_argument('--charge',  default=1.0,   type=float)
parse.add_argument('--spacing', default=1.0,   type=float)
parse.add_argument('--mu',      default=0.0, type=float)
parse.add_argument('--Nf',      default=1,   type=int)
parse.add_argument('--K',  default=2,   type=int)
parse.add_argument('--cgsteps', default=5,   type=int)
parse.add_argument('--Dcutz',   default=32,  type=int)
parse.add_argument('--Dcutxy',  default=32,  type=int)
parse.add_argument('--boundary_conditions', default="anti-periodic")
parse.add_argument('--clear_screen', default=False, action='store_true')
parse.add_argument('--cls', default=False, action='store_true')
parse.add_argument('--show_progress', default=False, action='store_true')

args = parse.parse_args()


#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

if args.clear_screen or args.cls:
    os.system('clear')

if args.show_progress:
    gtn.progress_bar_enabled = True
else:
    gtn.progress_bar_enabled = True #False
    
β = args.beta             # inverse coupling
m = args.mass             # mass
μ = args.mu               # chemical potential
q = args.charge           # charge
a = args.spacing          # lattice spacing
Nf = args.Nf              # fermion species
Nphi = args.K             # Z_K group
Zcut  = args.Dcutz        # Zcut for flavors
XYcut = args.Dcutxy       # XYcut for TRG
cgsteps = args.cgsteps    # the number of 2dtrgs
bc = args.boundary_conditions

print(
    " _parameters: β="+str(β)
    +", m="+str(m)
    +", κ="+str(gtn.clean_format(1.0/(2*m+4)))
    +", μ="+str(μ)
    +", q="+str(q)
    +", a="+str(a)
    +", Nf="+str(Nf)
    +", Z_"+str(Nphi)
    +", Dcutz="+str(Zcut)
    +", Dcutxy="+str(XYcut)
    +", "+bc
    )

t0 = time.time()
T, err = gauge.tensor_preparation(Nphi=Nphi, beta=β, Nf=Nf, spacing=a, mass=m, charge=q, mu=μ, mute=True)
logNorm = 0
vol = 1

print(" _initial_tensor:",(T.shape[0],T.shape[1]),"   ",'{:.3g}'.format(err),"   ",gtn.time_display(time.time()-t0))

#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
#                      Flavor Coarse-graining Procedure                       #
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

Log2Nf = int(np.log2(Nf))
intermediate_dcut=Zcut
for i in range(Log2Nf):
    
    t0 = time.time()
    T, Tnorm, err = gauge.hotrg3dz(T,T,Zcut,intermediate_dcut=intermediate_dcut,iternum=i
                                        ,error_test=True
                                        )
    logNorm = 2*logNorm + np.log(Tnorm)

    print(" _flavor_cg:",(T.shape[0],T.shape[1]),"   ",'{:.3g}'.format(err),"   ",gtn.time_display(time.time()-t0))

T = gauge.zcap(T)

#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
#                         2D Coarse-graining Procedure                        #
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

F = gauge.logZ(T,boundary_conditions=bc)+logNorm

cgxfirst = T.shape[0] > T.shape[1]

print(" trg:",vol,β,m,μ,q,a,Nf,Nphi,"   ",np.real(F),"   ",np.imag(F),"   ",(T.shape[0],T.shape[1]))
for i in range(cgsteps):
    t0 = time.time()
    direction=""
    if cgxfirst :
        if i%2==0 :
            T, Tnorm, err = gauge.atrg2dx(T,T,XYcut,iternum=i,error_test=True)
        else:
            T, Tnorm, err = gauge.atrg2dy(T,T,XYcut,iternum=i,error_test=True)
    else:
        if i%2==0 :
            T, Tnorm, err = gauge.atrg2dy(T,T,XYcut,iternum=i,error_test=True)
        else:
            T, Tnorm, err = gauge.atrg2dx(T,T,XYcut,iternum=i,error_test=True)

    vol = 2**(i+1)
    logNorm = 2*logNorm + np.log(Tnorm)
    F = (gauge.logZ(T,bc)+logNorm)/vol

    #print(gtn.clean_format(F))
    print(" trg:",vol,β,m,μ,q,a,Nf,Nphi,"   ",np.real(F),"   ",np.imag(F),"   ",(T.shape[0],T.shape[1]),"   ",'{:.3g}'.format(err),"   ",gtn.time_display(time.time()-t0))
