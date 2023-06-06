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

cI = complex(0,1)

from grassmanntn import grassmanntn as gtn
from grassmanntn import param
from grassmanntn.grassmanntn import sparse as sparse
from grassmanntn.grassmanntn import dense as dense

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
parse.add_argument('--Dcutc',   default=32,  type=int)
parse.add_argument('--Dcutz',   default=32,  type=int)
parse.add_argument('--Dcutxy',  default=32,  type=int)
parse.add_argument('--boundary_conditions', default="anti-periodic")
parse.add_argument('--clear_screen', default=False, action='store_true')
parse.add_argument('--cs', default=False, action='store_true')

args = parse.parse_args()


#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

if args.clear_screen or args.cs:
	os.system('clear')

def main():

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
		"parameters: β="+str(β)
		+", m="+str(m)
		+", μ="+str(μ)
		+", q="+str(q)
		+", a="+str(a)
		+", Nf="+str(Nf)
		+", K="+str(Nphi)
		+", Dz="+str(Zcut)
		+", Dxy="+str(XYcut)
		+", "+bc
		)

	t0 = time.time()
	T = tensor_preparation(Nphi=Nphi, beta=β, Nf=Nf, spacing=a, mass=m, charge=q, mu=μ, mute=True)
	logNorm = 0
	vol = 1

	print("tensor shape:",T.shape)
	F0 = logZhotrg3dz(T,T,boundary_conditions=bc)+2*logNorm
	print(F0)


	#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
	#                      Flavor Coarse-graining Procedure                       #
	#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

	Log2Nf = int(np.log2(Nf))
	for i in range(Log2Nf):

		F0 = logZhotrg3dz(T,T,boundary_conditions=bc)+2*logNorm

		T, Tnorm = gtn.hotrg3dz(T,T,Zcut,iternum=i)
		logNorm = 2*logNorm + np.log(Tnorm)

		F1 = logZ(zcap(T),boundary_conditions=bc)+logNorm
		print(F1)
		#print(" flavor_cg[Nf="+str(2**i)+"→ "+str(2**(i+1))+"]: err =",np.abs(F1-F0),",",gtn.time_display(time.time()-t0))
		t0 = time.time()
		print("tensor shape:",T.shape)

	T = zcap(T)

	#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
	#                         2D Coarse-graining Procedure                        #
	#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

	F = logZ(T,boundary_conditions=bc)+logNorm

	print(" trg:",vol,β,m,μ,q,a,Nf,Nphi,"   ",np.real(F),"   ",np.imag(F),"   ",gtn.time_display(time.time()-t0))
	for i in range(cgsteps):
		t0 = time.time()
		
		if i%2==0 :
			T, Tnorm = gtn.atrg2dx(T,T,XYcut,iternum=i)
		else:
			T, Tnorm = gtn.atrg2dy(T,T,XYcut,iternum=i)

		#print("                            atrg["+str(i)+"]: "+gtn.time_display(time.time()-t0))

		#T.info()

		vol = 2**(i+1)
		logNorm = 2*logNorm + np.log(Tnorm)
		F = (logZ(T,bc)+logNorm)/vol
		
		#print(gtn.clean_format(F))
		print(" trg:",vol,β,m,μ,q,a,Nf,Nphi,"   ",np.real(F),"   ",np.imag(F),"   ",gtn.time_display(time.time()-t0))
		print("tensor shape:",T.shape)

####################################################
##           Initial tensor compression           ##
####################################################

def tensor_preparation(Nphi, beta, Nf, spacing, mass, charge, mu, mute=True):

	if not mute:
		print()
		print(" :·:·:·:·:·:·:·:·:·:·:·:·:·:·: Tensor preparation :·:·:·:·:·:·:·:·:·:·:·:·:·:·: ")
		print()

	A, B = get_ABtensors(Nphi=Nphi, beta=beta, Nf=Nf, spacing=spacing, mass=mass, charge=charge, mu=mu)
	z1 = gtn.einsum("IJIJijij,jiji",B,A)

	t0 = time.time()
	B = fcompress_B(B)
	if not mute:
		print("                       B compression (1): "+gtn.time_display(time.time()-t0))

	t0 = time.time()
	B, [U1,U2,U3,U4] = compress_B(B)
	if not mute:
		print("                       B compression (2): "+gtn.time_display(time.time()-t0))

	t0 = time.time()
	A = compress_A(A,[U1,U2,U3,U4])
	if not mute:
		print("                           A compression: "+gtn.time_display(time.time()-t0))


	t0 = time.time()
	T = gtn.einsum('IJXYijklmn,XYKL->IJKLijklmn',A,B)
	if not mute:
		print("                             T formation: "+gtn.time_display(time.time()-t0))
	#z3 = gtn.einsum("IJIJjijiji",T)


	t0 = time.time()
	T = compress_T(T)
	if not mute:
		print("                           T compression: "+gtn.time_display(time.time()-t0))
	z4 = gtn.einsum("IJIJij,ij",T,sparse(np.full((Nphi,Nphi),1),statistic=(0,0)))



	trace_error = np.abs(1-z4/z1)
	if not mute:
		print("                       Compression error:",trace_error)
		T.info("Compressed tensor T",21)

	T = dense(T)

	return T

def myQuadrature(beta, npoints):
        
        xi = np.zeros([npoints],dtype=float)
        wi = np.zeros([npoints],dtype=float)
        for i in range(npoints):
            xi[i] = i*2*np.pi/npoints #-np.pi + 2*np.pi*(i+0.5)/npoints
            wi[i] = 2*np.pi/npoints
        
        return xi, wi

def get_ABtensors(Nphi=2, beta=1, Nf=1, spacing=1, mass=1, charge=1, mu=1):
	Npsi = 16 #this value is probably always fixed to 16
	Bshape = (Npsi,Npsi,Npsi,Npsi,Nphi,Nphi,Nphi,Nphi)
	A = np.zeros([Nphi,Nphi,Nphi,Nphi],dtype=float)
	psi1 = []
	psi2 = []
	psi3 = []
	psi4 = []
	phi1 = []
	phi2 = []
	phi3 = []
	phi4 = []
	B = []
	xi, wi = myQuadrature(beta=beta/Nf,npoints=Nphi)
	for i in range(Nphi):
		for j in range(Nphi):
			for k in range(Nphi):
				for l in range(Nphi):
					A[i,j,k,l]=np.exp(beta/Nf*(np.cos(xi[l]+xi[i]-xi[j]-xi[k])-1))*wi[k]*wi[l]
	for i in range(Nphi):
		p1 = xi[i]
		for j in range(Nphi):
			q1 = xi[j]
			for k in range(Nphi):
				P1 = xi[k]
				for l in range(Nphi):
					Q1 = xi[l]
					#There are 449 non-zero components for B.
					#Canonical ordering: (\[Eta]_{x,1;L}).(\[Eta]_{x,1;R}).(\[Eta]_{x+1,-1;L}).(\[Eta]_{x+1,-1;R}).(\[Eta]_{x,2;L}).(\[Eta]_{x,2;R}).(\[Eta]_{x+2,-2;L}).(\[Eta]_{x+2,-2;R}).(\[Eta]_{x-1,+1;L}).(\[Eta]_{x-1,+1;R}).(\[Eta]_{x,-1;L}).(\[Eta]_{x,-1;R}).(\[Eta]_{x-2,+2;L}).(\[Eta]_{x-2,+2;R}).(\[Eta]_{x,-2;L}).(\[Eta]_{x,-2;R})
					#Bosonic variables:
					#	p1 = \[CurlyPhi]_{x,1} = b(+x)
					#	q1 = \[CurlyPhi]_{x,2} = b(+y)
					#	P1 = \[CurlyPhi]_{x-1,1} = b(-x)
					#	Q1 = \[CurlyPhi]_{x-2,2} = b(-x)
					# Notation of the tensor: B[f(+x),f(+y),f(-x),f(-y);b(+x),b(+y),b(-x),b(-y)]
					psi1+=[0]; psi2+=[0]; psi3+=[0]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[4 + 4*spacing*mass + spacing**2*mass**2]  #1
					psi1+=[0]; psi2+=[0]; psi3+=[5]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-np.exp(cI*P1*charge) - (spacing*np.exp(cI*P1*charge)*mass)/2]  #2
					psi1+=[0]; psi2+=[0]; psi3+=[9]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[np.exp(cI*P1*charge) + (spacing*np.exp(cI*P1*charge)*mass)/2]  #3
					psi1+=[0]; psi2+=[0]; psi3+=[1]; psi4+=[4]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-np.exp(cI*P1*charge) - (spacing*np.exp(cI*P1*charge)*mass)/2]  #4
					psi1+=[0]; psi2+=[0]; psi3+=[1]; psi4+=[8]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[np.exp(cI*P1*charge) + (spacing*np.exp(cI*P1*charge)*mass)/2]  #5
					psi1+=[1]; psi2+=[0]; psi3+=[1]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-np.exp(cI*P1*charge) - (spacing*np.exp(cI*P1*charge)*mass)/2]  #6
					psi1+=[2]; psi2+=[0]; psi3+=[1]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[np.exp(cI*P1*charge) + (spacing*np.exp(cI*P1*charge)*mass)/2]  #7
					psi1+=[0]; psi2+=[1]; psi3+=[1]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-np.exp(cI*P1*charge) - (spacing*np.exp(cI*P1*charge)*mass)/2]  #8
					psi1+=[0]; psi2+=[2]; psi3+=[1]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[np.exp(cI*P1*charge) + (spacing*np.exp(cI*P1*charge)*mass)/2]  #9
					psi1+=[0]; psi2+=[0]; psi3+=[6]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[np.exp(cI*P1*charge) + (spacing*np.exp(cI*P1*charge)*mass)/2]  #10
					psi1+=[0]; psi2+=[0]; psi3+=[10]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-np.exp(cI*P1*charge) - (spacing*np.exp(cI*P1*charge)*mass)/2]  #11
					psi1+=[0]; psi2+=[0]; psi3+=[2]; psi4+=[4]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[np.exp(cI*P1*charge) + (spacing*np.exp(cI*P1*charge)*mass)/2]  #12
					psi1+=[0]; psi2+=[0]; psi3+=[2]; psi4+=[8]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-np.exp(cI*P1*charge) - (spacing*np.exp(cI*P1*charge)*mass)/2]  #13
					psi1+=[1]; psi2+=[0]; psi3+=[2]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[np.exp(cI*P1*charge) + (spacing*np.exp(cI*P1*charge)*mass)/2]  #14
					psi1+=[2]; psi2+=[0]; psi3+=[2]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-np.exp(cI*P1*charge) - (spacing*np.exp(cI*P1*charge)*mass)/2]  #15
					psi1+=[0]; psi2+=[1]; psi3+=[2]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[np.exp(cI*P1*charge) + (spacing*np.exp(cI*P1*charge)*mass)/2]  #16
					psi1+=[0]; psi2+=[2]; psi3+=[2]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-np.exp(cI*P1*charge) - (spacing*np.exp(cI*P1*charge)*mass)/2]  #17
					psi1+=[0]; psi2+=[0]; psi3+=[4]; psi4+=[1]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[np.exp(cI*Q1*charge + spacing*mu) + (spacing*np.exp(cI*Q1*charge + spacing*mu)*mass)/2]  #18
					psi1+=[0]; psi2+=[0]; psi3+=[8]; psi4+=[1]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[cI*np.exp(cI*Q1*charge + spacing*mu) + (cI/2)*spacing*np.exp(cI*Q1*charge + spacing*mu)*mass]  #19
					psi1+=[0]; psi2+=[0]; psi3+=[0]; psi4+=[5]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-np.exp(cI*Q1*charge + spacing*mu) - (spacing*np.exp(cI*Q1*charge + spacing*mu)*mass)/2]  #20
					psi1+=[0]; psi2+=[0]; psi3+=[0]; psi4+=[9]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-cI)*np.exp(cI*Q1*charge + spacing*mu) - (cI/2)*spacing*np.exp(cI*Q1*charge + spacing*mu)*mass]  #21
					psi1+=[1]; psi2+=[0]; psi3+=[0]; psi4+=[1]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-np.exp(cI*Q1*charge + spacing*mu) - (spacing*np.exp(cI*Q1*charge + spacing*mu)*mass)/2]  #22
					psi1+=[2]; psi2+=[0]; psi3+=[0]; psi4+=[1]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-cI)*np.exp(cI*Q1*charge + spacing*mu) - (cI/2)*spacing*np.exp(cI*Q1*charge + spacing*mu)*mass]  #23
					psi1+=[0]; psi2+=[1]; psi3+=[0]; psi4+=[1]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-np.exp(cI*Q1*charge + spacing*mu) - (spacing*np.exp(cI*Q1*charge + spacing*mu)*mass)/2]  #24
					psi1+=[0]; psi2+=[2]; psi3+=[0]; psi4+=[1]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-cI)*np.exp(cI*Q1*charge + spacing*mu) - (cI/2)*spacing*np.exp(cI*Q1*charge + spacing*mu)*mass]  #25
					psi1+=[0]; psi2+=[0]; psi3+=[4]; psi4+=[2]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-cI)*np.exp(cI*Q1*charge + spacing*mu) - (cI/2)*spacing*np.exp(cI*Q1*charge + spacing*mu)*mass]  #26
					psi1+=[0]; psi2+=[0]; psi3+=[8]; psi4+=[2]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[np.exp(cI*Q1*charge + spacing*mu) + (spacing*np.exp(cI*Q1*charge + spacing*mu)*mass)/2]  #27
					psi1+=[0]; psi2+=[0]; psi3+=[0]; psi4+=[6]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[cI*np.exp(cI*Q1*charge + spacing*mu) + (cI/2)*spacing*np.exp(cI*Q1*charge + spacing*mu)*mass]  #28
					psi1+=[0]; psi2+=[0]; psi3+=[0]; psi4+=[10]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-np.exp(cI*Q1*charge + spacing*mu) - (spacing*np.exp(cI*Q1*charge + spacing*mu)*mass)/2]  #29
					psi1+=[1]; psi2+=[0]; psi3+=[0]; psi4+=[2]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[cI*np.exp(cI*Q1*charge + spacing*mu) + (cI/2)*spacing*np.exp(cI*Q1*charge + spacing*mu)*mass]  #30
					psi1+=[2]; psi2+=[0]; psi3+=[0]; psi4+=[2]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-np.exp(cI*Q1*charge + spacing*mu) - (spacing*np.exp(cI*Q1*charge + spacing*mu)*mass)/2]  #31
					psi1+=[0]; psi2+=[1]; psi3+=[0]; psi4+=[2]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[cI*np.exp(cI*Q1*charge + spacing*mu) + (cI/2)*spacing*np.exp(cI*Q1*charge + spacing*mu)*mass]  #32
					psi1+=[0]; psi2+=[2]; psi3+=[0]; psi4+=[2]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-np.exp(cI*Q1*charge + spacing*mu) - (spacing*np.exp(cI*Q1*charge + spacing*mu)*mass)/2]  #33
					psi1+=[4]; psi2+=[0]; psi3+=[4]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-np.exp((-cI)*p1*charge) - (spacing*mass)/(2*np.exp(cI*p1*charge))]  #34
					psi1+=[4]; psi2+=[0]; psi3+=[8]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-np.exp((-cI)*p1*charge) - (spacing*mass)/(2*np.exp(cI*p1*charge))]  #35
					psi1+=[4]; psi2+=[0]; psi3+=[0]; psi4+=[4]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-np.exp((-cI)*p1*charge) - (spacing*mass)/(2*np.exp(cI*p1*charge))]  #36
					psi1+=[4]; psi2+=[0]; psi3+=[0]; psi4+=[8]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-np.exp((-cI)*p1*charge) - (spacing*mass)/(2*np.exp(cI*p1*charge))]  #37
					psi1+=[5]; psi2+=[0]; psi3+=[0]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-np.exp((-cI)*p1*charge) - (spacing*mass)/(2*np.exp(cI*p1*charge))]  #38
					psi1+=[6]; psi2+=[0]; psi3+=[0]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-np.exp((-cI)*p1*charge) - (spacing*mass)/(2*np.exp(cI*p1*charge))]  #39
					psi1+=[4]; psi2+=[1]; psi3+=[0]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[np.exp((-cI)*p1*charge) + (spacing*mass)/(2*np.exp(cI*p1*charge))]  #40
					psi1+=[4]; psi2+=[2]; psi3+=[0]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[np.exp((-cI)*p1*charge) + (spacing*mass)/(2*np.exp(cI*p1*charge))]  #41
					psi1+=[8]; psi2+=[0]; psi3+=[4]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-np.exp((-cI)*p1*charge) - (spacing*mass)/(2*np.exp(cI*p1*charge))]  #42
					psi1+=[8]; psi2+=[0]; psi3+=[8]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-np.exp((-cI)*p1*charge) - (spacing*mass)/(2*np.exp(cI*p1*charge))]  #43
					psi1+=[8]; psi2+=[0]; psi3+=[0]; psi4+=[4]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-np.exp((-cI)*p1*charge) - (spacing*mass)/(2*np.exp(cI*p1*charge))]  #44
					psi1+=[8]; psi2+=[0]; psi3+=[0]; psi4+=[8]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-np.exp((-cI)*p1*charge) - (spacing*mass)/(2*np.exp(cI*p1*charge))]  #45
					psi1+=[9]; psi2+=[0]; psi3+=[0]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-np.exp((-cI)*p1*charge) - (spacing*mass)/(2*np.exp(cI*p1*charge))]  #46
					psi1+=[10]; psi2+=[0]; psi3+=[0]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-np.exp((-cI)*p1*charge) - (spacing*mass)/(2*np.exp(cI*p1*charge))]  #47
					psi1+=[8]; psi2+=[1]; psi3+=[0]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[np.exp((-cI)*p1*charge) + (spacing*mass)/(2*np.exp(cI*p1*charge))]  #48
					psi1+=[8]; psi2+=[2]; psi3+=[0]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[np.exp((-cI)*p1*charge) + (spacing*mass)/(2*np.exp(cI*p1*charge))]  #49
					psi1+=[0]; psi2+=[4]; psi3+=[4]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-np.exp((-cI)*q1*charge - spacing*mu) - (spacing*np.exp((-cI)*q1*charge - spacing*mu)*mass)/2]  #50
					psi1+=[0]; psi2+=[4]; psi3+=[8]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[cI*np.exp((-cI)*q1*charge - spacing*mu) + (cI/2)*spacing*np.exp((-cI)*q1*charge - spacing*mu)*mass]  #51
					psi1+=[0]; psi2+=[4]; psi3+=[0]; psi4+=[4]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-np.exp((-cI)*q1*charge - spacing*mu) - (spacing*np.exp((-cI)*q1*charge - spacing*mu)*mass)/2]  #52
					psi1+=[0]; psi2+=[4]; psi3+=[0]; psi4+=[8]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[cI*np.exp((-cI)*q1*charge - spacing*mu) + (cI/2)*spacing*np.exp((-cI)*q1*charge - spacing*mu)*mass]  #53
					psi1+=[1]; psi2+=[4]; psi3+=[0]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-np.exp((-cI)*q1*charge - spacing*mu) - (spacing*np.exp((-cI)*q1*charge - spacing*mu)*mass)/2]  #54
					psi1+=[2]; psi2+=[4]; psi3+=[0]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[cI*np.exp((-cI)*q1*charge - spacing*mu) + (cI/2)*spacing*np.exp((-cI)*q1*charge - spacing*mu)*mass]  #55
					psi1+=[0]; psi2+=[5]; psi3+=[0]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-np.exp((-cI)*q1*charge - spacing*mu) - (spacing*np.exp((-cI)*q1*charge - spacing*mu)*mass)/2]  #56
					psi1+=[0]; psi2+=[6]; psi3+=[0]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[cI*np.exp((-cI)*q1*charge - spacing*mu) + (cI/2)*spacing*np.exp((-cI)*q1*charge - spacing*mu)*mass]  #57
					psi1+=[0]; psi2+=[8]; psi3+=[4]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-cI)*np.exp((-cI)*q1*charge - spacing*mu) - (cI/2)*spacing*np.exp((-cI)*q1*charge - spacing*mu)*mass]  #58
					psi1+=[0]; psi2+=[8]; psi3+=[8]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-np.exp((-cI)*q1*charge - spacing*mu) - (spacing*np.exp((-cI)*q1*charge - spacing*mu)*mass)/2]  #59
					psi1+=[0]; psi2+=[8]; psi3+=[0]; psi4+=[4]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-cI)*np.exp((-cI)*q1*charge - spacing*mu) - (cI/2)*spacing*np.exp((-cI)*q1*charge - spacing*mu)*mass]  #60
					psi1+=[0]; psi2+=[8]; psi3+=[0]; psi4+=[8]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-np.exp((-cI)*q1*charge - spacing*mu) - (spacing*np.exp((-cI)*q1*charge - spacing*mu)*mass)/2]  #61
					psi1+=[1]; psi2+=[8]; psi3+=[0]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-cI)*np.exp((-cI)*q1*charge - spacing*mu) - (cI/2)*spacing*np.exp((-cI)*q1*charge - spacing*mu)*mass]  #62
					psi1+=[2]; psi2+=[8]; psi3+=[0]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-np.exp((-cI)*q1*charge - spacing*mu) - (spacing*np.exp((-cI)*q1*charge - spacing*mu)*mass)/2]  #63
					psi1+=[0]; psi2+=[9]; psi3+=[0]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-cI)*np.exp((-cI)*q1*charge - spacing*mu) - (cI/2)*spacing*np.exp((-cI)*q1*charge - spacing*mu)*mass]  #64
					psi1+=[0]; psi2+=[10]; psi3+=[0]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-np.exp((-cI)*q1*charge - spacing*mu) - (spacing*np.exp((-cI)*q1*charge - spacing*mu)*mass)/2]  #65
					psi1+=[0]; psi2+=[0]; psi3+=[13]; psi4+=[1]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 - cI/4)*np.exp(cI*P1*charge + cI*Q1*charge + spacing*mu)]  #66
					psi1+=[0]; psi2+=[0]; psi3+=[5]; psi4+=[9]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 + cI/4)*np.exp(cI*P1*charge + cI*Q1*charge + spacing*mu)]  #67
					psi1+=[2]; psi2+=[0]; psi3+=[5]; psi4+=[1]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 + cI/4)*np.exp(cI*P1*charge + cI*Q1*charge + spacing*mu)]  #68
					psi1+=[0]; psi2+=[2]; psi3+=[5]; psi4+=[1]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 + cI/4)*np.exp(cI*P1*charge + cI*Q1*charge + spacing*mu)]  #69
					psi1+=[0]; psi2+=[0]; psi3+=[9]; psi4+=[5]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 - cI/4)*np.exp(cI*P1*charge + cI*Q1*charge + spacing*mu)]  #70
					psi1+=[1]; psi2+=[0]; psi3+=[9]; psi4+=[1]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 - cI/4)*np.exp(cI*P1*charge + cI*Q1*charge + spacing*mu)]  #71
					psi1+=[0]; psi2+=[1]; psi3+=[9]; psi4+=[1]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 - cI/4)*np.exp(cI*P1*charge + cI*Q1*charge + spacing*mu)]  #72
					psi1+=[0]; psi2+=[0]; psi3+=[1]; psi4+=[13]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 - cI/4)*np.exp(cI*P1*charge + cI*Q1*charge + spacing*mu)]  #73
					psi1+=[2]; psi2+=[0]; psi3+=[1]; psi4+=[5]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 - cI/4)*np.exp(cI*P1*charge + cI*Q1*charge + spacing*mu)]  #74
					psi1+=[0]; psi2+=[2]; psi3+=[1]; psi4+=[5]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 - cI/4)*np.exp(cI*P1*charge + cI*Q1*charge + spacing*mu)]  #75
					psi1+=[1]; psi2+=[0]; psi3+=[1]; psi4+=[9]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 + cI/4)*np.exp(cI*P1*charge + cI*Q1*charge + spacing*mu)]  #76
					psi1+=[0]; psi2+=[1]; psi3+=[1]; psi4+=[9]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 + cI/4)*np.exp(cI*P1*charge + cI*Q1*charge + spacing*mu)]  #77
					psi1+=[3]; psi2+=[0]; psi3+=[1]; psi4+=[1]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 - cI/4)*np.exp(cI*P1*charge + cI*Q1*charge + spacing*mu)]  #78
					psi1+=[1]; psi2+=[2]; psi3+=[1]; psi4+=[1]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 - cI/4)*np.exp(cI*P1*charge + cI*Q1*charge + spacing*mu)]  #79
					psi1+=[2]; psi2+=[1]; psi3+=[1]; psi4+=[1]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 + cI/4)*np.exp(cI*P1*charge + cI*Q1*charge + spacing*mu)]  #80
					psi1+=[0]; psi2+=[3]; psi3+=[1]; psi4+=[1]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 - cI/4)*np.exp(cI*P1*charge + cI*Q1*charge + spacing*mu)]  #81
					psi1+=[0]; psi2+=[0]; psi3+=[13]; psi4+=[2]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 + cI/4)*np.exp(cI*P1*charge + cI*Q1*charge + spacing*mu)]  #82
					psi1+=[0]; psi2+=[0]; psi3+=[5]; psi4+=[10]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 - cI/4)*np.exp(cI*P1*charge + cI*Q1*charge + spacing*mu)]  #83
					psi1+=[2]; psi2+=[0]; psi3+=[5]; psi4+=[2]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 - cI/4)*np.exp(cI*P1*charge + cI*Q1*charge + spacing*mu)]  #84
					psi1+=[0]; psi2+=[2]; psi3+=[5]; psi4+=[2]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 - cI/4)*np.exp(cI*P1*charge + cI*Q1*charge + spacing*mu)]  #85
					psi1+=[0]; psi2+=[0]; psi3+=[9]; psi4+=[6]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 + cI/4)*np.exp(cI*P1*charge + cI*Q1*charge + spacing*mu)]  #86
					psi1+=[1]; psi2+=[0]; psi3+=[9]; psi4+=[2]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 + cI/4)*np.exp(cI*P1*charge + cI*Q1*charge + spacing*mu)]  #87
					psi1+=[0]; psi2+=[1]; psi3+=[9]; psi4+=[2]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 + cI/4)*np.exp(cI*P1*charge + cI*Q1*charge + spacing*mu)]  #88
					psi1+=[0]; psi2+=[0]; psi3+=[1]; psi4+=[14]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 + cI/4)*np.exp(cI*P1*charge + cI*Q1*charge + spacing*mu)]  #89
					psi1+=[2]; psi2+=[0]; psi3+=[1]; psi4+=[6]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 + cI/4)*np.exp(cI*P1*charge + cI*Q1*charge + spacing*mu)]  #90
					psi1+=[0]; psi2+=[2]; psi3+=[1]; psi4+=[6]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 + cI/4)*np.exp(cI*P1*charge + cI*Q1*charge + spacing*mu)]  #91
					psi1+=[1]; psi2+=[0]; psi3+=[1]; psi4+=[10]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 - cI/4)*np.exp(cI*P1*charge + cI*Q1*charge + spacing*mu)]  #92
					psi1+=[0]; psi2+=[1]; psi3+=[1]; psi4+=[10]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 - cI/4)*np.exp(cI*P1*charge + cI*Q1*charge + spacing*mu)]  #93
					psi1+=[3]; psi2+=[0]; psi3+=[1]; psi4+=[2]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 + cI/4)*np.exp(cI*P1*charge + cI*Q1*charge + spacing*mu)]  #94
					psi1+=[1]; psi2+=[2]; psi3+=[1]; psi4+=[2]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 + cI/4)*np.exp(cI*P1*charge + cI*Q1*charge + spacing*mu)]  #95
					psi1+=[2]; psi2+=[1]; psi3+=[1]; psi4+=[2]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 - cI/4)*np.exp(cI*P1*charge + cI*Q1*charge + spacing*mu)]  #96
					psi1+=[0]; psi2+=[3]; psi3+=[1]; psi4+=[2]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 + cI/4)*np.exp(cI*P1*charge + cI*Q1*charge + spacing*mu)]  #97
					psi1+=[4]; psi2+=[0]; psi3+=[13]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[np.exp((-cI)*p1*charge + cI*P1*charge)/2]  #98
					psi1+=[4]; psi2+=[0]; psi3+=[5]; psi4+=[8]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[np.exp((-cI)*p1*charge + cI*P1*charge)/2]  #99
					psi1+=[6]; psi2+=[0]; psi3+=[5]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[np.exp((-cI)*p1*charge + cI*P1*charge)/2]  #100
					psi1+=[4]; psi2+=[2]; psi3+=[5]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-1/2*np.exp((-cI)*p1*charge + cI*P1*charge)]  #101
					psi1+=[4]; psi2+=[0]; psi3+=[9]; psi4+=[4]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-1/2*np.exp((-cI)*p1*charge + cI*P1*charge)]  #102
					psi1+=[5]; psi2+=[0]; psi3+=[9]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-1/2*np.exp((-cI)*p1*charge + cI*P1*charge)]  #103
					psi1+=[4]; psi2+=[1]; psi3+=[9]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[np.exp((-cI)*p1*charge + cI*P1*charge)/2]  #104
					psi1+=[4]; psi2+=[0]; psi3+=[1]; psi4+=[12]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[np.exp((-cI)*p1*charge + cI*P1*charge)/2]  #105
					psi1+=[6]; psi2+=[0]; psi3+=[1]; psi4+=[4]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[np.exp((-cI)*p1*charge + cI*P1*charge)/2]  #106
					psi1+=[4]; psi2+=[2]; psi3+=[1]; psi4+=[4]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-1/2*np.exp((-cI)*p1*charge + cI*P1*charge)]  #107
					psi1+=[5]; psi2+=[0]; psi3+=[1]; psi4+=[8]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-1/2*np.exp((-cI)*p1*charge + cI*P1*charge)]  #108
					psi1+=[4]; psi2+=[1]; psi3+=[1]; psi4+=[8]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[np.exp((-cI)*p1*charge + cI*P1*charge)/2]  #109
					psi1+=[7]; psi2+=[0]; psi3+=[1]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[np.exp((-cI)*p1*charge + cI*P1*charge)/2]  #110
					psi1+=[5]; psi2+=[2]; psi3+=[1]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-1/2*np.exp((-cI)*p1*charge + cI*P1*charge)]  #111
					psi1+=[6]; psi2+=[1]; psi3+=[1]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[np.exp((-cI)*p1*charge + cI*P1*charge)/2]  #112
					psi1+=[4]; psi2+=[3]; psi3+=[1]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[np.exp((-cI)*p1*charge + cI*P1*charge)/2]  #113
					psi1+=[8]; psi2+=[0]; psi3+=[13]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[np.exp((-cI)*p1*charge + cI*P1*charge)/2]  #114
					psi1+=[8]; psi2+=[0]; psi3+=[5]; psi4+=[8]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[np.exp((-cI)*p1*charge + cI*P1*charge)/2]  #115
					psi1+=[10]; psi2+=[0]; psi3+=[5]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[np.exp((-cI)*p1*charge + cI*P1*charge)/2]  #116
					psi1+=[8]; psi2+=[2]; psi3+=[5]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-1/2*np.exp((-cI)*p1*charge + cI*P1*charge)]  #117
					psi1+=[8]; psi2+=[0]; psi3+=[9]; psi4+=[4]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-1/2*np.exp((-cI)*p1*charge + cI*P1*charge)]  #118
					psi1+=[9]; psi2+=[0]; psi3+=[9]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-1/2*np.exp((-cI)*p1*charge + cI*P1*charge)]  #119
					psi1+=[8]; psi2+=[1]; psi3+=[9]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[np.exp((-cI)*p1*charge + cI*P1*charge)/2]  #120
					psi1+=[8]; psi2+=[0]; psi3+=[1]; psi4+=[12]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[np.exp((-cI)*p1*charge + cI*P1*charge)/2]  #121
					psi1+=[10]; psi2+=[0]; psi3+=[1]; psi4+=[4]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[np.exp((-cI)*p1*charge + cI*P1*charge)/2]  #122
					psi1+=[8]; psi2+=[2]; psi3+=[1]; psi4+=[4]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-1/2*np.exp((-cI)*p1*charge + cI*P1*charge)]  #123
					psi1+=[9]; psi2+=[0]; psi3+=[1]; psi4+=[8]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-1/2*np.exp((-cI)*p1*charge + cI*P1*charge)]  #124
					psi1+=[8]; psi2+=[1]; psi3+=[1]; psi4+=[8]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[np.exp((-cI)*p1*charge + cI*P1*charge)/2]  #125
					psi1+=[11]; psi2+=[0]; psi3+=[1]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[np.exp((-cI)*p1*charge + cI*P1*charge)/2]  #126
					psi1+=[9]; psi2+=[2]; psi3+=[1]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-1/2*np.exp((-cI)*p1*charge + cI*P1*charge)]  #127
					psi1+=[10]; psi2+=[1]; psi3+=[1]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[np.exp((-cI)*p1*charge + cI*P1*charge)/2]  #128
					psi1+=[8]; psi2+=[3]; psi3+=[1]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[np.exp((-cI)*p1*charge + cI*P1*charge)/2]  #129
					psi1+=[0]; psi2+=[4]; psi3+=[13]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 - cI/4)*np.exp((-cI)*q1*charge + cI*P1*charge - spacing*mu)]  #130
					psi1+=[0]; psi2+=[4]; psi3+=[5]; psi4+=[8]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 - cI/4)*np.exp((-cI)*q1*charge + cI*P1*charge - spacing*mu)]  #131
					psi1+=[2]; psi2+=[4]; psi3+=[5]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 - cI/4)*np.exp((-cI)*q1*charge + cI*P1*charge - spacing*mu)]  #132
					psi1+=[0]; psi2+=[6]; psi3+=[5]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 - cI/4)*np.exp((-cI)*q1*charge + cI*P1*charge - spacing*mu)]  #133
					psi1+=[0]; psi2+=[4]; psi3+=[9]; psi4+=[4]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 + cI/4)*np.exp((-cI)*q1*charge + cI*P1*charge - spacing*mu)]  #134
					psi1+=[1]; psi2+=[4]; psi3+=[9]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 + cI/4)*np.exp((-cI)*q1*charge + cI*P1*charge - spacing*mu)]  #135
					psi1+=[0]; psi2+=[5]; psi3+=[9]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 + cI/4)*np.exp((-cI)*q1*charge + cI*P1*charge - spacing*mu)]  #136
					psi1+=[0]; psi2+=[4]; psi3+=[1]; psi4+=[12]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 - cI/4)*np.exp((-cI)*q1*charge + cI*P1*charge - spacing*mu)]  #137
					psi1+=[2]; psi2+=[4]; psi3+=[1]; psi4+=[4]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 - cI/4)*np.exp((-cI)*q1*charge + cI*P1*charge - spacing*mu)]  #138
					psi1+=[0]; psi2+=[6]; psi3+=[1]; psi4+=[4]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 - cI/4)*np.exp((-cI)*q1*charge + cI*P1*charge - spacing*mu)]  #139
					psi1+=[1]; psi2+=[4]; psi3+=[1]; psi4+=[8]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 + cI/4)*np.exp((-cI)*q1*charge + cI*P1*charge - spacing*mu)]  #140
					psi1+=[0]; psi2+=[5]; psi3+=[1]; psi4+=[8]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 + cI/4)*np.exp((-cI)*q1*charge + cI*P1*charge - spacing*mu)]  #141
					psi1+=[3]; psi2+=[4]; psi3+=[1]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 - cI/4)*np.exp((-cI)*q1*charge + cI*P1*charge - spacing*mu)]  #142
					psi1+=[1]; psi2+=[6]; psi3+=[1]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 - cI/4)*np.exp((-cI)*q1*charge + cI*P1*charge - spacing*mu)]  #143
					psi1+=[2]; psi2+=[5]; psi3+=[1]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 + cI/4)*np.exp((-cI)*q1*charge + cI*P1*charge - spacing*mu)]  #144
					psi1+=[0]; psi2+=[7]; psi3+=[1]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 - cI/4)*np.exp((-cI)*q1*charge + cI*P1*charge - spacing*mu)]  #145
					psi1+=[0]; psi2+=[8]; psi3+=[13]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 + cI/4)*np.exp((-cI)*q1*charge + cI*P1*charge - spacing*mu)]  #146
					psi1+=[0]; psi2+=[8]; psi3+=[5]; psi4+=[8]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 + cI/4)*np.exp((-cI)*q1*charge + cI*P1*charge - spacing*mu)]  #147
					psi1+=[2]; psi2+=[8]; psi3+=[5]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 + cI/4)*np.exp((-cI)*q1*charge + cI*P1*charge - spacing*mu)]  #148
					psi1+=[0]; psi2+=[10]; psi3+=[5]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 + cI/4)*np.exp((-cI)*q1*charge + cI*P1*charge - spacing*mu)]  #149
					psi1+=[0]; psi2+=[8]; psi3+=[9]; psi4+=[4]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 - cI/4)*np.exp((-cI)*q1*charge + cI*P1*charge - spacing*mu)]  #150
					psi1+=[1]; psi2+=[8]; psi3+=[9]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 - cI/4)*np.exp((-cI)*q1*charge + cI*P1*charge - spacing*mu)]  #151
					psi1+=[0]; psi2+=[9]; psi3+=[9]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 - cI/4)*np.exp((-cI)*q1*charge + cI*P1*charge - spacing*mu)]  #152
					psi1+=[0]; psi2+=[8]; psi3+=[1]; psi4+=[12]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 + cI/4)*np.exp((-cI)*q1*charge + cI*P1*charge - spacing*mu)]  #153
					psi1+=[2]; psi2+=[8]; psi3+=[1]; psi4+=[4]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 + cI/4)*np.exp((-cI)*q1*charge + cI*P1*charge - spacing*mu)]  #154
					psi1+=[0]; psi2+=[10]; psi3+=[1]; psi4+=[4]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 + cI/4)*np.exp((-cI)*q1*charge + cI*P1*charge - spacing*mu)]  #155
					psi1+=[1]; psi2+=[8]; psi3+=[1]; psi4+=[8]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 - cI/4)*np.exp((-cI)*q1*charge + cI*P1*charge - spacing*mu)]  #156
					psi1+=[0]; psi2+=[9]; psi3+=[1]; psi4+=[8]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 - cI/4)*np.exp((-cI)*q1*charge + cI*P1*charge - spacing*mu)]  #157
					psi1+=[3]; psi2+=[8]; psi3+=[1]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 + cI/4)*np.exp((-cI)*q1*charge + cI*P1*charge - spacing*mu)]  #158
					psi1+=[1]; psi2+=[10]; psi3+=[1]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 + cI/4)*np.exp((-cI)*q1*charge + cI*P1*charge - spacing*mu)]  #159
					psi1+=[2]; psi2+=[9]; psi3+=[1]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 - cI/4)*np.exp((-cI)*q1*charge + cI*P1*charge - spacing*mu)]  #160
					psi1+=[0]; psi2+=[11]; psi3+=[1]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 + cI/4)*np.exp((-cI)*q1*charge + cI*P1*charge - spacing*mu)]  #161
					psi1+=[0]; psi2+=[0]; psi3+=[14]; psi4+=[1]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 + cI/4)*np.exp(cI*P1*charge + cI*Q1*charge + spacing*mu)]  #162
					psi1+=[0]; psi2+=[0]; psi3+=[6]; psi4+=[9]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 - cI/4)*np.exp(cI*P1*charge + cI*Q1*charge + spacing*mu)]  #163
					psi1+=[2]; psi2+=[0]; psi3+=[6]; psi4+=[1]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 - cI/4)*np.exp(cI*P1*charge + cI*Q1*charge + spacing*mu)]  #164
					psi1+=[0]; psi2+=[2]; psi3+=[6]; psi4+=[1]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 - cI/4)*np.exp(cI*P1*charge + cI*Q1*charge + spacing*mu)]  #165
					psi1+=[0]; psi2+=[0]; psi3+=[10]; psi4+=[5]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 + cI/4)*np.exp(cI*P1*charge + cI*Q1*charge + spacing*mu)]  #166
					psi1+=[1]; psi2+=[0]; psi3+=[10]; psi4+=[1]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 + cI/4)*np.exp(cI*P1*charge + cI*Q1*charge + spacing*mu)]  #167
					psi1+=[0]; psi2+=[1]; psi3+=[10]; psi4+=[1]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 + cI/4)*np.exp(cI*P1*charge + cI*Q1*charge + spacing*mu)]  #168
					psi1+=[0]; psi2+=[0]; psi3+=[2]; psi4+=[13]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 + cI/4)*np.exp(cI*P1*charge + cI*Q1*charge + spacing*mu)]  #169
					psi1+=[2]; psi2+=[0]; psi3+=[2]; psi4+=[5]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 + cI/4)*np.exp(cI*P1*charge + cI*Q1*charge + spacing*mu)]  #170
					psi1+=[0]; psi2+=[2]; psi3+=[2]; psi4+=[5]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 + cI/4)*np.exp(cI*P1*charge + cI*Q1*charge + spacing*mu)]  #171
					psi1+=[1]; psi2+=[0]; psi3+=[2]; psi4+=[9]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 - cI/4)*np.exp(cI*P1*charge + cI*Q1*charge + spacing*mu)]  #172
					psi1+=[0]; psi2+=[1]; psi3+=[2]; psi4+=[9]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 - cI/4)*np.exp(cI*P1*charge + cI*Q1*charge + spacing*mu)]  #173
					psi1+=[3]; psi2+=[0]; psi3+=[2]; psi4+=[1]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 + cI/4)*np.exp(cI*P1*charge + cI*Q1*charge + spacing*mu)]  #174
					psi1+=[1]; psi2+=[2]; psi3+=[2]; psi4+=[1]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 + cI/4)*np.exp(cI*P1*charge + cI*Q1*charge + spacing*mu)]  #175
					psi1+=[2]; psi2+=[1]; psi3+=[2]; psi4+=[1]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 - cI/4)*np.exp(cI*P1*charge + cI*Q1*charge + spacing*mu)]  #176
					psi1+=[0]; psi2+=[3]; psi3+=[2]; psi4+=[1]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 + cI/4)*np.exp(cI*P1*charge + cI*Q1*charge + spacing*mu)]  #177
					psi1+=[0]; psi2+=[0]; psi3+=[14]; psi4+=[2]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 - cI/4)*np.exp(cI*P1*charge + cI*Q1*charge + spacing*mu)]  #178
					psi1+=[0]; psi2+=[0]; psi3+=[6]; psi4+=[10]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 + cI/4)*np.exp(cI*P1*charge + cI*Q1*charge + spacing*mu)]  #179
					psi1+=[2]; psi2+=[0]; psi3+=[6]; psi4+=[2]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 + cI/4)*np.exp(cI*P1*charge + cI*Q1*charge + spacing*mu)]  #180
					psi1+=[0]; psi2+=[2]; psi3+=[6]; psi4+=[2]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 + cI/4)*np.exp(cI*P1*charge + cI*Q1*charge + spacing*mu)]  #181
					psi1+=[0]; psi2+=[0]; psi3+=[10]; psi4+=[6]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 - cI/4)*np.exp(cI*P1*charge + cI*Q1*charge + spacing*mu)]  #182
					psi1+=[1]; psi2+=[0]; psi3+=[10]; psi4+=[2]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 - cI/4)*np.exp(cI*P1*charge + cI*Q1*charge + spacing*mu)]  #183
					psi1+=[0]; psi2+=[1]; psi3+=[10]; psi4+=[2]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 - cI/4)*np.exp(cI*P1*charge + cI*Q1*charge + spacing*mu)]  #184
					psi1+=[0]; psi2+=[0]; psi3+=[2]; psi4+=[14]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 - cI/4)*np.exp(cI*P1*charge + cI*Q1*charge + spacing*mu)]  #185
					psi1+=[2]; psi2+=[0]; psi3+=[2]; psi4+=[6]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 - cI/4)*np.exp(cI*P1*charge + cI*Q1*charge + spacing*mu)]  #186
					psi1+=[0]; psi2+=[2]; psi3+=[2]; psi4+=[6]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 - cI/4)*np.exp(cI*P1*charge + cI*Q1*charge + spacing*mu)]  #187
					psi1+=[1]; psi2+=[0]; psi3+=[2]; psi4+=[10]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 + cI/4)*np.exp(cI*P1*charge + cI*Q1*charge + spacing*mu)]  #188
					psi1+=[0]; psi2+=[1]; psi3+=[2]; psi4+=[10]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 + cI/4)*np.exp(cI*P1*charge + cI*Q1*charge + spacing*mu)]  #189
					psi1+=[3]; psi2+=[0]; psi3+=[2]; psi4+=[2]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 - cI/4)*np.exp(cI*P1*charge + cI*Q1*charge + spacing*mu)]  #190
					psi1+=[1]; psi2+=[2]; psi3+=[2]; psi4+=[2]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 - cI/4)*np.exp(cI*P1*charge + cI*Q1*charge + spacing*mu)]  #191
					psi1+=[2]; psi2+=[1]; psi3+=[2]; psi4+=[2]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 + cI/4)*np.exp(cI*P1*charge + cI*Q1*charge + spacing*mu)]  #192
					psi1+=[0]; psi2+=[3]; psi3+=[2]; psi4+=[2]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 - cI/4)*np.exp(cI*P1*charge + cI*Q1*charge + spacing*mu)]  #193
					psi1+=[4]; psi2+=[0]; psi3+=[14]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-1/2*np.exp((-cI)*p1*charge + cI*P1*charge)]  #194
					psi1+=[4]; psi2+=[0]; psi3+=[6]; psi4+=[8]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-1/2*np.exp((-cI)*p1*charge + cI*P1*charge)]  #195
					psi1+=[6]; psi2+=[0]; psi3+=[6]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-1/2*np.exp((-cI)*p1*charge + cI*P1*charge)]  #196
					psi1+=[4]; psi2+=[2]; psi3+=[6]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[np.exp((-cI)*p1*charge + cI*P1*charge)/2]  #197
					psi1+=[4]; psi2+=[0]; psi3+=[10]; psi4+=[4]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[np.exp((-cI)*p1*charge + cI*P1*charge)/2]  #198
					psi1+=[5]; psi2+=[0]; psi3+=[10]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[np.exp((-cI)*p1*charge + cI*P1*charge)/2]  #199
					psi1+=[4]; psi2+=[1]; psi3+=[10]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-1/2*np.exp((-cI)*p1*charge + cI*P1*charge)]  #200
					psi1+=[4]; psi2+=[0]; psi3+=[2]; psi4+=[12]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-1/2*np.exp((-cI)*p1*charge + cI*P1*charge)]  #201
					psi1+=[6]; psi2+=[0]; psi3+=[2]; psi4+=[4]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-1/2*np.exp((-cI)*p1*charge + cI*P1*charge)]  #202
					psi1+=[4]; psi2+=[2]; psi3+=[2]; psi4+=[4]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[np.exp((-cI)*p1*charge + cI*P1*charge)/2]  #203
					psi1+=[5]; psi2+=[0]; psi3+=[2]; psi4+=[8]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[np.exp((-cI)*p1*charge + cI*P1*charge)/2]  #204
					psi1+=[4]; psi2+=[1]; psi3+=[2]; psi4+=[8]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-1/2*np.exp((-cI)*p1*charge + cI*P1*charge)]  #205
					psi1+=[7]; psi2+=[0]; psi3+=[2]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-1/2*np.exp((-cI)*p1*charge + cI*P1*charge)]  #206
					psi1+=[5]; psi2+=[2]; psi3+=[2]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[np.exp((-cI)*p1*charge + cI*P1*charge)/2]  #207
					psi1+=[6]; psi2+=[1]; psi3+=[2]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-1/2*np.exp((-cI)*p1*charge + cI*P1*charge)]  #208
					psi1+=[4]; psi2+=[3]; psi3+=[2]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-1/2*np.exp((-cI)*p1*charge + cI*P1*charge)]  #209
					psi1+=[8]; psi2+=[0]; psi3+=[14]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-1/2*np.exp((-cI)*p1*charge + cI*P1*charge)]  #210
					psi1+=[8]; psi2+=[0]; psi3+=[6]; psi4+=[8]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-1/2*np.exp((-cI)*p1*charge + cI*P1*charge)]  #211
					psi1+=[10]; psi2+=[0]; psi3+=[6]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-1/2*np.exp((-cI)*p1*charge + cI*P1*charge)]  #212
					psi1+=[8]; psi2+=[2]; psi3+=[6]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[np.exp((-cI)*p1*charge + cI*P1*charge)/2]  #213
					psi1+=[8]; psi2+=[0]; psi3+=[10]; psi4+=[4]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[np.exp((-cI)*p1*charge + cI*P1*charge)/2]  #214
					psi1+=[9]; psi2+=[0]; psi3+=[10]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[np.exp((-cI)*p1*charge + cI*P1*charge)/2]  #215
					psi1+=[8]; psi2+=[1]; psi3+=[10]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-1/2*np.exp((-cI)*p1*charge + cI*P1*charge)]  #216
					psi1+=[8]; psi2+=[0]; psi3+=[2]; psi4+=[12]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-1/2*np.exp((-cI)*p1*charge + cI*P1*charge)]  #217
					psi1+=[10]; psi2+=[0]; psi3+=[2]; psi4+=[4]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-1/2*np.exp((-cI)*p1*charge + cI*P1*charge)]  #218
					psi1+=[8]; psi2+=[2]; psi3+=[2]; psi4+=[4]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[np.exp((-cI)*p1*charge + cI*P1*charge)/2]  #219
					psi1+=[9]; psi2+=[0]; psi3+=[2]; psi4+=[8]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[np.exp((-cI)*p1*charge + cI*P1*charge)/2]  #220
					psi1+=[8]; psi2+=[1]; psi3+=[2]; psi4+=[8]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-1/2*np.exp((-cI)*p1*charge + cI*P1*charge)]  #221
					psi1+=[11]; psi2+=[0]; psi3+=[2]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-1/2*np.exp((-cI)*p1*charge + cI*P1*charge)]  #222
					psi1+=[9]; psi2+=[2]; psi3+=[2]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[np.exp((-cI)*p1*charge + cI*P1*charge)/2]  #223
					psi1+=[10]; psi2+=[1]; psi3+=[2]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-1/2*np.exp((-cI)*p1*charge + cI*P1*charge)]  #224
					psi1+=[8]; psi2+=[3]; psi3+=[2]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-1/2*np.exp((-cI)*p1*charge + cI*P1*charge)]  #225
					psi1+=[0]; psi2+=[4]; psi3+=[14]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 + cI/4)*np.exp((-cI)*q1*charge + cI*P1*charge - spacing*mu)]  #226
					psi1+=[0]; psi2+=[4]; psi3+=[6]; psi4+=[8]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 + cI/4)*np.exp((-cI)*q1*charge + cI*P1*charge - spacing*mu)]  #227
					psi1+=[2]; psi2+=[4]; psi3+=[6]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 + cI/4)*np.exp((-cI)*q1*charge + cI*P1*charge - spacing*mu)]  #228
					psi1+=[0]; psi2+=[6]; psi3+=[6]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 + cI/4)*np.exp((-cI)*q1*charge + cI*P1*charge - spacing*mu)]  #229
					psi1+=[0]; psi2+=[4]; psi3+=[10]; psi4+=[4]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 - cI/4)*np.exp((-cI)*q1*charge + cI*P1*charge - spacing*mu)]  #230
					psi1+=[1]; psi2+=[4]; psi3+=[10]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 - cI/4)*np.exp((-cI)*q1*charge + cI*P1*charge - spacing*mu)]  #231
					psi1+=[0]; psi2+=[5]; psi3+=[10]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 - cI/4)*np.exp((-cI)*q1*charge + cI*P1*charge - spacing*mu)]  #232
					psi1+=[0]; psi2+=[4]; psi3+=[2]; psi4+=[12]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 + cI/4)*np.exp((-cI)*q1*charge + cI*P1*charge - spacing*mu)]  #233
					psi1+=[2]; psi2+=[4]; psi3+=[2]; psi4+=[4]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 + cI/4)*np.exp((-cI)*q1*charge + cI*P1*charge - spacing*mu)]  #234
					psi1+=[0]; psi2+=[6]; psi3+=[2]; psi4+=[4]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 + cI/4)*np.exp((-cI)*q1*charge + cI*P1*charge - spacing*mu)]  #235
					psi1+=[1]; psi2+=[4]; psi3+=[2]; psi4+=[8]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 - cI/4)*np.exp((-cI)*q1*charge + cI*P1*charge - spacing*mu)]  #236
					psi1+=[0]; psi2+=[5]; psi3+=[2]; psi4+=[8]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 - cI/4)*np.exp((-cI)*q1*charge + cI*P1*charge - spacing*mu)]  #237
					psi1+=[3]; psi2+=[4]; psi3+=[2]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 + cI/4)*np.exp((-cI)*q1*charge + cI*P1*charge - spacing*mu)]  #238
					psi1+=[1]; psi2+=[6]; psi3+=[2]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 + cI/4)*np.exp((-cI)*q1*charge + cI*P1*charge - spacing*mu)]  #239
					psi1+=[2]; psi2+=[5]; psi3+=[2]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 - cI/4)*np.exp((-cI)*q1*charge + cI*P1*charge - spacing*mu)]  #240
					psi1+=[0]; psi2+=[7]; psi3+=[2]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 + cI/4)*np.exp((-cI)*q1*charge + cI*P1*charge - spacing*mu)]  #241
					psi1+=[0]; psi2+=[8]; psi3+=[14]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 - cI/4)*np.exp((-cI)*q1*charge + cI*P1*charge - spacing*mu)]  #242
					psi1+=[0]; psi2+=[8]; psi3+=[6]; psi4+=[8]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 - cI/4)*np.exp((-cI)*q1*charge + cI*P1*charge - spacing*mu)]  #243
					psi1+=[2]; psi2+=[8]; psi3+=[6]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 - cI/4)*np.exp((-cI)*q1*charge + cI*P1*charge - spacing*mu)]  #244
					psi1+=[0]; psi2+=[10]; psi3+=[6]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 - cI/4)*np.exp((-cI)*q1*charge + cI*P1*charge - spacing*mu)]  #245
					psi1+=[0]; psi2+=[8]; psi3+=[10]; psi4+=[4]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 + cI/4)*np.exp((-cI)*q1*charge + cI*P1*charge - spacing*mu)]  #246
					psi1+=[1]; psi2+=[8]; psi3+=[10]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 + cI/4)*np.exp((-cI)*q1*charge + cI*P1*charge - spacing*mu)]  #247
					psi1+=[0]; psi2+=[9]; psi3+=[10]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 + cI/4)*np.exp((-cI)*q1*charge + cI*P1*charge - spacing*mu)]  #248
					psi1+=[0]; psi2+=[8]; psi3+=[2]; psi4+=[12]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 - cI/4)*np.exp((-cI)*q1*charge + cI*P1*charge - spacing*mu)]  #249
					psi1+=[2]; psi2+=[8]; psi3+=[2]; psi4+=[4]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 - cI/4)*np.exp((-cI)*q1*charge + cI*P1*charge - spacing*mu)]  #250
					psi1+=[0]; psi2+=[10]; psi3+=[2]; psi4+=[4]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 - cI/4)*np.exp((-cI)*q1*charge + cI*P1*charge - spacing*mu)]  #251
					psi1+=[1]; psi2+=[8]; psi3+=[2]; psi4+=[8]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 + cI/4)*np.exp((-cI)*q1*charge + cI*P1*charge - spacing*mu)]  #252
					psi1+=[0]; psi2+=[9]; psi3+=[2]; psi4+=[8]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 + cI/4)*np.exp((-cI)*q1*charge + cI*P1*charge - spacing*mu)]  #253
					psi1+=[3]; psi2+=[8]; psi3+=[2]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 - cI/4)*np.exp((-cI)*q1*charge + cI*P1*charge - spacing*mu)]  #254
					psi1+=[1]; psi2+=[10]; psi3+=[2]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 - cI/4)*np.exp((-cI)*q1*charge + cI*P1*charge - spacing*mu)]  #255
					psi1+=[2]; psi2+=[9]; psi3+=[2]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 + cI/4)*np.exp((-cI)*q1*charge + cI*P1*charge - spacing*mu)]  #256
					psi1+=[0]; psi2+=[11]; psi3+=[2]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 - cI/4)*np.exp((-cI)*q1*charge + cI*P1*charge - spacing*mu)]  #257
					psi1+=[4]; psi2+=[0]; psi3+=[12]; psi4+=[1]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 - cI/4)*np.exp((-cI)*p1*charge + cI*Q1*charge + spacing*mu)]  #258
					psi1+=[4]; psi2+=[0]; psi3+=[4]; psi4+=[9]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 + cI/4)*np.exp((-cI)*p1*charge + cI*Q1*charge + spacing*mu)]  #259
					psi1+=[6]; psi2+=[0]; psi3+=[4]; psi4+=[1]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 + cI/4)*np.exp((-cI)*p1*charge + cI*Q1*charge + spacing*mu)]  #260
					psi1+=[4]; psi2+=[2]; psi3+=[4]; psi4+=[1]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 - cI/4)*np.exp((-cI)*p1*charge + cI*Q1*charge + spacing*mu)]  #261
					psi1+=[4]; psi2+=[0]; psi3+=[8]; psi4+=[5]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 - cI/4)*np.exp((-cI)*p1*charge + cI*Q1*charge + spacing*mu)]  #262
					psi1+=[5]; psi2+=[0]; psi3+=[8]; psi4+=[1]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 - cI/4)*np.exp((-cI)*p1*charge + cI*Q1*charge + spacing*mu)]  #263
					psi1+=[4]; psi2+=[1]; psi3+=[8]; psi4+=[1]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 + cI/4)*np.exp((-cI)*p1*charge + cI*Q1*charge + spacing*mu)]  #264
					psi1+=[4]; psi2+=[0]; psi3+=[0]; psi4+=[13]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 - cI/4)*np.exp((-cI)*p1*charge + cI*Q1*charge + spacing*mu)]  #265
					psi1+=[6]; psi2+=[0]; psi3+=[0]; psi4+=[5]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 - cI/4)*np.exp((-cI)*p1*charge + cI*Q1*charge + spacing*mu)]  #266
					psi1+=[4]; psi2+=[2]; psi3+=[0]; psi4+=[5]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 + cI/4)*np.exp((-cI)*p1*charge + cI*Q1*charge + spacing*mu)]  #267
					psi1+=[5]; psi2+=[0]; psi3+=[0]; psi4+=[9]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 + cI/4)*np.exp((-cI)*p1*charge + cI*Q1*charge + spacing*mu)]  #268
					psi1+=[4]; psi2+=[1]; psi3+=[0]; psi4+=[9]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 - cI/4)*np.exp((-cI)*p1*charge + cI*Q1*charge + spacing*mu)]  #269
					psi1+=[7]; psi2+=[0]; psi3+=[0]; psi4+=[1]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 - cI/4)*np.exp((-cI)*p1*charge + cI*Q1*charge + spacing*mu)]  #270
					psi1+=[5]; psi2+=[2]; psi3+=[0]; psi4+=[1]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 + cI/4)*np.exp((-cI)*p1*charge + cI*Q1*charge + spacing*mu)]  #271
					psi1+=[6]; psi2+=[1]; psi3+=[0]; psi4+=[1]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 - cI/4)*np.exp((-cI)*p1*charge + cI*Q1*charge + spacing*mu)]  #272
					psi1+=[4]; psi2+=[3]; psi3+=[0]; psi4+=[1]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 - cI/4)*np.exp((-cI)*p1*charge + cI*Q1*charge + spacing*mu)]  #273
					psi1+=[8]; psi2+=[0]; psi3+=[12]; psi4+=[1]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 - cI/4)*np.exp((-cI)*p1*charge + cI*Q1*charge + spacing*mu)]  #274
					psi1+=[8]; psi2+=[0]; psi3+=[4]; psi4+=[9]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 + cI/4)*np.exp((-cI)*p1*charge + cI*Q1*charge + spacing*mu)]  #275
					psi1+=[10]; psi2+=[0]; psi3+=[4]; psi4+=[1]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 + cI/4)*np.exp((-cI)*p1*charge + cI*Q1*charge + spacing*mu)]  #276
					psi1+=[8]; psi2+=[2]; psi3+=[4]; psi4+=[1]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 - cI/4)*np.exp((-cI)*p1*charge + cI*Q1*charge + spacing*mu)]  #277
					psi1+=[8]; psi2+=[0]; psi3+=[8]; psi4+=[5]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 - cI/4)*np.exp((-cI)*p1*charge + cI*Q1*charge + spacing*mu)]  #278
					psi1+=[9]; psi2+=[0]; psi3+=[8]; psi4+=[1]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 - cI/4)*np.exp((-cI)*p1*charge + cI*Q1*charge + spacing*mu)]  #279
					psi1+=[8]; psi2+=[1]; psi3+=[8]; psi4+=[1]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 + cI/4)*np.exp((-cI)*p1*charge + cI*Q1*charge + spacing*mu)]  #280
					psi1+=[8]; psi2+=[0]; psi3+=[0]; psi4+=[13]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 - cI/4)*np.exp((-cI)*p1*charge + cI*Q1*charge + spacing*mu)]  #281
					psi1+=[10]; psi2+=[0]; psi3+=[0]; psi4+=[5]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 - cI/4)*np.exp((-cI)*p1*charge + cI*Q1*charge + spacing*mu)]  #282
					psi1+=[8]; psi2+=[2]; psi3+=[0]; psi4+=[5]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 + cI/4)*np.exp((-cI)*p1*charge + cI*Q1*charge + spacing*mu)]  #283
					psi1+=[9]; psi2+=[0]; psi3+=[0]; psi4+=[9]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 + cI/4)*np.exp((-cI)*p1*charge + cI*Q1*charge + spacing*mu)]  #284
					psi1+=[8]; psi2+=[1]; psi3+=[0]; psi4+=[9]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 - cI/4)*np.exp((-cI)*p1*charge + cI*Q1*charge + spacing*mu)]  #285
					psi1+=[11]; psi2+=[0]; psi3+=[0]; psi4+=[1]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 - cI/4)*np.exp((-cI)*p1*charge + cI*Q1*charge + spacing*mu)]  #286
					psi1+=[9]; psi2+=[2]; psi3+=[0]; psi4+=[1]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 + cI/4)*np.exp((-cI)*p1*charge + cI*Q1*charge + spacing*mu)]  #287
					psi1+=[10]; psi2+=[1]; psi3+=[0]; psi4+=[1]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 - cI/4)*np.exp((-cI)*p1*charge + cI*Q1*charge + spacing*mu)]  #288
					psi1+=[8]; psi2+=[3]; psi3+=[0]; psi4+=[1]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 - cI/4)*np.exp((-cI)*p1*charge + cI*Q1*charge + spacing*mu)]  #289
					psi1+=[0]; psi2+=[4]; psi3+=[12]; psi4+=[1]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/2*cI)*np.exp((-cI)*q1*charge + cI*Q1*charge)]  #290
					psi1+=[0]; psi2+=[4]; psi3+=[4]; psi4+=[9]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(cI/2)*np.exp((-cI)*q1*charge + cI*Q1*charge)]  #291
					psi1+=[2]; psi2+=[4]; psi3+=[4]; psi4+=[1]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(cI/2)*np.exp((-cI)*q1*charge + cI*Q1*charge)]  #292
					psi1+=[0]; psi2+=[6]; psi3+=[4]; psi4+=[1]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(cI/2)*np.exp((-cI)*q1*charge + cI*Q1*charge)]  #293
					psi1+=[0]; psi2+=[4]; psi3+=[8]; psi4+=[5]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/2*cI)*np.exp((-cI)*q1*charge + cI*Q1*charge)]  #294
					psi1+=[1]; psi2+=[4]; psi3+=[8]; psi4+=[1]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/2*cI)*np.exp((-cI)*q1*charge + cI*Q1*charge)]  #295
					psi1+=[0]; psi2+=[5]; psi3+=[8]; psi4+=[1]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/2*cI)*np.exp((-cI)*q1*charge + cI*Q1*charge)]  #296
					psi1+=[0]; psi2+=[4]; psi3+=[0]; psi4+=[13]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/2*cI)*np.exp((-cI)*q1*charge + cI*Q1*charge)]  #297
					psi1+=[2]; psi2+=[4]; psi3+=[0]; psi4+=[5]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/2*cI)*np.exp((-cI)*q1*charge + cI*Q1*charge)]  #298
					psi1+=[0]; psi2+=[6]; psi3+=[0]; psi4+=[5]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/2*cI)*np.exp((-cI)*q1*charge + cI*Q1*charge)]  #299
					psi1+=[1]; psi2+=[4]; psi3+=[0]; psi4+=[9]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(cI/2)*np.exp((-cI)*q1*charge + cI*Q1*charge)]  #300
					psi1+=[0]; psi2+=[5]; psi3+=[0]; psi4+=[9]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(cI/2)*np.exp((-cI)*q1*charge + cI*Q1*charge)]  #301
					psi1+=[3]; psi2+=[4]; psi3+=[0]; psi4+=[1]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/2*cI)*np.exp((-cI)*q1*charge + cI*Q1*charge)]  #302
					psi1+=[1]; psi2+=[6]; psi3+=[0]; psi4+=[1]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/2*cI)*np.exp((-cI)*q1*charge + cI*Q1*charge)]  #303
					psi1+=[2]; psi2+=[5]; psi3+=[0]; psi4+=[1]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(cI/2)*np.exp((-cI)*q1*charge + cI*Q1*charge)]  #304
					psi1+=[0]; psi2+=[7]; psi3+=[0]; psi4+=[1]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/2*cI)*np.exp((-cI)*q1*charge + cI*Q1*charge)]  #305
					psi1+=[0]; psi2+=[8]; psi3+=[12]; psi4+=[1]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[np.exp((-cI)*q1*charge + cI*Q1*charge)/2]  #306
					psi1+=[0]; psi2+=[8]; psi3+=[4]; psi4+=[9]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-1/2*np.exp((-cI)*q1*charge + cI*Q1*charge)]  #307
					psi1+=[2]; psi2+=[8]; psi3+=[4]; psi4+=[1]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-1/2*np.exp((-cI)*q1*charge + cI*Q1*charge)]  #308
					psi1+=[0]; psi2+=[10]; psi3+=[4]; psi4+=[1]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-1/2*np.exp((-cI)*q1*charge + cI*Q1*charge)]  #309
					psi1+=[0]; psi2+=[8]; psi3+=[8]; psi4+=[5]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[np.exp((-cI)*q1*charge + cI*Q1*charge)/2]  #310
					psi1+=[1]; psi2+=[8]; psi3+=[8]; psi4+=[1]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[np.exp((-cI)*q1*charge + cI*Q1*charge)/2]  #311
					psi1+=[0]; psi2+=[9]; psi3+=[8]; psi4+=[1]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[np.exp((-cI)*q1*charge + cI*Q1*charge)/2]  #312
					psi1+=[0]; psi2+=[8]; psi3+=[0]; psi4+=[13]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[np.exp((-cI)*q1*charge + cI*Q1*charge)/2]  #313
					psi1+=[2]; psi2+=[8]; psi3+=[0]; psi4+=[5]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[np.exp((-cI)*q1*charge + cI*Q1*charge)/2]  #314
					psi1+=[0]; psi2+=[10]; psi3+=[0]; psi4+=[5]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[np.exp((-cI)*q1*charge + cI*Q1*charge)/2]  #315
					psi1+=[1]; psi2+=[8]; psi3+=[0]; psi4+=[9]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-1/2*np.exp((-cI)*q1*charge + cI*Q1*charge)]  #316
					psi1+=[0]; psi2+=[9]; psi3+=[0]; psi4+=[9]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-1/2*np.exp((-cI)*q1*charge + cI*Q1*charge)]  #317
					psi1+=[3]; psi2+=[8]; psi3+=[0]; psi4+=[1]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[np.exp((-cI)*q1*charge + cI*Q1*charge)/2]  #318
					psi1+=[1]; psi2+=[10]; psi3+=[0]; psi4+=[1]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[np.exp((-cI)*q1*charge + cI*Q1*charge)/2]  #319
					psi1+=[2]; psi2+=[9]; psi3+=[0]; psi4+=[1]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-1/2*np.exp((-cI)*q1*charge + cI*Q1*charge)]  #320
					psi1+=[0]; psi2+=[11]; psi3+=[0]; psi4+=[1]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[np.exp((-cI)*q1*charge + cI*Q1*charge)/2]  #321
					psi1+=[4]; psi2+=[0]; psi3+=[12]; psi4+=[2]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 - cI/4)*np.exp((-cI)*p1*charge + cI*Q1*charge + spacing*mu)]  #322
					psi1+=[4]; psi2+=[0]; psi3+=[4]; psi4+=[10]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 + cI/4)*np.exp((-cI)*p1*charge + cI*Q1*charge + spacing*mu)]  #323
					psi1+=[6]; psi2+=[0]; psi3+=[4]; psi4+=[2]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 + cI/4)*np.exp((-cI)*p1*charge + cI*Q1*charge + spacing*mu)]  #324
					psi1+=[4]; psi2+=[2]; psi3+=[4]; psi4+=[2]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 - cI/4)*np.exp((-cI)*p1*charge + cI*Q1*charge + spacing*mu)]  #325
					psi1+=[4]; psi2+=[0]; psi3+=[8]; psi4+=[6]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 - cI/4)*np.exp((-cI)*p1*charge + cI*Q1*charge + spacing*mu)]  #326
					psi1+=[5]; psi2+=[0]; psi3+=[8]; psi4+=[2]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 - cI/4)*np.exp((-cI)*p1*charge + cI*Q1*charge + spacing*mu)]  #327
					psi1+=[4]; psi2+=[1]; psi3+=[8]; psi4+=[2]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 + cI/4)*np.exp((-cI)*p1*charge + cI*Q1*charge + spacing*mu)]  #328
					psi1+=[4]; psi2+=[0]; psi3+=[0]; psi4+=[14]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 - cI/4)*np.exp((-cI)*p1*charge + cI*Q1*charge + spacing*mu)]  #329
					psi1+=[6]; psi2+=[0]; psi3+=[0]; psi4+=[6]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 - cI/4)*np.exp((-cI)*p1*charge + cI*Q1*charge + spacing*mu)]  #330
					psi1+=[4]; psi2+=[2]; psi3+=[0]; psi4+=[6]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 + cI/4)*np.exp((-cI)*p1*charge + cI*Q1*charge + spacing*mu)]  #331
					psi1+=[5]; psi2+=[0]; psi3+=[0]; psi4+=[10]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 + cI/4)*np.exp((-cI)*p1*charge + cI*Q1*charge + spacing*mu)]  #332
					psi1+=[4]; psi2+=[1]; psi3+=[0]; psi4+=[10]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 - cI/4)*np.exp((-cI)*p1*charge + cI*Q1*charge + spacing*mu)]  #333
					psi1+=[7]; psi2+=[0]; psi3+=[0]; psi4+=[2]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 - cI/4)*np.exp((-cI)*p1*charge + cI*Q1*charge + spacing*mu)]  #334
					psi1+=[5]; psi2+=[2]; psi3+=[0]; psi4+=[2]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 + cI/4)*np.exp((-cI)*p1*charge + cI*Q1*charge + spacing*mu)]  #335
					psi1+=[6]; psi2+=[1]; psi3+=[0]; psi4+=[2]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 - cI/4)*np.exp((-cI)*p1*charge + cI*Q1*charge + spacing*mu)]  #336
					psi1+=[4]; psi2+=[3]; psi3+=[0]; psi4+=[2]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 - cI/4)*np.exp((-cI)*p1*charge + cI*Q1*charge + spacing*mu)]  #337
					psi1+=[8]; psi2+=[0]; psi3+=[12]; psi4+=[2]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 - cI/4)*np.exp((-cI)*p1*charge + cI*Q1*charge + spacing*mu)]  #338
					psi1+=[8]; psi2+=[0]; psi3+=[4]; psi4+=[10]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 + cI/4)*np.exp((-cI)*p1*charge + cI*Q1*charge + spacing*mu)]  #339
					psi1+=[10]; psi2+=[0]; psi3+=[4]; psi4+=[2]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 + cI/4)*np.exp((-cI)*p1*charge + cI*Q1*charge + spacing*mu)]  #340
					psi1+=[8]; psi2+=[2]; psi3+=[4]; psi4+=[2]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 - cI/4)*np.exp((-cI)*p1*charge + cI*Q1*charge + spacing*mu)]  #341
					psi1+=[8]; psi2+=[0]; psi3+=[8]; psi4+=[6]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 - cI/4)*np.exp((-cI)*p1*charge + cI*Q1*charge + spacing*mu)]  #342
					psi1+=[9]; psi2+=[0]; psi3+=[8]; psi4+=[2]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 - cI/4)*np.exp((-cI)*p1*charge + cI*Q1*charge + spacing*mu)]  #343
					psi1+=[8]; psi2+=[1]; psi3+=[8]; psi4+=[2]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 + cI/4)*np.exp((-cI)*p1*charge + cI*Q1*charge + spacing*mu)]  #344
					psi1+=[8]; psi2+=[0]; psi3+=[0]; psi4+=[14]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 - cI/4)*np.exp((-cI)*p1*charge + cI*Q1*charge + spacing*mu)]  #345
					psi1+=[10]; psi2+=[0]; psi3+=[0]; psi4+=[6]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 - cI/4)*np.exp((-cI)*p1*charge + cI*Q1*charge + spacing*mu)]  #346
					psi1+=[8]; psi2+=[2]; psi3+=[0]; psi4+=[6]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 + cI/4)*np.exp((-cI)*p1*charge + cI*Q1*charge + spacing*mu)]  #347
					psi1+=[9]; psi2+=[0]; psi3+=[0]; psi4+=[10]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 + cI/4)*np.exp((-cI)*p1*charge + cI*Q1*charge + spacing*mu)]  #348
					psi1+=[8]; psi2+=[1]; psi3+=[0]; psi4+=[10]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 - cI/4)*np.exp((-cI)*p1*charge + cI*Q1*charge + spacing*mu)]  #349
					psi1+=[11]; psi2+=[0]; psi3+=[0]; psi4+=[2]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 - cI/4)*np.exp((-cI)*p1*charge + cI*Q1*charge + spacing*mu)]  #350
					psi1+=[9]; psi2+=[2]; psi3+=[0]; psi4+=[2]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 + cI/4)*np.exp((-cI)*p1*charge + cI*Q1*charge + spacing*mu)]  #351
					psi1+=[10]; psi2+=[1]; psi3+=[0]; psi4+=[2]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 - cI/4)*np.exp((-cI)*p1*charge + cI*Q1*charge + spacing*mu)]  #352
					psi1+=[8]; psi2+=[3]; psi3+=[0]; psi4+=[2]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 - cI/4)*np.exp((-cI)*p1*charge + cI*Q1*charge + spacing*mu)]  #353
					psi1+=[0]; psi2+=[4]; psi3+=[12]; psi4+=[2]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-1/2*np.exp((-cI)*q1*charge + cI*Q1*charge)]  #354
					psi1+=[0]; psi2+=[4]; psi3+=[4]; psi4+=[10]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[np.exp((-cI)*q1*charge + cI*Q1*charge)/2]  #355
					psi1+=[2]; psi2+=[4]; psi3+=[4]; psi4+=[2]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[np.exp((-cI)*q1*charge + cI*Q1*charge)/2]  #356
					psi1+=[0]; psi2+=[6]; psi3+=[4]; psi4+=[2]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[np.exp((-cI)*q1*charge + cI*Q1*charge)/2]  #357
					psi1+=[0]; psi2+=[4]; psi3+=[8]; psi4+=[6]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-1/2*np.exp((-cI)*q1*charge + cI*Q1*charge)]  #358
					psi1+=[1]; psi2+=[4]; psi3+=[8]; psi4+=[2]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-1/2*np.exp((-cI)*q1*charge + cI*Q1*charge)]  #359
					psi1+=[0]; psi2+=[5]; psi3+=[8]; psi4+=[2]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-1/2*np.exp((-cI)*q1*charge + cI*Q1*charge)]  #360
					psi1+=[0]; psi2+=[4]; psi3+=[0]; psi4+=[14]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-1/2*np.exp((-cI)*q1*charge + cI*Q1*charge)]  #361
					psi1+=[2]; psi2+=[4]; psi3+=[0]; psi4+=[6]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-1/2*np.exp((-cI)*q1*charge + cI*Q1*charge)]  #362
					psi1+=[0]; psi2+=[6]; psi3+=[0]; psi4+=[6]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-1/2*np.exp((-cI)*q1*charge + cI*Q1*charge)]  #363
					psi1+=[1]; psi2+=[4]; psi3+=[0]; psi4+=[10]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[np.exp((-cI)*q1*charge + cI*Q1*charge)/2]  #364
					psi1+=[0]; psi2+=[5]; psi3+=[0]; psi4+=[10]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[np.exp((-cI)*q1*charge + cI*Q1*charge)/2]  #365
					psi1+=[3]; psi2+=[4]; psi3+=[0]; psi4+=[2]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-1/2*np.exp((-cI)*q1*charge + cI*Q1*charge)]  #366
					psi1+=[1]; psi2+=[6]; psi3+=[0]; psi4+=[2]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-1/2*np.exp((-cI)*q1*charge + cI*Q1*charge)]  #367
					psi1+=[2]; psi2+=[5]; psi3+=[0]; psi4+=[2]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[np.exp((-cI)*q1*charge + cI*Q1*charge)/2]  #368
					psi1+=[0]; psi2+=[7]; psi3+=[0]; psi4+=[2]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[-1/2*np.exp((-cI)*q1*charge + cI*Q1*charge)]  #369
					psi1+=[0]; psi2+=[8]; psi3+=[12]; psi4+=[2]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/2*cI)*np.exp((-cI)*q1*charge + cI*Q1*charge)]  #370
					psi1+=[0]; psi2+=[8]; psi3+=[4]; psi4+=[10]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(cI/2)*np.exp((-cI)*q1*charge + cI*Q1*charge)]  #371
					psi1+=[2]; psi2+=[8]; psi3+=[4]; psi4+=[2]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(cI/2)*np.exp((-cI)*q1*charge + cI*Q1*charge)]  #372
					psi1+=[0]; psi2+=[10]; psi3+=[4]; psi4+=[2]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(cI/2)*np.exp((-cI)*q1*charge + cI*Q1*charge)]  #373
					psi1+=[0]; psi2+=[8]; psi3+=[8]; psi4+=[6]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/2*cI)*np.exp((-cI)*q1*charge + cI*Q1*charge)]  #374
					psi1+=[1]; psi2+=[8]; psi3+=[8]; psi4+=[2]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/2*cI)*np.exp((-cI)*q1*charge + cI*Q1*charge)]  #375
					psi1+=[0]; psi2+=[9]; psi3+=[8]; psi4+=[2]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/2*cI)*np.exp((-cI)*q1*charge + cI*Q1*charge)]  #376
					psi1+=[0]; psi2+=[8]; psi3+=[0]; psi4+=[14]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/2*cI)*np.exp((-cI)*q1*charge + cI*Q1*charge)]  #377
					psi1+=[2]; psi2+=[8]; psi3+=[0]; psi4+=[6]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/2*cI)*np.exp((-cI)*q1*charge + cI*Q1*charge)]  #378
					psi1+=[0]; psi2+=[10]; psi3+=[0]; psi4+=[6]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/2*cI)*np.exp((-cI)*q1*charge + cI*Q1*charge)]  #379
					psi1+=[1]; psi2+=[8]; psi3+=[0]; psi4+=[10]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(cI/2)*np.exp((-cI)*q1*charge + cI*Q1*charge)]  #380
					psi1+=[0]; psi2+=[9]; psi3+=[0]; psi4+=[10]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(cI/2)*np.exp((-cI)*q1*charge + cI*Q1*charge)]  #381
					psi1+=[3]; psi2+=[8]; psi3+=[0]; psi4+=[2]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/2*cI)*np.exp((-cI)*q1*charge + cI*Q1*charge)]  #382
					psi1+=[1]; psi2+=[10]; psi3+=[0]; psi4+=[2]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/2*cI)*np.exp((-cI)*q1*charge + cI*Q1*charge)]  #383
					psi1+=[2]; psi2+=[9]; psi3+=[0]; psi4+=[2]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(cI/2)*np.exp((-cI)*q1*charge + cI*Q1*charge)]  #384
					psi1+=[0]; psi2+=[11]; psi3+=[0]; psi4+=[2]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/2*cI)*np.exp((-cI)*q1*charge + cI*Q1*charge)]  #385
					psi1+=[4]; psi2+=[4]; psi3+=[12]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 + cI/4)*np.exp((-cI)*p1*charge - cI*q1*charge - spacing*mu)]  #386
					psi1+=[4]; psi2+=[4]; psi3+=[4]; psi4+=[8]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 + cI/4)*np.exp((-cI)*p1*charge - cI*q1*charge - spacing*mu)]  #387
					psi1+=[6]; psi2+=[4]; psi3+=[4]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 + cI/4)*np.exp((-cI)*p1*charge - cI*q1*charge - spacing*mu)]  #388
					psi1+=[4]; psi2+=[6]; psi3+=[4]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 - cI/4)*np.exp((-cI)*p1*charge - cI*q1*charge - spacing*mu)]  #389
					psi1+=[4]; psi2+=[4]; psi3+=[8]; psi4+=[4]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 - cI/4)*np.exp((-cI)*p1*charge - cI*q1*charge - spacing*mu)]  #390
					psi1+=[5]; psi2+=[4]; psi3+=[8]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 - cI/4)*np.exp((-cI)*p1*charge - cI*q1*charge - spacing*mu)]  #391
					psi1+=[4]; psi2+=[5]; psi3+=[8]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 + cI/4)*np.exp((-cI)*p1*charge - cI*q1*charge - spacing*mu)]  #392
					psi1+=[4]; psi2+=[4]; psi3+=[0]; psi4+=[12]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 + cI/4)*np.exp((-cI)*p1*charge - cI*q1*charge - spacing*mu)]  #393
					psi1+=[6]; psi2+=[4]; psi3+=[0]; psi4+=[4]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 + cI/4)*np.exp((-cI)*p1*charge - cI*q1*charge - spacing*mu)]  #394
					psi1+=[4]; psi2+=[6]; psi3+=[0]; psi4+=[4]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 - cI/4)*np.exp((-cI)*p1*charge - cI*q1*charge - spacing*mu)]  #395
					psi1+=[5]; psi2+=[4]; psi3+=[0]; psi4+=[8]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 - cI/4)*np.exp((-cI)*p1*charge - cI*q1*charge - spacing*mu)]  #396
					psi1+=[4]; psi2+=[5]; psi3+=[0]; psi4+=[8]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 + cI/4)*np.exp((-cI)*p1*charge - cI*q1*charge - spacing*mu)]  #397
					psi1+=[7]; psi2+=[4]; psi3+=[0]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 + cI/4)*np.exp((-cI)*p1*charge - cI*q1*charge - spacing*mu)]  #398
					psi1+=[5]; psi2+=[6]; psi3+=[0]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 - cI/4)*np.exp((-cI)*p1*charge - cI*q1*charge - spacing*mu)]  #399
					psi1+=[6]; psi2+=[5]; psi3+=[0]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 + cI/4)*np.exp((-cI)*p1*charge - cI*q1*charge - spacing*mu)]  #400
					psi1+=[4]; psi2+=[7]; psi3+=[0]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 + cI/4)*np.exp((-cI)*p1*charge - cI*q1*charge - spacing*mu)]  #401
					psi1+=[4]; psi2+=[8]; psi3+=[12]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 + cI/4)*np.exp((-cI)*p1*charge - cI*q1*charge - spacing*mu)]  #402
					psi1+=[4]; psi2+=[8]; psi3+=[4]; psi4+=[8]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 + cI/4)*np.exp((-cI)*p1*charge - cI*q1*charge - spacing*mu)]  #403
					psi1+=[6]; psi2+=[8]; psi3+=[4]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 + cI/4)*np.exp((-cI)*p1*charge - cI*q1*charge - spacing*mu)]  #404
					psi1+=[4]; psi2+=[10]; psi3+=[4]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 - cI/4)*np.exp((-cI)*p1*charge - cI*q1*charge - spacing*mu)]  #405
					psi1+=[4]; psi2+=[8]; psi3+=[8]; psi4+=[4]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 - cI/4)*np.exp((-cI)*p1*charge - cI*q1*charge - spacing*mu)]  #406
					psi1+=[5]; psi2+=[8]; psi3+=[8]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 - cI/4)*np.exp((-cI)*p1*charge - cI*q1*charge - spacing*mu)]  #407
					psi1+=[4]; psi2+=[9]; psi3+=[8]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 + cI/4)*np.exp((-cI)*p1*charge - cI*q1*charge - spacing*mu)]  #408
					psi1+=[4]; psi2+=[8]; psi3+=[0]; psi4+=[12]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 + cI/4)*np.exp((-cI)*p1*charge - cI*q1*charge - spacing*mu)]  #409
					psi1+=[6]; psi2+=[8]; psi3+=[0]; psi4+=[4]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 + cI/4)*np.exp((-cI)*p1*charge - cI*q1*charge - spacing*mu)]  #410
					psi1+=[4]; psi2+=[10]; psi3+=[0]; psi4+=[4]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 - cI/4)*np.exp((-cI)*p1*charge - cI*q1*charge - spacing*mu)]  #411
					psi1+=[5]; psi2+=[8]; psi3+=[0]; psi4+=[8]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 - cI/4)*np.exp((-cI)*p1*charge - cI*q1*charge - spacing*mu)]  #412
					psi1+=[4]; psi2+=[9]; psi3+=[0]; psi4+=[8]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 + cI/4)*np.exp((-cI)*p1*charge - cI*q1*charge - spacing*mu)]  #413
					psi1+=[7]; psi2+=[8]; psi3+=[0]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 + cI/4)*np.exp((-cI)*p1*charge - cI*q1*charge - spacing*mu)]  #414
					psi1+=[5]; psi2+=[10]; psi3+=[0]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 - cI/4)*np.exp((-cI)*p1*charge - cI*q1*charge - spacing*mu)]  #415
					psi1+=[6]; psi2+=[9]; psi3+=[0]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 + cI/4)*np.exp((-cI)*p1*charge - cI*q1*charge - spacing*mu)]  #416
					psi1+=[4]; psi2+=[11]; psi3+=[0]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 + cI/4)*np.exp((-cI)*p1*charge - cI*q1*charge - spacing*mu)]  #417
					psi1+=[8]; psi2+=[4]; psi3+=[12]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 + cI/4)*np.exp((-cI)*p1*charge - cI*q1*charge - spacing*mu)]  #418
					psi1+=[8]; psi2+=[4]; psi3+=[4]; psi4+=[8]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 + cI/4)*np.exp((-cI)*p1*charge - cI*q1*charge - spacing*mu)]  #419
					psi1+=[10]; psi2+=[4]; psi3+=[4]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 + cI/4)*np.exp((-cI)*p1*charge - cI*q1*charge - spacing*mu)]  #420
					psi1+=[8]; psi2+=[6]; psi3+=[4]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 - cI/4)*np.exp((-cI)*p1*charge - cI*q1*charge - spacing*mu)]  #421
					psi1+=[8]; psi2+=[4]; psi3+=[8]; psi4+=[4]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 - cI/4)*np.exp((-cI)*p1*charge - cI*q1*charge - spacing*mu)]  #422
					psi1+=[9]; psi2+=[4]; psi3+=[8]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 - cI/4)*np.exp((-cI)*p1*charge - cI*q1*charge - spacing*mu)]  #423
					psi1+=[8]; psi2+=[5]; psi3+=[8]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 + cI/4)*np.exp((-cI)*p1*charge - cI*q1*charge - spacing*mu)]  #424
					psi1+=[8]; psi2+=[4]; psi3+=[0]; psi4+=[12]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 + cI/4)*np.exp((-cI)*p1*charge - cI*q1*charge - spacing*mu)]  #425
					psi1+=[10]; psi2+=[4]; psi3+=[0]; psi4+=[4]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 + cI/4)*np.exp((-cI)*p1*charge - cI*q1*charge - spacing*mu)]  #426
					psi1+=[8]; psi2+=[6]; psi3+=[0]; psi4+=[4]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 - cI/4)*np.exp((-cI)*p1*charge - cI*q1*charge - spacing*mu)]  #427
					psi1+=[9]; psi2+=[4]; psi3+=[0]; psi4+=[8]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 - cI/4)*np.exp((-cI)*p1*charge - cI*q1*charge - spacing*mu)]  #428
					psi1+=[8]; psi2+=[5]; psi3+=[0]; psi4+=[8]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 + cI/4)*np.exp((-cI)*p1*charge - cI*q1*charge - spacing*mu)]  #429
					psi1+=[11]; psi2+=[4]; psi3+=[0]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 + cI/4)*np.exp((-cI)*p1*charge - cI*q1*charge - spacing*mu)]  #430
					psi1+=[9]; psi2+=[6]; psi3+=[0]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 - cI/4)*np.exp((-cI)*p1*charge - cI*q1*charge - spacing*mu)]  #431
					psi1+=[10]; psi2+=[5]; psi3+=[0]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 + cI/4)*np.exp((-cI)*p1*charge - cI*q1*charge - spacing*mu)]  #432
					psi1+=[8]; psi2+=[7]; psi3+=[0]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 + cI/4)*np.exp((-cI)*p1*charge - cI*q1*charge - spacing*mu)]  #433
					psi1+=[8]; psi2+=[8]; psi3+=[12]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 + cI/4)*np.exp((-cI)*p1*charge - cI*q1*charge - spacing*mu)]  #434
					psi1+=[8]; psi2+=[8]; psi3+=[4]; psi4+=[8]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 + cI/4)*np.exp((-cI)*p1*charge - cI*q1*charge - spacing*mu)]  #435
					psi1+=[10]; psi2+=[8]; psi3+=[4]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 + cI/4)*np.exp((-cI)*p1*charge - cI*q1*charge - spacing*mu)]  #436
					psi1+=[8]; psi2+=[10]; psi3+=[4]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 - cI/4)*np.exp((-cI)*p1*charge - cI*q1*charge - spacing*mu)]  #437
					psi1+=[8]; psi2+=[8]; psi3+=[8]; psi4+=[4]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 - cI/4)*np.exp((-cI)*p1*charge - cI*q1*charge - spacing*mu)]  #438
					psi1+=[9]; psi2+=[8]; psi3+=[8]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 - cI/4)*np.exp((-cI)*p1*charge - cI*q1*charge - spacing*mu)]  #439
					psi1+=[8]; psi2+=[9]; psi3+=[8]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 + cI/4)*np.exp((-cI)*p1*charge - cI*q1*charge - spacing*mu)]  #440
					psi1+=[8]; psi2+=[8]; psi3+=[0]; psi4+=[12]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 + cI/4)*np.exp((-cI)*p1*charge - cI*q1*charge - spacing*mu)]  #441
					psi1+=[10]; psi2+=[8]; psi3+=[0]; psi4+=[4]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 + cI/4)*np.exp((-cI)*p1*charge - cI*q1*charge - spacing*mu)]  #442
					psi1+=[8]; psi2+=[10]; psi3+=[0]; psi4+=[4]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 - cI/4)*np.exp((-cI)*p1*charge - cI*q1*charge - spacing*mu)]  #443
					psi1+=[9]; psi2+=[8]; psi3+=[0]; psi4+=[8]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 - cI/4)*np.exp((-cI)*p1*charge - cI*q1*charge - spacing*mu)]  #444
					psi1+=[8]; psi2+=[9]; psi3+=[0]; psi4+=[8]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 + cI/4)*np.exp((-cI)*p1*charge - cI*q1*charge - spacing*mu)]  #445
					psi1+=[11]; psi2+=[8]; psi3+=[0]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 + cI/4)*np.exp((-cI)*p1*charge - cI*q1*charge - spacing*mu)]  #446
					psi1+=[9]; psi2+=[10]; psi3+=[0]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(1/4 - cI/4)*np.exp((-cI)*p1*charge - cI*q1*charge - spacing*mu)]  #447
					psi1+=[10]; psi2+=[9]; psi3+=[0]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 + cI/4)*np.exp((-cI)*p1*charge - cI*q1*charge - spacing*mu)]  #448
					psi1+=[8]; psi2+=[11]; psi3+=[0]; psi4+=[0]; phi1+=[i]; phi2+=[j]; phi3+=[k]; phi4+=[l]
					B+=[(-1/4 + cI/4)*np.exp((-cI)*p1*charge - cI*q1*charge - spacing*mu)]  #449
	A = gtn.sparse(A, statistic=(0,0,0,0))
	B = sp.COO([psi1,psi2,psi3,psi4,phi1,phi2,phi3,phi4], B, shape=(Npsi,Npsi,Npsi,Npsi,Nphi,Nphi,Nphi,Nphi))
	B = gtn.sparse(B, statistic=(1,1,-1,-1,0,0,0,0))

	return A, B

def fcompress_B(B,cutoff=64,mute=True):

	process_name = "B compression (1)"
	process_length = 16
	s00 = time.time()
	print()

	if not mute:
		B.info("B (uncompressed)")

	# μ = 1 ===========================================================================

	step = 1
	step = gtn.show_progress(step,process_length,process_name,time=time.time()-s00)
	Qp = gtn.einsum('IJKLijkl -> JKLijkl I',B)
	step = gtn.show_progress(step,process_length,process_name,time=time.time()-s00)
	cQp = Qp.hconjugate("abcdefg x")
	step = gtn.show_progress(step,process_length,process_name,time=time.time()-s00)
	Mp = gtn.einsum('I abcdefg,abcdefg J -> IJ',cQp,Qp)
	
	del Qp, cQp
	gc.collect()

	step = gtn.show_progress(step,process_length,process_name,time=time.time()-s00)
	Qm = gtn.einsum('IJKLijkl -> K IJLijkl',B)
	step = gtn.show_progress(step,process_length,process_name,time=time.time()-s00)
	cQm = Qm.hconjugate("x abcdefg")
	step = gtn.show_progress(step,process_length,process_name,time=time.time()-s00)
	Mm = gtn.einsum('I abcdefg,abcdefg J -> IJ',Qm,cQm)

	del Qm, cQm
	gc.collect()

	step = gtn.show_progress(step,process_length,process_name,time=time.time()-s00)
	Λp, Up = Mp.eig("I J",cutoff)
	Λm, Um = Mm.eig("I J",cutoff)

	if Λp.shape[0] < Λm.shape[0] :
		U1 = Up.copy()
	else:
		U1 = Um.copy()

	step = gtn.show_progress(step,process_length,process_name,time=time.time()-s00)
	B = gtn.einsum('IA,IJKLijkl->AJKLijkl',U1,B)
	B = gtn.einsum('CK,AJKLijkl->AJCLijkl',U1.hconjugate('I J'),B)

	if not mute:
		gtn.clear_progress()
		U1.info("U1")

	# μ = 2 ===========================================================================
	
	step = gtn.show_progress(step,process_length,process_name,time=time.time()-s00)
	Qp = gtn.einsum('IJKLijkl -> IKLijkl J',B)
	step = gtn.show_progress(step,process_length,process_name,time=time.time()-s00)
	cQp = Qp.hconjugate("abcdefg x")
	step = gtn.show_progress(step,process_length,process_name,time=time.time()-s00)
	Mp = gtn.einsum('I abcdefg,abcdefg J -> IJ',cQp,Qp)
	
	del Qp, cQp
	gc.collect()

	step = gtn.show_progress(step,process_length,process_name,time=time.time()-s00)
	Qm = gtn.einsum('IJKLijkl -> L IJKijkl',B)
	step = gtn.show_progress(step,process_length,process_name,time=time.time()-s00)
	cQm = Qm.hconjugate("x abcdefg")
	step = gtn.show_progress(step,process_length,process_name,time=time.time()-s00)
	Mm = gtn.einsum('I abcdefg,abcdefg J -> IJ',Qm,cQm)
	del Qm, cQm
	gc.collect()

	step = gtn.show_progress(step,process_length,process_name,time=time.time()-s00)
	Λp, Up = Mp.eig("I J",cutoff)
	Λm, Um = Mm.eig("I J",cutoff)

	if Λp.shape[0] < Λm.shape[0] :
		U2 = Up.copy()
	else:
		U2 = Um.copy()

	step = gtn.show_progress(step,process_length,process_name,time=time.time()-s00)
	B = gtn.einsum('JB,AJCLijkl->ABCLijkl',U2,B)
	B = gtn.einsum('DL,ABCLijkl->ABCDijkl',U2.hconjugate('I J'),B)
	gtn.clear_progress()
	sys.stdout.write("\033[F")
	
	if not mute:
		U2.info("U2")

	return B

def compress_B(B,cutoff=64,mute=True):

	process_name = "B compression (2)"
	process_length = 4*8
	s00 = time.time()
	print()

	Npsi = B.shape[0]
	Nphi = B.shape[4]
	δ = np.zeros([Nphi,Nphi],dtype=int)
	for i in range(Nphi):
		δ[i,i] = 1
	δ = gtn.sparse(δ,statistic=(0,0))

	if not mute:
		B.info("B (uncompressed)")

	# μ = 1 ===========================================================================

	step = 1
	step = gtn.show_progress(step,process_length,process_name,time=time.time()-s00)
	Qp = gtn.einsum('IJKLijkl,km -> JKLjklm Ii',B,δ)
	step = gtn.show_progress(step,process_length,process_name,time=time.time()-s00)
	cQp = Qp.hconjugate("JKLjklm Ii")
	step = gtn.show_progress(step,process_length,process_name,time=time.time()-s00)
	Mp = gtn.einsum('Ii abcdefg,abcdefg Jj -> IiJj',cQp,Qp)
	
	del Qp, cQp
	gc.collect()

	step = gtn.show_progress(step,process_length,process_name,time=time.time()-s00)
	Qm = gtn.einsum('IJKLijkl,km -> Kk IJLijlm',B,δ)
	step = gtn.show_progress(step,process_length,process_name,time=time.time()-s00)
	cQm = Qm.hconjugate("Kk IJLijlm")
	step = gtn.show_progress(step,process_length,process_name,time=time.time()-s00)
	Mm = gtn.einsum('Ii abcdefg,abcdefg Jj -> IiJj',Qm,cQm)

	del Qm, cQm
	gc.collect()

	step = gtn.show_progress(step,process_length,process_name,time=time.time()-s00)
	Λp, Up = Mp.eig("Ii Jj",cutoff)
	Λm, Um = Mm.eig("Ii Jj",cutoff)

	if Λp.shape[0] < Λm.shape[0] :
		U1 = Up.copy()
	else:
		U1 = Um.copy()

	step = gtn.show_progress(step,process_length,process_name,time=time.time()-s00)
	Bfin = gtn.einsum('IJKLijkl,IiA->AJKLjkl',B,U1)

	if not mute:
		gtn.clear_progress()
		U1.info("U1")


	# μ = 2 ===========================================================================
	
	step = gtn.show_progress(step,process_length,process_name,time=time.time()-s00)
	Qp = gtn.einsum('IJKLijkl,lm -> IKLiklm Jj',B,δ)
	step = gtn.show_progress(step,process_length,process_name,time=time.time()-s00)
	cQp = Qp.hconjugate("JKLjklm Ii")
	step = gtn.show_progress(step,process_length,process_name,time=time.time()-s00)
	Mp = gtn.einsum('Ii abcdefg,abcdefg Jj -> IiJj',cQp,Qp)
	
	del Qp, cQp
	gc.collect()

	step = gtn.show_progress(step,process_length,process_name,time=time.time()-s00)
	Qm = gtn.einsum('IJKLijkl,lm -> Ll IJKijkm',B,δ)
	step = gtn.show_progress(step,process_length,process_name,time=time.time()-s00)
	cQm = Qm.hconjugate("Kk IJLijlm")
	step = gtn.show_progress(step,process_length,process_name,time=time.time()-s00)
	Mm = gtn.einsum('Ii abcdefg,abcdefg Jj -> IiJj',Qm,cQm)

	del Qm, cQm
	gc.collect()

	step = gtn.show_progress(step,process_length,process_name,time=time.time()-s00)
	Λp, Up = Mp.eig("Ii Jj",cutoff)
	Λm, Um = Mm.eig("Ii Jj",cutoff)

	if Λp.shape[0] < Λm.shape[0] :
		U2 = Up.copy()
	else:
		U2 = Um.copy()

	step = gtn.show_progress(step,process_length,process_name,time=time.time()-s00)
	Bfin = gtn.einsum('AJKLjkl,JjB->ABKLkl',Bfin,U2)
	
	if not mute:
		gtn.clear_progress()
		U2.info("U2")

	# μ = 3 ===========================================================================

	step = gtn.show_progress(step,process_length,process_name,time=time.time()-s00)
	Qp = gtn.einsum('IJKLijkl,im -> JKLjklm Ii',B,δ)
	step = gtn.show_progress(step,process_length,process_name,time=time.time()-s00)
	cQp = Qp.hconjugate("JKLjklm Ii")
	step = gtn.show_progress(step,process_length,process_name,time=time.time()-s00)
	Mp = gtn.einsum('Ii abcdefg,abcdefg Jj -> IiJj',cQp,Qp)
	
	del Qp, cQp
	gc.collect()

	step = gtn.show_progress(step,process_length,process_name,time=time.time()-s00)
	Qm = gtn.einsum('IJKLijkl,im -> Kk IJLijlm',B,δ)
	step = gtn.show_progress(step,process_length,process_name,time=time.time()-s00)
	cQm = Qm.hconjugate("Kk IJLijlm")
	step = gtn.show_progress(step,process_length,process_name,time=time.time()-s00)
	Mm = gtn.einsum('Ii abcdefg,abcdefg Jj -> IiJj',Qm,cQm)

	del Qm, cQm
	gc.collect()

	step = gtn.show_progress(step,process_length,process_name,time=time.time()-s00)
	Λp, Up = Mp.eig("Ii Jj",cutoff)
	Λm, Um = Mm.eig("Ii Jj",cutoff)

	if Λp.shape[0] < Λm.shape[0] :
		U3 = Up.copy()
	else:
		U3 = Um.copy()

	step = gtn.show_progress(step,process_length,process_name,time=time.time()-s00)
	Bfin = gtn.einsum('ABKLkl,CKk->ABCLl',Bfin,U3.hconjugate('ij k'))
	
	if not mute:
		gtn.clear_progress()
		U3.info("U3")


	# μ = 4 ===========================================================================

	step = gtn.show_progress(step,process_length,process_name,time=time.time()-s00)
	Qp = gtn.einsum('IJKLijkl,jm -> IKLiklm Jj',B,δ)
	step = gtn.show_progress(step,process_length,process_name,time=time.time()-s00)
	cQp = Qp.hconjugate("JKLjklm Ii")
	step = gtn.show_progress(step,process_length,process_name,time=time.time()-s00)
	Mp = gtn.einsum('Ii abcdefg,abcdefg Jj -> IiJj',cQp,Qp)
	
	del Qp, cQp
	gc.collect()

	step = gtn.show_progress(step,process_length,process_name,time=time.time()-s00)
	Qm = gtn.einsum('IJKLijkl,jm -> Ll IJKijkm',B,δ)
	step = gtn.show_progress(step,process_length,process_name,time=time.time()-s00)
	cQm = Qm.hconjugate("Kk IJLijlm")
	step = gtn.show_progress(step,process_length,process_name,time=time.time()-s00)
	Mm = gtn.einsum('Ii abcdefg,abcdefg Jj -> IiJj',Qm,cQm)

	del Qm, cQm
	gc.collect()

	step = gtn.show_progress(step,process_length,process_name,time=time.time()-s00)
	Λp, Up = Mp.eig("Ii Jj",cutoff)
	Λm, Um = Mm.eig("Ii Jj",cutoff)

	if Λp.shape[0] < Λm.shape[0] :
		U4 = Up.copy()
	else:
		U4 = Um.copy()

	step = gtn.show_progress(step,process_length,process_name,time=time.time()-s00)
	Bfin = gtn.einsum('ABCLl,DLl->ABCD',Bfin,U4.hconjugate('ij k'))
	gtn.clear_progress()
	sys.stdout.write("\033[F")
	
	if not mute:
		U1.info("U4")
		Bfin.info("B (compressed)")

	return Bfin, [U1,U2,U3,U4]

def compress_A(A,Upack,mute=True):
	
	[U1,U2,U3,U4] = Upack

	process_name = "A compression"
	process_length = 3
	s00 = time.time()
	print()

	Nphi = A.shape[0]
	δ = np.zeros([Nphi,Nphi],dtype=int)
	for i in range(Nphi):
		δ[i,i] = 1
	δ = gtn.sparse(δ,statistic=(0,0))

	if not mute:
		A.info("A (uncompressed)")

	step = 1
	step = gtn.show_progress(step,process_length,process_name,time=time.time()-s00)
	Ix = gtn.einsum('KXj,XjI->KIj',U1.hconjugate('ij k'),U3)
	step = gtn.show_progress(step,process_length,process_name,time=time.time()-s00)
	Iy = gtn.einsum('LYj,YjJ->LJj',U2.hconjugate('ij k'),U4)

	step = gtn.show_progress(step,process_length,process_name,time=time.time()-s00)
	Afin = gtn.einsum('ijkl,KIl,LJk,km,ln->IJKLijklmn',A,Ix,Iy,δ,δ)
	gtn.clear_progress()
	sys.stdout.write("\033[F")

	if not mute:
		Afin.info("A (compressed)")

	return Afin

def compress_T(T,cutoff=64,mute=True):

	process_name = "T compression"
	process_length = 16
	s00 = time.time()
	print()

	# x direction =====================================================================

	step = 1
	step = gtn.show_progress(step,process_length,process_name,time=time.time()-s00)
	Txp = gtn.einsum('IJKLijklmn -> JKLjklmn Ii',T)
	step = gtn.show_progress(step,process_length,process_name,time=time.time()-s00)
	Txm = gtn.einsum('IJKLijklmn -> Kk IJLijlmn',T)

	step = gtn.show_progress(step,process_length,process_name,time=time.time()-s00)
	cTxp = Txp.hconjugate('abcdefgh xy')
	step = gtn.show_progress(step,process_length,process_name,time=time.time()-s00)
	cTxm = Txm.hconjugate('xy abcdefgh')

	step = gtn.show_progress(step,process_length,process_name,time=time.time()-s00)
	Mp = gtn.einsum('Ii abcdefgh, abcdefgh Jj -> IiJj',cTxp,Txp)
	step = gtn.show_progress(step,process_length,process_name,time=time.time()-s00)
	Mm = gtn.einsum('Ii abcdefgh, abcdefgh Jj -> IiJj',Txm,cTxm)

	del Txp, Txm, cTxp, cTxm
	gc.collect()


	step = gtn.show_progress(step,process_length,process_name,time=time.time()-s00)
	Λp, Up = Mp.eig("Ii Jj",cutoff)
	Λm, Um = Mm.eig("Ii Jj",cutoff)

	if Λp.shape[0] < Λm.shape[0] :
		U = Up.copy()
	else:
		U = Um.copy()

	step = gtn.show_progress(step,process_length,process_name,time=time.time()-s00)

	Tfin = gtn.einsum('IJKLijklmn,IiA,CKk->ACJLjlmn',T,U,U.hconjugate('ij k'))

	if not mute:
		Tfin.info("T (x-compression)")

	# y direction =====================================================================

	step = gtn.show_progress(step,process_length,process_name,time=time.time()-s00)
	Typ = gtn.einsum('IKJLjlmn -> IKLlmn Jj',Tfin)
	step = gtn.show_progress(step,process_length,process_name,time=time.time()-s00)
	Tym = gtn.einsum('IKJLjlmn -> Ll IKJjmn',Tfin)

	step = gtn.show_progress(step,process_length,process_name,time=time.time()-s00)
	cTyp = Typ.hconjugate('abcdef xy')
	step = gtn.show_progress(step,process_length,process_name,time=time.time()-s00)
	cTym = Tym.hconjugate('xy abcdef')

	step = gtn.show_progress(step,process_length,process_name,time=time.time()-s00)
	Mp = gtn.einsum('Ii abcdef, abcdef Jj -> IiJj',cTyp,Typ)
	step = gtn.show_progress(step,process_length,process_name,time=time.time()-s00)
	Mm = gtn.einsum('Ii abcdef, abcdef Jj -> IiJj',Tym,cTym)

	del Typ, Tym, cTyp, cTym
	gc.collect()

	step = gtn.show_progress(step,process_length,process_name,time=time.time()-s00)
	Λp, Up = Mp.eig("Ii Jj",cutoff)
	Λm, Um = Mm.eig("Ii Jj",cutoff)

	if Λp.shape[0] < Λm.shape[0] :
		U = Up.copy()
	else:
		U = Um.copy()

	step = gtn.show_progress(step,process_length,process_name,time=time.time()-s00)
	Tfin = gtn.einsum('ACJLjlmn,JjB,DLl->ABCDmn',Tfin,U,U.hconjugate('ij k'))
	gtn.clear_progress()
	sys.stdout.write("\033[F")

	if not mute:
		Tfin.info("T (y-compression)")

	return Tfin

####################################################
##               2D Coarse graining               ##
####################################################

def zcap(T):
	Nphi = T.shape[4]
	if type(T)==dense :
		capper = dense(np.full((Nphi,Nphi),1),statistic=(0,0))
	else :
		capper = sparse(np.full((Nphi,Nphi),1),statistic=(0,0))
	return gtn.einsum("IJKLij,ij->IJKL",T,capper)

def logZ(T,boundary_conditions='periodic'):
	
	if type(T) == sparse:
		T = dense(T)
		if T.encoder == 'parity-preserving':
			T = T.switch_encoder()

	if boundary_conditions=='anti-periodic' :
		Tdat = T.data
		d = Tdat.shape[1]
		sgn = [ (-1)**param.gparity(i) for i in range(d) ]
		Tdat = np.einsum('IJKL,J->IJKL',Tdat,sgn)
		T.data = Tdat

	Z = gtn.einsum('IJIJ',T)

	return np.log(Z)

def logZhotrg3dz(T1,T2,boundary_conditions='periodic'):
	
	if type(T1) == sparse:
		T1 = dense(T1)
		if T1.encoder == 'parity-preserving':
			T1 = T1.switch_encoder()

	if type(T2) == sparse:
		T2 = dense(T2)
		if T2.encoder == 'parity-preserving':
			T2 = T2.switch_encoder()

	if boundary_conditions=='anti-periodic' :
		T1dat = T1.data
		T2dat = T2.data
		d = T1dat.shape[1]
		sgn = [ (-1)**param.gparity(i) for i in range(d) ]
		T1dat = np.einsum('IJKLmn,J->IJKLmn',T1dat,sgn)
		T2dat = np.einsum('IJKLmn,J->IJKLmn',T2dat,sgn)
		T1.data = T1dat
		T2.data = T2dat

	Z = gtn.einsum('IJIJmn,KLKLmn',T1,T2)

	return np.log(Z)

main()
