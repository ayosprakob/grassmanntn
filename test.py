import os
# Clearing the Screen
os.system('clear')
import time
import numpy as np
import math
import random
import sys
cI = complex(0,1)

from grassmanntn import grassmanntn as gtn
from grassmanntn.grassmanntn import dense
from grassmanntn.grassmanntn import sparse
from grassmanntn import param

def check_einsum_simple():

	m = 4
	n = 2

	# dense

	shape = (m,n)
	A = dense(np.random.rand(*shape)+cI*np.random.rand(*shape), statistic = (1,1))
	A = gtn.trim_grassmann_odd(A)
	if(type(A)==sparse):
		A = A.remove_zeros()
	A.switch_encoder().display("A = original")

	C = gtn.einsum('mn->nm',A)
	C.switch_encoder().display("C = dense")

	# sparse

	sA = sparse(A)
	sC = gtn.einsum('mn->nm',sA)
	sC.switch_encoder().display("sC = sparse")

	print()
	print("|C-sC| =",(C-sC).norm)
	print()

	# manual

	mA = A.data
	sgn = np.zeros([m,n],dtype=int)
	for mi in range(m):
		for ni in range(n):
			sgn[mi,ni] = (-1)**(param.gparity[mi]*param.gparity[ni])
	mC = dense(
		np.einsum('mn,mn->nm',mA,sgn)
		,statistic=(1,1))
	mC.switch_encoder().display("mC = manual")

	print()
	print("|C-mC| =",(C-mC).norm)
	print()

def check_einsum():

	m = 4
	n = 4
	p = 4
	q = n

	# dense

	shape = (m,n,p)
	A = dense(np.random.rand(*shape)+cI*np.random.rand(*shape), statistic = (1,1,-1))
	A = gtn.trim_grassmann_odd(A)
	if(type(A)==sparse):
		A = A.remove_zeros()


	shape = (n,p,m)
	B = dense(np.random.rand(*shape)+cI*np.random.rand(*shape), statistic=(1,1,-1))
	B = gtn.trim_grassmann_odd(B)
	if(type(B)==sparse):
		B = B.remove_zeros()

	C = gtn.einsum('mnp,qpm->qn',A,B)

	C.display("C = dense*dense")

	# sparse

	sA = sparse(A)
	sB = sparse(B)

	sC = gtn.einsum('mnp,qpm->qn',sA,sB)

	sC.display("sC = sparse*sparse")

	print()
	print("|C-sC| =",(C-sC).norm)
	print()

	# manual
	q = n
	sgn = np.zeros([m,q,p,n],dtype=int)
	for mi in range(m):
		for qi in range(q): 
			for pi in range(p):
				for ni in range(n):
					sgn[mi,qi,pi,ni] = param.sgn[mi]*param.sgn[pi]*(
						(-1)**(
							+ param.gparity[pi]*(param.gparity[pi]+param.gparity[qi])
							+ param.gparity[qi]*param.gparity[ni]
							)
						)

	mA = A.data
	mB = B.data
	mC = np.einsum('mnp,qpm,mqpn->qn',mA,mB,sgn)

	mC = dense(mC,statistic=(1,1))

	mC.display("mC = manual contraction")

	print()
	print("|C-mC| =",(C-mC).norm)
	print()

def check_join_and_split():

	m = 4
	n = 4
	p = 4
	q = n

	# dense

	shape = (m,n,p)
	A = dense(np.random.rand(*shape)+cI*np.random.rand(*shape), statistic = (1,1,-1))
	A = gtn.trim_grassmann_odd(A)
	if(type(A)==sparse):
		A = A.remove_zeros()


	shape = (p,m,n)
	B = dense(np.random.rand(*shape)+cI*np.random.rand(*shape), statistic=(1,-1,-1))
	B = gtn.trim_grassmann_odd(B)
	if(type(B)==sparse):
		B = B.remove_zeros()

	C = gtn.einsum('mnp,pxy->mnxy',A,B)
	D = gtn.einsum('mnp,qmn->pq',A,B)

	C.info("C = dense*dense")
	D.info("D = dense*dense")

	jA = A.join_legs('mn p',intermediate_stat=(1,-1),make_format='matrix')
	jB = B.join_legs('p mn',intermediate_stat=(1,-1),make_format='matrix')
	C3 = gtn.einsum('ij,jk->ik',jA,jB)
	D3 = gtn.einsum('ip,qi->pq',jA,jB)
	
	C3.info("C3")
	D3.info("D3")
	print()
	print("|D-D3| =",(D-D3).norm)
	print()

	sA = jA.split_legs('mn p',intermediate_stat=(1,-1),final_stat=A.statistic,final_shape=A.shape)
	sB = jB.split_legs('p mn',intermediate_stat=(1,-1),final_stat=B.statistic,final_shape=B.shape)
	C4 = gtn.einsum('mnp,pxy->mnxy',sA,sB)

	C4.info("C4")
	print()
	print("|C-C4| =",(C-C4).norm)
	print()

def check_hybrid_join():

	m = 8
	n = 5

	A = np.reshape(np.arange(m*n),(m,n))
	A = dense(A, statistic = (1,0))
	A = gtn.trim_grassmann_odd(A)
	if(type(A)==sparse):
		A = A.remove_zeros()

	A.switch_encoder().display()

	B = A.join_legs('(ij)',intermediate_stat=(1))

	B.switch_encoder().display()

def check_conjugate():

	m = 2
	n = 2
	p = 4
	q = n

	# dense

	shape = (m,n,p)
	A = dense(np.random.rand(*shape)+cI*np.random.rand(*shape), statistic = (1,1,-1))
	A = gtn.trim_grassmann_odd(A)
	if(type(A)==sparse):
		A = A.remove_zeros()

	cA = A.hconjugate('mnp')
	jA = A.join_legs('(mnp)')

	A.display("A")
	jA.display("jA")

def check_svd():

	# dense

	def f():
		return (-1)**int(round(random.uniform(0, 1),0))
	def g():
		return int(round(random.uniform(0, 2),0)-1)

	s1 = 0
	s2 = 0
	s3 = 0
	s4 = 0
	s5 = 0
	while [s1,s2,s3]==[0,0,0] or [s4,s5]==[0,0] :
		s1 = g()
		s2 = g()
		s3 = g()
		s4 = g()
		s5 = g()

	shape     = (4,4,4,4,4)
	statistic = (s1,s2,s3,s4,s5)
	print()
	print("  generated statistic = ",statistic)
	print()

	A = dense(np.random.rand(*shape)+cI*np.random.rand(*shape), statistic = statistic)
	A = gtn.trim_grassmann_odd(A)
	if(type(A)==sparse):
		A = A.remove_zeros()
	
	U, S, V = A.svd('ijx kl')

	USV = gtn.einsum('ijxa,ab,bkl->ijxkl',U,S,V)
	
	err = (A-USV).norm
	print('  |A-U•S•V| =',	err)
	if err>1.0e-8 :
		print("\n!\n!\n!\n!\n!\n!\n!\n!\n!\n!\n!\n!\n!\n!\n!")

	cU = U.hconjugate('ijx a')
	cV = V.hconjugate('a ij')

	cU = gtn.einsum('aijx->xjai',cU)

	gtn.einsum('xjai,ijxb->ab',cU,U).switch_format().display("U†•U")

	gtn.einsum('aij,ijb->ab',V,cV).switch_format().display("V•V†")

	#S.display("Λ")

def check_join():
    # dense
    shape = ( 4, 3, 2, 2, 4,2)
    stats = ( 1, 0, 1,-1,0,1)
    A = dense(np.random.rand(*shape), statistic = stats)
    A = gtn.trim_grassmann_odd(A)
    if(type(A)==sparse):
        A = A.remove_zeros()
    #A.display()
    
    A.info("original")
    A0 = A.copy()
    A = A.join_legs('ijk b de',intermediate_stat=(1,-1,1))
    A.info("joined")

    A = A.split_legs('ijk b de',stats,shape,intermediate_stat=(1,-1,1))
    A.info("split")
    
    (A-A0).info("|original-split|")
    
def check_eig():

	# dense

	def f():
		return (-1)**int(round(random.uniform(0, 1),0))
	def g():
		return int(round(random.uniform(0, 2),0)-1)

	s1 = 0
	s2 = 0
	s3 = 0
	s4 = 0
	s5 = 0
	while [s1,s2]==[0,0] or [s4,s5]==[0,0] :
		s1 = g()
		s2 = g()
		s3 = g()
		s4 = g()
		s5 = g()

	shape     = (4,8,4,8)
	statistic = (s1,s2,-s1,-s2)
	print()
	print("  generated statistic = ",statistic)
	print()

	A = dense(np.random.rand(*shape)+cI*np.random.rand(*shape), statistic = statistic)
	A = gtn.trim_grassmann_odd(A)
	if(type(A)==sparse):
		A = A.remove_zeros()
	cA = A.hconjugate('ij kl')
	A = gtn.einsum('ijxy,xykl->ijkl',A,cA)
	S, U = A.eig('ij kl')
	cU = U.hconjugate('ij a')

	US = gtn.einsum('ija,ab->ijb',U,S)
	UScU = gtn.einsum('ija,akl->ijkl',US,cU)
	
	err = (A-UScU).norm
	print('  |A-U•S•U†| =',	err)
	if err>1.0e-8 :
		print("\n!\n!\n!\n!\n!\n!\n!\n!\n!\n!\n!\n!\n!\n!\n!")

	gtn.einsum('aij,ijb->ab',cU,U).switch_format().display("U†•U")

	S.display("Λ")

def bosonic_svd():

	shape = (2,2,2,2)
	stat = (-1,-1,1,1)

	A = dense(np.random.rand(*shape)+cI*np.random.rand(*shape), statistic = stat)
	A = gtn.trim_grassmann_odd(A)
	if(type(A)==sparse):
		A = A.remove_zeros()


	U,S,V = A.svd("ij,kl")

	U.info()
	S.info()
	V.info()

	USV = gtn.einsum('ija,ab,bkl->ijkl',U,S,V)
	
	err = (A-USV).norm
	print()
	print()
	print('  |A-U•S•V| =',	err)
	if err>1.0e-8 :
		A.display("A")
		USV.display("USV")

	cU = U.hconjugate('ij a')
	cV = V.hconjugate('a ij')

	cU = gtn.einsum('aij->jai',cU)

	gtn.einsum('jai,ijb->ab',cU,U).switch_format().display("U†•U")

	gtn.einsum('aij,ijb->ab',V,cV).switch_format().display("V•V†")

# ---------------------------- main ----------------------------
bosonic_svd()
