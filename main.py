import os
# Clearing the Screen
os.system('clear')
import time
import numpy as np
import math
import random
cI = complex(0,1)

from grassmannTN import grassmannTN as gTN
from grassmannTN.grassmannTN import dense
from grassmannTN.grassmannTN import sparse
from grassmannTN import param

def check_einsum_simple():

	m = 4
	n = 2

	# dense

	shape = (m,n)
	A = dense(np.random.rand(*shape)+cI*np.random.rand(*shape), statistic = (1,1))
	A = gTN.trim_grassmann_odd(A)
	if(type(A)==sparse):
		A = A.remove_zeros()
	A.switch_encoder().display("A = original")

	C = gTN.einsum('mn->nm',A)
	C.switch_encoder().display("C = dense")

	# sparse

	sA = sparse(A)
	sC = gTN.einsum('mn->nm',sA)
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
	A = gTN.trim_grassmann_odd(A)
	if(type(A)==sparse):
		A = A.remove_zeros()


	shape = (n,p,m)
	B = dense(np.random.rand(*shape)+cI*np.random.rand(*shape), statistic=(1,1,-1))
	B = gTN.trim_grassmann_odd(B)
	if(type(B)==sparse):
		B = B.remove_zeros()

	C = gTN.einsum('mnp,qpm->qn',A,B)

	C.display("C = dense*dense")

	# sparse

	sA = sparse(A)
	sB = sparse(B)

	sC = gTN.einsum('mnp,qpm->qn',sA,sB)

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
	A = gTN.trim_grassmann_odd(A)
	if(type(A)==sparse):
		A = A.remove_zeros()


	shape = (p,m,n)
	B = dense(np.random.rand(*shape)+cI*np.random.rand(*shape), statistic=(1,-1,-1))
	B = gTN.trim_grassmann_odd(B)
	if(type(B)==sparse):
		B = B.remove_zeros()

	C = gTN.einsum('mnp,pxy->mnxy',A,B)
	D = gTN.einsum('mnp,qmn->pq',A,B)

	C.info("C = dense*dense")
	D.info("D = dense*dense")

	jA = A.join_legs('mn p',statistic=(1,-1),format='matrix')
	jB = B.join_legs('p mn',statistic=(1,-1),format='matrix')
	C3 = gTN.einsum('ij,jk->ik',jA,jB)
	D3 = gTN.einsum('ip,qi->pq',jA,jB)
	
	C3.info("C3")
	D3.info("D3")
	print()
	print("|D-D3| =",(D-D3).norm)
	print()

	sA = jA.split_legs('mn p',A.statistic,shape=A.shape)
	sB = jB.split_legs('p mn',B.statistic,shape=B.shape)
	C4 = gTN.einsum('mnp,pxy->mnxy',sA,sB)

	C4.info("C4")
	print()
	print("|C-C4| =",(C-C4).norm)
	print()

def check_hybrid_join():

	m = 8
	n = 5

	A = np.reshape(np.arange(m*n),(m,n))
	A = dense(A, statistic = (1,0))
	A = gTN.trim_grassmann_odd(A)
	if(type(A)==sparse):
		A = A.remove_zeros()

	A.switch_encoder().display()

	B = A.join_legs('(ij)')

	B.switch_encoder().display()

def check_conjugate():

	m = 2
	n = 2
	p = 4
	q = n

	# dense

	shape = (m,n,p)
	A = dense(np.random.rand(*shape)+cI*np.random.rand(*shape), statistic = (1,1,-1))
	A = gTN.trim_grassmann_odd(A)
	if(type(A)==sparse):
		A = A.remove_zeros()

	cA = A.hconjugate('mnp')
	jA = A.join_legs('(mnp)')

	A.display("A")
	jA.display("jA")

def check_svd():

	m = 4
	n = 2
	p = 2
	q = 4

	# dense

	shape = (m,n,p,q)

	def f():
		return (-1)**int(round(random.uniform(0, 1),0))
	def g():
		return int(round(random.uniform(0, 2),0)-1)

	s1 = 0
	s2 = 0
	s3 = 0
	s4 = 0
	while [s1,s2]==[0,0] or [s3,s4]==[0,0] :
		s1 = f()
		s2 = f()
		s3 = f()
		s4 = f()

	statistic = (s1,s2,s3,s4)
	print(statistic)

	A = dense(np.random.rand(*shape)+cI*np.random.rand(*shape), statistic = statistic)
	A = gTN.trim_grassmann_odd(A)
	if(type(A)==sparse):
		A = A.remove_zeros()
	
	U, S, V = gTN.SVD(A,'ij kl')

	US = gTN.einsum('ija,ab->ijb',U,S)
	USV = gTN.einsum('ija,akl->ijkl',US,V)
	
	err = (A-USV).norm
	print('|A-USV| =',	err)
	if err>1.0e-8 :
		print("\n!\n!\n!\n!\n!\n!\n!\n!\n!\n!\n!\n!\n!\n!\n!")

	#U.info("U")
	#S.info("S")
	#V.info("V")

	cU = U.hconjugate('ij a')
	cV = V.hconjugate('a ij')

	#gTN.einsum('iaj,jkb->iakb',U,cU).switch_format().display("UcU")
	gTN.einsum('aij,ijb->ab',cU,U).switch_format().display("cUU")

	gTN.einsum('aij,ijb->ab',V,cV).switch_format().display("VcV")
	#gTN.einsum('iaj,jkb->iakb',cV,V).switch_format().display("cVV")

# ---------------------------- main ----------------------------
check_svd()