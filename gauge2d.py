import numpy as np
import math
import grassmanntn as gtn
from grassmanntn import param
import sparse as sp
import opt_einsum as oe
import time
import sys
import gc
import tracemalloc
import os
from datetime import datetime

cI = complex(0,1)

####################################################
##           Initial tensor compression           ##
####################################################

def tensor_preparation(Nphi, beta, Nf, spacing, mass, charge, mu, mute=True, Gauss=False):

    if not mute:
        print()
        print(" :·:·:·:·:·:·:·:·:·:·:·:·:·:·: Tensor preparation :·:·:·:·:·:·:·:·:·:·:·:·:·:·: ")
        print()

    A, B = get_ABtensors(Nphi=Nphi, beta=beta, Nf=Nf, spacing=spacing, mass=mass, charge=charge, mu=mu, Gauss=Gauss)
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

    t0 = time.time()
    T = compress_T(T)
    if not mute:
        print("                           T compression: "+gtn.time_display(time.time()-t0))
    z4 = gtn.einsum("IJIJij,ij",T,gtn.sparse(np.full((Nphi,Nphi),1),statistics=(0,0)))

    trace_error = np.abs(1-z4/z1)
    if not mute:
        print("                       Compression error:",trace_error)
        T.info("Compressed tensor T",21)

    T = gtn.dense(T)

    return T, trace_error

def myQuadrature(beta, Nf, npoints, Gauss=False):
    
    if not Gauss :
        xi = np.array( [ i*2*np.pi/npoints for i in range(npoints) ] )
        wi = np.array( [ 2*np.pi/npoints for i in range(npoints) ] )
    else:
        if npoints==1:
            xi = np.array( [ 0.0 ] )
            wi = np.array( [ 1.0 ] )
        elif npoints==2:
            xi = np.array( [ -1.81379936423422,1.81379936423422 ] )
            wi = np.array( [ 3.14159265358979,3.14159265358979 ] )
        elif npoints==3:
            xi = np.array( [ -2.43346720558417,0,2.43346720558417 ] )
            wi = np.array( [ 1.74532925199433,2.79252680319093,1.74532925199433 ] )
        elif npoints==4:
            xi = np.array( [ -2.70533951024329,-1.06808194888598,1.06808194888598,2.70533951024329 ] )
            wi = np.array( [ 1.09281822599944,2.04877442759035,2.04877442759035,1.09281822599944 ] )
        elif npoints==5:
            xi = np.array( [ -2.84684794683204,-1.69165122881158,0,1.69165122881158,2.84684794683204 ] )
            wi = np.array( [ 0.744327761530440,1.50365631503826,1.78721715404219,1.50365631503826,0.744327761530440 ] )
        elif npoints==6:
            xi = np.array( [ -2.92943937551707,-2.07725055100703,-0.749644282004547,0.749644282004547,2.07725055100703,2.92943937551707 ] )
            wi = np.array( [ 0.538231766638403,1.13336590758553,1.46999497936586,1.46999497936586,1.13336590758553,0.538231766638403 ] )
        elif npoints==7:
            xi = np.array( [ -2.98171044487996,-2.32958892508679,-1.27500014606227,0,1.27500014606227,2.32958892508679,2.98171044487995 ] )
            wi = np.array( [ 0.406789018466448,0.878720403072167,1.19955448158670,1.31305750092896,1.19955448158670,0.878720403072167,0.406789018466448 ] )
        elif npoints==8:
            xi = np.array( [ -3.01683955848946,-2.50280155280391,-1.65100875821648,-0.576276925278204,0.576276925278204,1.65100875821648,2.50280155280391,3.01683955848946 ] )
            wi = np.array( [ 0.318018825943497,0.698630624136419,0.985538494072264,1.13940470943761,1.13940470943761,0.985538494072264,0.698630624136419,0.318018825943497 ] )
        elif npoints==9:
            xi = np.array( [ -3.04156509593489,-2.62646918494990,-1.92696318689402,-1.01867217286675,0,1.01867217286675,1.92696318689402,2.62646918494990,3.04156509593489 ] )
            wi = np.array( [ 0.255331021401727,0.567522934523472,0.818732649266382,0.981267282599117,1.03747753159819,0.981267282599117,0.818732649266382,0.567522934523472,0.255331021401727 ] )
        elif npoints==10:
            xi = np.array( [ -3.05961759527269,-2.71767671767977,-2.13442810854683,-1.36155178629610,-0.467702529652729,0.467702529652729,1.36155178629610,2.13442810854683,2.71767671767977,3.05961759527268 ] )
            wi = np.array( [ 0.209454205485129,0.469515260560545,0.688280106981920,0.845926347240510,0.928416733321688,0.928416733321688,0.845926347240510,0.688280106981920,0.469515260560545,0.209454205485129 ] )
        elif npoints==11:
            xi = np.array( [ -3.07319596596265,-2.78678934670571,-2.29384017671529,-1.63078858602302,-0.846794798565295,0,0.846794798565295,1.63078858602302,2.29384017671529,2.78678934670571,3.07319596596265 ] )
            wi = np.array( [ 0.174887961488048,0.394522366146033,0.585247958086261,0.732599817705145,0.825624826363402,0.857419447601808,0.825624826363402,0.732599817705145,0.585247958086261,0.394522366146033,0.174887961488048 ] )
        elif npoints==12:
            xi = np.array( [ -3.08366367760243,-2.84036813059724,-2.41872058522796,-1.84511377050822,-1.15557673501160,-0.393432356163640,0.393432356163640,1.15557673501160,1.84511377050822,2.41872058522796,2.84036813059724,3.08366367760243 ] )
            wi = np.array( [ 0.148205690222500,0.335959800926734,0.502900900950709,0.638269295241925,0.733538437456940,0.782718528790984,0.782718528790984,0.733538437456940,0.638269295241925,0.502900900950709,0.335959800926734,0.148205690222500 ] )
        elif npoints==13:
            xi = np.array( [ -3.09190225449148,-2.88272038994466,-2.51823184112630,-2.01799996582403,-1.40898153184438,-0.724006152363327,0,0.724006152363327,1.40898153184438,2.01799996582403,2.51823184112630,2.88272038994466,3.09190225449148 ] )
            wi = np.array( [ 0.127184251958611,0.289408227127881,0.436283999484711,0.559662104428278,0.652873368239956,0.710889576744852,0.730582251211008,0.710889576744852,0.652873368239956,0.559662104428278,0.436283999484711,0.289408227127881,0.127184251958611 ] )
        elif npoints==14:
            xi = np.array( [ -3.09850196775647,-2.91676420985398,-2.59872957446299,-2.15919434062078,-1.61870133075494,-1.00252107389349,-0.339464633043013,0.339464633043013,1.00252107389349,1.61870133075494,2.15919434062078,2.59872957446299,2.91676420985398,3.09850196775647 ] )
            wi = np.array( [ 0.110331038576265,0.251824057746911,0.381761848947849,0.493868315065231,0.582886066475514,0.644649986154735,0.676271340623288,0.676271340623288,0.644649986154735,0.582886066475514,0.493868315065231,0.381761848947849,0.251824057746911,0.110331038576265 ] )
        elif npoints==15:
            xi = np.array( [ -3.10387003641484,-2.94453120397124,-2.66471957116870,-2.27582542297129,-1.79376198287119,-1.23826297638139,-0.632069887647995,0,0.632069887647995,1.23826297638139,1.79376198287119,2.27582542297129,2.66471957116870,2.94453120397124,3.10387003641484 ] )
            wi = np.array( [ 0.0966141591290766,0.221061457850790,0.336650619784075,0.438474216429353,0.522350115512877,0.584842030033819,0.623390896544564,0.636418316610479,0.623390896544564,0.584842030033819,0.522350115512877,0.438474216429353,0.336650619784075,0.221061457850790,0.0966141591290766 ] )
        elif npoints==16:
            xi = np.array( [ -3.10829470882464,-2.96746995325128,-2.71946062613971,-2.37317293977742,-1.94111547024300,-1.43890214390881,-0.884683646352920,-0.298490602905050,0.298490602905050,0.884683646352920,1.43890214390882,1.94111547024300,2.37317293977742,2.71946062613971,2.96746995325128,3.10829470882464 ] )
            wi = np.array( [ 0.0853019670148604,0.195575213465732,0.298949281228256,0.391533460520839,0.469969659472659,0.531420878638161,0.573665547225541,0.595176646023746,0.595176646023746,0.573665547225541,0.531420878638161,0.469969659472659,0.391533460520839,0.298949281228256,0.195575213465732,0.0853019670148604 ] )
        elif npoints==17:
            xi = np.array( [ -3.11198463607399,-2.98663523513640,-2.76535285875079,-2.45519865331974,-2.06613488227304,-1.61066482487588,-1.10342712777409,-0.560724593369343,0,0.560724593369343,1.10342712777409,1.61066482487588,2.06613488227304,2.45519865331974,2.76535285875079,2.98663523513640,3.11198463607399 ] )
            wi = np.array( [ 0.0758641308884886,0.174231250052868,0.267148938842824,0.351493472398162,0.424543422413522,0.483949031315556,0.527800453107652,0.554688098078883,0.563747712983677,0.554688098078883,0.527800453107652,0.483949031315556,0.424543422413522,0.351493472398162,0.267148938842824,0.174231250052868,0.0758641308884886 ] )
        elif npoints==18:
            xi = np.array( [ -3.11509384866672,-3.00280949809868,-2.80419335132485,-2.52491359476176,-2.17299893306165,-1.75857193059577,-1.29355442395873,-0.791323916172894,-0.266328558179894,0.266328558179894,0.791323916172894,1.29355442395873,1.75857193059577,2.17299893306165,2.52491359476176,2.80419335132485,3.00280949809868,3.11509384866672 ] )
            wi = np.array( [ 0.0679087092946904,0.156182861584973,0.240098512713994,0.317118784202649,0.385018537063959,0.441842747508773,0.485956238999599,0.516089794493472,0.531376467727684,0.531376467727684,0.516089794493472,0.485956238999599,0.441842747508773,0.385018537063959,0.317118784202649,0.240098512713994,0.156182861584973,0.0679087092946904 ] )
        elif npoints==19:
            xi = np.array( [ -3.11773804999124,-3.01658287666381,-2.83734795184256,-2.58463432097774,-2.26498204620307,-1.88666871727298,-1.45949202817948,-0.994515450836004,-0.503781543082941,0,0.503781543082941,0.994515450836005,1.45949202817948,1.88666871727298,2.26498204620307,2.58463432097774,2.83734795184256,3.01658287666381,3.11773804999124 ] )
            wi = np.array( [ 0.0611410109282240,0.140788045583429,0.216909828235038,0.287424379805861,0.350496954037162,0.404492502834154,0.448012167901271,0.479928675472095,0.499415350556115,0.505967476472886,0.499415350556115,0.479928675472095,0.448012167901271,0.404492502834154,0.350496954037162,0.287424379805861,0.216909828235038,0.140788045583429,0.0611410109282240 ] )
        elif npoints==20:
            xi = np.array( [ -3.12000551126982,-3.02840712500309,-2.86586897814605,-2.63616371417920,-2.34467083447487,-1.99822157065917,-1.60493602029016,-1.17403230291014,-0.715610356538290,-0.240415156597779,0.240415156597779,0.715610356538290,1.17403230291014,1.60493602029016,1.99822157065917,2.34467083447487,2.63616371417920,2.86586897814605,3.02840712500309,3.12000551126982 ] )
            wi = np.array( [ 0.0553360354286418,0.127553153586131,0.196890046631862,0.261621599552271,0.320222915597370,0.371319073304790,0.413712059113176,0.446408093138322,0.468640758416383,0.479888918820847,0.479888918820847,0.468640758416383,0.446408093138322,0.413712059113176,0.371319073304790,0.320222915597370,0.261621599552271,0.196890046631862,0.127553153586131,0.0553360354286418 ] )
        else:
            gtn.error("Cannot accomodate Gaussian quadrature with more than 20 points!")
    wi = wi**(1.0/Nf)
    return xi, wi

def get_ABtensors(Nphi=2, beta=1, Nf=1, spacing=1, mass=1, charge=1, mu=1, Gauss=False):
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
    xi, wi = myQuadrature(beta=beta,Nf=Nf,npoints=Nphi, Gauss=Gauss)
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
                    #    p1 = \[CurlyPhi]_{x,1} = b(+x)
                    #    q1 = \[CurlyPhi]_{x,2} = b(+y)
                    #    P1 = \[CurlyPhi]_{x-1,1} = b(-x)
                    #    Q1 = \[CurlyPhi]_{x-2,2} = b(-x)
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
    A = gtn.sparse(A, statistics=(0,0,0,0))
    B = sp.COO([psi1,psi2,psi3,psi4,phi1,phi2,phi3,phi4], B, shape=(Npsi,Npsi,Npsi,Npsi,Nphi,Nphi,Nphi,Nphi))
    B = gtn.sparse(B, statistics=(1,1,-1,-1,0,0,0,0))

    return A, B

def fcompress_B(B,cutoff=64,mute=True):

    process_name = "B compression (1)"
    process_length = 16
    s00 = time.time()
    gtn.progress_space()

    if not mute:
        B.info("B (uncompressed)")

    # μ = 1 ===========================================================================

    step = 1
    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",time=time.time()-s00)
    Qp = gtn.einsum('IJKLijkl -> JKLijkl I',B)
    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",time=time.time()-s00)
    cQp = Qp.hconjugate("abcdefg|x")
    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",time=time.time()-s00)
    Mp = gtn.einsum('I abcdefg,abcdefg J -> IJ',cQp,Qp)
    
    del Qp, cQp
    gc.collect()

    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",time=time.time()-s00)
    Qm = gtn.einsum('IJKLijkl -> K IJLijkl',B)
    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",time=time.time()-s00)
    cQm = Qm.hconjugate("x|abcdefg")
    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",time=time.time()-s00)
    Mm = gtn.einsum('I abcdefg,abcdefg J -> IJ',Qm,cQm)

    del Qm, cQm
    gc.collect()

    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",time=time.time()-s00)
    Up, Λp, cUp = Mp.eig("I|J",cutoff)
    Um, Λm, cUm = Mm.eig("I|J",cutoff)

    if Λp.shape[0] < Λm.shape[0] :
        U1 = Up.copy()
        cU1 = cUp.copy()
    else:
        U1 = Um.copy()
        cU1 = cUm.copy()

    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",time=time.time()-s00)
    B = gtn.einsum('IA,IJKLijkl->AJKLijkl',U1,B)
    B = gtn.einsum('CK,AJKLijkl->AJCLijkl',U1.hconjugate('I|J'),B)

    if not mute:
        gtn.clear_progress()
        U1.info("U1")

    # μ = 2 ===========================================================================
    
    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",time=time.time()-s00)
    Qp = gtn.einsum('IJKLijkl -> IKLijkl J',B)
    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",time=time.time()-s00)
    cQp = Qp.hconjugate("abcdefg|x")
    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",time=time.time()-s00)
    Mp = gtn.einsum('I abcdefg,abcdefg J -> IJ',cQp,Qp)
    
    del Qp, cQp
    gc.collect()

    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",time=time.time()-s00)
    Qm = gtn.einsum('IJKLijkl -> L IJKijkl',B)
    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",time=time.time()-s00)
    cQm = Qm.hconjugate("x|abcdefg")
    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",time=time.time()-s00)
    Mm = gtn.einsum('I abcdefg,abcdefg J -> IJ',Qm,cQm)
    del Qm, cQm
    gc.collect()

    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",time=time.time()-s00)
    Up, Λp, cUp = Mp.eig("I|J",cutoff)
    Um, Λm, cUm = Mm.eig("I|J",cutoff)

    if Λp.shape[0] < Λm.shape[0] :
        U2 = Up.copy()
        cU2 = cUp.copy()
    else:
        U2 = Um.copy()
        cU2 = cUm.copy()

    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",time=time.time()-s00)
    B = gtn.einsum('JB,AJCLijkl->ABCLijkl',U2,B)
    B = gtn.einsum('DL,ABCLijkl->ABCDijkl',cU2,B)
    gtn.clear_progress()
    gtn.tab_up()
    
    if not mute:
        U2.info("U2")

    return B

def compress_B(B,cutoff=64,mute=True):

    process_name = "B compression (2)"
    process_length = 4*8
    s00 = time.time()
    gtn.progress_space()

    Npsi = B.shape[0]
    Nphi = B.shape[4]
    δ = np.zeros([Nphi,Nphi],dtype=int)
    for i in range(Nphi):
        δ[i,i] = 1
    δ = gtn.sparse(δ,statistics=(0,0))

    if not mute:
        B.info("B (uncompressed)")

    # μ = 1 ===========================================================================

    step = 1
    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",time=time.time()-s00)
    Qp = gtn.einsum('IJKLijkl,km -> JKLjklm Ii',B,δ)
    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",time=time.time()-s00)
    cQp = Qp.hconjugate("JKLjklm|Ii")
    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",time=time.time()-s00)
    Mp = gtn.einsum('Ii abcdefg,abcdefg Jj -> IiJj',cQp,Qp)
    
    del Qp, cQp
    gc.collect()

    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",time=time.time()-s00)
    Qm = gtn.einsum('IJKLijkl,km -> Kk IJLijlm',B,δ)
    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",time=time.time()-s00)
    cQm = Qm.hconjugate("Kk|IJLijlm")
    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",time=time.time()-s00)
    Mm = gtn.einsum('Ii abcdefg,abcdefg Jj -> IiJj',Qm,cQm)

    del Qm, cQm
    gc.collect()

    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",time=time.time()-s00)
    Up, Λp, cUp = Mp.eig("Ii|Jj",cutoff)
    Um, Λm, cUm = Mm.eig("Ii|Jj",cutoff)

    if Λp.shape[0] < Λm.shape[0] :
        U1 = Up.copy()
    else:
        U1 = Um.copy()

    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",time=time.time()-s00)
    Bfin = gtn.einsum('IJKLijkl,IiA->AJKLjkl',B,U1)

    if not mute:
        gtn.clear_progress()
        U1.info("U1")


    # μ = 2 ===========================================================================
    
    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",time=time.time()-s00)
    Qp = gtn.einsum('IJKLijkl,lm -> IKLiklm Jj',B,δ)
    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",time=time.time()-s00)
    cQp = Qp.hconjugate("JKLjklm|Ii")
    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",time=time.time()-s00)
    Mp = gtn.einsum('Ii abcdefg,abcdefg Jj -> IiJj',cQp,Qp)
    
    del Qp, cQp
    gc.collect()

    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",time=time.time()-s00)
    Qm = gtn.einsum('IJKLijkl,lm -> Ll IJKijkm',B,δ)
    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",time=time.time()-s00)
    cQm = Qm.hconjugate("Kk|IJLijlm")
    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",time=time.time()-s00)
    Mm = gtn.einsum('Ii abcdefg,abcdefg Jj -> IiJj',Qm,cQm)

    del Qm, cQm
    gc.collect()

    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",time=time.time()-s00)
    Up, Λp, cUp = Mp.eig("Ii|Jj",cutoff)
    Um, Λm, cUm = Mm.eig("Ii|Jj",cutoff)

    if Λp.shape[0] < Λm.shape[0] :
        U2 = Up.copy()
    else:
        U2 = Um.copy()

    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",time=time.time()-s00)
    Bfin = gtn.einsum('AJKLjkl,JjB->ABKLkl',Bfin,U2)
    
    if not mute:
        gtn.clear_progress()
        U2.info("U2")

    # μ = 3 ===========================================================================

    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",time=time.time()-s00)
    Qp = gtn.einsum('IJKLijkl,im -> JKLjklm Ii',B,δ)
    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",time=time.time()-s00)
    cQp = Qp.hconjugate("JKLjklm|Ii")
    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",time=time.time()-s00)
    Mp = gtn.einsum('Ii abcdefg,abcdefg Jj -> IiJj',cQp,Qp)
    
    del Qp, cQp
    gc.collect()

    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",time=time.time()-s00)
    Qm = gtn.einsum('IJKLijkl,im -> Kk IJLijlm',B,δ)
    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",time=time.time()-s00)
    cQm = Qm.hconjugate("Kk|IJLijlm")
    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",time=time.time()-s00)
    Mm = gtn.einsum('Ii abcdefg,abcdefg Jj -> IiJj',Qm,cQm)

    del Qm, cQm
    gc.collect()

    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",time=time.time()-s00)
    Up, Λp, cUp = Mp.eig("Ii|Jj",cutoff)
    Um, Λm, cUm = Mm.eig("Ii|Jj",cutoff)

    if Λp.shape[0] < Λm.shape[0] :
        U3 = Up.copy()
    else:
        U3 = Um.copy()

    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",time=time.time()-s00)
    Bfin = gtn.einsum('ABKLkl,CKk->ABCLl',Bfin,U3.hconjugate('ij|k'))
    
    if not mute:
        gtn.clear_progress()
        U3.info("U3")


    # μ = 4 ===========================================================================

    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",time=time.time()-s00)
    Qp = gtn.einsum('IJKLijkl,jm -> IKLiklm Jj',B,δ)
    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",time=time.time()-s00)
    cQp = Qp.hconjugate("JKLjklm|Ii")
    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",time=time.time()-s00)
    Mp = gtn.einsum('Ii abcdefg,abcdefg Jj -> IiJj',cQp,Qp)
    
    del Qp, cQp
    gc.collect()

    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",time=time.time()-s00)
    Qm = gtn.einsum('IJKLijkl,jm -> Ll IJKijkm',B,δ)
    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",time=time.time()-s00)
    cQm = Qm.hconjugate("Kk|IJLijlm")
    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",time=time.time()-s00)
    Mm = gtn.einsum('Ii abcdefg,abcdefg Jj -> IiJj',Qm,cQm)

    del Qm, cQm
    gc.collect()

    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",time=time.time()-s00)
    Up, Λp, cUp = Mp.eig("Ii|Jj",cutoff)
    Um, Λm, cUm = Mm.eig("Ii|Jj",cutoff)

    if Λp.shape[0] < Λm.shape[0] :
        U4 = Up.copy()
    else:
        U4 = Um.copy()

    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",time=time.time()-s00)
    Bfin = gtn.einsum('ABCLl,DLl->ABCD',Bfin,U4.hconjugate('ij|k'))
    gtn.clear_progress()
    gtn.tab_up()
    
    if not mute:
        U1.info("U4")
        Bfin.info("B (compressed)")

    return Bfin, [U1,U2,U3,U4]

def compress_A(A,Upack,mute=True):
    
    [U1,U2,U3,U4] = Upack

    process_name = "A compression"
    process_length = 3
    s00 = time.time()
    gtn.progress_space()

    Nphi = A.shape[0]
    δ = np.zeros([Nphi,Nphi],dtype=int)
    for i in range(Nphi):
        δ[i,i] = 1
    δ = gtn.sparse(δ,statistics=(0,0))

    if not mute:
        A.info("A (uncompressed)")

    step = 1
    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",time=time.time()-s00)
    Ix = gtn.einsum('KXj,XjI->KIj',U1.hconjugate('ij|k'),U3)
    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",time=time.time()-s00)
    Iy = gtn.einsum('LYj,YjJ->LJj',U2.hconjugate('ij|k'),U4)

    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",time=time.time()-s00)
    Afin = gtn.einsum('ijkl,KIl,LJk,km,ln->IJKLijklmn',A,Ix,Iy,δ,δ)
    gtn.clear_progress()
    gtn.tab_up()

    if not mute:
        Afin.info("A (compressed)")

    return Afin

def compress_T(T,cutoff=64,mute=True):

    process_name = "T compression"
    process_length = 16
    s00 = time.time()
    gtn.progress_space()

    # x direction =====================================================================

    step = 1
    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",time=time.time()-s00)
    Txp = gtn.einsum('IJKLijklmn -> JKLjklmn Ii',T)
    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",time=time.time()-s00)
    Txm = gtn.einsum('IJKLijklmn -> Kk IJLijlmn',T)

    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",time=time.time()-s00)
    cTxp = Txp.hconjugate('abcdefgh|xy')
    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",time=time.time()-s00)
    cTxm = Txm.hconjugate('xy|abcdefgh')

    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",time=time.time()-s00)
    Mp = gtn.einsum('Ii abcdefgh, abcdefgh Jj -> IiJj',cTxp,Txp)
    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",time=time.time()-s00)
    Mm = gtn.einsum('Ii abcdefgh, abcdefgh Jj -> IiJj',Txm,cTxm)

    del Txp, Txm, cTxp, cTxm
    gc.collect()


    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",time=time.time()-s00)
    Up, Λp, cUp = Mp.eig("Ii|Jj",cutoff)
    Um, Λm, cUm = Mm.eig("Ii|Jj",cutoff)

    if Λp.shape[0] < Λm.shape[0] :
        U = Up.copy()
    else:
        U = Um.copy()

    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",time=time.time()-s00)

    Tfin = gtn.einsum('IJKLijklmn,IiA,CKk->ACJLjlmn',T,U,U.hconjugate('ij|k'))

    if not mute:
        Tfin.info("T (x-compression)")

    # y direction =====================================================================

    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",time=time.time()-s00)
    Typ = gtn.einsum('IKJLjlmn -> IKLlmn Jj',Tfin)
    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",time=time.time()-s00)
    Tym = gtn.einsum('IKJLjlmn -> Ll IKJjmn',Tfin)

    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",time=time.time()-s00)
    cTyp = Typ.hconjugate('abcdef|xy')
    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",time=time.time()-s00)
    cTym = Tym.hconjugate('xy|abcdef')

    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",time=time.time()-s00)
    Mp = gtn.einsum('Ii abcdef, abcdef Jj -> IiJj',cTyp,Typ)
    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",time=time.time()-s00)
    Mm = gtn.einsum('Ii abcdef, abcdef Jj -> IiJj',Tym,cTym)

    del Typ, Tym, cTyp, cTym
    gc.collect()

    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",time=time.time()-s00)
    Up, Λp, cUp = Mp.eig("Ii|Jj",cutoff)
    Um, Λm, cUm = Mm.eig("Ii|Jj",cutoff)

    if Λp.shape[0] < Λm.shape[0] :
        U = Up.copy()
    else:
        U = Um.copy()

    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",time=time.time()-s00)
    Tfin = gtn.einsum('ACJLjlmn,JjB,DLl->ABCDmn',Tfin,U,U.hconjugate('ij|k'))
    gtn.clear_progress()
    gtn.tab_up()

    if not mute:
        Tfin.info("T (y-compression)")

    return Tfin

####################################################
##               2D Coarse graining               ##
####################################################

def zcap(T):
    Nphi = T.shape[4]
    if type(T)==gtn.dense :
        capper = gtn.dense(np.full((Nphi,Nphi),1),statistics=(0,0))
    else :
        capper = gtn.sparse(np.full((Nphi,Nphi),1),statistics=(0,0))
    return gtn.einsum("IJKLij,ij->IJKL",T,capper)

def logZ(T,boundary_conditions='periodic'):
    
    if type(T) == gtn.sparse:
        T = gtn.dense(T)
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
    
    if type(T1) == gtn.sparse:
        T1 = gtn.dense(T1)
        if T1.encoder == 'parity-preserving':
            T1 = T1.switch_encoder()

    if type(T2) == gtn.sparse:
        T2 = gtn.dense(T2)
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

####################################################
##                       TRG                      ##
####################################################

def trg(T,dcut=64,iternum=None,error_test=False):

    process_name = "trg"
    if iternum != None:
        process_name = process_name+"["+str(iternum)+"]"
    process_length = 10
    process_color = "purple"
    step = 1
    s00 = time.time()
    gtn.progress_space() # << Don't remove this. This is for the gtn.show_progress!

    # mandatory properties of T:
    #    - shape = (nx,ny,nx,ny)
    #    - statistics = (1,1,-1,-1)

    if [T.shape[0],T.shape[1]] != [T.shape[2],T.shape[3]] :
        gtn.error("Error[trg]: The shape must be of the form (m,n,m,n)!")

    if gtn.make_list(T.statistics) != [1,1,-1,-1] :
        gtn.error("Error[trg]: The statistics must be (1,1,-1,-1)!")

    #===============================================================================#
    #   Step 1: Rearrange the tensor legs in two ways                               #
    #===============================================================================#
    
    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",color=process_color,time=time.time()-s00)
    T1 = gtn.einsum('ijkl->jkli',T)
    T2 = gtn.einsum('ijkl->klij',T)

    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",color=process_color,time=time.time()-s00)
    U1,S1,V1 = T1.svd('ab|cd',dcut)
    U2,S2,V2 = T2.svd('ab|cd',dcut)

    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",color=process_color,time=time.time()-s00)
    #
    #                             j                                     j
    #                             ↑                                     ↑
    #        j                    ↑                                     ↑
    #        ↑              k → →(U1)                                 (V2)→ → i
    #        ↑                      ↘                                 ↗
    #  k → →(T)→ → i    =             ↘              =              ↗
    #        ↑                          ↘                         ↗
    #        ↑                          (V1)→ → i         k → →(U2)
    #        l                            ↑                     ↑
    #                                     ↑                     ↑
    #                                     l                     l
    #

    #===============================================================================#
    #   Step 2: Multiply sqrt(S) into U and V                                       #
    #===============================================================================#
    
    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",color=process_color,time=time.time()-s00)
    sqrtS = gtn.sqrt(S1)
    U1 = gtn.einsum('abx,xc->abc',U1,sqrtS)
    V1 = gtn.einsum('ax,xbc->abc',sqrtS,V1)

    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",color=process_color,time=time.time()-s00)
    sqrtS = gtn.sqrt(S2)
    U2 = gtn.einsum('abx,xc->abc',U2,sqrtS)
    V2 = gtn.einsum('ax,xbc->abc',sqrtS,V2)
    
    #===============================================================================#
    #   Step 3: Renormalization                                                     #
    #===============================================================================#

    #
    #      k                       j
    #        ↘                   ↗
    #          ↘               ↗
    #          (V1)→ → z → →(U2)
    #            ↑           ↑
    #            ↑           ↑
    #            w           y
    #            ↑           ↑
    #            ↑           ↑
    #          (V2)→ → x → →(U1)
    #          ↗               ↘
    #        ↗                   ↘
    #      l                       i
    #
    
    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",color=process_color,time=time.time()-s00)
    VV = gtn.einsum('kwz,lxw->lxzk',V1,V2);
    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",color=process_color,time=time.time()-s00)
    UU = gtn.einsum('yxi,zyj->jzxi',U1,U2);
    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",color=process_color,time=time.time()-s00)
    T2 = gtn.einsum('lxzk,jzxi->ijkl',VV,UU);
    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",color=process_color,time=time.time()-s00)

    if error_test :
        Z1 = gtn.einsum('ijkl,klij',T,T);
        Z2 = gtn.einsum('ijij',T2);
        err = np.abs(1-Z2/Z1)
    #print("Error:",err)
    
    gtn.clear_progress()
    gtn.tab_up()
    
    Tnorm = T2.norm
    T2 = gtn.dense(T2)
    T2.data = T2.data/Tnorm
    if type(T) == gtn.sparse :
        T2 = gtn.sparse(T2)
    
    if error_test :
        return T2, Tnorm, err
    else:
        return T2, Tnorm

####################################################
##                     2D ATRG                    ##
####################################################

def atrg2dy(T1,T2,dcut=64,intermediate_dcut=None,iternum=None,error_test=False,alignment="y"):
    
    T1ori = T1.copy()
    T2ori = T2.copy()
    
    process_name = "atrg2d"+alignment
    if iternum != None:
        process_name = process_name+"["+str(iternum)+"]"
    process_length = 8
    process_color = "purple"
    step = 1
    s00 = time.time()
    gtn.progress_space() # << Don't remove this. This is for the gtn.show_progress!

    if intermediate_dcut==None:
        intermediate_dcut=dcut

    T1 = gtn.einsum("ijkl->li jk",T1)
    T2 = gtn.einsum("ijkl->li jk",T2)

    U1, S1, V1 = T1.svd("li|jk",intermediate_dcut)
    U2, S2, V2 = T2.svd("li|jk",intermediate_dcut)

    #
    #        j                    j
    #        :                    :
    #        :                    :
    #  k --- X --- i        k --- V
    #        :                    :
    #        :                    S
    #        l                    :
    #                             U --- i
    #                             :
    #                             :
    #                             l
    #
    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",color=process_color,time=time.time()-s00)
    A = V1.copy()
    B = gtn.einsum("lia,ab->lib",U1,S1)
    C = gtn.einsum("ab,bjk->ajk",S2,V2)
    D = U2.copy()

    M = gtn.einsum("ajk,jib->aibk",C,B)

    del U1.data,S1.data,V1.data,U2.data,S2.data,V2.data,B.data,C.data
    del U1,S1,V1,U2,S2,V2,B,C
    gc.collect()

    #
    #        b                    
    #        :                    
    #        :                   
    #  k --- M --- i        
    #        :                    
    #        :                    
    #        a                    
    #

    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",color=process_color,time=time.time()-s00)
    U, S, V = M.svd("ai|bk",intermediate_dcut)

    sqrtS = gtn.sqrt(S)
    Y = gtn.einsum('abx,xc->abc',U,sqrtS)
    X = gtn.einsum('ax,xbc->abc',sqrtS,V)

    del U.data,S.data,V.data,sqrtS.data
    del U,S,V,sqrtS
    gc.collect()

    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",color=process_color,time=time.time()-s00)
    Q1 = gtn.einsum('iax,xbj->ijab',D,Y)
    Q2 = gtn.einsum('kya,ylb->abkl',X,A)

    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",color=process_color,time=time.time()-s00)
    Q = gtn.einsum('ijab,abkl->ijkl',Q1,Q2)
    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",color=process_color,time=time.time()-s00)
    
    U,S,V = Q.svd("ij|kl",dcut)

    sqrtS = gtn.sqrt(S)
    H = gtn.einsum('abx,xc->abc',U,sqrtS)
    G = gtn.einsum('ax,xbc->abc',sqrtS,V)

    del U.data,S.data,V.data,sqrtS.data
    del U,S,V,sqrtS
    gc.collect()

    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",color=process_color,time=time.time()-s00)

    H = gtn.einsum('lai->ila',H)
    G = gtn.einsum('kaj->ajk',G)
    T = gtn.einsum('ila,ajk->ijkl',H,G)

    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",color=process_color,time=time.time()-s00)
    if error_test :
        Z1 = gtn.einsum('IJIK,iKiJ',T1ori,T2ori)
        Z2 = gtn.einsum('IJIJ',T)
        error = np.abs(1-Z2/Z1)
    
    gtn.clear_progress()
    gtn.tab_up()
    
    Tnorm = T.norm
    T.data = T.data/Tnorm

    if error_test :
        return T, Tnorm, error
    else :
        return T, Tnorm

def atrg2dx(T1,T2,dcut=64,intermediate_dcut=None,iternum=None,error_test=False):
    T1 = gtn.einsum('ijkl->jikl',T1)
    T1 = gtn.einsum('jikl->jilk',T1)
    T2 = gtn.einsum('ijkl->jikl',T2)
    T2 = gtn.einsum('jikl->jilk',T2)
    if error_test :
        T, Tnorm, err = atrg2dy(T1,T2,dcut,intermediate_dcut,iternum,True,alignment="x")
    else:
        T, Tnorm = atrg2dy(T1,T2,dcut,intermediate_dcut,iternum,alignment="x")
    T = gtn.einsum('ijkl->jikl',T)
    T = gtn.einsum('jikl->jilk',T)
    if error_test :
        return T, Tnorm, err
    else :
        return T, Tnorm

####################################################
##                     3D ATRG                    ##
####################################################

def hotrg3dz(T1,T2,dcut=64,intermediate_dcut=None,iternum=None,error_test=False,svd_only=False,print_svd=False):
    dt_string = datetime.now().strftime("%y%m%d.%H%M%S.%f")

    if print_svd :
        if(not os.path.exists("data/")):
            os.mkdir("data/")
        directory = "data/hotrg3dz_svd"+dt_string+".txt"
        fsvd = open(directory, "a")
        print(" Singular values will be printed into <../"+directory+">.")
        
    if svd_only :
        print(" Most of the CG process will be skipped in favor of getting the singular values.")

    T1ori = T1.copy()
    T2ori = T2.copy()

    process_name = "hotrg3d"
    if iternum != None:
        process_name = process_name+"["+str(iternum)+"]"
    process_color = "purple"
    process_length = 38
    step = 1
    s00 = time.time()
    gtn.progress_space() # << Don't remove this. This is for the gtn.show_progress!

    if intermediate_dcut==None:
        intermediate_dcut=dcut

    #=================================================================================================
    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",color=process_color,time=time.time()-s00) #1
    T1 = gtn.einsum('i1 i2 i3 i4 mn-> i1 i3 mn i2 i4',T1)
    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",color=process_color,time=time.time()-s00)
    X1,S1,Y1 = T1.svd('(i1 i3 m)(n i2 i4)',intermediate_dcut)
    sqrtS = gtn.sqrt(S1)
    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",color=process_color,time=time.time()-s00)
    X1 = gtn.einsum('i1 i3 m a,ab->i1 i3 b m',X1,sqrtS)
    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",color=process_color,time=time.time()-s00)
    Y1 = gtn.einsum('ab,b n i2 i4->n a i2 i4',sqrtS,Y1)

    X1,S1,P1 = X1.svd('(i1 i3)(a m)',intermediate_dcut)
    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",color=process_color,time=time.time()-s00)
    X1 = gtn.einsum('i1 i3 x, xy -> i1 i3 y',X1,S1)
    #P1 = gtn.einsum('yx,xam->yam',S1,P1)

    Q1,S1,Y1 = Y1.svd('(na)(i2 i4)',intermediate_dcut)
    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",color=process_color,time=time.time()-s00)
    Y1 = gtn.einsum('xy, y i2 i4 -> x i2 i4',S1,Y1)
    #Q1 = gtn.einsum('nax,xy->nay',Q1,S1)

    #=================================================================================================
    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",color=process_color,time=time.time()-s00) #7
    T2 = gtn.einsum('i1 i2 i3 i4 mn-> i1 i3 mn i2 i4',T2)
    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",color=process_color,time=time.time()-s00)
    X2,S2,Y2 = T2.svd('(i1 i3 m)(n i2 i4)',intermediate_dcut)
    sqrtS = gtn.sqrt(S2)
    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",color=process_color,time=time.time()-s00)
    X2 = gtn.einsum('i1 i3 m a,ab->i1 i3 b m',X2,sqrtS)
    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",color=process_color,time=time.time()-s00)
    Y2 = gtn.einsum('ab,b n i2 i4->n a i2 i4',sqrtS,Y2)
    
    X2,S2,P2 = X2.svd('(i1 i3)(a m)',intermediate_dcut)
    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",color=process_color,time=time.time()-s00)
    X2 = gtn.einsum('i1 i3 x, xy -> i1 i3 y',X2,S2)
    #P2 = gtn.einsum('yx,xam->yam',S2,P2)

    Q2,S2,Y2 = Y2.svd('(na)(i2 i4)',intermediate_dcut)
    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",color=process_color,time=time.time()-s00)
    Y2 = gtn.einsum('xy, y i2 i4 -> x i2 i4',S2,Y2)
    #Q2 = gtn.einsum('nax,xy->nay',Q2,S2)

    #=================================================================================================

    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",color=process_color,time=time.time()-s00)
    XX = gtn.einsum('i1 i3 a, j1 j3 b -> i1 i3 ab j1 j3',X1,X2) #-> i3 j3 ab m i1 j1
    XX = gtn.einsum('i1 i3 ab j1 j3 -> i1 i3 ab j3 j1 ',XX)
    XX = gtn.einsum('i1 i3 ab j3 j1 -> i1 i3 j3 ab j1 ',XX)
    XX = gtn.einsum('i1 i3 j3 ab j1 -> i3 j3 i1 ab j1 ',XX)
    XX = gtn.einsum('i3 j3 i1 ab j1 -> i3 j3 ab i1 j1 ',XX)

    del X1.data,X2.data
    del X1,X2
    gc.collect()

    # below here is the one different from the svd version
    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",color=process_color,time=time.time()-s00) #14
    cXX1 = XX.hconjugate('(i3 j3 ab)(i1 j1)')
    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",color=process_color,time=time.time()-s00)
    M1 = gtn.einsum(' I1 J1 i3 j3 ab, i3 j3 ab i1 j1 -> I1 J1 i1 j1',cXX1,XX)
    del cXX1.data
    del cXX1
    gc.collect()
    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",color=process_color,time=time.time()-s00) #16
    U1 , S1, _ = M1.eig('(I J)(i j)',dcut)
    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",color=process_color,time=time.time()-s00)
    cXX3 = XX.hconjugate('(i3 j3)(ab i1 j1)')
    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",color=process_color,time=time.time()-s00)
    M3 = gtn.einsum(' I3 J3 ab i1 j1, ab i1 j1 i3 j3  -> I3 J3 i3 j3',XX,cXX3)
    del cXX3.data
    del cXX3
    gc.collect()
    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",color=process_color,time=time.time()-s00) #19
    U3, S3,  _ = M3.eig('(I J)(i j)',dcut)
    if S1.shape[0] < S3.shape[0] :
        Ux = U1.copy()
        Sx = S1.copy()
    else:
        Ux = U3.copy()
        Sx = S3.copy()

    if not svd_only :
        step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",color=process_color,time=time.time()-s00) #20
        cUx = Ux.hconjugate('ij|a')
        #switch the i and j
        step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",color=process_color,time=time.time()-s00)
        XX = gtn.einsum('i3 j3 ab i1 j1 -> j3 i3 ab i1 j1',XX)
        step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",color=process_color,time=time.time()-s00)
        XX = gtn.einsum('j3 i3 ab i1 j1 -> j3 i3 ab j1 i1',XX)

        #=================================================================================================
        step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",color=process_color,time=time.time()-s00) #23
        Xprime = gtn.einsum('s i3 j3,j3 i3 kl j1 i1 -> s kl j1 i1',cUx,XX)
        
        del XX.data
        del XX
        gc.collect()

        step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",color=process_color,time=time.time()-s00) #24
        Xprime = gtn.einsum('s kl j1 i1, i1 j1 t -> s kl t',Xprime,Ux)
        Xprime = gtn.einsum('s kl t -> t s kl',Xprime)

        Xprime = gtn.einsum('t s kl , kam -> t s al m',Xprime, P1)
        Xprime = gtn.einsum('t s al m , lbm -> t s ab m',Xprime, P2)
    
    #=================================================================================================
    # Do SVD instead of Eig because doing conjugate is slow
    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",color=process_color,time=time.time()-s00) #25
    YY = gtn.einsum('b j2 j4, a i2 i4 -> j2 j4 ba i2 i4 ',Y2,Y1) #-> i4 j4 n b a i2 j2
    YY = gtn.einsum('j2 j4 ba i2 i4 -> j2 j4 i4 ba i2 ',YY)
    YY = gtn.einsum('j2 j4 i4 ba i2 -> j4 i4 j2 ba i2 ',YY)
    YY = gtn.einsum('j4 i4 j2 ba i2 -> j4 i4 ba i2 j2 ',YY)
    YY = gtn.einsum('j4 i4 ba i2 j2 -> i4 j4 ba i2 j2 ',YY)

    del Y1.data,Y2.data
    del Y1,Y2
    gc.collect()

    # below here is the one different from the svd version
    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",color=process_color,time=time.time()-s00) #26
    cYY2 = YY.hconjugate('(i4 j4 ba)(i2 j2)')
    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",color=process_color,time=time.time()-s00)
    M2 = gtn.einsum(' I2 J2 i4 j4 ba, i4 j4 ba i2 j2 -> I2 J2 i2 j2',cYY2,YY)
    del cYY2.data
    del cYY2
    gc.collect()
    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",color=process_color,time=time.time()-s00) #28
    U2 , S2, _ = M2.eig('(I J)(i j)',dcut)
    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",color=process_color,time=time.time()-s00)
    cYY4 = YY.hconjugate('(i4 j4)(ba i2 j2)')
    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",color=process_color,time=time.time()-s00)
    M4 = gtn.einsum(' I4 J4 ba i2 j2, ba i2 j2 i4 j4  -> I4 J4 i4 j4',YY,cYY4)
    del cYY4.data
    del cYY4
    gc.collect()
    step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",color=process_color,time=time.time()-s00) #31
    U4, S4,  _ = M4.eig('(I J)(i j)',dcut)
    if S2.shape[0] < S4.shape[0] :
        Uy = U2.copy()
        Sy = S2.copy()
    else:
        Uy = U4.copy()
        Sy = S4.copy()


    if not svd_only :
        step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",color=process_color,time=time.time()-s00) #32
        cUy = Uy.hconjugate('ij|a')

        #switch the i and j
        step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",color=process_color,time=time.time()-s00) #33
        YY = gtn.einsum('i4 j4 b a i2 j2 -> j4 i4 b a i2 j2',YY)
        step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",color=process_color,time=time.time()-s00)
        YY = gtn.einsum('j4 i4 b a i2 j2 -> j4 i4 b a j2 i2',YY)
        
        #=================================================================================================
        step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",color=process_color,time=time.time()-s00) #35
        Yprime = gtn.einsum('s i4 j4,j4 i4 lk j2 i2 -> s lk j2 i2',cUy,YY)

        del YY.data
        del YY
        gc.collect()

        step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",color=process_color,time=time.time()-s00) #36
        Yprime = gtn.einsum('s lk j2 i2, i2 j2 t -> s lk t',Yprime,Uy)
        Yprime = gtn.einsum('s lk t -> lk t s',Yprime)

        Yprime = gtn.einsum('nak, lk t s -> n la t s',Q1,Yprime)
        Yprime = gtn.einsum('nbl, n la t s -> n ba t s',Q2,Yprime)
    
        step = gtn.show_progress(step,process_length,process_name+" "+"<"+gtn.current_memory_display()+">",color=process_color,time=time.time()-s00) #37
        T = gtn.einsum(' t1 t3 kl m, n lk t2 t4 -> t1 t2 t3 t4 mn ',Xprime,Yprime)
        del Xprime.data,Yprime.data
        del Xprime,Yprime
        gc.collect()
    
    gtn.clear_progress()
    gtn.tab_up()

    if print_svd :
        Sx = np.sort(np.diag(Sx.force_format("matrix").data))[::-1]
        Sy = np.sort(np.diag(Sy.force_format("matrix").data))[::-1]

        Sxmax = 0
        for s in Sx:
            if np.abs(s) > np.abs(Sxmax):
                Sxmax = s
        Symax = 0
        for s in Sy:
            if np.abs(s) > np.abs(Symax):
                Symax = s

        nsvd = max(len(Sx),len(Sy))
        fsvd.write("\n")
        fsvd.write("   i "+"                  Sx          "+"                  Sy          "+"\n")
        for i in range(nsvd):
            display = '{:5d}'.format(i+1)
            if i<len(Sx) and Sxmax > 10*numer_cutoff:
                display += '{:30g}'.format(np.real(Sx[i]/Sxmax))
            else:
                display += '{:30g}'.format(0)

            if i<len(Sy) and Symax > 10*numer_cutoff:
                display += '{:30g}'.format(np.real(Sy[i]/Symax))
            else:
                display += '{:30g}'.format(0)
            fsvd.write(display+"\n")
            fsvd.flush()
        fsvd.write("\n")
    
    if svd_only :
        print("\n The code is terminated due to the svd_only option.")
        print("Maximum memory usage:",memory_display(tracemalloc.get_traced_memory()[1]))
        exit()
    
    if error_test :
        Z1 = gtn.einsum('IJIJmn,KLKLmn->mn',T1ori,T2ori)
        Z2 = gtn.einsum('IJIJmn->mn',T)
        error = (Z1-Z2).norm/Z1.norm
    
    Tnorm = T.norm
    T.data = T.data/Tnorm
    
    if error_test :
        return T, Tnorm, error
    else :
        return T, Tnorm
