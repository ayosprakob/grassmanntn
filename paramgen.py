import sys
import numpy as np
import math

####################################################
##             Sign factors generator             ##
####################################################

def generate_param(bitsize):
    def to_bin_canonical(i,n):
        if(i>=2**n):
            print("error[generate_param.to_bin_canonical]: Hilbert space dimension is probably wrong with values.")
            exit()
        ret = np.zeros([n],dtype=int)
        num = i
        for p in range(n):
            ret[n-1-p] = math.floor(num*(2**(p-n+1)))
            num = num - ret[n-1-p]*2**(n-1-p)
        return ret;
    def from_bin_canonical(vec):
        ret=0
        for i in range(vec.size):
            ret = ret + vec[i]*(2**i)
        return ret
    def to_bin_akiyama(i,n):
        for j in range(2**n):
            v = to_bin_canonical(j,n)
            I = from_bin_akiyama(v)
            if(I==i):
                return v
        print("error[generate_param.to_bin_akiyama]: cannot convert the decimal index to binary.")
        exit()
    def from_bin_akiyama(vec):
        condition = 0
        for k in range(1,vec.size):
            condition += vec[k]
        condition %= 2
        I = 0
        if(condition==0):
            for k in range(vec.size):
                I += (2**k)*vec[k]
        else:
            I = 1-vec[0]
            for k in range(1,vec.size):
                I += (2**k)*vec[k]
        return I
    def sgn(i,n):
        I = to_bin_canonical(i,n)
        p = np.sum(I)

        return (-1)**round(p*(p-1)/2)
    def sgnn(i1,i2,n1,n2):
        return sgn(i1+(2**n1)*i2,n1+n2)
    def gparity(i,n):
        # Grassmann parity
        Ia = to_bin_canonical(i,n)
        p = 0
        for i in range(Ia.size):
            p = p + Ia[i]
        return p
    param = open("param.generated.py", "w")
    nbts = bitsize
    param.write('info=["var:gparity","var:encoder","var:sgn","var:sgnn"]\n')
    param.write("gparity=["+str(gparity(0,nbts)))
    for i in range(1,2**nbts):
        param.write(","+str(gparity(i,nbts)))
    param.write("]\n")
    print("gparity done")
    binary = to_bin_canonical(0,nbts)
    aki = from_bin_akiyama(binary)
    param.write("encoder=["+str(aki))
    for i in range(1,2**nbts):
        binary = to_bin_canonical(i,nbts)
        aki = from_bin_akiyama(binary)
        #binary2 = to_bin_akiyama(aki,nbts)
        #j = from_bin_canonical(binary2)
        param.write(","+str(aki))
    param.write("]\n")
    print("encoder done")
    param.write("sgn=["+str(sgn(0,nbts)))
    for i in range(1,2**nbts):
        param.write(","+str(sgn(i,nbts)))
    param.write("]\n")
    print("sgn done")
    param.write("sgnn=[["+str(sgnn(0,0,nbts,nbts)))
    for i in range(1,2**nbts):
        param.write(","+str(sgnn(0,i,nbts,nbts)))
    param.write("]")
    for i in range(1,2**nbts):
        param.write(",["+str(sgnn(i,0,nbts,nbts)))
        for j in range(1,2**nbts):
            param.write(","+str(sgnn(i,j,nbts,nbts)))
        param.write("]")
        param.flush()
        print(i,"/",2**nbts)
    param.write("]\n")
    print("sgnn done")


#generate_param(8)
