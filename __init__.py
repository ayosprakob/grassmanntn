from grassmanntn.param import *
from grassmanntn.gauge2d import *
from grassmanntn.gauge2d_block import *
from grassmanntn.arith import *

import numpy as np
import math
from grassmanntn import param
import sparse as sp
import opt_einsum as oe
import time
import copy
import sys
import gc
import tracemalloc
import os
from datetime import datetime

hybrid_symbol = "*"
separator_list = ("|",":",";",",",".")
number_character = ("0","1","2","3","4","5","6","7","8","9")

skip_parity_blocking_check = False
skip_power_of_two_check = False
allowed_stat = (0,1,-1,hybrid_symbol)
fermi_type = (1,-1)
bose_type = (0,hybrid_symbol)
encoder_type = ("canonical","parity-preserving")
format_type = ("standard","matrix")
numer_cutoff = 1.0e-14
numer_display_cutoff = 1000*numer_cutoff
char_list = (
    "a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"
    ,"A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"
    ,"α","β","Γ","γ","Δ","δ","ε","ζ","η","Θ","θ","ι","κ","λ","μ","ν","Ξ","ξ","Π","π","ρ","Σ","σ","ς","τ","υ"
    ,"Φ","ϕ","φ","χ","Ψ","ψ","Ω","ω"
    )

progress_bar_enabled = False

####################################################
##                Random Utilities                ##
####################################################

def make_tuple(obj):
    if np.isscalar(obj):
        return tuple([obj])
    else:
        return tuple(list(obj))

def make_list(obj):
    if np.isscalar(obj):
        return [obj]
    else:
        return list(obj)

def error(text="Error[]: Unknown error."):
    print()
    print(text)
    print()
    print("\t__________look_below_for_the_error_path__________\n")
    #intentional sabotage!!!
    __________look_above_for_the_general_error_information__________

def get_char(string):
    for char in char_list:
        if char not in string:
            return char
    error("Error[get_char]: Running out of index character!")    

def show_progress(step_inp,total_inp,process_name = "", ratio=True, color="blue", time=0):

    if progress_bar_enabled:

        bar_length = 37
        step = int(np.floor(bar_length*step_inp/total_inp))
        total = bar_length

        color = color.lower()
        if   color=="black":
            color="0"
        elif color=="red":
            color="1"
        elif color=="green":
            color="2"
        elif color=="yellow":
            color="3"
        elif color=="blue":
            color="4"
        elif color=="purple":
            color="5"
        elif color=="cyan":
            color="6"

        if step > total:
            step = total
        print("\r",end="")

        if ratio :
            progress = str(step_inp)+"/"+str(total_inp)
        else :
            progress = '{:.4g} %'.format(step_inp/total_inp*100)

        time_text = ""
        if time>1.0e-10 :
            time_text = "("+time_display(time)+")"

        if step_inp/total_inp > 0.75 :
            left_text = progress+" "+time_text+" "+process_name+" "
            rght_text = ""
        elif step_inp/total_inp > 0.5:
            left_text = time_text+" "+process_name+" "
            rght_text = " "+progress
        elif step_inp/total_inp > 0.25:
            left_text = process_name+" "
            rght_text = " "+progress+" "+time_text
        else:
            left_text = ""
            rght_text = " "+process_name+" "+progress+" "+time_text

        if len(left_text) > 2*step :
            left_text = left_text[(len(left_text)-2*step):]

        styled_left_text = "\u001b[1;37;4"+color+"m"+left_text+"\u001b[0;0m"

        if 2*step-len(left_text)+len(left_text)+len(rght_text) > 2*total :
            rght_text = rght_text[:(2*total-2*step)]
        styled_rght_text = "\u001b[1;3"+color+";47m"+rght_text+"\u001b[0;0m"

        filled_bar = "\u001b[0;;4"+color+"m \u001b[0;0m"
        blank_bar = "\u001b[0;;47m \u001b[0;0m"


        n_filled = 2*step-len(left_text)
        n_blank  = 2*total-n_filled-len(left_text)-len(rght_text)

        total = n_filled+len(left_text)+len(rght_text)+n_blank

        #print("   progress: ",end="")
        print("   ",end="")
        for i in range(n_filled):
            print(filled_bar,end="")
        
        print(styled_left_text,end="")
        print(styled_rght_text,end="")

        for i in range(n_blank):
            print(blank_bar,end="")
        return step_inp+1

def clear_progress():

    if progress_bar_enabled:
        print("\r",end="")
        for i in range(90):
            print(" ",end="")
        print("\r",end="")
        return 1

def progress_space():
    if progress_bar_enabled:
        print()

def tab_up():
    if progress_bar_enabled:
        sys.stdout.write("\033[F")

def time_display(time_seconds):

    if time_seconds < 60 :
        ret = '{:.4g}'.format(time_seconds)+" s"
    elif time_seconds < 60*60 :
        minutes = int(np.floor(time_seconds/60))
        seconds = time_seconds-60*minutes
        ret = str(minutes)+" m "+'{:.4g}'.format(seconds)+" s"
    elif time_seconds < 60*60*24 :
        hours = int(np.floor(time_seconds/60/60))
        minutes = int(np.floor(time_seconds/60-60*hours))
        seconds = time_seconds-60*minutes-60*60*hours
        ret = str(hours)+" hr "+str(minutes)+" m "+'{:.4g}'.format(seconds)+" s"
    else:
        days = int(np.floor(time_seconds/60/60/24))
        hours = int(np.floor(time_seconds/60/60-24*days))
        minutes = int(np.floor(time_seconds/60-60*hours-60*24*days))
        seconds = time_seconds-60*minutes-60*60*hours-60*60*24*days
        ret = str(days)+" d "+str(hours)+" hr "+str(minutes)+" m "+'{:.4g}'.format(seconds)+" s"

    return ret

def clean_format(number):
    order = -int(np.log10(numer_display_cutoff))
    if np.abs(np.real(number))>numer_cutoff and np.abs(np.imag(number))<numer_cutoff:
        return round(np.real(number),order)
    elif np.abs(np.real(number))<numer_cutoff and np.abs(np.imag(number))>numer_cutoff:
        return round(np.imag(number),order)*complex(0,1)
    elif np.abs(np.real(number))<numer_cutoff and np.abs(np.imag(number))<numer_cutoff:
        return 0
    else:
        return round(np.real(number),order)+round(np.imag(number),order)*complex(0,1)

def memory_display(raw_memory):
    if raw_memory<2**10:
        return '{:.4g}'.format(raw_memory)+" B"
    elif raw_memory<2**20:
        return '{:.4g}'.format(raw_memory/(2**10))+" KiB"
    elif raw_memory<2**30:
        return '{:.4g}'.format(raw_memory/(2**20))+" MiB"
    elif raw_memory<2**40:
        return '{:.4g}'.format(raw_memory/(2**30))+" GiB"
    else:
        return '{:.4g}'.format(raw_memory/(2**40))+" TiB"

def current_memory_display():
    return memory_display(tracemalloc.get_traced_memory()[0])+"/"+memory_display(tracemalloc.get_traced_memory()[1])

def getsize(shape):
    ret = 1
    for d in shape:
        ret *= d
    return ret

def none(shape,initial=None):
    ret=None
    for i in range(len(shape)):
        temp = []
        for j in range(shape[len(shape)-i-1]):
            if isinstance(ret,list):
                temp += [copy.deepcopy(ret)]
            else:
                temp += [None]
        ret = copy.deepcopy(temp)
    ret = np.array(ret,dtype=object)
    it = np.nditer(ret, flags=['multi_index','refs_ok'])
    for val in it:
        ret[it.multi_index]=copy.deepcopy(initial)
    return ret

####################################################
##              Block Grassmann Array             ##
####################################################

class block:

    __array_priority__ = 1000000
    

    def __init__(self, data=None):
        
        #import from dense only
        dat = data.copy()
        if type(data)==sparse:
            dat = dense(data)
        dat = dat.force_encoder('parity-preserving')

        datatype = data.data.dtype

        # create a Z2-blocking structure
        # cell is a larger matrix containing smaller blocks
        grading = 2

        # pick up the shape of the fermionic sector
        fshape = []
        for i,d in enumerate(dat.shape):
            if dat.statistics[i] in fermi_type:
                fshape += [d]
        fshape = tuple(fshape)

        cell_shape = tuple([grading]*len(fshape))

        # compute the base block shape
        base_block_shape = []
        for i,d in enumerate(dat.shape):
            if dat.statistics[i] in fermi_type:
                base_block_shape += [max(1,int(d/grading))]
            else:
                base_block_shape += [d]
        base_block_shape = tuple(base_block_shape)

        # initialize the cells and sgn_cells =============================================
        cells = none(cell_shape,np.zeros(base_block_shape,dtype=datatype))

        # create a blank list
        sgn = []
        for p in [0,1]:
            sgn_p = []
            for d,stat in zip(dat.shape,dat.statistics):
                if stat in fermi_type:
                    dim = max(int(d/grading),1)
                else:
                    dim = d
                sgn_p+=[np.zeros([dim],dtype=int)]
            sgn += [sgn_p]

        for axis,(d,stat) in enumerate(zip(dat.shape,dat.statistics)):
            for i in range(d):
                if stat in bose_type:
                    sgn[0][axis][i]=1
                    sgn[1][axis][i]=1
                else:
                    block = i%2
                    sub_i = int(math.floor(i/2))
                    sign = param.sgn(param.encoder(i))
                    sgn[block][axis][sub_i]=sign

        # Consider using the fact that all odd axis use the same sign factor (also the even axis)

        # now iterate through the dat tensor and separate it into blocks
        it = np.nditer(dat, flags=['multi_index'])
        for val in it:
            
            #total index
            index = it.multi_index

            #fermionic index
            findex = []
            for i,d in enumerate(index):
                if dat.statistics[i] in fermi_type:
                    findex += [d]

            #determine the block number
            blocknum = [ elem%grading for elem in findex ]
            blocknum = tuple(blocknum)

            blockelem = [ int((d-d%grading)/grading) for d in findex ]

            #determine the block matrix elements
            blockelem = []
            for i,d in enumerate(index):
                if dat.statistics[i] in fermi_type:
                    blockelem += [int((d-d%grading)/grading)]
                else:
                    blockelem += [d]
            blockelem = tuple(blockelem)
            cells[blocknum][blockelem] += val.item()

            # compute the sigma sign factor
            #for i,dim in enumerate(blockelem):
            #    σ = 1
            #    if dat.statistics[i] in fermi_type:
            #        canonical_index = param.encoder(index[i])
            #        σ = param.sgn(canonical_index)
            #    sgn_cells[blocknum][i][dim]=σ

        self.data = cells.copy()
        self.sgn  = copy.deepcopy(sgn)
        self.statistics = dat.statistics
        self.format = dat.format
        self.shape = dat.shape # not the actual shape, but the 'physical' shape
        self.marked_as_joined = False

    def display(self,name=None,indent_size=0):
        indent = ""
        for i in range(indent_size):
            indent+=" "

        mem = 0

        it = np.nditer(self.data, flags=['multi_index','refs_ok'])
        for val in it:
            block = it.multi_index
            dat = self.data[block]
            mem += sys.getsizeof(dat)

        print()
        if name != None:
            print(indent+"            name:",name)
        print(indent+"      array type: block")
        print(indent+"     total shape:", self.shape)
        print(indent+" effective shape:", self.effective_shape)
        print(indent+"      even shape:", self.even_shape)
        print(indent+"       odd shape:", self.odd_shape)
        print(indent+"      statistics:", self.statistics)
        print(indent+"          format:", self.format)
        print(indent+"         encoder: block")
        print(indent+"          memory:",memory_display(mem+sys.getsizeof(self)))
        print(indent+"            norm:",self.norm)
        print()

        print(" :::::::::::::::::::::: Sign factor :::::::::::::::::::::")
        print("          Even sector:")
        for _ in self.sgn[0]:
            print("            ",_)
        print("           Odd sector:")
        for _ in self.sgn[1]:
            print("            ",_)

        print(" ::::::::::::::::::: Block information ::::::::::::::::::")

        i=0
        it = np.nditer(self.data, flags=['multi_index','refs_ok'])
        for val in it:
            block = it.multi_index
            dat = self.data[block]
            
            if(np.linalg.norm(dat)>numer_display_cutoff):
                if i>0:
                    print(" --------------------------------------------------------")
                print()
                block_symbol = [ "even" if i==0 else "odd" for i in block ]
                block_symbol = "("+",".join(block_symbol)+")"
                print("  Block number:",block_symbol,"\n")
                print("   Block shape:",dat.shape,"\n")
                print("  Block tensor:\n",dat,"\n")
                i+=1
        print("==========================================================")

    def info(self,name=None,indent_size=0):

        indent = ""
        for i in range(indent_size):
            indent+=" "

        mem = 0

        it = np.nditer(self.data, flags=['multi_index','refs_ok'])
        for val in it:
            block = it.multi_index
            dat = self.data[block]
            mem += sys.getsizeof(dat)

        print()
        if name != None:
            print(indent+"            name:",name)
        print(indent+"      array type: block")
        print(indent+"     total shape:", self.shape)
        print(indent+" effective shape:", self.effective_shape)
        print(indent+"      even shape:", self.even_shape)
        print(indent+"       odd shape:", self.odd_shape)
        print(indent+"      statistics:", self.statistics)
        print(indent+"          format:", self.format)
        print(indent+"         encoder: block")
        print(indent+"          memory:",memory_display(mem+sys.getsizeof(self)))
        print(indent+"            norm:",self.norm)
        print()

    @property
    def norm(self):
        ret = 0
        it = np.nditer(self.data, flags=['multi_index','refs_ok'])
        for val in it:
            ret += np.linalg.norm(val.item())**2
        return np.sqrt(ret)

    @property
    def effective_shape(self):
        ret = np.array(self.even_shape)+np.array(self.odd_shape)
        for i in range(self.ndim):
            if self.statistics[i] in bose_type:
                ret[i] = self.shape[i]
        return make_tuple(ret)

    @property
    def even_shape(self):
        ret = [ len(v) if v[0]!=0 else 0 for v in self.sgn[0] ]
        return make_tuple(ret)

    @property
    def odd_shape(self):
        ret = [ len(v) if v[0]!=0 else 0 for v in self.sgn[1] ]
        return make_tuple(ret)

    @property
    def ndim(self):
        return len(self.shape)
    
    @property
    def dtype(self):
        it = np.nditer(self.data, flags=['multi_index','refs_ok'])
        for val in it:
            #total index
            return val.item().dtype

    def copy(self):

        ret = block( random(shape=(2,2),statistics=(1,1)) )

        ret.data = copy.deepcopy(self.data)
        ret.sgn  = copy.deepcopy(self.sgn)
        ret.statistics = self.statistics
        ret.format = self.format
        ret.shape = self.shape # not the actual shape, but the 'physical' shape
        ret.marked_as_joined = self.marked_as_joined

        return ret

    def __add__(self, other):
        if(type(self) != type(other)
                or self.even_shape!=other.even_shape
                or self.odd_shape!=other.odd_shape
                or self.shape!=other.shape
                or self.statistics!=other.statistics
                or self.format!=other.format):
            error("Error[block.+]: Inconsistent object properties")
        
        if self.marked_as_joined or other.marked_as_joined :
            error("Error[block.+]: You cannot add a joined object to ther object.")

        ret = self.copy()
        it = np.nditer(self.data, flags=['multi_index','refs_ok'])
        for val in it:
            block = it.multi_index
            ret.data[block] = self.data[block]+other.data[block]
        return ret
        
    def __mul__(self, other):
        if not np.isscalar(other):
            error("Error[block.*]: Only scalar multiplication is allowed.")

        ret = self.copy()
        it = np.nditer(self.data, flags=['multi_index','refs_ok'])
        for val in it:
            block = it.multi_index
            ret.data[block] = self.data[block]*other

        return ret
        
    def __truediv__(self, other):
        
        if np.isscalar(other):
            # dividing the scalar
            ret = self.copy()
            ret.data = self.data*(1.0/other)
            return ret
        else:
            error("Error[block./]: Only scalar division is allowed.")
            exit()

    def __pos__(self):                 # do not modify this
        return self
    def __neg__(self):                 # do not modify this
        return self*(-1)
    def __radd__(self, other):         # do not modify this
        return self+other
    def __sub__(self, other):          # do not modify this
        return self+(-1)*other
    def __rsub__(self, other):         # do not modify this
        return other+(-1)*self
        
    def __rmul__(self, other):
        return self*other
        
    def switch_format(self):

        def mult_along_axis(A, v, axis):
            # see [https://stackoverflow.com/questions/30031828/multiply-numpy-ndarray-with-1d-array-along-a-given-axis]
            # for explanation
            # ensure we're working with Numpy arrays
            A = np.array(A)
            v = np.array(v)
            shape = np.swapaxes(A, A.ndim-1, axis).shape
            v_brc = np.broadcast_to(v, shape)
            v_brc = np.swapaxes(v_brc, A.ndim-1, axis)
            return A * v_brc

        #print()
        #print("statistics = ",self.statistics)
        ret = self.copy()
        it = np.nditer(self.data, flags=['multi_index','refs_ok'])
        for val in it:
            #total index
            block = it.multi_index
            #print("block = ",block)
            val_replace = val.item().copy()
            fd = 0
            for d in range(self.ndim):
                if self.statistics[d] == -1:
                    #print(val_replace.shape,self.statistics[d],self.sgn[block[d]][d].shape,d)
                    val_replace = mult_along_axis(val_replace,self.sgn[block[fd]][d],d)
                if self.statistics[d] in fermi_type:
                    fd += 1
            ret.data[block] = val_replace.copy()

        if self.format == "standard":
            ret.format = "matrix"
        else:
            ret.format = "standard"

        return ret

    def force_format(self,format):
        if self.format == format:
            return self.copy()
        else:
            return self.switch_format()

    def todense(self,encoder="canonical",skip_joined_check=False):
        return todense(self,encoder,skip_joined_check)

    def tosparse(self,encoder="canonical"):
        return sparse(todense(self,encoder))

    def join_legs(self,string_inp,final_stat):
        return join_legs_block(self,string_inp,final_stat)

    def split_legs(self,string_inp,final_stat,final_shape,final_even_shape,final_odd_shape):
        return split_legs_block(self,string_inp,final_stat,final_shape,final_even_shape,final_odd_shape)

    def hconjugate(self,string,save_memory=False):
        return hconjugate_block(self,string,save_memory)

    def op_hconjugate(self):
        return op_hconjugate(self)

    def svd(self,string,cutoff=None,save_memory=False):
        return svd_block(self,string,cutoff,save_memory)

    def eig(self,string,cutoff=None,save_memory=False):
        return eig_block(self,string,cutoff,save_memory)

def zero_block(effective_shape,statistics,format='standard',dtype=float):
    shape = [ 2**int(np.ceil(np.log2(dim))) if stat in fermi_type else dim for dim,stat in zip(effective_shape,statistics) ]
    oeshape = [ max(1,int(round(dim/2))) if stat in fermi_type else dim for dim,stat in zip(effective_shape,statistics) ]
    ret = block(dense(np.zeros(shape,dtype=dtype),statistics=statistics,format=format))
    it = np.nditer(ret.data, flags=['multi_index','refs_ok'])
    for val in it:
        #total index
        blk = it.multi_index
        ret.data[blk] = np.zeros(oeshape,dtype=dtype)
    new_esgn = [ sgnv[:oeshape[i]] for i,sgnv in enumerate(ret.sgn[0]) ]
    new_osgn = [ sgnv[:oeshape[i]] for i,sgnv in enumerate(ret.sgn[1]) ]
    ret.sgn = [new_esgn,new_osgn]
    return ret

def zero_block_eo(even_shape,odd_shape,statistics,format='standard',dtype=float):
    effective_shape = [ edim+odim if stat in fermi_type else edim for edim,odim,stat in zip(even_shape,odd_shape,statistics) ]
    shape = [ 2**int(np.ceil(np.log2(dim))) if stat in fermi_type else dim for dim,stat in zip(effective_shape,statistics) ]
    ret = block(dense(np.zeros(shape,dtype=dtype),statistics=statistics,format=format))
    it = np.nditer(ret.data, flags=['multi_index','refs_ok'])
    for val in it:
        #total index
        blk = it.multi_index
        blk_shape = []
        fi = 0
        for i in range(len(even_shape)):
            if statistics[i] in fermi_type:
                if blk[fi] == 0:
                    blk_shape += [even_shape[i]]
                else:
                    blk_shape += [odd_shape[i]]
                fi += 1
            else:
                blk_shape += [odd_shape[i]]
        ret.data[blk] = np.zeros(blk_shape,dtype=dtype)
    new_esgn = [ sgnv[:even_shape[i]] for i,sgnv in enumerate(ret.sgn[0]) ]
    new_osgn = [ sgnv[:odd_shape[i]] for i,sgnv in enumerate(ret.sgn[1]) ]
    ret.sgn = [new_esgn,new_osgn]
    return ret

def random_block(effective_shape,statistics,format='standard',dtype=float,skip_trimming=False):
    shape = [ 2**int(np.ceil(np.log2(dim))) if stat in fermi_type else dim for dim,stat in zip(effective_shape,statistics) ]
    oeshape = [ max(1,int(round(dim/2))) if stat in fermi_type else dim for dim,stat in zip(effective_shape,statistics) ]
    ret = block(random(shape=shape,statistics=statistics,dtype=dtype,skip_trimming=skip_trimming,format=format))
    it = np.nditer(ret.data, flags=['multi_index','refs_ok'])
    for val in it:
        #total index
        blk = it.multi_index
        ret.data[blk] = np.random.rand(*oeshape)
        if dtype == complex:
            ret.data[blk] = complex(1,0)*ret.data[blk] + complex(0,1)*np.random.rand(*oeshape)
    new_esgn = [ sgnv[:oeshape[i]] for i,sgnv in enumerate(ret.sgn[0]) ]
    new_osgn = [ sgnv[:oeshape[i]] for i,sgnv in enumerate(ret.sgn[1]) ]
    ret.sgn = [new_esgn,new_osgn]
    return ret

def todense(obj,encoder,skip_joined_check=False):

    if type(obj)==sparse or type(obj)==dense :
        return dense(obj)

    if not skip_joined_check:
        if obj.marked_as_joined :
            error("Error[todense]: Split the legs first!")

    ret = dense(np.zeros(obj.shape,dtype=obj.dtype),statistics=obj.statistics,encoder='parity-preserving',format=obj.format)

    it = np.nditer(obj.data, flags=['multi_index','refs_ok'])
    for _ in it:
        block = it.multi_index
        dat = obj.data[block]
        #print("block:",block)
        sub_it = np.nditer(dat, flags=['multi_index','refs_ok'])
        for val in sub_it:
            coords = sub_it.multi_index
            fcoords = [ c for i,c in enumerate(coords) if obj.statistics[i] in fermi_type ]
            new_coords = []
            fi = 0
            for i,c in enumerate(coords):
                if obj.statistics[i] in fermi_type:
                    if block[fi]==0:
                        new_coords += [2*c]
                    else:
                        new_coords += [2*c+1]
                    fi += 1
                else:
                    new_coords += [c]

            new_coords = tuple(new_coords)
            #print("  M",coords,"= M'",new_coords,"=",val)

            ret.data[new_coords] = val

    ret = ret.force_encoder(encoder)

    return ret

def incorrect_sign_size(InpObj):
    
    obj = InpObj.copy()

    it = np.nditer(obj.data, flags=['multi_index','refs_ok'])
    for _ in it:

        edim_sgn = [ len(sgnv) for sgnv,stat in zip(obj.sgn[0],obj.statistics) if stat in fermi_type]
        odim_sgn = [ len(sgnv) for sgnv,stat in zip(obj.sgn[1],obj.statistics) if stat in fermi_type]
        dim_sgn = [edim_sgn,odim_sgn]

        block = it.multi_index
        shape_block = [ dim for dim,stat in zip(_.item().shape,obj.statistics) if stat in fermi_type]
        shape_sgn = make_tuple([ dim_sgn[p][i] for i,p in enumerate(block) ])

        shape_sgn = make_list(shape_sgn)
        shape_block = make_list(shape_block)

        if shape_sgn!=shape_block:

            # no need to do anything if the inconsistent part is of dimension 1
            is_ok = True
            for s,b in zip(shape_sgn,shape_block):
                if not (s==b or (s==1 and b==0)):
                    is_ok = False
                    break
            if is_ok :
                return False

            return True

    return False

def correct_sign_size(InpObj):
    
    obj = InpObj.copy()

    it = np.nditer(obj.data, flags=['multi_index','refs_ok'])
    for _ in it:

        edim_sgn = [ len(sgnv) for sgnv,stat in zip(obj.sgn[0],obj.statistics) if stat in fermi_type]
        odim_sgn = [ len(sgnv) for sgnv,stat in zip(obj.sgn[1],obj.statistics) if stat in fermi_type]
        dim_sgn = [edim_sgn,odim_sgn]

        block = it.multi_index
        shape_block = [ dim for dim,stat in zip(_.item().shape,obj.statistics) if stat in fermi_type]
        shape_sgn = make_tuple([ dim_sgn[p][i] for i,p in enumerate(block) ])

        shape_sgn = make_list(shape_sgn)
        shape_block = make_list(shape_block)

        if shape_sgn!=shape_block:

            # no need to do anything if the inconsistent part is of dimension 1
            is_ok = True
            for s,b in zip(shape_sgn,shape_block):
                if not (s==b or (s==1 and b==0)):
                    is_ok = False
                    break
            if is_ok :
                return obj

            # get the block number but artifically add the bosonic one as well
            full_block = []
            fi = 0
            for i,stat in enumerate(obj.statistics):
                if stat in fermi_type:
                    full_block += [block[fi]]
                    fi += 1
                else:
                    full_block += [0]

            #modify the length
            this_shape = _.item().shape
            for i,p in enumerate(full_block):
                obj.sgn[p][i] = obj.sgn[p][i][:this_shape[i]]

    return obj

####################################################
##             Densed Grassmann Array             ##
####################################################

class dense:

    __array_priority__ = 1000000
    
    def __init__(self, data=None, encoder = "canonical", format = "standard", statistics=None):
    
        #copy dense properties
        self.data = None
        self.statistics = None
        self.format = format
        self.encoder = encoder
    
        default = True
    
        if(encoder not in encoder_type):
            error("Error[dense]: Unknown encoder.")
            
        if(format not in format_type):
            error("Error[dense]: Unknown format.")
            
        if(type(data)==np.array or type(data)==np.ndarray or type(data)==list):
            self.data = np.array(data)
            default = False
        elif(type(data)==sp.COO):
            self.data  = data.todense()
            default = False
        elif(type(data)==sparse):
            #copy dense properties
            self.data = data.data.todense()
            self.statistics = data.statistics
            self.format = data.format
            self.encoder = data.encoder
            default = False
        elif(type(data)==dense):
            #copy dense properties
            self.data = data.data.copy()
            self.statistics = data.statistics
            self.format = data.format
            self.encoder = data.encoder
            default = False
        elif(np.isscalar(data)):
            self.data = np.array(list([data]))
            default = False
        elif(data==None):
            "nothing to see here"
        else:
            error("Error[dense]: Invalid initialized data.")
            
        
        if statistics != None:
            self.statistics = make_tuple(statistics)
            
        if not default and not skip_power_of_two_check:
            for i,dim in enumerate(self.data.shape):
                if self.statistics[i] in fermi_type and dim != int(2**math.floor(np.log2(dim))):
                    error("Error[dense]: Some of the fermionic tensor shapes are not a power of two."
                        +"\n              Have you added the <statistics> argument when calling this function?")
                
    def __getitem__(self, index):
        return self.data[index]
    
    def __setitem__(self, index, value):
        self.data[index] = value
        
    @property
    def shape(self):
        return self.data.shape

    @property
    def size(self):
        return self.data.size

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def norm(self):
        array_form = self.data
        return np.linalg.norm(array_form)

    @property
    def nnz(self):
        iterator = np.nditer(self, flags=['multi_index'])
        n = 0
        for element in iterator:
            coords = iterator.multi_index
            if(np.abs(element.item())>numer_display_cutoff):
                n+=1
        return n

    def display(self,name=None,indent_size=0):

        indent = ""
        for i in range(indent_size):
            indent+=" "

        print()
        if name != None:
            print(indent+"        name:",name)
        print(indent+"  array type: dense")
        print(indent+"       shape:",self.shape)
        print(indent+"     density:",self.nnz,"/",self.size,"~",self.nnz/self.size*100,"%")
        print(indent+"  statistics:",self.statistics)
        print(indent+"      format:",self.format)
        print(indent+"     encoder:",self.encoder)
        print(indent+"      memory:",memory_display(sys.getsizeof(self.data)+sys.getsizeof(self)))
        print(indent+"        norm:",self.norm)
        print(indent+"     entries:")
        print(indent+" [coords]  [values]")
        iterator = np.nditer(self, flags=['multi_index'])
        for element in iterator:
            coords = iterator.multi_index
            if(np.abs(element.item())>numer_display_cutoff):
                print(indent+"\t",coords,clean_format(element.item()))
        print()

    def info(self,name=None,indent_size=0):

        indent = ""
        for i in range(indent_size):
            indent+=" "

        print()
        if name != None:
            print(indent+"        name:",name)
        print(indent+"  array type: dense")
        print(indent+"       shape:",self.shape)
        print(indent+"     density:",self.nnz,"/",self.size,"~",self.nnz/self.size*100,"%")
        print(indent+"  statistics:",self.statistics)
        print(indent+"      format:",self.format)
        print(indent+"     encoder:",self.encoder)
        print(indent+"      memory:",memory_display(sys.getsizeof(self.data)+sys.getsizeof(self)))
        print(indent+"        norm:",self.norm)
        print()

    def copy(self):
        #copy dense properties
        ret = dense()
        ret.data = self.data.copy()
        ret.statistics = self.statistics
        ret.format = self.format
        ret.encoder = self.encoder
        return ret
    
    def __add__(self, other):
        if(self.shape!=other.shape
            or self.statistics!=other.statistics
             or self.format!=other.format
              or self.encoder!=other.encoder):
            error("Error[dense.+]: Inconsistent object properties")
            
        ret = self.copy()
        ret.data = ret.data+other.data
        return ret
        
    def __mul__(self, other):
        if not np.isscalar(other):
            error("Error[dense.*]: Only scalar multiplication is allowed.")
        ret = self.copy()
        ret.data = ret.data*other
        return ret
        
    def __truediv__(self, other):
        
        if np.isscalar(other):
            # dividing the scalar
            ret = self.copy()
            ret.data = self.data/other
            return ret
        else:
            error("Error[dense./]: Only scalar division is allowed.")
            exit()

    def __pos__(self):                 # do not modify this
        return self
    def __neg__(self):                 # do not modify this
        return self*(-1)
    def __radd__(self, other):         # do not modify this
        return self+other
    def __sub__(self, other):          # do not modify this
        return self+(-1)*other
    def __rsub__(self, other):         # do not modify this
        return other+(-1)*self
        
    def __rmul__(self, other):
        return self*other
        
    def __len__(self):
        return len(self.data)
    
    def __str__(self):
        return str(self.data)
    
    def __repr__(self):
        return repr(self.data)
        
    def switch_format(self,save_memory=False):
        # multiply sign factor sigma[i] to every conjugated indices i

        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        # WORK IN THE FOLLOWING CONDITIONS      :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        # (WILL CONVERT AUTOMATICALLY)          :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #   - canonical encoder                 :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

        if save_memory :
            ret = self
        else:
            ret = self.copy()
        if(self.encoder=='parity-preserving'):
            ret = ret.switch_encoder(save_memory=True)
        to_calc = []
        for i in range(self.ndim):
            to_calc += (ret.statistics[i]==-1),
            if(ret.statistics[i]==hybrid_symbol):
                error("Error[switch_format]: Cannot switch format with a hybrid index.\n                      Split them into bosonic and fermionic ones first!")
        
        k=0
        kmax = ret.ndim
        s0 = time.time()
        s00 = s0
        progress_space()

        dat = ret.data
        for i in range(ret.ndim):
            if not to_calc[i] :
                continue

            ss = [None] * ret.ndim    # slice object
            ss[i] = slice(None)       # sum over axis i
            ss = make_tuple(ss)
            d = ret.shape[i]
            v = np.array([param.sgn(j) for j in range(d)])
            dat = dat*v[ss]

            if time.time()-s0 > 2 :
                show_progress(i,kmax,process_name = "switch_format",ratio = False,color="blue",time=time.time()-s00)
                s0 = time.time()

        clear_progress()
        tab_up()

        ret.data = dat

        if(ret.format=='standard'):
            ret.format = 'matrix'
        elif(ret.format=='matrix'):
            ret.format = 'standard'
        else:
            error("Error[switch_format]: unknown format")
            
        if(self.encoder=='parity-preserving'):
            ret = ret.switch_encoder(save_memory=True)
        return ret

    def switch_encoder(self,save_memory=False):

        if save_memory :
            ret = self
        else:
            ret = self.copy()

        k=0
        kmax = ret.ndim
        s0 = time.time()
        s00 = s0
        progress_space()

        dat = ret.data
        for axis in range(ret.ndim):
            d = ret.shape[axis]
            if ret.statistics[axis] in fermi_type :
                dat = dat.take(indices=np.array([param.encoder(i) for i in range(d)]),axis=axis)

            if time.time()-s0 > 2 :
                show_progress(axis,kmax,process_name = "switch_encoder",ratio = False,color="cyan",time=time.time()-s00)
                s0 = time.time()
            
        clear_progress()
        tab_up()

        ret.data = dat

        if(ret.encoder=='canonical'):
            ret.encoder='parity-preserving'
        else:
            ret.encoder='canonical'
        return ret

    def force_encoder(self,target="canonical"):
        if target not in encoder_type:
            error("Error[dense.force_encoder]: Unrecognized target encoder.")
        if target != self.encoder :
            return self.switch_encoder(save_memory=True)
        else :
            return self.copy()

    def force_format(self,target="standard"):
        if target not in format_type:
            error("Error[dense.force_format]: Unrecognized target format.")
        if target != self.format :
            return self.switch_format(save_memory=True)
        else :
            return self.copy()

    def join_legs(self,string_inp,make_format='standard',intermediate_stat=None,save_memory=False):
        return join_legs(self,string_inp,make_format,intermediate_stat,save_memory)

    def split_legs(self,string_inp,final_stat,final_shape,intermediate_stat=None,save_memory=False):
        return split_legs(self,string_inp,final_stat,final_shape,intermediate_stat,save_memory)

    def hconjugate(self,input_string,save_memory=False):
        return hconjugate(self,input_string,save_memory)

    def op_hconjugate(self):
        return op_hconjugate(self)

    def svd(self,string_inp,cutoff=None,save_memory=False):
        return svd(self,string_inp,cutoff,save_memory)

    def eig(self,string_inp,cutoff=None,debug_mode=False,save_memory=False):
        return eig(self,string_inp,cutoff,debug_mode,save_memory)

    def toblock(self):
        return block(self)

####################################################
##            Sparse Grassmann arrays             ##
####################################################

class sparse:

    __array_priority__ = 1000000
    
    def __init__(self, data=None, encoder = "canonical", format = "standard", statistics = None):
    
        #copy sparse properties
        self.data = None
        self.statistics = None
        self.format = format
        self.encoder = encoder

        default = True
    
        if(encoder not in encoder_type):
            error("Error[sparse]: Unknown encoder.")
            
        if(format not in format_type):
            error("Error[sparse]: Unknown format.")
            
        if(type(data)==np.array or type(data)==np.ndarray or type(data)==list):
            self.data  = sp.COO.from_numpy(np.array(data))
            default = False
        elif(type(data)==sp.COO):
            self.data  = data.copy()
            default = False
        elif(type(data)==dense):
            #copy sparse properties
            self.data  = sp.COO.from_numpy(data.data)
            self.statistics = data.statistics
            self.format = data.format
            self.encoder = data.encoder
            default = False
        elif(type(data)==sparse):
            #copy sparse properties
            self.data = data.data.copy()
            self.statistics = data.statistics
            self.format = data.format
            self.encoder = data.encoder
            default = False
        elif data==None:
            "nothing to see here"
        else:
            error("Error[sparse]: Invalid initialized data")
            
        
        if statistics != None:
            self.statistics = make_tuple(statistics)
        
        if not default and not skip_power_of_two_check:
            for i,dim in enumerate(self.data.shape):
                if self.statistics[i] in fermi_type and dim != int(2**math.floor(np.log2(dim))):
                    error("Error[sparse]: Some of the fermionic tensor shapes are not a power of two.\n               Have you added the <statistics> argument when calling this function?")
               
    @property
    def nnz(self):
        return self.data.nnz

    @property
    def shape(self):
        return self.data.shape

    @property
    def size(self):
        return self.data.size

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def coords(self):
        coords_list = []
        for entry in range(self.data.nnz):
            coords = []
            for axis in range(self.data.ndim):
                coords = coords + [self.data.coords[axis][entry]]
            coords_list = coords_list + [tuple(coords)]
        return coords_list

    @property
    def value(self):
        value_list = []
        for entry in range(self.data.nnz):
            value_list = value_list + [self.data.data[entry]]
        return value_list

    @property
    def norm(self):
        array_form = self.data.todense()
        return np.linalg.norm(array_form)

    def display(self, name=None,indent_size=0):

        indent = ""
        for i in range(indent_size):
            indent+=" "

        print()
        if name != None:
            print(indent+"        name:",name)
        print(indent+"  array type: sparse")
        print(indent+"       shape:",self.shape)
        print(indent+"     density:",self.nnz,"/",self.size,"~",self.nnz/self.size*100,"%")
        print(indent+"  statistics:",self.statistics)
        print(indent+"      format:",self.format)
        print(indent+"     encoder:",self.encoder)
        print(indent+"      memory:",memory_display(sys.getsizeof(self.data)+sys.getsizeof(self)))
        print(indent+"        norm:",self.norm)
        print(indent+"     entries:")
        print(indent+" [coords]  [values]")

        C = self.coords
        V = self.value
        for elem in range(self.nnz):
            if(np.abs(V[elem])>numer_display_cutoff):
                print(indent+"\t",C[elem],clean_format(V[elem]))
        print()

    def info(self,name=None,indent_size=0):

        indent = ""
        for i in range(indent_size):
            indent+=" "

        print()
        if name != None:
            print(indent+"        name:",name)
        print(indent+"  array type: sparse")
        print(indent+"       shape:",self.shape)
        print(indent+"     density:",self.nnz,"/",self.size,"~",self.nnz/self.size*100,"%")
        print(indent+"  statistics:",self.statistics)
        print(indent+"      format:",self.format)
        print(indent+"     encoder:",self.encoder)
        print(indent+"      memory:",memory_display(sys.getsizeof(self.data)+sys.getsizeof(self)))
        print(indent+"        norm:",self.norm)
        print()

    def set_value(self,entry,value):
        self.data.data[entry] = value

    def remove_entry(self,entry):
        popped_coords = []
        for axis in range(self.data.ndim):
            popped_coords = popped_coords + [np.delete(self.data.coords[axis],entry)]
        popped_coords = np.array(popped_coords)
        popped_value = np.delete(self.data.data,entry)
        
        self.data.coords = popped_coords
        self.data.data = popped_value

    def append_entry(self,coords,value):
        appended_coords = []
        for axis in range(self.data.ndim):
            appended_coords = appended_coords + [np.append(self.data.coords[axis],coords[axis])]
        appended_coords = np.array(appended_coords)
        appended_value = np.append(self.data.data,value)
        
        self.data.coords = appended_coords
        self.data.data = appended_value

    def remove_zeros(self):
        ret = self.copy()
        zero_entry_list = []
        for entry in range(ret.data.nnz):
            if np.abs(ret.data.data[entry]) < numer_cutoff:
                zero_entry_list = zero_entry_list + [entry]

        for i,zero_entry in enumerate(zero_entry_list):
            ret.remove_entry(zero_entry-i)
        return ret

    def copy(self):
        #copy sparse properties
        ret = sparse()
        ret.data = self.data.copy()
        ret.statistics = self.statistics
        ret.format = self.format
        ret.encoder = self.encoder
        return ret
    
    def __add__(self, other):
        if(self.shape!=other.shape
            or self.statistics!=other.statistics
             or self.format!=other.format
              or self.encoder!=other.encoder):
            error("Error[sparse.+]: Inconsistent object properties")
            
        ret = self.copy()
        ret.data = ret.data+other.data
        return ret
        
    def __mul__(self, other):
        if not np.isscalar(other):
            error("Error[sparse.*]: Only scalar multiplication is allowed.")
        ret = self.copy()
        ret.data = ret.data*other
        return ret
        
    def __truediv__(self, other):
        
        if np.isscalar(other):
            # dividing the scalar
            ret = self.copy()
            ret.data = self.data/other
            return ret
        else:
            error("Error[sparse./]: Only scalar division is allowed.")
            exit()

    def __pos__(self):                 # do not modify this
        return self
    def __neg__(self):                 # do not modify this
        return self*(-1)
    def __radd__(self, other):         # do not modify this
        return self+other
    def __sub__(self, other):          # do not modify this
        return self+(-1)*other
    def __rsub__(self, other):         # do not modify this
        return other+(-1)*self
        
    def __rmul__(self, other):
        return self*other
        
    def __str__(self):
        return str(self.data)
    
    def __repr__(self):
        return repr(self.data)

    def switch_format(self,save_memory=False):
        return sparse(dense(self).switch_format(save_memory=save_memory))

    def switch_encoder(self,save_memory=False):
        return sparse(dense(self).switch_encoder(save_memory=save_memory))

    def force_encoder(self,target="canonical"):
        if target not in encoder_type:
            error("Error[sparse.force_encoder]: Unrecognized target encoder.")
        if target != self.encoder :
            return self.switch_encoder()
        else :
            return self.copy()

    def force_format(self,target="standard"):
        if target not in format_type:
            error("Error[sparse.force_format]: Unrecognized target format.")
        if target != self.format :
            return self.switch_format()
        else :
            return self.copy()

    def join_legs(self,string_inp,make_format='standard',intermediate_stat=None,save_memory=False):
        return join_legs(self,string_inp,make_format,intermediate_stat,save_memory)

    def split_legs(self,string_inp,final_stat,final_shape,intermediate_stat=None,save_memory=False):
        return split_legs(self,string_inp,final_stat,final_shape,intermediate_stat,save_memory)

    def hconjugate(self,input_string,save_memory=False):
        return hconjugate(self,input_string,save_memory)

    def op_hconjugate(self):
        return op_hconjugate(self)

    def svd(self,string_inp,cutoff=None,save_memory=False):
        return svd(self,string_inp,cutoff,save_memory)

    def eig(self,string_inp,cutoff=None,debug_mode=False,save_memory=False):
        return eig(self,string_inp,cutoff,debug_mode,save_memory)

    def toblock(self):
        return block(self)

####################################################
##       Parity Calculation (internal tools)      ##
####################################################

def absolute_sign(object_set, parity):
    """
    Compute the absolute sign of an object_set, assuming that some elements always commute with every other element.
    
    Parameters:
    object_set (list[int]): A list object.
    parity (list[int]): A list of object's grassmann parity.

    Returns:
    int: 1 if the permutation is even, -1 if the permutation is odd.
    """
    noncommutative_elements = [x for i,x in enumerate(object_set)
                                if parity[i]%2==1]
    inversions = 0
    for i in range(len(noncommutative_elements)):
        for j in range(i+1, len(noncommutative_elements)):
            if noncommutative_elements[i] > noncommutative_elements[j]:
                inversions += 1
    return (-1)**inversions

def relative_sign_int(object_set1, object_set2, parity1):
    """
    Compute the relative sign from permuting object_set1 to object_set2 with the parity given by parity1 of the list object_set1
    
    Parameters:
    object_set1 (list): first list
    object_set2 (list): target list
    parity1 (list): Grassmann parity of the first list
    
    Returns:
    int: relative parity of the permutation
    """
    
    def permute_c(a, b, c):
        """
        Permute elements of c according to the permutation that maps a to b.
        """
        x = list(np.argsort(a))
        xc = [ c[i] for i in x ]
        bxc = [ xc[i] for i in b ]
        return bxc
    def get_noncommutative(object_set, parity):
        """
        Return a list of noncommutative elements
        """
        noncommutative_elements = [x for i,x in enumerate(object_set)
                                    if parity[i]%2==1]
        return noncommutative_elements
    # get parity of object_set2
    parity2 = permute_c(object_set1, object_set2, parity1)
    
    noncommutative_elements1 = get_noncommutative(object_set1,parity1)
    noncommutative_elements2 = get_noncommutative(object_set2,parity2)
    
    absolute_sign1 = absolute_sign(object_set1, parity1)
    absolute_sign2 = absolute_sign(object_set2, parity2)
    return absolute_sign1*absolute_sign2

def relative_sign_single_input(string, parity1):
    """
    the string version of relative_sign_int
    the sign factor version of single input einsum
    """
    [string1,string2] = list(string.split("->"))
    unique_chars = list(set(string1+string2))
    char_to_int = {char: i for i, char in enumerate(unique_chars)}
    permutation1 = [char_to_int[char] for char in string1]
    permutation2 = [char_to_int[char] for char in string2]
    return relative_sign_int(permutation1, permutation2, parity1)

def relative_sign(string, parity):
    """
    Imagine doing Grassmann version of Einsum.
    This function returns the sign factor of that sum.
    You have to enter the parity of the input indices in [parity]
    The grassmann indices must not be added or removed!
    Use a different function to integrate them out!
    Examples:
    >> relative_sign( "abCdE,aXdYb->XCEY", [0,0,1,0,1,0,1,0,1,0] )
    >> 1
    
    >> relative_sign( "abCdE,aXYb->CXEY", [0,0,1,0,1,0,1,1,0] )
    >> -1
    
    """
    [string_input,string_output] = list(string.split("->"))
    string_list = list(string_input.split(","))
    
    join_string = ""
    for i in range(len(string_list)):
        join_string = join_string + string_list[i]
        
    if(len(join_string)!=len(parity)):
        error("Error[relative_sign]: The number of input list and parity list are not consistent!")
        
        
    #remove the summed indices
    def remove_duplicates(list1, list2):
        new_list1 = []
        new_list2 = []
        for i, val in enumerate(list1):
            if list1.count(val)==1 :
                new_list1.append(val)
                new_list2.append(list2[i])
        return new_list1, new_list2
        
    join_string_list,parity = remove_duplicates(list(join_string), parity)
    join_string = ''.join(join_string_list)+"->"+string_output
    return relative_sign_single_input(join_string, parity)

def reordering(stringa,stringb,mylist):

    rC = []
    for b in stringb:
        
        # locate the location of b in the original string
        index = stringa.index(b)

        # add mylist's element at this index to the list
        rC += [mylist[index]]

    return rC

####################################################
##                     Einsums                    ##
####################################################

def denumerate(string):
    # turn numbered indices to just single indices --------------------------------------------------
    index_char = []
    for c in string :
        if c not in number_character:
            index_char += c,
        else:
            index_char[len(index_char)-1] += c
    unique_tokens = list(dict.fromkeys(index_char))
    to_be_replaced = []
    for item in unique_tokens:
        if len(item) > 1 :
            to_be_replaced += item,
    to_be_replaced = sorted(to_be_replaced, key=len, reverse=True)
    token_list = string
    for i in range(len(to_be_replaced)) :
        new_char = get_char(token_list)
        token_list += new_char
        to_be_replaced[i] = [to_be_replaced[i],new_char]
    for [c1,c2] in to_be_replaced :
        string = string.replace(c1,c2)
    return string

def einsum_ds(*args,format="standard",encoder="canonical",ignore_anticommutation=False,debug_mode=False):

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #                     Important variables and its meanings
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #
    #               subscripts = the whole input text with spaces removed
    #               has_output = True if the <input_string> has the output section
    #             input_string = the left side of the arrow
    #            output_string = the right side of the arrow
    #                  summand = <input_string> without commas
    #                     nobj = number of input objects
    #           obj_index_list = list of individual index strings of input objects
    #                 obj_list = list of input objects
    #               stats_list = input object's stats joined into a single list
    #               shape_list = input object's shape joined into a single list
    #        summed_index_info = [ [char, conjugated_char], [index locations in <input_string>] ]


    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #              Step 0: Input processing
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    # remove spaces ---------------------------------------------------------------------------------
    subscripts = args[0].replace(" ","")
    subscripts = denumerate(subscripts)

    has_output = subscripts.count("->") > 0
    if subscripts.count("->") > 1 :
        error("Error[einsum]: Only one arrow is allowed in the input string!")

    if has_output:
        input_string, output_string = subscripts.split("->")
        if output_string.count(",") > 0 :
            error("Error[einsum]: Output string must not contain commas (',') !")
    else:
        input_string = subscripts

    summand = input_string.replace(",","")

    obj_index_list = input_string.split(",")
    nobj = len(obj_index_list)

    obj_list = make_list(args[1:1+nobj])
    stats_list = sum([ make_list(obj.statistics) for obj in obj_list ],[])
    shape_list = sum([ make_list(obj.shape) for obj in obj_list ],[])

    # force everything to be canonical and standard -------------------------------------------------
    # also force everything to be of the same type

    this_type = type(obj_list[0])
    this_encoder = obj_list[0].encoder
    this_format = obj_list[0].format
    for i in range(len(obj_list)):
        obj_list[i] = this_type(obj_list[i].force_encoder('canonical').force_format('standard')).data

    # get some information about the summed indices -------------------------------------------------
    # [ f=<char>, [index locations in lf], [statistics] ]
    summed_index_info = []

    for i,char in enumerate(summand):

        # skip if bosonic
        if stats_list[i] in bose_type:
            continue

        lf =  input_string.count(char)
        if has_output :
            rf = output_string.count(char)
            if lf>2 or rf>1 or (lf+rf)%2==1 :
                if lf>2 :
                    reason = "lf>2"
                elif rf>1 :
                    reason = "rf>1"
                else :
                    reason = "(lf+rf)%2==1"
                error("Error[einsum]: Inconsistent index statistics. (reason:"+reason+")")
        else :
            if lf>2:
                reason = "lf>2"
                error("Error[einsum]: Inconsistent index statistics. (reason:"+reason+")")

        if lf==1:
            continue

        # add a new entry if it is a new character
        if summand[:i].count(char)==0 :
            summed_index_info += [ [ char, [i] , [stats_list[i]] ] ]
        else:
            for k in range(len(summed_index_info)):
                if(summed_index_info[k][0]==char):
                    summed_index_info[k][1] += [i]
                    summed_index_info[k][2] += [stats_list[i]]

    def is_statwise_inconsistent(stat1,stat2):
        if [stat1,stat2] == [0,0]:
            return False
        elif [stat1,stat2] == [1,-1]:
            return False
        elif [stat1,stat2] == [-1,1]:
            return False
        elif [stat1,stat2] == [hybrid_symbol,hybrid_symbol]:
            return False
        else:
            return True

    for [char,location,stats] in summed_index_info:
        if len(stats)==2 and is_statwise_inconsistent(stats[0],stats[1]):
            error("Error[einsum]: The contracted indices have inconsistent statistics!")
    
    if debug_mode :
        print()
        print("input:",subscripts)
        print("obj indices:")
        for elem in obj_index_list:
            print(" ",elem)
    
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #              Step 1: replace the conjugated variable's index with a new character
    #                      and remove bosonic string
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    fsummand   = ""
    fsummand_u = ""
    fstats_list = []
    fshape_list = []
    charlist = summand
    for i,char in enumerate(summand):
        if stats_list[i] == 0:
            continue

        fsummand    += char
        fstats_list += stats_list[i],
        fshape_list += shape_list[i],

        if stats_list[i] == -1 and summand.count(char)>1:
            new_char = get_char(charlist)
            charlist += new_char
            fsummand_u += new_char

            # edit the summed_index_info, with the position swapped a little bit as well
            for j,[char2,location,stats] in enumerate(summed_index_info):
                if char2==char :
                    summed_index_info[j][0] = [char,new_char]
                    if stats == [-1,1]:
                        summed_index_info[j][1] = [ summed_index_info[j][1][1], summed_index_info[j][1][0] ]
                        summed_index_info[j][2] = [ summed_index_info[j][2][1], summed_index_info[j][2][0] ]
        else:
            fsummand_u += char
    fstats = fstats_list.copy()
    fshape = fshape_list.copy()
    fstats_u = fstats_list.copy()
    fshape_u = fshape_list.copy()

    # remove the stat entry as it is redundant
    for i,elem in enumerate(summed_index_info):
        summed_index_info[i] = elem[:2]

    if debug_mode :
        print()
        print(" :::::::::: replace conjugated indices by unique characters :::::::::: ")
        print(fsummand,"<--- fsummand = original indices")
        print(fsummand_u,"<--- fsummand_u = replace conjugated indices by unique characters")
        print(fstats_u,"<--- fstats_u")
        print(fshape_u,"<--- fshape_u")
        print("summed indices info:")
        for elem in summed_index_info:
            print(" "," = ".join(elem[0]),"( location =",elem[1],")")
        
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #              Step 2: Arrange the indices (presum) and compute the sign factor
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    # the summed indices are now next to each other with the conjugated one to the right

    fsummand_ur = fsummand_u
    for [ [c1,c2], loc ] in summed_index_info:
        fsummand_ur = fsummand_ur.replace(c2,"")
        fsummand_ur = fsummand_ur.replace(c1,c1+c2)

    fstats_ur = reordering(fsummand_u,fsummand_ur,fstats_list)
    fshape_ur = reordering(fsummand_u,fsummand_ur,fshape_list)

    if debug_mode :
        print()
        print(" :::::::::::: move the conjugated to the right of its pair ::::::::::: ")
        print(fsummand_u,"<--- fsummand_u")
        print(fsummand_ur,"<--- fsummand_ur = move the conjugated to the right of its pair")
        print(fstats_ur,"<--- fstats_ur")
        print(fshape_ur,"<--- fshape_ur")
        
    # remove the leading and trailing chars that is the same as the pre-rearranged string

    nall_u = len(fsummand_u)

    nl = 0
    for i in range(nall_u):
        if fsummand_u[i] != fsummand_ur[i]:
            break
        nl+=1
    nr = 0
    for i in range(nall_u):
        if fsummand_u[nall_u-1-i] != fsummand_ur[nall_u-1-i]:
            break
        nr+=1
    nr = nall_u-nr

    fsummand_ut  = fsummand_u[nl:nr]
    fstats_ut   = fstats_u[nl:nr]
    fshape_ut   = fshape_u[nl:nr]
    fsummand_urt  = fsummand_ur[nl:nr]
    fstats_urt   = fstats_ur[nl:nr]
    fshape_urt   = fshape_ur[nl:nr]

    if nl>nr :
        skip_S1 = True
    else:
        skip_S1 = False

    if debug_mode :
        print()
        print(" :::::::::::::::::::::::: trim left and right :::::::::::::::::::::::: ")
        if skip_S1 :
            print("fsummand_u and fsummand_ur are identical.")
            print("S1 calculation is skipped.")
        else :
            print(fsummand_u,"<--- fsummand_u")
            print(fsummand_ut,"<--- fsummand_ut = trim fsummand_u left and right")
            print(fstats_ut,"<--- fstats_ut")
            print(fshape_ut,"<--- fshape_ut")
            print()
            print(fsummand_ur,"<--- fsummand_ur")
            print(fsummand_urt,"<--- fsummand_urt = trim fsummand_ur left and right")
            print(fstats_urt,"<--- fstats_urt")
            print(fshape_urt,"<--- fshape_urt")
        
    S1_sgn_computation_string = fsummand_ut+"->"+fsummand_urt
    
    if debug_mode and not skip_S1 :
        print()
        print(S1_sgn_computation_string,"<--- string used in S1 computation")
    
    # remove the conjugated character if both the nonnconjugated and conjugated are present
    
    # do the ut only
    fsummand_utx = ""
    fstats_utx   = []
    fshape_utx   = []
    nall_ut = len(fsummand_ut)
    for i in range(nall_ut):
        c = fsummand_ut[i]
        to_skip = False
        if fstats_ut[i]==-1 :
            for [ [c1,c2], loc ] in summed_index_info:
                if c==c2 and fsummand_ut.count(c1)>0:
                    to_skip = True
        if to_skip :
            continue
        fsummand_utx += fsummand_ut[i]
        fstats_utx   += fstats_ut[i],
        fshape_utx   += fshape_ut[i],
    S1dim = len(fsummand_utx)

    if debug_mode and not skip_S1 :
        print()
        print(" ::::::: remove the conjugated if both of the pair are present ::::::: ")
        print(fsummand_ut,"<--- fsummand_ut")
        print(fsummand_utx,"<--- fsummand_utx = remove on of the pair")
        print(fstats_utx,"<--- fstats_utx")
        print(fshape_utx,"<--- fshape_utx")
    
    # make the copy map
    copy_map = []
    fsummand_utx_r = list(fsummand_utx) #this is used in the mock reconstruction
    for i_ut, c in enumerate(fsummand_ut):
        if c not in fsummand_utx:
            #find the location of its pair first
            c_pair = "?"
            for [[c1,c2],_] in summed_index_info:
                if c==c2 :
                    i_utx = fsummand_utx_r.index(c1)
                    c_pair = c1
                    break
            copy_map += [i_ut,i_utx],
            fsummand_utx_r.insert(i_ut,c)
    
    def use_copy_map(the_map,the_list,apply_negative=False):
        new_list = list(the_list).copy()
        for [i2,i1] in the_map :
            if apply_negative :
                new_list.insert(i2,-new_list[i1])
            else :
                new_list.insert(i2,new_list[i1])
        return new_list
    
    if debug_mode and not skip_S1:
        print()
        print(" :::::::::::::::::::::::::: testing copy_map ::::::::::::::::::::::::: ")
        fsummand_ut2 = ''.join(use_copy_map(copy_map,list(fsummand_utx)))
        fstats_ut2 = use_copy_map(copy_map,fstats_utx,apply_negative=True)
        fshape_ut2 = use_copy_map(copy_map,fshape_utx)
        print(" ",fsummand_ut2,"vs",fsummand_ut)
        print(" ",fstats_ut2,"vs",fstats_ut)
        print(" ",fshape_ut2,"vs",fshape_ut)
        
    # fsummand_utx needs to be replaced with the original indices to obtain S1_index_string
    S1_index_string  = ""
    for c in fsummand_utx:
        if c in summand:
            S1_index_string += c
        else:
            for [ [c1,c2], loc ] in summed_index_info:
                if c==c2 :
                    S1_index_string += c1
                    break
    S1_shape = []
    for c in S1_index_string:
        ind = fsummand.index(c)
        S1_shape += fshape[ind],
    S1_shape = make_tuple(S1_shape)
    
    if debug_mode and not skip_S1 :
        print()
        print(" :::::::::::::::::::::::::::::: S1 stuff ::::::::::::::::::::::::::::: ")
        print(S1_index_string,"<--- S1's index string")
        print(S1_shape,"<--- S1's shape")
        
    # sign factor computation -------------------------------------------------------------------------
    
    if S1dim>0 and not skip_S1:

        S1 = np.zeros(S1_shape,dtype=int)
        iterator = np.nditer(S1, flags=['multi_index'])
        parity_list = []
        sgn_list = []

        k=1
        kmax = getsize(S1_shape)

        s0 = time.time()
        s00 = s0
        progress_space() # << Don't remove this. This is for the show_progress!

        for element in iterator:
            coords = iterator.multi_index
            dupped_coords = use_copy_map(copy_map,coords)
            
            parity = [ param.gparity(i)%2 for i in dupped_coords ]

            if parity not in parity_list:
                parity_list += parity,
                sgn = relative_sign(S1_sgn_computation_string,parity)
                sgn_list += sgn,
            else:
                index = parity_list.index(parity)
                sgn = sgn_list[index]

            S1[coords] = sgn if not ignore_anticommutation else 1

            if time.time()-s0 > 2 :
                show_progress(k,kmax+1,process_name = "einsum_sgn1",ratio = False,color="red",time=time.time()-s00)
                s0 = time.time()
            
            k+=1

        clear_progress()
        tab_up()
        
    #  ::: Summary :::
    #
    #  S1 is the sign factor from the initial rearrangement
    #  Its index string is S1_index_string

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #              Step 3: Sign factor from the summed pair
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    S2_dimlist = []
    S2_index_string = ""
    summed_indices = fsummand_ur
    for [ [c1,c2], loc ] in summed_index_info:
        S2_dimlist += shape_list[ summand.index(c1) ],
        S2_index_string += ","+c1
        summed_indices = summed_indices.replace(c1+c2,"")
    S2_index_string = S2_index_string[1:]
    nS2 = len(S2_dimlist)
    
    
    skip_S2 = nS2 == 0
    
    if debug_mode :
        print()
        print(" :::::::::::::::::::::::::::::: S2 stuff ::::::::::::::::::::::::::::: ")
        if skip_S2 :
            print("There is no summed indices.")
            print("S2 calculation is skipped.")
        else :
            print(fsummand,"<--- fsummand")
            print(S2_index_string,"<--- S2 index string")
            print(S2_dimlist,"<--- S2's dim list")
    
    if not skip_S2 :

        S2_list = []
        for i in range(nS2):
            S2i = [ (param.sgn(j) if not ignore_anticommutation else 1) for j in range(S2_dimlist[i]) ]
            S2_list += np.array(S2i),

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #              Step 4: rearrange to the final form
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    
    skip_S3 = True
    if has_output :

        foutput = ""
        for c in output_string:
            if stats_list[ summand.index(c) ] == 0 :
                continue
            else:
                foutput+=c

        S3_shape=[]
        for c in summed_indices:
            d = shape_list[ summand.index(c) ]
            S3_shape += d,

        # again... remove the leading and trailing duplicates

        nall_ur = len(summed_indices)
        nl = 0
        for i in range(nall_ur):
            if summed_indices[i] != foutput[i]:
                break
            nl+=1

        nr = 0
        for i in range(nall_ur):
            if summed_indices[nall_ur-1-i] != foutput[nall_ur-1-i]:
                break
            nr+=1
        nr = nall_ur-nr

        final = foutput[nl:nr]
        prefinal = summed_indices[nl:nr]
        S3_shape   = S3_shape[nl:nr]

        if nl>nr :
            skip_S3 = True
        else:
            skip_S3 = False

        S3_sgn_computation_string = prefinal+"->"+final
        S3_index_string = prefinal

        if not skip_S3 :

            k=1
            kmax = 1
            for d in S3_shape:
                kmax*=d

            s0 = time.time()
            s00 = s0
            progress_space() # << Don't remove this. This is for the show_progress!

            S3 = np.zeros(S3_shape,dtype=int)
            iterator = np.nditer(S3, flags=['multi_index'])
            parity_list = []
            sgn_list = []
            for element in iterator:
                coords = iterator.multi_index

                parity = [ param.gparity(i)%2 for i in coords ]

                if parity not in parity_list:
                    parity_list += parity,
                    sgn = relative_sign(S3_sgn_computation_string,parity)
                    sgn_list += sgn,
                else:
                    index = parity_list.index(parity)
                    sgn = sgn_list[index]

                S3[coords] = sgn if not ignore_anticommutation else 1

                if time.time()-s0 > 0.5 :
                    show_progress(k,kmax+1,process_name = "einsum_sgn3",ratio = False,color="red",time=time.time()-s00)
                    s0 = time.time()

                k+=1
            clear_progress()
            tab_up()

            skip_S3 = len(S3_shape)==0

    #  ::: Summary :::
    #
    #  S3 is the sign factor from the final rearrangement
    #  Its index string is S3_index_string
    
    if debug_mode and not skip_S3:
        print()
        print(" :::::::::::::::::::::::::::::: S3 stuff ::::::::::::::::::::::::::::: ")
        print(S3_sgn_computation_string,"<--- string used in the S3 computation")
        print(S3_index_string,"<--- S3_index_string")
        print(S3_shape,"<--- S3_shape")
    
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #              Step 5: add the vertices
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    
    einsum_input = input_string
    einsum_obj_list = obj_list

    if S1dim>0 and not skip_S1:
        einsum_input += ","+S1_index_string
        einsum_obj_list += S1,

    if nS2>0 :
        einsum_input += S2_index_string
        einsum_obj_list += S2_list

    if has_output :
        if not skip_S3 :
            einsum_input += ","+S3_index_string
            einsum_obj_list += S3,

    if this_type == sparse :
        

        instruction_all_indices = subscripts.replace(",","")
        instruction_all_indices = instruction_all_indices.replace("->","")
        instruction_all_indices = instruction_all_indices.replace(" ","")
        if not skip_S1:
            instruction_all_indices=instruction_all_indices+S1_index_string
        if not skip_S2:
            instruction_all_indices=instruction_all_indices+S2_index_string
        if not skip_S3:
            instruction_all_indices=instruction_all_indices+S3_index_string

        # make a listing of duplicated indices
        einsum_input_unique = ""
        vertex_list = [] # [ indices , shape , number_of_legs ]
        unique_char = ""
        for i,c in enumerate(instruction_all_indices):
            if c=="," :
                continue
            if instruction_all_indices[:i].count(c) == 0 :
                vertex_list += [ c, shape_list[ summand.index(c) ] , instruction_all_indices.count(c) ],
                unique_char += c

        for i,[char,dim,nlegs] in enumerate(vertex_list) :
            legchar = char
            for leg in range(nlegs-1):
                new_char = get_char(unique_char)
                unique_char += new_char
                legchar += new_char
            vertex_list[i][0] = legchar

        vertex_list_final = []
        for elem in vertex_list :
            if elem[2] > 2:
                vertex_list_final += elem[:2],
                
        for i,elem in enumerate(vertex_list_final):
            x = list(elem[0])
            x.reverse()
            vertex_list_final[i][0] = ''.join(x)
        
        if debug_mode :   
            print()
            print(" :::::::::::::::::::::::::::::: vertices ::::::::::::::::::::::::::::: ")
            print(instruction_all_indices,"<-- the instruction strings with all char combined into one string")
            print("vertex list:")
            for vert in vertex_list_final:
                print(" ",vert)
    
    
    # now replace the einsum string with these new characters
    
    #get the correct einsum string first
    einsum_string = input_string
    if not skip_S1 :
        einsum_string += ","+S1_index_string
    if not skip_S2 :
        einsum_string += ","+S2_index_string
    if not skip_S3 :
        einsum_string += ","+S3_index_string
    if debug_mode :   
        print()  
        print(einsum_string,"<-- einsum_string = einsum string without vertices")
    
    if this_type == sparse :
        einsum_string_replaced = einsum_string
        for [vert,dim] in vertex_list_final:
            c = vert[len(vert)-1]
            nreplace = einsum_string_replaced.count(c)
            for i in range(nreplace):
                einsum_string_replaced = einsum_string_replaced.replace(c,vert[i],1)
            einsum_string_replaced+=","+vert
            
        if debug_mode :   
            print()
            print(einsum_string_replaced,"<-- einsum_string with replacement")
        
        # construct the vertex tensors
        vertex_obj_list = []
        for [string,dim] in vertex_list_final :
            shape=tuple([dim]*len(string))
            coords = np.array([ [ i for i in range(dim) ] for j in range(len(string))])
            data = np.array( [ 1 for i in range(dim) ] )
            vertex = sp.COO( coords, data, shape=shape )
            vertex_obj_list += vertex.copy(),

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #              Step 6: get the final statistics
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    final_stats = []
    if has_output :
        for c in output_string :
            final_stats += stats_list[ summand.index(c) ],

    if debug_mode and has_output:
        print()
        print(" :::::::::::::::::::::::::: final statistics ::::::::::::::::::::::::: ")
        print(final_stats,"<--- final stats")
        
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #              Step 7: the actual sum
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    if this_type==sparse :
        einsum_obj_list += vertex_obj_list
        einsum_string = einsum_string_replaced
    
    if has_output :
        einsum_string += "->"+output_string

    # convert all object to this_type
    if this_type == sparse :
        for i, obj in enumerate(einsum_obj_list):
            if type(obj) == np.ndarray:
                einsum_obj_list[i] = sp.COO.from_numpy(obj)
    if this_type == dense :
        for i, obj in enumerate(einsum_obj_list):
            if type(obj) != np.ndarray:
                einsum_obj_list[i] = sp.COO.todense(obj)
    
    if debug_mode :
        print(" ::::::::::::::::::::::::::: the actual sum :::::::::::::::::::::::::: ")
        print(einsum_string,"<-- einsum string")
        print("[shape,type]:")
        for obj in einsum_obj_list:
            print(" ",[obj.shape,type(obj)])

    # [7-May-24] hot fix in case everything is bosonic
    einsum_string = einsum_string.replace(",->","->")
    # [8-May-24] the same as above but for complete contraction
    if einsum_string[-1]==',':
        einsum_string=einsum_string[:-1]
    
    ret = oe.contract(*tuple([einsum_string]+einsum_obj_list))

    if has_output :
        return this_type(ret,statistics=final_stats).force_encoder(this_encoder).force_format(this_format)
    else:
        if this_type == sparse :
            if type(ret.data)==memoryview:
                return ret
            else:
                return ret.data[0]
        else:
            return np.array(ret).flatten()[0]

def einsum(*args,ignore_anticommutation=False):
    subscripts = args[0].replace(" ","")
    subscripts = denumerate(subscripts)

    before=""
    after=""
    return_scalar = True
    if subscripts.count("->")==1 :
        before,after = subscripts.split("->")
        return_scalar = False
    elif subscripts.count("->")>1 :
        print(error("Error[einsum]: There should be no more than one '->' indicator in the subscript input."))
    else:
        before = subscripts
    has_output = not return_scalar

    summand = before.replace(",","")

    obj_index_list = before.split(",")
    nobj = len(obj_index_list)

    obj_list = make_list(args[1:1+nobj])

    if type(obj_list[0]) == block:
        return einsum_block(*args,ignore_anticommutation=ignore_anticommutation)
    else:
        return einsum_ds(*args,ignore_anticommutation=ignore_anticommutation)

def einsum_block(*args,ignore_anticommutation=False):

    # =============================================================================================#
    # Step I: Pre-generate the cells of the result                                                 #
    # =============================================================================================#

    global skip_power_of_two_check
    skip_power_of_two_check = True
    subscripts = args[0].replace(" ","")
    subscripts = denumerate(subscripts)

    before=""
    after=""
    return_scalar = True
    if subscripts.count("->")==1 :
        before,after = subscripts.split("->")
        return_scalar = False
    elif subscripts.count("->")>1 :
        print(error("Error[einsum_block]: There should be no more than one '->' indicator in the subscript input."))
    else:
        before = subscripts
    has_output = not return_scalar

    summand = before.replace(",","")

    obj_index_list = before.split(",")
    nobj = len(obj_index_list)

    obj_list = make_list(args[1:1+nobj])
    stats_list = sum([ make_list(obj.statistics) for obj in obj_list ],[])
    shape_list = sum([ make_list(obj.effective_shape) for obj in obj_list ],[])
    eshape_list = sum([ make_list(obj.even_shape) for obj in obj_list ],[])
    oshape_list = sum([ make_list(obj.odd_shape) for obj in obj_list ],[])
    
    has_int = False
    has_float = False
    has_complex = False
    first_format = obj_list[0].format
    for i,obj in enumerate(obj_list):
        if type(obj)==dense or type(obj)==sparse:
            error("Error[einsum_block]: This function only works with block data format.")
        obj_list[i] = obj_list[i].force_format("standard")
        has_int = has_int or obj_list[i].dtype==int
        has_float = has_float or obj_list[i].dtype==float
        has_complex = has_complex or obj_list[i].dtype==complex

    rettype = int
    if has_float:
        rettype = float
    if has_complex:
        rettype = complex

    feshape_list = make_list( shape for i,shape in enumerate(eshape_list) if stats_list[i] in fermi_type )
    foshape_list = make_list( shape for i,shape in enumerate(oshape_list) if stats_list[i] in fermi_type )

    # the block version of before
    fbefore = ""
    for i,c in enumerate(summand):
        if stats_list[i] in fermi_type:
            fbefore += c
    obj_findex_list = obj_index_list.copy()
    for i, findex in enumerate(obj_findex_list):
        new_findex = ""
        for c in findex:
            if c in fbefore:
                new_findex+=c
        obj_findex_list[i] = new_findex
    if has_output:
        # the block version of after
        fafter = ""
        for c in after:
            if c in fbefore:
                fafter += c
        fsubscripts = fbefore+"->"+fafter

        # <fbefore> is <summand> without bose-type indices
        # <fafter> is <after> without bose-type indices

        newstats = reordering(summand,after,stats_list)   # tensor stats
        newshape = reordering(summand,after,shape_list)   # tensor shape
        neweshape = reordering(summand,after,eshape_list)   # tensor even shape
        newoshape = reordering(summand,after,oshape_list)   # tensor odd shape
        #newfshape = reordering(fbefore,fafter,fshape_list) # cell shape

        newshape_final = make_tuple([
                dim if stat in bose_type else 2**int(np.ceil(np.log2(dim)))
                    for dim,stat in zip(newshape,newstats) ])
        # correct ret's shape in the end
        #ret = zero_block(effective_shape=newshape,statistics=newstats,format=first_format,dtype=rettype)
        ret = zero_block_eo(neweshape,newoshape,statistics=newstats,format=first_format,dtype=rettype)

        # copy the sign factors from here
        ret_sgn_e = []
        ret_sgn_o = []
        for c in after:
            found = False
            obj_num = -1
            ind_loc = -1
            str_sample = ""
            for obj_num_temp,obj_ind in enumerate(obj_index_list):
                if c in obj_ind:
                    obj_num = obj_num_temp
                    ind_loc = obj_ind.index(c)
                    str_sample = obj_ind
                    break
            if obj_num==-1 or ind_loc==-1:
                error("Error[einsum_block]: cannot locate the final subscript in the original objects' subscripts")
            ret_sgn_e += [obj_list[obj_num].sgn[0][ind_loc]]
            ret_sgn_o += [obj_list[obj_num].sgn[1][ind_loc]]

            #print("Found",c,"in "+str(ind_loc)+"-th index of "+str(obj_num)+"-th object;",str_sample)

        ret.sgn = [ret_sgn_e,ret_sgn_o]

    # =============================================================================================#
    # Step II: Arrange the indices (presum) and generate the sign-factor computation string        #
    #          Most of the stuff here are copied from einsum()                                     #
    # =============================================================================================#

    # get some information about the summed indices -------------------------------------------------
    # [ f=<char>, [index locations in lf], [statistics] ]
    summed_index_info = []

    for i,char in enumerate(summand):

        # skip if bosonic
        if stats_list[i] in bose_type:
            continue

        lf =  before.count(char)
        if has_output :
            rf = after.count(char)
            if lf>2 or rf>1 or (lf+rf)%2==1 :
                if lf>2 :
                    reason = "lf>2"
                elif rf>1 :
                    reason = "rf>1"
                else :
                    reason = "(lf+rf)%2==1"
                error("Error[einsum]: Inconsistent index statistics. (reason:"+reason+")")
        else :
            if lf>2:
                reason = "lf>2"
                error("Error[einsum]: Inconsistent index statistics. (reason:"+reason+")")

        if lf==1:
            continue

        # add a new entry if it is a new character
        if summand[:i].count(char)==0 :
            summed_index_info += [ [ char, [i] , [stats_list[i]] ] ]
        else:
            for k in range(len(summed_index_info)):
                if(summed_index_info[k][0]==char):
                    summed_index_info[k][1] += [i]
                    summed_index_info[k][2] += [stats_list[i]]

    def is_statwise_inconsistent(stat1,stat2):
        if [stat1,stat2] == [0,0]:
            return False
        elif [stat1,stat2] == [1,-1]:
            return False
        elif [stat1,stat2] == [-1,1]:
            return False
        elif [stat1,stat2] == [hybrid_symbol,hybrid_symbol]:
            return False
        else:
            return True

    for [char,location,stats] in summed_index_info:
        if len(stats)==2 and is_statwise_inconsistent(stats[0],stats[1]):
            error("Error[einsum]: The contracted indices have inconsistent statistics!")
    
    fsummand   = ""
    fsummand_u = ""
    fstats_list = []
    fshape_list = []
    charlist = summand
    for i,char in enumerate(summand):
        if stats_list[i] == 0:
            continue

        fsummand    += char
        fstats_list += stats_list[i],
        fshape_list += shape_list[i],

        if stats_list[i] == -1 and summand.count(char)>1:
            new_char = get_char(charlist)
            charlist += new_char
            fsummand_u += new_char

            # edit the summed_index_info, with the position swapped a little bit as well
            for j,[char2,location,stats] in enumerate(summed_index_info):
                if char2==char :
                    summed_index_info[j][0] = [char,new_char]
                    if stats == [-1,1]:
                        summed_index_info[j][1] = [ summed_index_info[j][1][1], summed_index_info[j][1][0] ]
                        summed_index_info[j][2] = [ summed_index_info[j][2][1], summed_index_info[j][2][0] ]
        else:
            fsummand_u += char
    fstats = fstats_list.copy()
    fshape = fshape_list.copy()
    fstats_u = fstats_list.copy()
    fshape_u = fshape_list.copy()

    # remove the stat entry as it is redundant
    for i,elem in enumerate(summed_index_info):
        summed_index_info[i] = elem[:2]

    # the summed indices are now next to each other with the conjugated one to the right

    fsummand_ur = fsummand_u
    for [ [c1,c2], loc ] in summed_index_info:
        fsummand_ur = fsummand_ur.replace(c2,"")
        fsummand_ur = fsummand_ur.replace(c1,c1+c2)

    fstats_ur = reordering(fsummand_u,fsummand_ur,fstats_list)
    fshape_ur = reordering(fsummand_u,fsummand_ur,fshape_list)

    # remove the leading and trailing chars that is the same as the pre-rearranged string

    nall_u = len(fsummand_u)

    nl = 0
    for i in range(nall_u):
        if fsummand_u[i] != fsummand_ur[i]:
            break
        nl+=1
    nr = 0
    for i in range(nall_u):
        if fsummand_u[nall_u-1-i] != fsummand_ur[nall_u-1-i]:
            break
        nr+=1
    nr = nall_u-nr

    fsummand_ut  = fsummand_u[nl:nr]
    fstats_ut   = fstats_u[nl:nr]
    fshape_ut   = fshape_u[nl:nr]
    fsummand_urt  = fsummand_ur[nl:nr]
    fstats_urt   = fstats_ur[nl:nr]
    fshape_urt   = fshape_ur[nl:nr]

    if nl>nr :
        skip_S1 = True
    else:
        skip_S1 = False

    if ignore_anticommutation :
        skip_S1 = True
        
    S1_sgn_computation_string = fsummand_ut+"->"+fsummand_urt
    
    # =============================================================================================#
    # Step III: rearrange to the final form                                                        #
    # =============================================================================================#

    S2_dimlist = []
    S2_index_string = ""
    summed_indices = fsummand_ur
    for [ [c1,c2], loc ] in summed_index_info:
        S2_dimlist += shape_list[ summand.index(c1) ],
        S2_index_string += ","+c1
        summed_indices = summed_indices.replace(c1+c2,"")
    S2_index_string = S2_index_string[1:]
    nS2 = len(S2_dimlist)
    
    skip_S3 = True
    if has_output :

        foutput = ""
        for c in after:
            if stats_list[ summand.index(c) ] == 0 :
                continue
            else:
                foutput+=c

        S3_shape=[]
        for c in summed_indices:
            d = shape_list[ summand.index(c) ]
            S3_shape += d,

        # again... remove the leading and trailing duplicates

        nall_ur = len(summed_indices)
        nl = 0
        for i in range(nall_ur):
            if summed_indices[i] != foutput[i]:
                break
            nl+=1

        nr = 0
        for i in range(nall_ur):
            if summed_indices[nall_ur-1-i] != foutput[nall_ur-1-i]:
                break
            nr+=1
        nr = nall_ur-nr

        final = foutput[nl:nr]
        prefinal = summed_indices[nl:nr]
        S3_shape   = S3_shape[nl:nr]

        if nl>nr :
            skip_S3 = True
        else:
            skip_S3 = False

        if ignore_anticommutation :
            skip_S3 = True

        S3_sgn_computation_string = prefinal+"->"+final

    # =============================================================================================#
    # Step IV: Determine the einsum string                                                         #
    # =============================================================================================#

    before_plus_sigma = before
    for elem in summed_index_info:
        before_plus_sigma+=(","+elem[0][0])
    final_einsum_string = before_plus_sigma+"->"+after

    # =============================================================================================#
    # Step V: Iterate over all parity config                                                       #
    #          For each parity config, the sign factor is computed                                 #
    # =============================================================================================#

    if has_output:
        # count unique indices
        unique_fsummand = "".join(sorted(list(set(fbefore))))
        nsum = len(unique_fsummand)

        # the comprehensive list of sign factors (fermion only)
        all_sgn = [[],[]]
        for obj in obj_list:
            for ind_parity in [0,1]:
                for s in obj.sgn[ind_parity]:
                    all_sgn[ind_parity] += [s]

        block_list = [[]] # this is the set of all blocks
        for char in unique_fsummand:
            block_list_E = [ config+[0] for config in block_list ]
            block_list_O = [ config+[1] for config in block_list ]
            block_list = block_list_E+block_list_O

        block_list = [ tuple(block) for block in block_list ]

        for block in block_list:
            #print()
            #print(unique_fsummand,"=",block)

            # ======================================================================================================
            # now you have the summand list and the block list, you can do whatever you want here
            # unique_fsummand is the ordered list of summed indices
            # block is the value of each index
            # ======================================================================================================

            # ======================================================================================================
            # This part determines the block number of each object
            # ======================================================================================================
            obj_block_list = []
            for i,findex in enumerate(obj_findex_list):
                sub_block = []
                for c in findex:
                    if c not in unique_fsummand:
                        error("Error[einsum_block]: the character c is not contained in unique_fsummand. Unexplanable error! Possibly a bug.")
                    c_location = unique_fsummand.index(c)
                    sub_block += [block[c_location]]
                obj_block_list += [tuple(sub_block)]
            
            #print(obj_findex_list,"=",obj_block_list)

            # ======================================================================================================
            # get the specified block from each object
            # ======================================================================================================
            blocked_obj_list = [ obj.data[obj_block_list[i]] for i,obj in enumerate(obj_list) ]

            sigma_list = []

            for ic, c in enumerate(unique_fsummand):
                c_index_in_summand = summand.index(c)
                sgn_c = all_sgn[block[ic]][c_index_in_summand]
                if summand.count(c) > 1:
                    err = 0
                    for ic2, c2 in enumerate(summand):
                        if c2==c :
                            sgn_c2 = all_sgn[block[ic]][ic2]
                            err += np.linalg.norm(sgn_c-sgn_c2)
                    if err>numer_cutoff :
                        error("Error[einsum_block]: some sign factors are not consistent.")
                sigma_list += [ sgn_c if not ignore_anticommutation else 1 ]

            # ======================================================================================================
            # evaluate the block sign factor
            # ======================================================================================================

            # step1: from moving conjugated summed index to the non-conjugated counterpart
            S1 = 1
            #print("compute S1:", not skip_S1, end="; S1 = ")
            if not skip_S1:
                S1_com_before = S1_sgn_computation_string.split("->")[0]
                # replaced the new index with the old index
                for elem in summed_index_info:
                    S1_com_before = S1_com_before.replace(elem[0][1],elem[0][0])
                # get the parity list
                parity_list = [ block[unique_fsummand.index(c)] for c in S1_com_before ]
                S1 = relative_sign(S1_sgn_computation_string,parity_list)
                #print("S1:",S1_sgn_computation_string,"=",S1)
                #print(S1)

            # step2: from moving the result after the summation to the specified output form
            S3 = 1
            #print("compute S3:", not skip_S3, end="; S3 = ")
            if not skip_S3:
                S3_com_before = S3_sgn_computation_string.split("->")[0]
                parity_list = [ block[unique_fsummand.index(c)] for c in S3_com_before ]
                S3 = relative_sign(S3_sgn_computation_string,parity_list)
                #print("S3:",S3_sgn_computation_string,"=",S3)
                #print(S3)

            block_sign = S1*S3
            #print("block sign = (",S1,")*(",S3,") =",block_sign)

            # ======================================================================================================
            # obtain the sign vectors
            # ======================================================================================================
            sigma_summed_list = []
            for elem in summed_index_info:
                c = elem[0][0]
                sigma_summed_list += [ sigma_list[unique_fsummand.index(c)] ]
            
            # ======================================================================================================
            # the summation
            # ======================================================================================================
            einsum_obj_list = blocked_obj_list + sigma_summed_list
            for i,elem in enumerate(einsum_obj_list):
                if elem.dtype == object :
                    newelem = np.array(elem.tolist())
                    einsum_obj_list[i] = newelem

            sum_result = block_sign*np.einsum(*tuple([final_einsum_string]+einsum_obj_list))


            # ======================================================================================================
            # obtain the output block
            # ======================================================================================================

            out_block = tuple([ block[unique_fsummand.index(c)] for c in fafter ])
            #print("output block:",fafter,";",out_block)
            ret.data[out_block] += sum_result
            ret.shape = newshape_final

            skip_power_of_two_check = False
            
    else:
        # if returns scalar
        # count unique indices
        unique_fsummand = "".join(sorted(list(set(fbefore))))
        nsum = len(unique_fsummand)

        # the comprehensive list of sign factors (fermion only)
        all_sgn = [[],[]]
        for obj in obj_list:
            for ind_parity in [0,1]:
                for s in obj.sgn[ind_parity]:
                    all_sgn[ind_parity] += [s]

        block_list = [[]] # this is the set of all blocks
        for char in unique_fsummand:
            block_list_E = [ config+[0] for config in block_list ]
            block_list_O = [ config+[1] for config in block_list ]
            block_list = block_list_E+block_list_O

        block_list = [ tuple(block) for block in block_list ]

        scalar_output=0
        for block in block_list:
            #print()
            #print(unique_fsummand,"=",block)

            # ======================================================================================================
            # now you have the summand list and the block list, you can do whatever you want here
            # unique_fsummand is the ordered list of summed indices
            # block is the value of each index
            # ======================================================================================================

            # ======================================================================================================
            # This part determines the block number of each object
            # ======================================================================================================
            obj_block_list = []
            for i,findex in enumerate(obj_findex_list):
                sub_block = []
                for c in findex:
                    if c not in unique_fsummand:
                        error("Error[einsum_block]: the character c is not contained in unique_fsummand. Unexplanable error! Possibly a bug.")
                    c_location = unique_fsummand.index(c)
                    sub_block += [block[c_location]]
                obj_block_list += [tuple(sub_block)]
            
            #print(obj_findex_list,"=",obj_block_list)

            # ======================================================================================================
            # get the specified block from each object
            # ======================================================================================================
            '''
            blocked_obj_list = [ obj.data[obj_block_list[i]] for i,obj in enumerate(obj_list) ]
            blocked_sgn_list = sum([ obj.sgn[obj_block_list[i]].tolist()  for i,obj in enumerate(obj_list) ],[])
            blocked_sgn_list = [ np.array(elem) for elem in blocked_sgn_list ]
            #
            # /\
            # ||
            #
            # dont use this one
            #
            # use this one instead
            #
            # |
            # V
            #
            sigma_list = []
            for c in unique_fsummand:
                # check if there are multiples
                sgn_c = blocked_sgn_list[summand.index(c)]
                if summand.count(c) > 1:
                    err = 0
                    for c2 in summand:
                        if c2==c :
                            sgn_c2 = blocked_sgn_list[summand.index(c2)]
                            err += np.linalg.norm(sgn_c-sgn_c2)
                    if err>numer_cutoff :
                        error("Error[einsum_block]: some sign factors are not consistent.")
                sigma_list += [ sgn_c ]
            '''
            blocked_obj_list = [ obj.data[obj_block_list[i]] for i,obj in enumerate(obj_list) ]

            sigma_list = []
            #print(summand)
            #print(all_sgn[0])
            #print(unique_fsummand)
            #print(block)
            
            for ic, c in enumerate(unique_fsummand):
                c_index_in_summand = summand.index(c)
                sgn_c = all_sgn[block[ic]][c_index_in_summand]
                if summand.count(c) > 1:
                    err = 0
                    for ic2, c2 in enumerate(summand):
                        if c2==c :
                            sgn_c2 = all_sgn[block[ic]][ic2]
                            err += np.linalg.norm(sgn_c-sgn_c2)
                    if err>numer_cutoff :
                        error("Error[einsum_block]: some sign factors are not consistent.")
                sigma_list += [ sgn_c ]

            # ======================================================================================================
            # evaluate the block sign factor
            # ======================================================================================================

            # step1: from moving conjugated summed index to the non-conjugated counterpart
            S1 = 1
            #print("compute S1:", not skip_S1, end="; S1 = ")
            if not skip_S1:
                S1_com_before = S1_sgn_computation_string.split("->")[0]
                # replaced the new index with the old index
                for elem in summed_index_info:
                    S1_com_before = S1_com_before.replace(elem[0][1],elem[0][0])
                # get the parity list
                parity_list = [ block[unique_fsummand.index(c)] for c in S1_com_before ]
                S1 = relative_sign(S1_sgn_computation_string,parity_list)
                #print("S1:",S1_sgn_computation_string,"=",S1)
                #print(S1)

            block_sign = S1
            #print("block sign = (",S1,")*(",S3,") =",block_sign)

            # ======================================================================================================
            # obtain the sign vectors
            # ======================================================================================================
            sigma_summed_list = []
            for elem in summed_index_info:
                c = elem[0][0]
                sigma_summed_list += [ sigma_list[unique_fsummand.index(c)] ]
            
            # ======================================================================================================
            # the summation
            # ======================================================================================================
            einsum_obj_list = blocked_obj_list + sigma_summed_list
            sum_result = block_sign*np.einsum(*tuple([final_einsum_string]+einsum_obj_list))

            scalar_output+=sum_result

            skip_power_of_two_check = False

        return scalar_output

    if incorrect_sign_size(ret):
        ret = correct_sign_size(ret)
        if incorrect_sign_size(ret):
            error("Error[??]: Unsuccessful attempt at truncating the sgn vectors.")
    
    return ret.force_format(first_format)

####################################################
##                     Reshape                    ##
####################################################

def join_legs(InpObj,string_inp,make_format='standard',intermediate_stat=None,save_memory=False):

    process_name = "join_legs"
    process_length = 6
    process_color="green"
    step = 1
    s00 = time.time()
    progress_space() # << Don't remove this. This is for the show_progress!

    string_inp = string_inp.replace(" ","")
    string_inp = denumerate(string_inp)

    intermediate_stat = make_tuple(intermediate_stat)
    
    step = show_progress(step,process_length,process_name+" "+"<"+current_memory_display()+">",color=process_color,time=time.time()-s00)
    # Always output the parity-preserving encoder
    #===============================================================================#
    #   Step 0: Preconditioning the initial Object to standard & canonical          #
    #===============================================================================#
    
    Obj = InpObj.copy()
    this_type = type(Obj)
    this_format = Obj.format
    this_encoder = Obj.encoder
    XObj_stats = Obj.statistics
    XObj_shape = Obj.shape

    if save_memory :
        del InpObj.data
        del InpObj
        gc.collect()
        
    #if this_type   == sparse:
    #    Obj = dense(Obj)
    if this_format == 'matrix':
        #force convert to standard
        Obj = Obj.switch_format(save_memory=True)
    if this_encoder == 'parity-preserving':
        #force convert to standard
        Obj = Obj.switch_encoder(save_memory=True)

    step = show_progress(step,process_length,process_name+" "+"<"+current_memory_display()+">",color=process_color,time=time.time()-s00)
    #===============================================================================#
    #   Step 1: Move bosonic indices to the left of each group                      #
    #===============================================================================#
    
    # get the grouping info first
    group_info, sorted_group_info = get_group_info(string_inp, Obj.statistics, Obj.shape)
    #group_info contains the index_string, statistics, and shape of each group
    #sorted_group_info is the same, but with bosonic indices sorted to the left
    
    n_indices = sum([ len(indices) for [indices,stats,shape] in group_info ])
    
    if n_indices != Obj.ndim :
        error("Error[join_legs]: The number of indices is not consistent with the object's shape.")
    
    sign_factors_list = get_grouping_sign_factors(sorted_group_info, intermediate_stat)
    
    unsorted_string = ''.join( [ indices for [indices, stats, shape] in group_info ] )
    sorted_string = ''.join( [ indices for [indices, stats, shape] in sorted_group_info ] )
    npeinsum_string = unsorted_string+sign_factors_list[0]+"->"+sorted_string
    npeinsum_obj = [Obj.data] + sign_factors_list[1]
    
    sorted_stat = make_tuple(sum(
            [ make_list(stats) for [indices, stats, shape] in sorted_group_info ] ,[]
        ))
    
    #sorted tensor
    Obj.data = oe.contract(*make_tuple( [npeinsum_string] + npeinsum_obj ))
    Obj.statistics = sorted_stat
    
    step = show_progress(step,process_length,process_name+" "+"<"+current_memory_display()+">",color=process_color,time=time.time()-s00)
    
    #===============================================================================#
    #   Step 2: Join fermionic indices with np.reshape                              #
    #===============================================================================#
    
    new_stats, new_shape, final_stats, final_shape = get_intermediate_info(sorted_group_info,intermediate_stat)
    #intermediate_tensor
    
    Obj.data = np.reshape(Obj.data,new_shape)
    Obj.statistics = new_stats
    
    step = show_progress(step,process_length,process_name+" "+"<"+current_memory_display()+">",color=process_color,time=time.time()-s00)
    #===============================================================================#
    #   Step 3: Switch format if make_format='matrix'                               #
    #===============================================================================#
    
    if make_format == 'matrix':
        Obj = Obj.switch_format(save_memory=True)
    
    step = show_progress(step,process_length,process_name+" "+"<"+current_memory_display()+">",color=process_color,time=time.time()-s00)
    #===============================================================================#
    #   Step 4: Switch encoder                                                      #
    #===============================================================================#

    Obj = Obj.switch_encoder(save_memory=True)
    
    step = show_progress(step,process_length,process_name+" "+"<"+current_memory_display()+">",color=process_color,time=time.time()-s00)
    #===============================================================================#
    #   Step 5: Merge bosons and fermions                                           #
    #===============================================================================#
    
    Obj.data = np.reshape(Obj.data,final_shape)
    Obj.statistics = final_stats
    
    clear_progress()
    tab_up()
    
    # check if there is a hybrid index or not
    can_convert = True
    for stat in Obj.statistics:
        if stat == hybrid_symbol:
            can_convert = False
            break
    '''
    if can_convert :
        Obj = Obj.force_format(this_format)
        Obj = Obj.force_encoder(this_encoder)
    '''

    return Obj
    
def split_legs(InpObj,string_inp,final_stat,final_shape,intermediate_stat=None,save_memory=False):

    process_name = "split_legs"
    process_length = 6
    process_color="green"
    step = 1
    s00 = time.time()
    progress_space() # << Don't remove this. This is for the show_progress!

    string_inp = string_inp.replace(" ","")
    string_inp = denumerate(string_inp)

    intermediate_stat = make_tuple(intermediate_stat)
    final_stat = make_tuple(final_stat)
    final_shape = make_tuple(final_shape)

    step = show_progress(step,process_length,process_name+" "+"<"+current_memory_display()+">",color=process_color,time=time.time()-s00)
    #===============================================================================#
    #   Step 0: Preparation                                                         #
    #===============================================================================#
    
    Obj = InpObj.copy()
    this_type = type(Obj)
    this_format = Obj.format
    this_encoder = Obj.encoder
    XObj_stats = Obj.statistics
    XObj_shape = Obj.shape

    if save_memory :
        del InpObj.data
        del InpObj
        gc.collect()

    step = show_progress(step,process_length,process_name+" "+"<"+current_memory_display()+">",color=process_color,time=time.time()-s00)
    #===============================================================================#
    #   Step 5: Split bosons and fermions                                           #
    #===============================================================================#
    
    group_info, sorted_group_info = get_group_info(string_inp, final_stat, final_shape)
    new_stats, new_shape, _, _ = get_intermediate_info(sorted_group_info,intermediate_stat)
    sign_factors_list = get_grouping_sign_factors(sorted_group_info, intermediate_stat)
    
    Obj.data = np.reshape(Obj.data,new_shape)
    Obj.statistics = new_stats
    
    step = show_progress(step,process_length,process_name+" "+"<"+current_memory_display()+">",color=process_color,time=time.time()-s00)
    #===============================================================================#
    #   Step 4: Switch encoder                                                      #
    #===============================================================================#
    
    if this_encoder == 'parity-preserving':
        #force convert to standard
        Obj = Obj.switch_encoder(save_memory=True)
    
    step = show_progress(step,process_length,process_name+" "+"<"+current_memory_display()+">",color=process_color,time=time.time()-s00)
    #===============================================================================#
    #   Step 3: Switch format if this_format='matrix'                               #
    #===============================================================================#
    
    if this_format == 'matrix':
        Obj = Obj.switch_format(save_memory=True)
        
    step = show_progress(step,process_length,process_name+" "+"<"+current_memory_display()+">",color=process_color,time=time.time()-s00)
    #===============================================================================#
    #   Step 2: Split fermionic indices with np.reshape                              #
    #===============================================================================#
    
    new_stats = []
    new_shape = []
    for [indices,stats,shape] in sorted_group_info:
        new_stats += make_list(stats)
        new_shape += make_list(shape)
    new_stats = make_tuple(new_stats)
    new_shape = make_tuple(new_shape)
    
    Obj.data = np.reshape(Obj.data,new_shape)
    Obj.statistics = new_stats
    
    step = show_progress(step,process_length,process_name+" "+"<"+current_memory_display()+">",color=process_color,time=time.time()-s00)
    #===============================================================================#
    #   Step 1: Move bosonic indices to the left of each group                      #
    #===============================================================================#
    
    unsorted_string = ''.join( [ indices for [indices, stats, shape] in group_info ] )
    sorted_string = ''.join( [ indices for [indices, stats, shape] in sorted_group_info ] )
    npeinsum_string = sorted_string+sign_factors_list[0]+"->"+unsorted_string
    npeinsum_obj = [Obj.data] + sign_factors_list[1]
    Obj.data = oe.contract(*make_tuple( [npeinsum_string] + npeinsum_obj ))
    Obj.statistics = final_stat
    
    clear_progress()
    tab_up()

    if this_format=='matrix':
        return Obj.switch_format(save_memory=True)
    
    if this_encoder == 'parity-preserving':
        #force convert to standard
        Obj = Obj.switch_encoder(save_memory=True)

    return Obj
    
def get_group_info(grouping_string, ungroup_stat, ungroup_shape):
    #print("original string:",grouping_string)
    formatted_string = grouping_string
    
    # check the string format --------------------------------------------------------
    if formatted_string.count("(") == 0 and formatted_string.count(")") == 0 :
        #this is the abbreviated format
        for separator in separator_list:
            formatted_string = formatted_string.replace(separator,")(")
        formatted_string = "("+formatted_string+")"
    else:
        separator_count = 0
        for separator in separator_list:
            separator_count += formatted_string.count(separator)
        if separator_count > 0 :
            error("Error[get_grouping_info]: Do not mix the string format.")
    #print("formatted string:",formatted_string)
    
    # parse the string ---------------------------------------------------------------
    group_info = [] # [string text, statistics, shape]
    is_outside = True
    location = 0
    for char in formatted_string:
        if char == "(" and is_outside :
            is_outside = False
            group_info += [
                [ "", [], [] ] #make a blank template for this group
            ]
            continue
        if char == ")" and not is_outside :
            is_outside = True
            continue
        if (char == "(" and not is_outside) or (char == ")" and is_outside) :
            error("Error[get_grouping_info]: No nested parenthesis allowed!")
        
        if is_outside :
            #add an element of just one index
            group_info += [
                [ char, [ungroup_stat[location]], [ungroup_shape[location]] ]
            ]
        else:
            n = len(group_info)-1
            group_info[n][0] += char
            group_info[n][1] += [ungroup_stat[location]]
            group_info[n][2] += [ungroup_shape[location]]
        location += 1
    for n in range(len(group_info)) :
        group_info[n][1] = make_tuple(group_info[n][1])
        group_info[n][2] = make_tuple(group_info[n][2])
    
    #print("group_info:")
    #for elem in group_info:
    #    print(elem)
        
    # make the sorted_group_info -----------------------------------------------------
    sorted_group_info = []
    for [indices, stats, shape] in group_info :
        d = len(indices)
        sorted_indices = ""
        sorted_stats = []
        sorted_shape = []
        for i in range(d):
            if stats[i] in bose_type:
                sorted_indices = indices[i] + sorted_indices
                sorted_stats = [stats[i]] + sorted_stats
                sorted_shape = [shape[i]] + sorted_shape
            else:
                sorted_indices = sorted_indices + indices[i]
                sorted_stats = sorted_stats + [stats[i]]
                sorted_shape = sorted_shape + [shape[i]]
        sorted_group_info += [[ sorted_indices,make_tuple(sorted_stats),make_tuple(sorted_shape) ]]
    
    #print("sorted_group_info:")
    #for elem in sorted_group_info:
    #    print(elem)
    
    return group_info, sorted_group_info
    
def get_grouping_sign_factors(sorted_group_info, intermediate_stat):
    #print(intermediate_stat)
    
    if len(sorted_group_info) != len(intermediate_stat) :
        error("Error[get_grouping_sign_factors]: Inconsistent number of intermediate_stat and groupings!")
    
    sign_factors_info = [ "" , [] ] #indices and dimensions
    for i,[strings,stats,shapes] in enumerate(sorted_group_info):
        
        for d in range(len(stats)):
            if stats[d] == 1 and intermediate_stat[i] == -1 :
                sign_factors_info[0] += ","+strings[d]
                sign_factors_info[1] += [shapes[d]]
        #print(stats,"--->",intermediate_stat[i])
    
    #print(sign_factors_info)
    
    # generate the sign factor tensor (separately)
    sign_factors_list = [ sign_factors_info[0], [] ]
    for d in sign_factors_info[1] :
        sgn = np.array([ (-1)**param.gparity(ind) for ind in range(d) ])
        sign_factors_list[1] += [sgn]
    
    #print(sign_factors_list)
    
    return sign_factors_list

def get_intermediate_info(sorted_group_info,intermediate_stat):
    new_shape = []
    new_stats = []
    final_shape = []
    final_stats = []
    
    if len(sorted_group_info) != len(intermediate_stat):
        error("Error[get_intermediate_info]: Inconsistent number of intermediate_stat and groupings!")
    for i,[indices, stats, shape] in enumerate(sorted_group_info):
        fermi_d = 1
        bose_d  = 1
        fermi_n = 0
        bose_n  = 0
        for d in range(len(stats)):
            if stats[d] in fermi_type :
                fermi_d *= shape[d]
                fermi_n += 1
            else:
                bose_d *= shape[d]
                bose_n += 1
        if bose_n > 0 :
            new_shape += [bose_d]
            new_stats += [0]
        if fermi_n > 0 :
            new_shape += [fermi_d]
            new_stats += [intermediate_stat[i]]
            
        final_shape += [bose_d*fermi_d]
        
        if bose_n > 0 and fermi_n > 0 :
            final_stats += [hybrid_symbol]
        else:
            final_stats += [intermediate_stat[i]]
            
    new_stats = make_tuple(new_stats)
    new_shape = make_tuple(new_shape)
    final_stats = make_tuple(final_stats)
    final_shape = make_tuple(final_shape)
    
    return new_stats, new_shape, final_stats, final_shape

####################################################
##                 Reshape (block)                ##
####################################################

def join_index(*args):

    # this function tells the information about the block this product belongs to

    # x = (p,d,s) = (parity,dimension,sigma)
    ndim   = len(args)
    pi     = [ _[0] for _ in args ]
    di     = [[ _[1] for _ in args ],
              [ _[2] for _ in args ]]
    sgni   = [ _[3] for _ in args ]

    # determine which sub-block this belongs to
    block = sum(pi)%2
    sub_block = -1
    n_sub_block = 2**ndim
    sub_block_shift = 0
    sub_block_size = 0
    for ip in range(n_sub_block):

        ib = param.to_bin_parity_preserving(ip,ndim).tolist()
        if ip%2!=block :
            continue

        # compute the block size
        sub_block_size = 1
        for i in range(ndim):
            sub_block_size *= di[ib[i]][i]

        # check if this is the correct block
        if ib == pi:
            sub_block = int(math.floor(ip/2))
            break

        sub_block_shift += sub_block_size

    sgn_sigma_perm = 1
    for i in range(ndim):
        for j in range(i+1,ndim):
            sgn_sigma_perm *= (-1)**(pi[i]*pi[j])

    einsum_string_before = ",".join(char_list[:ndim])
    einsum_string_after = "".join(char_list[:ndim])
    einsum_string = einsum_string_before+"->"+einsum_string_after
    einsum_obj = [einsum_string]+sgni
    new_sgn = sgn_sigma_perm*np.reshape(np.einsum(*tuple(einsum_obj)),make_tuple(sub_block_size))

    #print("            block:",block)
    #print("        sub-block:",sub_block,"/",n_sub_block)
    #print("   sub-block size:",sub_block_size)
    #print("  sub-block shift:",sub_block_shift)
    #print("   sgn_correction:",sgn_correction)

    return (block,sub_block_shift,sub_block_size,new_sgn)

def join_legs_block(InpObj,string_inp,final_stat):

    #
    #                    IMPORTANT NOTE
    #  THIS FUNCTION IS NOT INTENDED TO BE USED BY USER ! !
    #  ONLY USED THIS IN SVD, EIG, OR HCONJUGATE ONLY ! ! !
    #  ALSO, SGN MUST BE REMEMBERED SEPARATELY  ! ! ! ! ! !
    #  IF YOU WANT TO SPLIT THE TENSOR LATER  ! ! ! ! ! ! !
    #

    string_inp = string_inp.replace(" ","")
    string_inp = denumerate(string_inp)

    final_stat = make_tuple(final_stat)
    
    Obj = InpObj.copy()

    #===============================================================================#
    #   Step 1: Get the grouping information                                        #
    #===============================================================================#
    
    # get the grouping info first
    group_info, sorted_group_info = get_group_info(string_inp, Obj.statistics, Obj.effective_shape)
    group_einfo, sorted_group_info = get_group_info(string_inp, Obj.statistics, Obj.even_shape)
    group_oinfo, sorted_group_info = get_group_info(string_inp, Obj.statistics, Obj.odd_shape)
    #group_info contains the index_string, statistics, and shape of each group
    #sorted_group_info (not used) is the same, but with bosonic indices sorted to the left

    # check if there is any hybrid index here
    if '*' in final_stat:
        error("Error[join_legs_block]: Hybrid joining is not allowed for block format.")
    for elem in group_info:
        elem_stat = elem[1]
        if ( 1 in elem_stat or -1 in elem_stat ) and ( 0 in elem_stat or '*' in elem_stat ):
            error("Error[join_legs_block]: Hybrid joining is not allowed for block format.")
    
    newshape = []
    neweshape = []
    newoshape = []
    for elem, eelem, oelem in zip(group_info,group_einfo,group_oinfo):
        dim = 1
        for d in elem[2]:
            dim*=d
        newshape+=[dim]

        if 0 in elem[1]:
            neweshape+=[dim]
            newoshape+=[dim]
            continue

        def get_deo(dimdv):
            deven = 0
            dodd = 0
            for num in range(2**len(dimdv[0])):
                config = param.to_bin_canonical(num,len(dimdv[0])).tolist()
                parity = sum(config)%2
                d = 1
                for i,p in enumerate(config):
                    d*=dimdv[p][i]
                if parity==0:
                    deven += d
                else:
                    dodd += d
            return deven,dodd

        edim, odim = get_deo([eelem[2],oelem[2]])
        neweshape+=[edim]
        newoshape+=[odim]

    newshape = tuple(newshape)
    neweshape = tuple(neweshape)
    newoshape = tuple(newoshape)

    n_indices = sum([ len(indices) for [indices,stats,shape] in group_info ])
    # the number of indices after the joining
    
    if n_indices != Obj.ndim :
        error("Error[join_legs_block]: The number of indices is not consistent with the object's shape.")

    if Obj.marked_as_joined :
        error("Error[join_legs_block]: The tensor can only be joined once!")
    
    sign_factors_list = get_grouping_sign_factors(sorted_group_info, final_stat)
    corrected_indices = sign_factors_list[0].replace(",","")
    # this one tells which index we need to multiply the correction sign factor
    # probably only need the first element (corrected_indices)

    #print(newshape,neweshape,newoshape)

    #ret = zero_block(newshape,final_stat,Obj.format,dtype=Obj.dtype)
    ret = zero_block_eo(neweshape,newoshape,final_stat,Obj.format,dtype=Obj.dtype)
    ret.statistics = final_stat

    #--clear the sgn list first
    fi = 0
    for axis in range(ret.ndim):
        if ret.statistics[axis] in fermi_type:
            ret.sgn[0][axis] = ret.sgn[1][axis]*0
            ret.sgn[1][axis] = ret.sgn[1][axis]*0
            fi += 1
        else:
            ndimaxis = len(ret.sgn[0][axis])
            ret.sgn[0][axis] = np.array(copy.deepcopy([1]*ndimaxis),dtype=int)
            ret.sgn[1][axis] = np.array(copy.deepcopy([1]*ndimaxis),dtype=int)



    group_block_info = []
    for elem in group_info:
        if 0 not in elem[1]:
            group_block_info += [elem]

    it = np.nditer(Obj.data, flags=['multi_index','refs_ok'])
    for _ in it:
        # look into each block and determine where it should be in ret

        block = it.multi_index ;
        #print(block,":",_.item().shape)
        
        block_group = []
        i = 0
        for elem in group_block_info:
            n = len(elem[0])
            block_group += [tuple(block[i:i+n])]
            i+=n
        # block_group is just the regrouping of block according the grouping info
        block_group = tuple(block_group); #print(block_group)

        # blocknum is the corresponding block in ret
        blocknum = tuple([ sum(list(blk))%2 for blk in block_group ]); 
        #print(blocknum)

        # ================= Explanation of the sub block ================= #
        #   The product 1x0 and 0x1 both results in the block 1            #
        #   So each of them are the sub block of 1                         #
        # ================================================================ #

        indices_all = string_inp.replace("(","").replace(")","").replace(",","")
        check_block = []
        fblock_shift = []
        subblocksize = []
        new_sgn_list = []
        fielem = 0
        for ielem, elem in enumerate(group_info):
            if (0 in elem[1]) or (hybrid_symbol in elem[1]):
                continue
            axis_loc = [ indices_all.index(c) for c in elem[0] ]

            d0 = [ len(Obj.sgn[0][axis]) for axis in axis_loc ]
            d1 = [ len(Obj.sgn[1][axis]) for axis in axis_loc ]

            sgn_list = [ Obj.sgn[p][axis] for axis,p in zip(axis_loc,block_group[fielem]) ]

            x_list = zip(block_group[fielem],d0,d1,sgn_list)

            _block,_shift,_size,_sgn = join_index(*x_list)
            check_block += [_block]
            fblock_shift += [_shift]
            subblocksize += [_size]
            new_sgn_list += [_sgn]
            fielem += 1

        subblocksize = make_tuple(subblocksize)
        fblock_shift = make_tuple(fblock_shift)

        # this is the size of the subblock including the bosonic axes
        subblocksizefull = []
        fcount = 0
        for i,d in enumerate(ret.data[blocknum].shape):
            if ret.statistics[i] in fermi_type:
                subblocksizefull += [subblocksize[fcount]]
                fcount+=1
            else:
                subblocksizefull += [d]
        subblocksizefull = tuple(subblocksizefull)

        # the shifting factor
        block_shift = 0*np.array(subblocksizefull)
        fcount=0
        for i in range(len(block_shift)):
            if ret.statistics[i] in fermi_type :
                block_shift[i] += fblock_shift[fcount]
                fcount += 1
        block_shift = tuple(block_shift)

        # reshape the block
        reshaped_block = np.reshape(_.item(),subblocksizefull)

        fstat = [ s for s in ret.statistics if s in fermi_type ]

        group_stat = []
        for elem in group_info:
            if 0 in elem[1]:
                continue
            else:
                group_stat += [elem[1]]

        sgn_correction = 1
        for grp_num,blk in enumerate(block_group):
            #correction sign
            if fstat[grp_num] == -1:
                for j,stat in enumerate(group_stat[grp_num]):
                    if stat == 1:
                        sgn_correction *= (-1)**blk[j]

        subit = np.nditer(reshaped_block, flags=['multi_index','refs_ok'])
        for val in subit:
            block_index = subit.multi_index
            newblock_index = tuple((np.array(block_index)+np.array(block_shift)).tolist())
            ret.data[blocknum][newblock_index] += sgn_correction*val.item()

        # make a full sgn list (with the bosonic sgn)
        fi = 0
        for axis in range(ret.ndim):
            if ret.statistics[axis] in fermi_type:
                s = new_sgn_list[fi]

                for index in range(len(s)):
                    v0 = ret.sgn[blocknum[fi]][axis][index+block_shift[axis]]
                    v1 = new_sgn_list[fi][index]
                    if v0!=0 and v0!=v1:
                        error("Error[join_legs_block]: Sign factor is not consistent!")
                    ret.sgn[blocknum[fi]][axis][index+block_shift[axis]]=v1

                #ret.sgn[blocknum[fi]][axis] += new_sgn_list[fi]
                fi += 1
    
    ret.marked_as_joined = True
    
    if incorrect_sign_size(ret):
        ret = correct_sign_size(ret)
        if incorrect_sign_size(ret):
            error("Error[??]: Unsuccessful attempt at truncating the sgn vectors.")
    return ret

def split_legs_block(InpObj,string_inp,final_stat,final_shape,final_even_shape,final_odd_shape):

    #
    #                    IMPORTANT NOTE
    #  THIS FUNCTION IS NOT INTENDED TO BE USED BY USER ! !
    #  ONLY USED THIS IN SVD, EIG, OR HCONJUGATE ONLY ! ! !
    #  ALSO, SGN MUST BE REMEMBERED SEPARATELY  ! ! ! ! ! !
    #  IF YOU WANT TO SPLIT THE TENSOR LATER  ! ! ! ! ! ! !
    #

    string_inp = string_inp.replace(" ","")
    string_inp = denumerate(string_inp)

    final_stat = make_tuple(final_stat)
    final_shape = make_tuple(final_shape)
    final_even_shape = make_tuple(final_even_shape)
    final_odd_shape = make_tuple(final_odd_shape)
    
    Obj = InpObj.copy()

    final_fstat = make_tuple([ stat for stat in final_stat if stat in fermi_type ])
    fstat = make_tuple([ stat for stat in Obj.statistics if stat in fermi_type ])
    
    #===============================================================================#
    #   Step 1: Get the grouping information                                        #
    #===============================================================================#
    
    # get the grouping info first
    final_effective_stat = np.array(final_even_shape)+np.array(final_odd_shape)
    for i in range(len(final_stat)):
        if final_stat[i] in bose_type:
            final_effective_stat[i] = final_shape[i]
    group_info, sorted_group_info = get_group_info(string_inp, final_stat, final_effective_stat)
    #group_info contains the index_string, statistics, and shape of each group
    #sorted_group_info (not used) is the same, but with bosonic indices sorted to the left

    # check if there is any hybrid index here
    if '*' in final_stat:
        error("Error[split_legs_block]: Hybrid joining is not allowed for block format.")
    for elem in group_info:
        elem_stat = elem[1]
        if ( 1 in elem_stat or -1 in elem_stat ) and ( 0 in elem_stat or '*' in elem_stat ):
            error("Error[split_legs_block]: Hybrid joining is not allowed for block format.")
    
    n_indices = sum([ len(indices) for [indices,stats,shape] in group_info ])
    # the number of indices after the joining

    if not Obj.marked_as_joined :
        error("Error[split_legs_block]: You can only split the joined tensor!")

    sign_factors_list = get_grouping_sign_factors(sorted_group_info, InpObj.statistics)
    corrected_indices = sign_factors_list[0].replace(",","")
    # this one tells which index we need to multiply the correction sign factor
    # probably only need the first element (corrected_indices)

    ret = zero_block(final_shape,final_stat,Obj.format,dtype=Obj.dtype)
    ret.statistics = final_stat
    
    # reassign each block to be of the correct shape
    it = np.nditer(ret.data, flags=['multi_index','refs_ok'])
    for _ in it:
        block = it.multi_index ; #print(block)
        sub_block_shape = []
        faxis = 0
        for axis in range(_.item().ndim):
            if ret.statistics[axis] in bose_type:
                sub_block_shape+=[final_even_shape[axis]]
            else:
                if block[faxis]==0:
                    sub_block_shape+=[final_even_shape[axis]]
                else:
                    sub_block_shape+=[final_odd_shape[axis]]
                faxis+=1
        sub_block_shape = make_tuple(sub_block_shape)
        ret.data[block] = np.zeros(sub_block_shape)
    
    for axis in range(ret.ndim):
        ret.sgn[0][axis] = ret.sgn[0][axis][:final_even_shape[axis]]
        ret.sgn[1][axis] = ret.sgn[1][axis][:final_odd_shape[axis]]

    #--clear the sgn list first
    fi = 0
    for axis in range(ret.ndim):
        if ret.statistics[axis] in fermi_type:
            #ret.sgn[0][axis] = ret.sgn[1][axis]*0
            #ret.sgn[1][axis] = ret.sgn[1][axis]*0
            fi += 1
        else:
            ndimaxis = len(ret.sgn[0][axis])
            ret.sgn[0][axis] = np.array(copy.deepcopy([1]*ndimaxis),dtype=int)
            ret.sgn[1][axis] = np.array(copy.deepcopy([1]*ndimaxis),dtype=int)

    group_block_info = []
    for elem in group_info:
        if 0 not in elem[1]:
            group_block_info += [elem]
    
    findex_loc = [ i for i,stat in enumerate(Obj.statistics) if stat in fermi_type ]
    
    fgroup_info = []
    for elem in group_info:
        if 0 in elem[1] or hybrid_symbol in elem[1]:
            continue
        else:
            fgroup_info += [elem]
    
    fndim_list = [ len(elem[0]) for elem in fgroup_info]
    fstat_group = [ elem[1] for elem in fgroup_info]
    
    subshape = [final_even_shape,final_odd_shape]
    final_even_fshape = [ dim for dim,stat in zip(final_even_shape,final_stat) if stat in fermi_type ]
    final_odd_fshape = [ dim for dim,stat in zip(final_odd_shape,final_stat) if stat in fermi_type ]
    fsubshape = [final_even_fshape,final_odd_fshape]

    it = np.nditer(Obj.data, flags=['multi_index','refs_ok'])
    for _ in it:
        # look into each block and determine where it should be in ret
        #print()
        
        blocknum = it.multi_index ; #print(block)
        #print(blocknum)
        
        block_mat = _.item()
        
        # get the fermionic shape for this block
        fshape = make_tuple([
            _.item().shape[i]
            for i in range(_.item().ndim) if Obj.statistics[i] in fermi_type ])
        #print(fshape)
        
        # list the possible products
        product_list = []
        for fdim,parity in zip(fndim_list,blocknum):
            sub_axis = []
            for ip in range(2**fdim):
                if ip%2==parity:
                    sub_axis += [ make_tuple(param.to_bin_parity_preserving(ip,fdim)) ]
            product_list += [sub_axis]
            #print()
        #print(product_list)
        
        
        # compute the size of the sub block
        sub_block_size_list = []
        axis_shift = 0
        for fi,group in enumerate(product_list):
            axis_list = []
            #print("- ",group)
            for config in group:
                #print("---- ",config)
                dim=1
                for axis,val in enumerate(config):
                #    print("------- ",fsubshape[val][axis+axis_shift],(val,axis+axis_shift))
                    dim *= fsubshape[val][axis+axis_shift]
                axis_list += [dim]
            sub_block_size_list += [axis_list]
            axis_shift += len(group[0])
        #print(sub_block_size_list)
        
        coords_list = []
        index_list = []
        coords_dim = [ len(elem) for elem in product_list ]
        
        def gen_all_coords(c_index,c_dim,c_index_list,coord_rep=None):
            complete = True
            reset_prev = False
            incr_axis = 0
            for axis in range(len(c_index)):
                if c_index[axis]+1<c_dim[axis]:
                    complete = False
                    incr_axis = axis
                    break
            if coord_rep == None:
                item = c_index.copy()
            else:
                item = [ coord_rep[i][ index ] for i,index in enumerate(c_index) ]
            c_index_list += [item]
            if not complete:
                for axis in range(incr_axis):
                    c_index[axis]=0
                c_index[incr_axis]+=1
                gen_all_coords(c_index,c_dim,c_index_list,coord_rep)
        
        coords_index = [0]*len(product_list)
        gen_all_coords(coords_index,coords_dim,coords_list,product_list)
        coords_index = [0]*len(product_list)
        gen_all_coords(coords_index,coords_dim,index_list)
        
        for index,coords in zip(index_list,coords_list):
            #print()
            #print("blocknum:",blocknum)
            #print("sub block index:",index)
            #print("new block groups:",coords)
            
            block = make_tuple(sum( [ make_list(c) for c in coords ] ,[]))
            #print("block:",block)
            
            # compute the shifting factor
            coords_shift = []
            for grp,i in enumerate(index):
                shift = 0
                for j in range(i):
                    shift += sub_block_size_list[grp][j]
                coords_shift += [shift]
            sub_block_size = [ sub_block_size_list[grp][conf] for grp,conf in enumerate(index) ]
            
            #print("shift axis:",findex_loc)
            #print("coords shift:",coords_shift)
            #print("sub block size:",sub_block_size)
            
            extracted_mat = block_mat
            
            # original = 4(ab)
            # after = 2(2*ab)
            
            
            #print("joined_shape:",extracted_mat.shape)
            #print("splitted_shape:",ret.data[block].shape)
            for ig, group in enumerate(index):
                index_from = coords_shift[ig]
                index_to = index_from + sub_block_size[ig]
                taken_axis = findex_loc[ig]
                #print("take index from", index_from,"to",index_to,"in the axis",taken_axis)
                extracted_mat = extracted_mat.take(
                    indices=range(index_from,index_to),axis=taken_axis
                    )
                #print(extracted_mat.shape)
            
            extracted_mat = np.reshape(extracted_mat,ret.data[block].shape)
            
            # compute the correction sign factor
            #print(blocknum,block)
            #print(fstat,final_fstat)
            #print("----")
            sign_correction = 1
            for joined, splitted,jstat,sstat in zip(blocknum,coords,fstat,fstat_group):
                #print()
                #print(joined,"->",splitted)
                #print(jstat,"->",sstat)
                
                for prt,st in zip(splitted,sstat):
                    if st==1 and jstat==-1:
                        sign_correction*=(-1)**prt
                
            ret.data[block] = sign_correction*extracted_mat
        
    # compute the new sigma factors
    
    ret.marked_as_joined = False
    
    if incorrect_sign_size(ret):
        ret = correct_sign_size(ret)
        if incorrect_sign_size(ret):
            error("Error[??]: Unsuccessful attempt at truncating the sgn vectors.")
    return ret

####################################################
##          Trim the Grassmann-odd parts          ##
##               (for testing only)               ##
####################################################

def trim_grassmann_odd(Obj):
    objtype=type(Obj)
    objencoder=Obj.encoder

    if(Obj.encoder == 'canonical'):
        Obj = Obj.switch_encoder(save_memory=True)
        
    if(objtype==dense):
        Obj = sparse(Obj)
    C = Obj.coords
    s0 = time.time()
    s00 = s0
    progress_space() # << Don't remove this. This is for the show_progress!

    for i in range(Obj.nnz):
        fcoords = [ ind for j,ind in enumerate(C[i]) if (Obj.statistics[j] in fermi_type)]
        if(sum(fcoords)%2 == 1):
            Obj.data.data[i] = 0

        if time.time()-s0 > 0.5 :
            show_progress(i,Obj.nnz,"trim_grassmann_odd",time=time.time()-s00)

    clear_progress()
    tab_up()


    if(objtype==dense):
        Obj = dense(Obj)

    if(objencoder == 'canonical'):
        Obj = Obj.switch_encoder(save_memory=True)

    return Obj

def is_grassmann_even(Obj):
    if Obj.encoder == 'canonical':
        Obj = Obj.switch_encoder(save_memory=True)
    if type(Obj) == dense:
        Obj = sparse(Obj)

    C = Obj.coords
    for x in C:
        parity = sum([ ind for i,ind in enumerate(x) if Obj.statistics[i] in fermi_type ])
        if parity%2!=0 :
            return False
    return True

####################################################
##          Singular value decomposition          ##
####################################################

def SortedSVD(M,cutoff=None):
    U, Λ, V = np.linalg.svd(M, full_matrices=False)


    nnz = 0
    #if np.abs(Λ[0]) < numer_cutoff:
    #    error("Error[SortedSVD]: np.linalg.svd() returns zero singular value vector!")

    for i,s in enumerate(Λ):
        if np.abs(s/(np.abs(Λ[0])+numer_cutoff)) > numer_cutoff:
            nnz+=1

    if cutoff!=None and cutoff < nnz:
        nnz = cutoff

    Λ = Λ[:nnz]
    U = U[:,:nnz]
    V = V[:nnz,:]


    return U, Λ, V
# I = cUU = VcV
def BlockSVD(Obj,cutoff=None):
    
    # performing an svd of a matrix block by block

    if(type(Obj)!=np.array and type(Obj)!=np.ndarray):
        error("Error[BlockSVD]: An input must be of type numpy.array or numpy.ndarray only!")
        

    if(Obj.ndim!=2):
        error("Error[BlockSVD]: An input must be a matrix only!")
        

    m = Obj.shape[0]
    n = Obj.shape[1]

    if(m%2!=0 and n%2!=0):
        error("Error[BlockSVD]: The matrix dimensions must be even!")
        


    if(m==0 and n==0):
        error("Error[BlockSVD]: The matrix dimensions must be at least 2!")
        

    parity_norm = 0
    for i in range(m):
        for j in range(n):
            if (i+j)%2!=0:
                parity_norm += np.linalg.norm(Obj[i,j])
    if( (not skip_parity_blocking_check) and parity_norm/(m*n/2)>1.0e-14):
        error("Error[BlockSVD]: This matrix is not constructed from a Grassmann-even tensor.")
        print("                 (Or that one of the indices are non-fermionic.)")
        

    # At this point the matrix is well-behaved

    # time to separate the blocks
    ME = np.zeros([int(m/2),int(n/2)],dtype=type(Obj[0][0]))
    MO = np.zeros([int(m/2),int(n/2)],dtype=type(Obj[0][0]))

    for i in range(int(m/2)):
        for j in range(int(n/2)):
            ME[i,j] = Obj[2*i,2*j]
            MO[i,j] = Obj[2*i+1,2*j+1]

    halfcutoff = None
    if cutoff!=None :
        halfcutoff = int(cutoff/2)

    UE, ΛE, VE = SortedSVD(ME,halfcutoff)
    UO, ΛO, VO = SortedSVD(MO,halfcutoff)

    d = max(len(ΛE),len(ΛO))
    d = int(2**math.ceil(np.log2(d)))

    def padding(Ux, Λx, Vx, padding_dimension):
        Ux = np.pad(Ux,((0,0),(0,padding_dimension)),'constant',constant_values=((0,0),(0,0)))
        Λx = np.diag(np.pad(Λx,(0,padding_dimension),'constant',constant_values=   (0,0)    ))
        Vx = np.pad(Vx,((0,padding_dimension),(0,0)),'constant',constant_values=((0,0),(0,0)))
        return Ux, Λx, Vx

    UE, ΛE, VE = padding(UE, ΛE, VE, d-len(ΛE))
    UO, ΛO, VO = padding(UO, ΛO, VO, d-len(ΛO))

    def get_full_matrix(AE, AO):
        mhalf,nhalf = AE.shape
        A = np.zeros([2*mhalf,2*nhalf],dtype=type(AE[0][0]))
        for i in range(mhalf):
            for j in range(nhalf):
                A[2*i,2*j] = AE[i,j]
                A[2*i+1,2*j+1] = AO[i,j]
        return A

    U = get_full_matrix(UE,UO)
    Λ = get_full_matrix(ΛE,ΛO)
    V = get_full_matrix(VE,VO)
    

    return U, Λ, V

def svd(InpObj,string,cutoff=None,save_memory=False):

    process_name = "svd"
    process_color = "yellow"
    process_length = 6
    step = 1
    s00 = time.time()
    progress_space()

    global skip_power_of_two_check

    # the string is of the form aaaa|bbb

    string = string.replace(" ","")
    string = denumerate(string)

    
    Obj = InpObj.copy()
    this_type = type(Obj)
    this_format = Obj.format
    this_encoder = Obj.encoder
    XObj_stats = Obj.statistics
    XObj_shape = Obj.shape

    if save_memory :
        del InpObj.data
        del InpObj
        gc.collect()
        
        
    if(this_type==sparse):
        Obj = dense(Obj)
    if(this_type not in [dense,sparse]):
        error("Error[svd]: Object type must only be dense or sparse!")
        
    # check if Obj.statistics or final_statistics is weird or not
    for stat in Obj.statistics:
        if(stat not in allowed_stat):
            error("Error[svd]: The input object contains illegal statistics. (0, 1, -1, or "+hybrid_symbol+" only)")

    if string.count("(")==string.count(")") and string.count("(")>0:
        string = string.replace(" ","")
        string = string.replace(")(","|")
        if string.count("(")>1 or string.count(")")<1:
            error("Error[svd]: Parentheses don't match")
        string = string.replace(")","")
        string = string.replace("(","")

    partition_count = 0
    for partition in separator_list:
        partition_count += string.count(partition)
    if(partition_count!=1):
        partition_string = ""
        for i, partition in enumerate(separator_list):
            if(i==0):
                partition_string += "( "
            elif(i==len(separator_list)-1):
                partition_string += ", or "
            else:
                partition_string += ", "

            partition_string += "'"+partition+"'"

            if(i==len(separator_list)-1):
                partition_string += " )"

        error("Error[svd]: The input string must contain one and only one partition "+partition_string+" in it.")

    step = show_progress(step,process_length,process_name+" "+"<"+current_memory_display()+">",color=process_color,time=time.time()-s00) #1

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #:::::      STEP 1 - JOIN LEGS BAESD ON THE GROUPINGS                                       :::::
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    # count the number of indices in the two groups
    n_left = 0
    n_right = 0
    partition_found = False
    for char in string:
        if char in separator_list :
            partition_found = True
            continue
        if(partition_found):
            n_right+=1
        else:
            n_left+=1

    join_legs_string_input = string
    for partition in separator_list:
        join_legs_string_input = join_legs_string_input.replace(partition,")(")
    join_legs_string_input = "("+join_legs_string_input+")"

    shape_left  = Obj.shape[:n_left]
    stats_left  = Obj.statistics[:n_left]
    shape_right = Obj.shape[n_left:]
    stats_right = Obj.statistics[n_left:]

    # check if one of the legs (or both) are fermion with dimension 1

    left_bond1  = list(stats_left)  in [[-1],[1]] and list(shape_left)  == [1]
    right_bond1 = list(stats_right) in [[-1],[1]] and list(shape_right) == [1]

    if left_bond1 or right_bond1:

        Norm = Obj.norm
        M = Obj.force_format("matrix")

        if left_bond1 :
            U = dense([[1]],statistics=(*stats_left,1),format='matrix')
            Λ = dense([[Norm]],statistics=(-1,1),format='matrix')
            V = M/Norm
            Vstat = list(V.statistics)
            Vstat[0] = -1
            V.statistics = tuple(Vstat)
        else :
            U = M/Norm
            Λ = dense([[Norm]],statistics=(-1,1),format='matrix')
            V = dense([[1]],statistics=(-1,*stats_right),format='matrix')
            Ustat = list(U.statistics)
            Ustat[-1] = 1
            U.statistics = tuple(Ustat)

        if(this_format == 'standard'):
            U = U.switch_format(save_memory=True)
            Λ = Λ.switch_format(save_memory=True)
            V = V.switch_format(save_memory=True)

        if(this_encoder == 'parity-preserving'):
            U = U.switch_encoder(save_memory=True)
            Λ = Λ.switch_encoder(save_memory=True)
            V = V.switch_encoder(save_memory=True)

        if(this_type==sparse):
            U = sparse(U)
            Λ = sparse(Λ)
            V = sparse(V)

        return U,Λ,V

    def zero_or_else(vector,value):
        for elem in vector:
            if elem!=0:
                return value
        return 0
    def get_stat(vector,prefer=None):
        boson_count = 0
        fermi_count = 0
        for elem in vector:
            if elem in bose_type :
                boson_count += 1
            if elem in fermi_type :
                fermi_count += 1

        if(boson_count==0 and fermi_count>0):
            if prefer != None:
                return prefer
            else:
                return 1
        elif(boson_count>0 and fermi_count==0):
            if prefer != None:
                return prefer
            else:
                return 0
        elif(boson_count>0 and fermi_count>0):
            return hybrid_symbol
    
    intermediate_stat = ( zero_or_else(stats_left,-1),zero_or_else(stats_right,1) )

    step = show_progress(step,process_length,process_name+" "+"<"+current_memory_display()+">",color=process_color,time=time.time()-s00) #2
    Obj = Obj.join_legs(join_legs_string_input,"matrix",intermediate_stat=intermediate_stat,save_memory=True)
    Obj = Obj.force_encoder("parity-preserving")

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #:::::      STEP 2 - BLOCK SVD (MAKE SURE IT'S PARITY-PRESERVING!)                          :::::
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    step = show_progress(step,process_length,process_name+" "+"<"+current_memory_display()+">",color=process_color,time=time.time()-s00) #3
    if Obj.statistics[0]==0 or Obj.statistics[1]==0:
        U, Λ, V = SortedSVD(Obj.data,cutoff)
        Λ = np.diag(Λ)
    else:
        U, Λ, V = BlockSVD(Obj.data,cutoff)

    step = show_progress(step,process_length,process_name+" "+"<"+current_memory_display()+">",color=process_color,time=time.time()-s00) #4
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #:::::      STEP 3 - RECONSTRUCT U, Λ, and V AS GRASSMANN TENSORS                           :::::
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    skip_power_of_two_check = True

    #the first way is to form the tensor first, then split
    Λstatleft = -1
    Λstatright = +1
    
    if Obj.statistics[0]==0:
        U = dense(U,encoder="parity-preserving",format="matrix",statistics=(0,0))
        Λ = dense(Λ,encoder="parity-preserving",format="matrix",statistics=(0,0))
        V = dense(V,encoder="parity-preserving",format="matrix",statistics=(0,Obj.statistics[1]))
        Λstatleft = 0
        Λstatright = 0
    elif Obj.statistics[1]==0:
        U = dense(U,encoder="parity-preserving",format="matrix",statistics=(Obj.statistics[0],0))
        Λ = dense(Λ,encoder="parity-preserving",format="matrix",statistics=(0,0))
        V = dense(V,encoder="parity-preserving",format="matrix",statistics=(0,0))
        Λstatleft = 0
        Λstatright = 0
    else:
        U = dense(U,encoder="parity-preserving",format="matrix",statistics=(Obj.statistics[0],1))
        Λ = dense(Λ,encoder="parity-preserving",format="matrix",statistics=(-1,1))
        V = dense(V,encoder="parity-preserving",format="matrix",statistics=(-1,Obj.statistics[1]))
    dΛ = Λ.shape[0]

    skip_power_of_two_check = False
    step = show_progress(step,process_length,process_name+" "+"<"+current_memory_display()+">",color=process_color,time=time.time()-s00) #5
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #:::::      STEP 4 - Split the legs                                                         :::::
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    # count the number of indices in the two groups
    Uind = ""
    Vind = ""
    Ustats = []
    Vstats = []
    Ushape = []
    Vshape = []
    partition_found = False
    for i,char in enumerate(string):
        if char in separator_list :
            partition_found = True
            continue
        if(partition_found):
            Vind+=char
            Vstats+=[XObj_stats[i-1]]
            Vshape+=[XObj_shape[i-1]]
        else:
            Uind+=char
            Ustats+=[XObj_stats[i]]
            Ushape+=[XObj_shape[i]]

    new_ind1 = ""
    for char in char_list:
        if char not in Uind+Vind:
            new_ind1 = char
            break

    Uind   = "("+Uind+")" + new_ind1
    Vind   = new_ind1 + "("+Vind+")"
    Ustats = tuple(Ustats + [-Λstatleft])
    Vstats = tuple([-Λstatright] + Vstats)
    Ushape = tuple(Ushape + [dΛ])
    Vshape = tuple([dΛ] + Vshape)
    
    step = show_progress(step,process_length,process_name+" "+"<"+current_memory_display()+">",color=process_color,time=time.time()-s00) #6
    U = U.split_legs(Uind,Ustats,Ushape,intermediate_stat=(-1,1),save_memory=True)
    Λ = Λ.switch_encoder(save_memory=True)
    V = V.split_legs(Vind,Vstats,Vshape,intermediate_stat=(-1,1),save_memory=True)
    
    if(this_format == 'standard'):
        U = U.switch_format(save_memory=True)
        Λ = Λ.switch_format(save_memory=True)
        V = V.switch_format(save_memory=True)

    if(this_encoder == 'parity-preserving'):
        U = U.switch_encoder(save_memory=True)
        Λ = Λ.switch_encoder(save_memory=True)
        V = V.switch_encoder(save_memory=True)

    if(this_type==sparse):
        U = sparse(U)
        Λ = sparse(Λ)
        V = sparse(V)

    clear_progress()
    tab_up()

    return U, Λ, V

####################################################
##            Eigen value decomposition           ##
####################################################

def SortedEig(M,cutoff=None,debug_mode=False):

    NonHeritianNorm = np.linalg.norm(M-np.conjugate(np.transpose(M)))/np.linalg.norm(M)
    if NonHeritianNorm>numer_cutoff :
        error("Error[SortedEig]: The input matrix is not Hermitian!")

    U, Λ, V = np.linalg.svd(M, full_matrices=False)

    nnz = 0
    #if np.abs(Λ[0]) < numer_cutoff:
    #    error("Error[SortedSVD]: np.linalg.svd() returns zero singular value vector!")

    for i,s in enumerate(Λ):
        if np.abs(s/(np.abs(Λ[0])+numer_cutoff)) > numer_cutoff:
            nnz+=1

    if cutoff!=None and cutoff < nnz:
        nnz = cutoff

    Λ = Λ[:nnz]
    U = U[:,:nnz]
    V = V[:nnz,:]
    cU = np.conjugate(np.transpose(U))
    ΛVU = np.einsum('i,ij,jk->ik',Λ,V,U)
    Λ = np.einsum('i,ij,jk->k',Λ,V,U)

    return U, Λ, cU
# I = cUU
def BlockEig(Obj,cutoff=None,debug_mode=False):
    
    # performing an svd of a matrix block by block

    if(type(Obj)!=np.array and type(Obj)!=np.ndarray):
        error("Error[BlockEig]: An input must be of type numpy.array or numpy.ndarray only!")
        

    if(Obj.ndim!=2):
        error("Error[BlockEig]: An input must be a matrix only!")
        

    m = Obj.shape[0]
    n = Obj.shape[1]

    if(m%2!=0 and n%2!=0):
        error("Error[BlockEig]: The matrix dimensions must be even!")
        


    if(m==0 and n==0):
        error("Error[BlockEig]: The matrix dimensions must be at least 2!")
        

    parity_norm = 0
    for i in range(m):
        for j in range(n):
            if (i+j)%2!=0:
                parity_norm += np.linalg.norm(Obj[i,j])
    if( (not skip_parity_blocking_check) and parity_norm/(m*n/2)>1.0e-14):
        error("Error[BlockEig]: This matrix is not constructed from a Grassmann-even tensor.")
        print("                 (Or that one of the indices are non-fermionic.)")
        

    # At this point the matrix is well-behaved

    # time to separate the blocks
    ME = np.zeros([int(m/2),int(n/2)],dtype=type(Obj[0][0]))
    MO = np.zeros([int(m/2),int(n/2)],dtype=type(Obj[0][0]))

    for i in range(int(m/2)):
        for j in range(int(n/2)):
            ME[i,j] = Obj[2*i,2*j]
            MO[i,j] = Obj[2*i+1,2*j+1]

    halfcutoff = None
    if cutoff!=None :
        halfcutoff = int(cutoff/2)

    UE, ΛE, cUE = SortedEig(ME,halfcutoff,debug_mode)
    UO, ΛO, cUO = SortedEig(MO,halfcutoff,debug_mode)

    d = max(len(ΛE),len(ΛO))
    d = int(2**math.ceil(np.log2(d)))

    def padding(Ux, Λx, cUx, padding_dimension):
        Ux = np.pad(Ux,((0,0),(0,padding_dimension)),'constant',constant_values=((0,0),(0,0)))
        Λx = np.diag(np.pad(Λx,(0,padding_dimension),'constant',constant_values=(0,0)       ))
        cUx = np.pad(cUx,((0,padding_dimension),(0,0)),'constant',constant_values=((0,0),(0,0)))
        return Ux, Λx, cUx

    UE, ΛE, cUE = padding(UE, ΛE, cUE, d-len(ΛE))
    UO, ΛO, cUO = padding(UO, ΛO, cUO, d-len(ΛO))

    def get_full_matrix(AE, AO):
        mhalf,nhalf = AE.shape
        A = np.zeros([2*mhalf,2*nhalf],dtype=type(AE[0][0]))
        for i in range(mhalf):
            for j in range(nhalf):
                A[2*i,2*j] = AE[i,j]
                A[2*i+1,2*j+1] = AO[i,j]
        return A

    U = get_full_matrix(UE,UO)
    Λ = get_full_matrix(ΛE,ΛO)
    cU = get_full_matrix(cUE,cUO)
    

    return U, Λ, cU

def eig(InpObj,string,cutoff=None,debug_mode=False,save_memory=False):

    process_name = "eig"
    process_color = "yellow"
    process_length = 6
    step = 1
    s00 = time.time()
    progress_space()

    global skip_power_of_two_check

    # the string is of the form aaaa|bbb

    string = string.replace(" ","")
    string = denumerate(string)

    
    Obj = InpObj.copy()
    this_type = type(Obj)
    this_format = Obj.format
    this_encoder = Obj.encoder
    XObj_stats = Obj.statistics
    XObj_shape = Obj.shape

    if save_memory :
        del InpObj.data
        del InpObj
        gc.collect()
        
        
    if(this_type==sparse):
        Obj = dense(Obj)
    if(this_type not in [dense,sparse]):
        error("Error[eig]: Object type must only be dense or sparse!")
        
    # check if Obj.statistics or final_statistics is weird or not
    for stat in Obj.statistics:
        if(stat not in allowed_stat):
            error("Error[eig]: The input object contains illegal statistics. (0, 1, -1, or "+hybrid_symbol+" only)")

    if string.count("(")==string.count(")") and string.count("(")>0:
        string = string.replace(" ","")
        string = string.replace(")(","|")
        if string.count("(")>1 or string.count(")")<1:
            error("Error[eig]: Parentheses don't match")
        string = string.replace(")","")
        string = string.replace("(","")

    partition_count = 0
    for partition in separator_list:
        partition_count += string.count(partition)
    if(partition_count!=1):
        partition_string = ""
        for i, partition in enumerate(separator_list):
            if(i==0):
                partition_string += "( "
            elif(i==len(separator_list)-1):
                partition_string += ", or "
            else:
                partition_string += ", "

            partition_string += "'"+partition+"'"

            if(i==len(separator_list)-1):
                partition_string += " )"

        error("Error[eig]: The input string must contain one and only one partition "+partition_string+" in it.")
    
    step = show_progress(step,process_length,process_name+" "+"<"+current_memory_display()+">",color=process_color,time=time.time()-s00) #1
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #:::::      STEP 1 - JOIN LEGS BAESD ON THE GROUPINGS                                       :::::
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    # count the number of indices in the two groups
    n_left = 0
    n_right = 0
    partition_found = False
    for char in string:
        if char in separator_list :
            partition_found = True
            continue
        if(partition_found):
            n_right+=1
        else:
            n_left+=1

    join_legs_string_input = string
    for partition in separator_list:
        join_legs_string_input = join_legs_string_input.replace(partition,")(")
    join_legs_string_input = "("+join_legs_string_input+")"

    shape_left  = Obj.shape[:n_left]
    stats_left  = Obj.statistics[:n_left]
    shape_right = Obj.shape[n_left:]
    stats_right = Obj.statistics[n_left:]

    left_bond1  = list(stats_left)  in [[-1],[1]] and list(shape_left)  == [1]
    right_bond1 = list(stats_right) in [[-1],[1]] and list(shape_right) == [1]

    if left_bond1 or right_bond1:

        Norm = Obj.norm
        M = Obj.force_format("matrix")

        if left_bond1 :
            U = dense([[1]],statistics=(*stats_left,1),format='matrix')
            Λ = dense([[Norm]],statistics=(-1,1),format='matrix')
            V = M/Norm
            Vstat = list(V.statistics)
            Vstat[0] = -1
            V.statistics = tuple(Vstat)
        else :
            U = M/Norm
            Λ = dense([[Norm]],statistics=(-1,1),format='matrix')
            V = dense([[1]],statistics=(-1,*stats_right),format='matrix')
            Ustat = list(U.statistics)
            Ustat[-1] = 1
            U.statistics = tuple(Ustat)

        if(this_format == 'standard'):
            U = U.switch_format(save_memory=True)
            Λ = Λ.switch_format(save_memory=True)
            V = V.switch_format(save_memory=True)

        if(this_encoder == 'parity-preserving'):
            U = U.switch_encoder(save_memory=True)
            Λ = Λ.switch_encoder(save_memory=True)
            V = V.switch_encoder(save_memory=True)

        if(this_type==sparse):
            U = sparse(U)
            Λ = sparse(Λ)
            V = sparse(V)

        return U,Λ,V

    def zero_or_else(vector,value):
        for elem in vector:
            if elem!=0:
                return value
        return 0
    def get_stat(vector,prefer=None):
        boson_count = 0
        fermi_count = 0
        for elem in vector:
            if elem in bose_type :
                boson_count += 1
            if elem in fermi_type :
                fermi_count += 1

        if(boson_count==0 and fermi_count>0):
            if prefer != None:
                return prefer
            else:
                return 1
        elif(boson_count>0 and fermi_count==0):
            if prefer != None:
                return prefer
            else:
                return 0
        elif(boson_count>0 and fermi_count>0):
            return hybrid_symbol
    
    step = show_progress(step,process_length,process_name+" "+"<"+current_memory_display()+">",color=process_color,time=time.time()-s00) #2
    intermediate_stat = ( zero_or_else(stats_left,-1),zero_or_else(stats_right,1) )
    Obj = Obj.join_legs(join_legs_string_input,"matrix",intermediate_stat=intermediate_stat,save_memory=True)
    Obj = Obj.force_encoder("parity-preserving")
    #print("eig",Obj.encoder,Obj.format)
    #exit()

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #:::::      STEP 2 - BLOCK SVD (MAKE SURE IT'S PARITY-PRESERVING!)                          :::::
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    step = show_progress(step,process_length,process_name+" "+"<"+current_memory_display()+">",color=process_color,time=time.time()-s00) #3
    if Obj.statistics[0]==0 or Obj.statistics[1]==0:
        U, Λ, V = SortedEig(Obj.data,cutoff,debug_mode)
        Λ = np.diag(Λ)
    else:
        U, Λ, V = BlockEig(Obj.data,cutoff,debug_mode)

    step = show_progress(step,process_length,process_name+" "+"<"+current_memory_display()+">",color=process_color,time=time.time()-s00) #4
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #:::::      STEP 3 - RECONSTRUCT U, Λ, and V AS GRASSMANN TENSORS                           :::::
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    skip_power_of_two_check = True

    #the first way is to form the tensor first, then split
    Λstatleft = -1
    Λstatright = +1
    
    if Obj.statistics[0]==0:
        U = dense(U,encoder="parity-preserving",format="matrix",statistics=(0,0))
        Λ = dense(Λ,encoder="parity-preserving",format="matrix",statistics=(0,0))
        V = dense(V,encoder="parity-preserving",format="matrix",statistics=(0,Obj.statistics[1]))
        Λstatleft = 0
        Λstatright = 0
    elif Obj.statistics[1]==0:
        U = dense(U,encoder="parity-preserving",format="matrix",statistics=(Obj.statistics[0],0))
        Λ = dense(Λ,encoder="parity-preserving",format="matrix",statistics=(0,0))
        V = dense(V,encoder="parity-preserving",format="matrix",statistics=(0,0))
        Λstatleft = 0
        Λstatright = 0
    else:
        U = dense(U,encoder="parity-preserving",format="matrix",statistics=(Obj.statistics[0],1))
        Λ = dense(Λ,encoder="parity-preserving",format="matrix",statistics=(-1,1))
        V = dense(V,encoder="parity-preserving",format="matrix",statistics=(-1,Obj.statistics[1]))
    dΛ = Λ.shape[0]

    step = show_progress(step,process_length,process_name+" "+"<"+current_memory_display()+">",color=process_color,time=time.time()-s00) #5
    skip_power_of_two_check = False
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #:::::      STEP 4 - Split the legs                                                         :::::
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    # count the number of indices in the two groups
    Uind = ""
    Vind = ""
    Ustats = []
    Vstats = []
    Ushape = []
    Vshape = []
    partition_found = False
    for i,char in enumerate(string):
        if char in separator_list :
            partition_found = True
            continue
        if(partition_found):
            Vind+=char
            Vstats+=[XObj_stats[i-1]]
            Vshape+=[XObj_shape[i-1]]
        else:
            Uind+=char
            Ustats+=[XObj_stats[i]]
            Ushape+=[XObj_shape[i]]

    new_ind1 = ""
    for char in char_list:
        if char not in Uind+Vind:
            new_ind1 = char
            break

    Uind   = "("+Uind+")" + new_ind1
    Vind   = new_ind1 + "("+Vind+")"
    Ustats = tuple(Ustats + [-Λstatleft])
    Vstats = tuple([-Λstatright] + Vstats)
    Ushape = tuple(Ushape + [dΛ])
    Vshape = tuple([dΛ] + Vshape)
    
    step = show_progress(step,process_length,process_name+" "+"<"+current_memory_display()+">",color=process_color,time=time.time()-s00) #6
    U = U.split_legs(Uind,Ustats,Ushape,intermediate_stat=(-1,1),save_memory=True)
    Λ = Λ.switch_encoder(save_memory=True)
    V = V.split_legs(Vind,Vstats,Vshape,intermediate_stat=(-1,1),save_memory=True)
    
    if(this_format == 'standard'):
        U = U.switch_format(save_memory=True)
        Λ = Λ.switch_format(save_memory=True)
        V = V.switch_format(save_memory=True)

    if(this_encoder == 'parity-preserving'):
        U = U.switch_encoder(save_memory=True)
        Λ = Λ.switch_encoder(save_memory=True)
        V = V.switch_encoder(save_memory=True)

    if(this_type==sparse):
        U = sparse(U)
        Λ = sparse(Λ)
        V = sparse(V)

    clear_progress()
    tab_up()
    
    return U, Λ, V

####################################################
##               Block decomposition              ##
####################################################

def decompose_block(InpObj,string,cutoff=None,save_memory=False,option=None):

    string = string.replace(" ","")
    string = denumerate(string)

    Obj = InpObj.copy()
    this_format = Obj.format

    if save_memory :
        del InpObj.data
        del InpObj
        gc.collect()

    # check if Obj.statistics or final_statistics is weird or not
    for stat in Obj.statistics:
        if stat not in allowed_stat :
            error("Error[svd_block]: The input object contains illegal statistics. (0, 1, -1, or "+hybrid_symbol+" only)")

    # convert to standard format first
    Obj = Obj.force_format("standard")

    if string.count("(")==string.count(")") and string.count("(")>0:
        string = string.replace(" ","")
        string = string.replace(")(","|")
        if string.count("(")>1 or string.count(")")<1:
            error("Error[svd_block]: Parentheses don't match")
        string = string.replace(")","")
        string = string.replace("(","")

    partition_count = 0
    for partition in separator_list:
        partition_count += string.count(partition)
    if(partition_count!=1):
        partition_string = ""
        for i, partition in enumerate(separator_list):
            if(i==0):
                partition_string += "( "
            elif(i==len(separator_list)-1):
                partition_string += ", or "
            else:
                partition_string += ", "

            partition_string += "'"+partition+"'"

            if(i==len(separator_list)-1):
                partition_string += " )"

        error("Error[svd_block]: The input string must contain one and only one partition "+partition_string+" in it.")

    for separator in separator_list:
        if separator == "|":
            continue
        else:
            string = string.replace(separator,"|")


    # ===========================================================================
    # Patch 1.4.2 --- accomodate the case with bond dimension 1
    # ===========================================================================

    # count the number of indices in the two groups
    n_left = 0
    n_right = 0
    partition_found = False
    for char in string:
        if char in separator_list :
            partition_found = True
            continue
        if(partition_found):
            n_right+=1
        else:
            n_left+=1

    join_legs_string_input = string
    for partition in separator_list:
        join_legs_string_input = join_legs_string_input.replace(partition,")(")
    join_legs_string_input = "("+join_legs_string_input+")"

    shape_left  = Obj.shape[:n_left]
    stats_left  = Obj.statistics[:n_left]
    shape_right = Obj.shape[n_left:]
    stats_right = Obj.statistics[n_left:]

    # check if one of the legs (or both) are fermion with dimension 1

    left_bond1  = list(stats_left)  in [[-1],[1]] and list(shape_left)  == [1]
    right_bond1 = list(stats_right) in [[-1],[1]] and list(shape_right) == [1]

    if left_bond1 or right_bond1:

        Norm = Obj.norm
        M = Obj.force_format("matrix")

        if left_bond1 :
            U = dense([[1]],statistics=(*stats_left,1),format='matrix').toblock()
            Λ = dense([[Norm]],statistics=(-1,1),format='matrix').toblock()
            V = M/Norm
            Vstat = list(V.statistics)
            Vstat[0] = -1
            V.statistics = tuple(Vstat)
        else :
            U = M/Norm
            Λ = dense([[Norm]],statistics=(-1,1),format='matrix').toblock()
            V = dense([[1]],statistics=(-1,*stats_right),format='matrix').toblock()
            Ustat = list(U.statistics)
            Ustat[-1] = 1
            U.statistics = tuple(Ustat)

        if(this_format == 'standard'):
            U = U.switch_format()
            Λ = Λ.switch_format()
            V = V.switch_format()

        return U,Λ,V


    # ===========================================================================
    # These are the index info before the rearrangement
    # ===========================================================================

    index_num = 0
    subscripts1 = ["",""]
    stats1 = [[],[]]
    shape1 = [[],[]]
    even_shape1 = [[],[]]
    odd_shape1 = [[],[]]

    lr = 0
    for c in string:
        if c=="|":
            lr = 1
            continue
        else:
            subscripts1[lr] += c
            stats1[lr] += [Obj.statistics[index_num]]
            shape1[lr] += [Obj.shape[index_num]]
            even_shape1[lr] += [Obj.even_shape[index_num]]
            odd_shape1[lr] += [Obj.odd_shape[index_num]]
            index_num += 1

    if False:
        print("       Set 1")
        print("  subscripts:",subscripts1)
        print("  statistics:",stats1)
        print("  full shape:",shape1)
        print("  even shape:",even_shape1)
        print("   odd shape:",odd_shape1)



    # ===========================================================================
    # Now move bosonic indices to the left of fermionic indices
    # ===========================================================================

    subscripts2 = ["",""]
    stats2 = [[],[]]
    shape2 = [[],[]]
    even_shape2 = [[],[]]
    odd_shape2 = [[],[]]

    for lr in [0,1]:
        for char,stat,dim,edim,odim in zip(subscripts1[lr],stats1[lr],shape1[lr],even_shape1[lr],odd_shape1[lr]):
            if stat == 0:
                subscripts2[lr] = char+subscripts2[lr]
                stats2[lr] = [stat]+stats2[lr]
                shape2[lr] = [dim]+shape2[lr]
                even_shape2[lr] = [edim]+even_shape2[lr]
                odd_shape2[lr] = [odim]+odd_shape2[lr]
            else:
                subscripts2[lr] += char
                stats2[lr] += [stat]
                shape2[lr] += [dim]
                even_shape2[lr] += [edim]
                odd_shape2[lr] += [odim]

    if False:
        print("       Set 2")
        print("  subscripts:",subscripts2)
        print("  statistics:",stats2)
        print("  full shape:",shape2)
        print("  even shape:",even_shape2)
        print("   odd shape:",odd_shape2)

    # ===========================================================================
    # Merge the infos into one vector
    # ===========================================================================

    subscripts3 = subscripts2[0]+subscripts2[1]
    stats3 = stats2[0]+stats2[1]
    shape3 = shape2[0]+shape2[1]
    even_shape3 = even_shape2[0]+even_shape2[1]
    odd_shape3 = odd_shape2[0]+odd_shape2[1]

    if False:
        print("       Set 3")
        print("  subscripts:",subscripts3)
        print("  statistics:",stats3)
        print("  full shape:",shape3)
        print("  even shape:",even_shape3)
        print("   odd shape:",odd_shape3)


    # ===========================================================================
    # Rearrange with einsum
    # ===========================================================================

    einsum_string = string.replace("|","")+"->"+subscripts3
    Obj3 = einsum_block(einsum_string,Obj)


    if save_memory :
        del Obj.data
        del Obj
        gc.collect()

    # ===========================================================================
    # join the indices into four groups
    # ===========================================================================

    group_string = ""
    group_string_left = ""
    group_string_right = ""
    is_left = True
    for i,c in enumerate(subscripts3):

        add_middle = False
        if i==len(subscripts2[0]):
            add_middle = True
            is_left = False
            group_string += ","

        if i>0 :
            if (
                ((Obj3.statistics[i] in bose_type) and (Obj3.statistics[i-1] in fermi_type))
                or
                ((Obj3.statistics[i] in fermi_type) and (Obj3.statistics[i-1] in bose_type))
                ):
                if not add_middle:
                    group_string += ","
                    if is_left:
                        group_string_left += ","
                    else:
                        group_string_right += ","
        group_string += c

        if is_left:
            group_string_left += c
        else:
            group_string_right += c

    final_stat4 = []
    group_substring = ""
    if 0 in stats2[0]:
        final_stat4+=[0]
        group_substring += "x"
    if 1 in stats2[0] or -1 in stats2[0]:
        final_stat4+=[-1]
        group_substring += "y"
    group_substring += "|"
    if 0 in stats2[1]:
        final_stat4+=[0]
        group_substring += "z"
    if 1 in stats2[1] or -1 in stats2[1]:
        final_stat4+=[1]
        group_substring += "w"
    final_stat4 = make_tuple(final_stat4)

    group_string_left = "("+group_string_left.replace(",",")(")+")"
    group_string_right = "("+group_string_right.replace(",",")(")+")"

    if False:
        print("-------------------- -2 -------------------------------------------------------")
        Obj3.force_format("standard").display()

    Obj4 = join_legs_block(Obj3,group_string,final_stat4)


    l = 0
    r = 0
    if 0 in stats2[0]:
        l+=1
    if (1 in stats2[0]) or (-1  in stats2[0]):
        l+=1
    if 0 in stats2[1]:
        r+=1
    if (1 in stats2[1]) or (-1 in stats2[1]):
        r+=1

    stats4 = [[],[]]
    shape4 = [[],[]]
    even_shape4 = [[],[]]
    odd_shape4  = [[],[]]

    index = 0
    for il in range(l):
        stats4[0] += Obj4.statistics[index],
        shape4[0] += Obj4.shape[index],
        even_shape4[0] += Obj4.even_shape[index],
        odd_shape4[0] += Obj4.odd_shape[index],
        index+=1
    for ir in range(r): 
        stats4[1] += Obj4.statistics[index],
        shape4[1] += Obj4.shape[index],
        even_shape4[1] += Obj4.even_shape[index],
        odd_shape4[1] += Obj4.odd_shape[index],
        index+=1

    if False:
        print("       Set 4")
        print("  statistics:",stats4)
        print("  full shape:",shape4)
        print("  even shape:",even_shape4)
        print("   odd shape:",odd_shape4)


    if save_memory :
        del Obj3.data
        del Obj3
        gc.collect()

    # ===========================================================================
    # Convert to matrix format
    # ===========================================================================

    #Obj4.info("Set 4")

    Obj5 = Obj4.force_format("matrix")
    sgn5 = Obj5.sgn

    if save_memory :
        del Obj4.data
        del Obj4
        gc.collect()

    if False:
        print("-------------------- -1 -------------------------------------------------------")
        Obj5.force_format("standard").display()

    # ===========================================================================
    # Extract the block matrix
    # ===========================================================================

    it = np.nditer(Obj5.data, flags=['multi_index','refs_ok'])
    for val in it:
        #total index
        block = it.multi_index
        norm = np.linalg.norm(val.item())
        parity = sum(block)%2
        
        if parity==1 and norm > numer_cutoff:
            error("Error[SVD/EigV_block]: This function only works for even tensors!")

    def prod(x):
        x = make_list(x)
        ret = 1
        for val in x:
            ret *= val
        return ret

    merged_shape_even = ( prod(even_shape4[0]),prod(even_shape4[1]) )
    merged_shape_odd = ( prod(odd_shape4[0]),prod(odd_shape4[1]) )

    try:
        ME = np.reshape(Obj5.data[0,0],merged_shape_even)
        MO = np.reshape(Obj5.data[1,1],merged_shape_odd)
    except:
        error("Error[SVD/EigV_block]: This function do not support the case where one of the decomposed branches is totally bosonic.\nPlease do it in the dense/sparse format instead. I plan to solve this issue in the future.")

    # ===========================================================================
    # SVD
    # ===========================================================================

    if cutoff==None :
        chiE = None
        chiO = None
    else:
        chiE = int(math.ceil(cutoff/2))
        chiO = int(math.floor(cutoff/2))

    if option=="SVD":
        UE,SE,VE = SortedSVD(ME,cutoff=chiE)
        UO,SO,VO = SortedSVD(MO,cutoff=chiO)
    elif option=="Eig":
        UE,SE,VE = SortedEig(ME,cutoff=chiE)
        UO,SO,VO = SortedEig(MO,cutoff=chiO)
    else:
        error("Error[decompose_block]: Unknown decomposition type")


    SE = np.diag(SE)
    SO = np.diag(SO)

    NE = SE.shape[0]
    NO = SO.shape[0]

    Nfull = 2**int(1+np.ceil(np.log2(max(NE,NO))))

    # ===========================================================================
    # Construct the template matrix
    # ===========================================================================

    U_even_shape = make_tuple(even_shape4[0]+[NE])
    S_even_shape = (NE,NE)
    V_even_shape = make_tuple([NE]+even_shape4[1])

    U_odd_shape = make_tuple(odd_shape4[0]+[NO])
    S_odd_shape = (NO,NO)
    V_odd_shape = make_tuple([NO]+odd_shape4[1])

    U_shape = make_tuple(shape4[0]+[Nfull])
    S_shape = (Nfull,Nfull)
    V_shape = make_tuple([Nfull]+shape4[1])

    U_stat = make_tuple(stats4[0]+[1])
    S_stat = (-1,1)
    V_stat = make_tuple([-1]+stats4[1])

    # Don't forget the off-diagonal matrix !!!!!

    U_01_shape = make_tuple(even_shape4[0]+[NO])
    U_10_shape = make_tuple(odd_shape4[0]+[NE])

    S_01_shape = (NE,NO)
    S_10_shape = (NO,NE)

    V_01_shape = make_tuple([NE]+odd_shape4[1])
    V_10_shape = make_tuple([NO]+even_shape4[1])

    UE = np.reshape(UE,U_even_shape)
    VE = np.reshape(VE,V_even_shape)
    UO = np.reshape(UO,U_odd_shape)
    VO = np.reshape(VO,V_odd_shape)

    U = zero_block(U_shape,U_stat,format='matrix')
    S = zero_block(S_shape,S_stat,format='matrix')
    V = zero_block(V_shape,V_stat,format='matrix')

    U.data[0,0] = UE
    U.data[1,1] = UO
    U.data[0,1] = np.zeros(U_01_shape)
    U.data[1,0] = np.zeros(U_10_shape)

    S.data[0,0] = SE
    S.data[1,1] = SO
    S.data[0,1] = np.zeros(S_01_shape)
    S.data[1,0] = np.zeros(S_10_shape)

    V.data[0,0] = VE
    V.data[1,1] = VO
    V.data[0,1] = np.zeros(V_01_shape)
    V.data[1,0] = np.zeros(V_10_shape)

    U.marked_as_joined = True
    V.marked_as_joined = True

    # ===========================================================================
    # The sign factor is not correct!!!
    # ===========================================================================

    sgn_E = S.sgn[0][0][:NE]
    sgn_O = S.sgn[1][0][:NO]

    U.sgn[0][l] = sgn_E.copy()
    U.sgn[1][l] = sgn_O.copy()

    S.sgn[0][0] = sgn_E.copy()
    S.sgn[1][0] = sgn_O.copy()

    S.sgn[0][1] = sgn_E.copy()
    S.sgn[1][1] = sgn_O.copy()

    V.sgn[0][0] = sgn_E.copy()
    V.sgn[1][0] = sgn_O.copy()

    for axis in range(l):
        U.sgn[0][axis] = sgn5[0][axis]
        U.sgn[1][axis] = sgn5[1][axis]

    for axis in range(r):
        V.sgn[0][1+axis] = sgn5[0][l+axis]
        V.sgn[1][1+axis] = sgn5[1][l+axis]

    if False:
        print("-------------------- 1 -------------------------------------------------------")
        X0 = einsum_block('ij,jk,kl->il',U,S,V).force_format("standard")
        X0.marked_as_joined = True
        X0.display()

    U = U.force_format(this_format)
    S = S.force_format(this_format)
    V = V.force_format(this_format)

    # ===========================================================================
    # Split back
    # ===========================================================================

    U_even_shape = make_tuple(even_shape2[0]+[NE])
    V_even_shape = make_tuple([NE]+even_shape2[1])

    U_odd_shape = make_tuple(odd_shape2[0]+[NO])
    V_odd_shape = make_tuple([NO]+odd_shape2[1])

    U_shape = make_tuple(shape2[0]+[Nfull])
    V_shape = make_tuple([Nfull]+shape2[1])

    U_stat = make_tuple(stats2[0]+[1])
    V_stat = make_tuple([-1]+stats2[1])

    strL = group_string_left.replace("(","").replace(")","")
    strR = group_string_right.replace("(","").replace(")","")

    string_U = group_string_left+"("+get_char(strL)+")"
    string_V = "("+get_char(strR)+")"+group_string_right

    U = split_legs_block(U,string_U,U_stat,U_shape,U_even_shape,U_odd_shape)
    V = split_legs_block(V,string_V,V_stat,V_shape,V_even_shape,V_odd_shape)

    if False:
        print("-------------------- 2 -------------------------------------------------------")
        X1 = einsum_block('ij,jk,kl->il',U,S,V)
        X1.force_format("standard").display()

        X11 = split_legs_block(X0,'i,j',X1.statistics,X1.shape,X1.even_shape,X1.odd_shape)
        X11.force_format("standard").display()

    # ===========================================================================
    # Rearrange back
    # ===========================================================================

    cU = get_char(subscripts2[0])
    strU = subscripts2[0]+cU+"->"+subscripts1[0]+cU

    cV = get_char(subscripts2[1])
    strV = cV+subscripts2[1]+"->"+cV+subscripts1[1]

    if False:
        U.info("U")
        V.info("V")
        print(strU,strV)

    global skip_power_of_two_check

    skip_power_of_two_check = True

    U = einsum_block(strU,U)
    V = einsum_block(strV,V)

    skip_power_of_two_check = False

    if False:
        U.info("U")
        V.info("V")

    if incorrect_sign_size(U):
        U = correct_sign_size(U)
        if incorrect_sign_size(U):
            error("Error[??]: Unsuccessful attempt at truncating the sgn vectors.")
    if incorrect_sign_size(S):
        S = correct_sign_size(S)
        if incorrect_sign_size(S):
            error("Error[??]: Unsuccessful attempt at truncating the sgn vectors.")
    if incorrect_sign_size(V):
        V = correct_sign_size(V)
        if incorrect_sign_size(V):
            error("Error[??]: Unsuccessful attempt at truncating the sgn vectors.")

    return U, S, V

def svd_block(InpObj,string,cutoff=None,save_memory=False):
    return decompose_block(InpObj,string,cutoff,save_memory,option="SVD")

def eig_block(InpObj,string,cutoff=None,save_memory=False):
    return decompose_block(InpObj,string,cutoff,save_memory,option="Eig")

####################################################
##                   Conjugation                  ##
####################################################

def op_hconjugate(InpObj):
    # operator hermitian conjugate
    # 1. In standard format, flip all indices (ignore anticommutations)
    # 2. convert all fermionic indices stat to -1 and switch the format
    # 3. forcefully convert to standard format
    # 4. Assign correct statistics

    Obj = InpObj.copy()

    this_format = Obj.format
    original_stat = Obj.statistics
    new_stat = tuple([ -i for i in list(original_stat)[::-1]])
    conj_stat = tuple([ -1 if i*i>0 else 0 for i in new_stat])

    str1 = "".join(char_list[:len(original_stat)])
    str2 = str1[::-1]
    einsum_str = str1+"->"+str2

    # 1 -------
    ret = Obj.force_format("standard")
    ret = einsum(einsum_str,ret,ignore_anticommutation=True)

    # 2 -------
    ret.statistics = conj_stat
    ret = ret.force_format("matrix")

    # 3 -------
    ret.format = "standard"

    # 4 -------
    ret.statistics = new_stat

    ret.data = np.conjugate(ret.data)
    ret = ret.force_format(this_format)

    return ret

def hconjugate(InpObj,string,save_memory=False):

    process_name = "hconjugate"
    process_length = 6
    process_color="yellow"
    step = 1
    s00 = time.time()
    progress_space() # << Don't remove this. This is for the show_progress!

    string = string.replace(" ","")
    string = denumerate(string)
    
    if string.count("(")==string.count(")") and string.count("(")>0:
        string = string.replace(" ","")
        string = string.replace(")(","|")
        if string.count("(")>1 or string.count(")")<1:
            error("Error[hconjugate]: Parentheses don't match")
        string = string.replace(")","")
        string = string.replace("(","")


    # the string is of the form aaaa|bbb

    Obj = InpObj.copy()
    this_type = type(Obj)
    this_format = Obj.format
    this_encoder = Obj.encoder
    XObj_stats = Obj.statistics
    XObj_shape = Obj.shape

    if save_memory :
        del InpObj.data
        del InpObj
        gc.collect()

    #if this_type==sparse :
    #    Obj = dense(Obj)
    if this_type not in [dense,sparse] :
        error("Error[hconjugate]: Object type must only be dense or sparse!")
        
    # check if Obj.statistics or final_statistics is weird or not
    for stat in Obj.statistics:
        if(stat not in allowed_stat):
            error("Error[hconjugate]: The input object contains illegal statistics. (0, 1, -1, or "+hybrid_symbol+" only)")
            
    partition_count = 0
    for partition in separator_list:
        partition_count += string.count(partition)
    if(partition_count!=1):
        partition_string = ""
        for i, partition in enumerate(separator_list):
            if(i==0):
                partition_string += "( "
            elif(i==len(separator_list)-1):
                partition_string += ", or "
            else:
                partition_string += ", "

            partition_string += "'"+partition+"'"

            if(i==len(separator_list)-1):
                partition_string += " )"

        error("Error[hconjugate]: The input string must contain one and only one partition "+partition_string+" in it.")
    
    step = show_progress(step,process_length,process_name+" "+"<"+current_memory_display()+">",color=process_color,time=time.time()-s00)
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #:::::      STEP 1 - JOIN LEGS BAESD ON THE GROUPINGS                                       :::::
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    # count the number of indices in the two groups
    n_left = 0
    n_right = 0
    partition_found = False
    for char in string:
        if char in separator_list :
            partition_found = True
            continue
        if(partition_found):
            n_right+=1
        else:
            n_left+=1

    join_legs_string_input = string
    for partition in separator_list:
        join_legs_string_input = join_legs_string_input.replace(partition,")(")
    join_legs_string_input = "("+join_legs_string_input+")"

    shape_left  = Obj.shape[:n_left]
    stats_left  = Obj.statistics[:n_left]
    shape_right = Obj.shape[n_left:]
    stats_right = Obj.statistics[n_left:]
    def prod(vector):
        ret = 1
        for elem in vector:
            ret *= elem
        return ret
    def get_stat(vector,prefer=None):
        boson_count = 0
        fermi_count = 0
        for elem in vector:
            if elem in bose_type :
                boson_count += 1
            if elem in fermi_type :
                fermi_count += 1

        if(boson_count==0 and fermi_count>0):
            if prefer != None:
                return prefer
            else:
                return 1
        elif(boson_count>0 and fermi_count==0):
            if prefer != None:
                return prefer
            else:
                return 0
        elif(boson_count>0 and fermi_count>0):
            return hybrid_symbol
    step = show_progress(step,process_length,process_name+" "+"<"+current_memory_display()+">",color=process_color,time=time.time()-s00)
    Obj = Obj.join_legs(join_legs_string_input,"matrix",intermediate_stat=(-1,1),save_memory=True)
    Obj = Obj.force_encoder("parity-preserving")

    
    step = show_progress(step,process_length,process_name+" "+"<"+current_memory_display()+">",color=process_color,time=time.time()-s00)
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #:::::      STEP 2 - Perform Hermitian Conjugation                                          :::::
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    Obj.data = np.conjugate(oe.contract('ij->ji',Obj.data))
    
    new_stat = [1,1]
    
    if Obj.statistics[0] in fermi_type :
        new_stat[1] = -Obj.statistics[0]
    else:
        new_stat[1] = Obj.statistics[0]
        
    if Obj.statistics[1] in fermi_type :
        new_stat[0] = -Obj.statistics[1]
    else:
        new_stat[0] = Obj.statistics[1]
    
    new_stat = make_tuple(new_stat)
    
    Obj.statistics = new_stat
    
    step = show_progress(step,process_length,process_name+" "+"<"+current_memory_display()+">",color=process_color,time=time.time()-s00)
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #:::::      STEP 3 - Split legs                                                             :::::
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    
    
    # count the number of indices in the two groups
    Uind = ""
    Vind = ""
    Ustats = []
    Vstats = []
    Ushape = []
    Vshape = []
    partition_found = False
    for i,char in enumerate(string):
        if char in separator_list :
            partition_found = True
            continue
        if(partition_found):
            Vind+=char
            Vstats+=[XObj_stats[i-1]]
            Vshape+=[XObj_shape[i-1]]
        else:
            Uind+=char
            Ustats+=[XObj_stats[i]]
            Ushape+=[XObj_shape[i]]
    
    new_ind = "("+Vind+")("+Uind+")"
    new_stats = Vstats + Ustats
    new_shape = make_tuple(Vshape + Ushape)
    for i in range(len(new_stats)):
        if new_stats[i] in fermi_type :
            new_stats[i]*=-1
    new_stats = make_tuple(new_stats)
    step = show_progress(step,process_length,process_name+" "+"<"+current_memory_display()+">",color=process_color,time=time.time()-s00)
    Obj = Obj.split_legs(new_ind,new_stats,new_shape,intermediate_stat=(-1,1),save_memory=True)
    
    #if this_type==sparse :
    #    Obj = sparse(Obj)
    if this_format!=Obj.format :
        Obj = Obj.switch_format(save_memory=True)
    if this_encoder!=Obj.encoder :
        Obj = Obj.switch_encoder(save_memory=True)

    clear_progress()
    tab_up()

    return Obj

def hconjugate_block(InpObj,string,save_memory=False):

    string = string.replace(" ","")
    string = denumerate(string)

    
    Obj = InpObj.copy()
    this_format = Obj.format

    if save_memory :
        del InpObj.data
        del InpObj
        gc.collect()

    # check if Obj.statistics or final_statistics is weird or not
    for stat in Obj.statistics:
        if stat not in allowed_stat :
            error("Error[svd_block]: The input object contains illegal statistics. (0, 1, -1, or "+hybrid_symbol+" only)")

    # convert to standard format first
    Obj = Obj.force_format("standard")

    if string.count("(")==string.count(")") and string.count("(")>0:
        string = string.replace(" ","")
        string = string.replace(")(","|")
        if string.count("(")>1 or string.count(")")<1:
            error("Error[svd_block]: Parentheses don't match")
        string = string.replace(")","")
        string = string.replace("(","")

    partition_count = 0
    for partition in separator_list:
        partition_count += string.count(partition)
    if(partition_count!=1):
        partition_string = ""
        for i, partition in enumerate(separator_list):
            if(i==0):
                partition_string += "( "
            elif(i==len(separator_list)-1):
                partition_string += ", or "
            else:
                partition_string += ", "

            partition_string += "'"+partition+"'"

            if(i==len(separator_list)-1):
                partition_string += " )"

        error("Error[svd_block]: The input string must contain one and only one partition "+partition_string+" in it.")

    for separator in separator_list:
        if separator == "|":
            continue
        else:
            string = string.replace(separator,"|")

    # ===========================================================================
    # Patch 1.4.2 --- accomodate the case with bond dimension 1
    # ===========================================================================

    # count the number of indices in the two groups
    n_left = 0
    n_right = 0
    partition_found = False
    for char in string:
        if char in separator_list :
            partition_found = True
            continue
        if(partition_found):
            n_right+=1
        else:
            n_left+=1

    join_legs_string_input = string
    for partition in separator_list:
        join_legs_string_input = join_legs_string_input.replace(partition,")(")
    join_legs_string_input = "("+join_legs_string_input+")"

    shape_left  = Obj.shape[:n_left]
    stats_left  = Obj.statistics[:n_left]
    shape_right = Obj.shape[n_left:]
    stats_right = Obj.statistics[n_left:]

    # check if one of the legs (or both) are fermion with dimension 1

    left_bond1  = list(stats_left)  in [[-1],[1]] and list(shape_left)  == [1]
    right_bond1 = list(stats_right) in [[-1],[1]] and list(shape_right) == [1]

    if left_bond1 or right_bond1:
        # forcefully make the dimension to 2

        if right_bond1:
            Obj.sgn[1][-1] = [1]
            Obj.shape = tuple(list(Obj.shape[:-1])+[2])

            ret = Obj.hconjugate(string)

            ret.sgn[1][0] = [0]
            ret.shape = tuple([1]+list(Obj.shape[:-1]))
        else:
            Obj.sgn[1][0] = [1]
            Obj.shape = tuple([2]+list(Obj.shape[1:]))
            
            ret = Obj.hconjugate(string)

            ret.sgn[1][-1] = [0]
            ret.shape = tuple(list(Obj.shape[1:])+[1])

        return ret


    # ===========================================================================
    # These are the index info before the rearrangement
    # ===========================================================================

    index_num = 0
    subscripts1 = ["",""]
    stats1 = [[],[]]
    shape1 = [[],[]]
    even_shape1 = [[],[]]
    odd_shape1 = [[],[]]

    lr = 0
    for c in string:
        if c=="|":
            lr = 1
            continue
        else:
            subscripts1[lr] += c
            stats1[lr] += [Obj.statistics[index_num]]
            shape1[lr] += [Obj.shape[index_num]]
            even_shape1[lr] += [Obj.even_shape[index_num]]
            odd_shape1[lr] += [Obj.odd_shape[index_num]]
            index_num += 1

    if False:
        print("       Set 1")
        print("  subscripts:",subscripts1)
        print("  statistics:",stats1)
        print("  full shape:",shape1)
        print("  even shape:",even_shape1)
        print("   odd shape:",odd_shape1)



    # ===========================================================================
    # Now move bosonic indices to the left of fermionic indices
    # ===========================================================================

    subscripts2 = ["",""]
    stats2 = [[],[]]
    shape2 = [[],[]]
    even_shape2 = [[],[]]
    odd_shape2 = [[],[]]

    for lr in [0,1]:
        for char,stat,dim,edim,odim in zip(subscripts1[lr],stats1[lr],shape1[lr],even_shape1[lr],odd_shape1[lr]):
            if stat == 0:
                subscripts2[lr] = char+subscripts2[lr]
                stats2[lr] = [stat]+stats2[lr]
                shape2[lr] = [dim]+shape2[lr]
                even_shape2[lr] = [edim]+even_shape2[lr]
                odd_shape2[lr] = [odim]+odd_shape2[lr]
            else:
                subscripts2[lr] += char
                stats2[lr] += [stat]
                shape2[lr] += [dim]
                even_shape2[lr] += [edim]
                odd_shape2[lr] += [odim]

    if False:
        print("       Set 2")
        print("  subscripts:",subscripts2)
        print("  statistics:",stats2)
        print("  full shape:",shape2)
        print("  even shape:",even_shape2)
        print("   odd shape:",odd_shape2)

    # ===========================================================================
    # Merge the infos into one vector
    # ===========================================================================

    subscripts3 = subscripts2[0]+subscripts2[1]
    stats3 = stats2[0]+stats2[1]
    shape3 = shape2[0]+shape2[1]
    even_shape3 = even_shape2[0]+even_shape2[1]
    odd_shape3 = odd_shape2[0]+odd_shape2[1]

    if False:
        print("       Set 3")
        print("  subscripts:",subscripts3)
        print("  statistics:",stats3)
        print("  full shape:",shape3)
        print("  even shape:",even_shape3)
        print("   odd shape:",odd_shape3)


    # ===========================================================================
    # Rearrange with einsum
    # ===========================================================================

    einsum_string = string.replace("|","")+"->"+subscripts3
    Obj3 = einsum_block(einsum_string,Obj)


    if save_memory :
        del Obj.data
        del Obj
        gc.collect()

    # ===========================================================================
    # join the indices into four groups
    # ===========================================================================

    group_string = ""
    group_string_left = ""
    group_string_right = ""
    is_left = True
    for i,c in enumerate(subscripts3):

        add_middle = False
        if i==len(subscripts2[0]):
            add_middle = True
            is_left = False
            group_string += ","

        if i>0 :
            if (
                ((Obj3.statistics[i] in bose_type) and (Obj3.statistics[i-1] in fermi_type))
                or
                ((Obj3.statistics[i] in fermi_type) and (Obj3.statistics[i-1] in bose_type))
                ):
                if not add_middle:
                    group_string += ","
                    if is_left:
                        group_string_left += ","
                    else:
                        group_string_right += ","
        group_string += c

        if is_left:
            group_string_left += c
        else:
            group_string_right += c

    final_stat4 = []
    group_substring = ""
    if 0 in stats2[0]:
        final_stat4+=[0]
        group_substring += "x"
    if 1 in stats2[0] or -1 in stats2[0]:
        final_stat4+=[-1]
        group_substring += "y"
    group_substring += "|"
    if 0 in stats2[1]:
        final_stat4+=[0]
        group_substring += "z"
    if 1 in stats2[1] or -1 in stats2[1]:
        final_stat4+=[1]
        group_substring += "w"
    final_stat4 = make_tuple(final_stat4)

    group_string_left = "("+group_string_left.replace(",",")(")+")"
    group_string_right = "("+group_string_right.replace(",",")(")+")"

    Obj4 = join_legs_block(Obj3,group_string,final_stat4)


    l = 0
    r = 0
    if 0 in stats2[0]:
        l+=1
    if (1 in stats2[0]) or (-1  in stats2[0]):
        l+=1
    if 0 in stats2[1]:
        r+=1
    if (1 in stats2[1]) or (-1 in stats2[1]):
        r+=1

    stats4 = [[],[]]
    shape4 = [[],[]]
    even_shape4 = [[],[]]
    odd_shape4  = [[],[]]

    index = 0
    for il in range(l):
        stats4[0] += Obj4.statistics[index],
        shape4[0] += Obj4.shape[index],
        even_shape4[0] += Obj4.even_shape[index],
        odd_shape4[0] += Obj4.odd_shape[index],
        index+=1
    for ir in range(r): 
        stats4[1] += Obj4.statistics[index],
        shape4[1] += Obj4.shape[index],
        even_shape4[1] += Obj4.even_shape[index],
        odd_shape4[1] += Obj4.odd_shape[index],
        index+=1

    if False:
        print("       Set 4")
        print("  statistics:",stats4)
        print("  full shape:",shape4)
        print("  even shape:",even_shape4)
        print("   odd shape:",odd_shape4)


    if save_memory :
        del Obj3.data
        del Obj3
        gc.collect()

    # ===========================================================================
    # Convert to matrix format
    # ===========================================================================

    #Obj4.info("Set 4")

    Obj5 = Obj4.force_format("matrix")
    sgn5 = copy.deepcopy(Obj5.sgn)

    if save_memory :
        del Obj4.data
        del Obj4
        gc.collect()

    if False:
        print("-------------------- -1 -------------------------------------------------------")
        Obj5.force_format("standard").display()

    # ===========================================================================
    # Extract the block matrix
    # ===========================================================================

    it = np.nditer(Obj5.data, flags=['multi_index','refs_ok'])
    for val in it:
        #total index
        block = it.multi_index
        norm = np.linalg.norm(val.item())
        parity = sum(block)%2
        
        if parity==1 and norm > numer_cutoff:
            error("Error[SVD_block]: This function only works for even tensors!")

    def prod(x):
        x = make_list(x)
        ret = 1
        for val in x:
            ret *= val
        return ret

    merged_shape_even = ( prod(even_shape4[0]),prod(even_shape4[1]) )
    merged_shape_odd = ( prod(odd_shape4[0]),prod(odd_shape4[1]) )

    ME = np.reshape(Obj5.data[0,0],merged_shape_even)
    MO = np.reshape(Obj5.data[1,1],merged_shape_odd)

    # ===========================================================================
    # conjugate
    # ===========================================================================

    MEh = np.conjugate(np.einsum('ij->ji',ME))
    MOh = np.conjugate(np.einsum('ij->ji',MO))

    unmerged_shape_even = make_tuple(even_shape4[1]+even_shape4[0])
    unmerged_shape_odd = make_tuple(odd_shape4[1]+odd_shape4[0])

    TEh = np.reshape(MEh,unmerged_shape_even)
    TOh = np.reshape(MOh,unmerged_shape_odd)

    # make a rough template
    einstr = "".join(char_list[:l])+"".join(char_list[l:l+r])+"->"+"".join(char_list[l:l+r])+"".join(char_list[:l])
    ret = einsum_block(einstr,Obj5)
    newstat = make_tuple( [ -stat if stat in fermi_type else stat for stat in ret.statistics ] )
    ret.statistics = newstat

    # -------------------------------------
    # RECALCULATE THE SIGN FACTORS !!!
    # -------------------------------------
    sgn_0 = sgn5[0]
    sgn_1 = sgn5[1]

    new_sgn_0 = sgn_0[l:l+r]+sgn_0[:l]
    new_sgn_1 = sgn_1[l:l+r]+sgn_1[:l]

    new_sgn = [new_sgn_0,new_sgn_1]

    ret.sgn = copy.deepcopy(new_sgn)

    ret.data[0,0] = TEh
    ret.data[1,1] = TOh
    ret.data[0,1] = ret.data[0,1]*0
    ret.data[1,0] = ret.data[1,0]*0

    ret = ret.force_format(this_format)
    ret.marked_as_joined = True

    # ===========================================================================
    # split
    # ===========================================================================

    def swap(vec):
        return make_tuple(vec[1]+vec[0])

    stats6 = swap(stats2)
    shape6 = swap(shape2)
    even_shape6 = swap(even_shape2)
    odd_shape6 = swap(odd_shape2)

    split_string = group_string_right+group_string_left

    stats6 = make_tuple( [ -stat if stat in fermi_type else stat for stat in stats6 ] )

    ret = split_legs_block(ret,split_string,stats6,shape6,even_shape6,odd_shape6)
    
    # ===========================================================================
    # revert the subscript's locations
    # ===========================================================================

    revert_str = subscripts2[1]+subscripts2[0]+"->"+subscripts1[1]+subscripts1[0]

    ret = einsum_block(revert_str,ret)

    if incorrect_sign_size(ret):
        ret = correct_sign_size(ret)
        if incorrect_sign_size(ret):
            error("Error[??]: Unsuccessful attempt at truncating the sgn vectors.")
    return ret

####################################################
##                    Utilities                   ##
####################################################

def random(shape,statistics,tensor_type=dense,encoder="canonical",format="standard",dtype=float,skip_trimming=False):
    X = np.random.rand(*shape)
    if dtype == complex :
        X = complex(1,0)*X + complex(0,1)*np.random.rand(*shape)
    A = dense(X,statistics=statistics,encoder=encoder,format=format)
    if not skip_trimming:
        A = trim_grassmann_odd(A)
    if tensor_type==sparse:
        A = sparse(A)
        A = A.remove_zeros()
    return A

def zeros(shape,statistics,tensor_type=dense,encoder="canonical",format="standard",dtype=float):
    return 0.0*random(shape,statistics,tensor_type,encoder,format,dtype,True)

def power(T,p):
    if type(T)==block:
        return power_block(T,p)
    else:
        return power_ds(T,p)

def sqrt(T):
    return power(T,0.5)

def power_ds(T,p):
    this_type = type(T)
    this_format = T.format
    this_encoder = T.encoder
    T = dense(T).force_format("matrix").force_encoder("canonical")
    T.data = np.power(T.data,p)
    T = T.force_format(this_format).force_encoder(this_encoder)
    if this_type==sparse :
        T = sparse(T)
    return T

def power_block(T,p):
    this_format = T.format
    T = T.force_format("matrix")
    
    it = np.nditer(T.data, flags=['multi_index','refs_ok'])
    for val in it:
        block = it.multi_index
        T.data[block] = np.power(val.item(),p)

    T = T.force_format(this_format)
    return T
