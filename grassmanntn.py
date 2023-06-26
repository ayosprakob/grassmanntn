import numpy as np
import math
from grassmanntn import param
import sparse as sp
import opt_einsum as oe
import time
import sys
import gc
import tracemalloc
import os
from datetime import datetime

hybrid_symbol = "*"
separator_list = ("|",":",";",",","."," ")
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
    ,"α","β","Γ","γ","Δ","δ","ε","ζ","η","Θ","θ","ι","κ","λ","μ","ν","Ξ","ξ","Π","π","ρ","Σ","σ","ς","τ","υ","Φ","ϕ","φ","χ","Ψ","ψ","Ω","ω"
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
    print("\t__________look_below_for_the_error_information__________")
    #intentional sabotage!!!
    __________look_above_for_the_error_information__________

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
            #gtn.error("Error[show_progress]: <step> cannot be larger than <total>!")
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

####################################################
##             Densed Grassmann Array             ##
####################################################

class dense:
    def __init__(self, data=None, encoder = "canonical", format = "standard", statistic=None):
    
        #copy dense properties
        self.data = None
        self.statistic = None
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
            self.statistic = data.statistic
            self.format = data.format
            self.encoder = data.encoder
            default = False
        elif(type(data)==dense):
            #copy dense properties
            self.data = data.data.copy()
            self.statistic = data.statistic
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
            
        
        if statistic != None:
            self.statistic = make_tuple(statistic)
            
        if not default and not skip_power_of_two_check:
            for i,dim in enumerate(self.data.shape):
                if self.statistic[i] in fermi_type and dim != int(2**math.floor(np.log2(dim))):
                    error("Error[dense]: Some of the fermionic tensor shapes are not a power of two.\n              Have you added the <statistic> argument when calling this function?")
                
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
        print(indent+"   statistic:",self.statistic)
        print(indent+"      format:",self.format)
        print(indent+"     encoder:",self.encoder)
        print(indent+"      memory:",memory_display(sys.getsizeof(self.data)+sys.getsizeof(self)))
        print(indent+"        norm:",self.norm)
        print(indent+"     entries:")
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
        print(indent+"   statistic:",self.statistic)
        print(indent+"      format:",self.format)
        print(indent+"     encoder:",self.encoder)
        print(indent+"      memory:",memory_display(sys.getsizeof(self.data)+sys.getsizeof(self)))
        print(indent+"        norm:",self.norm)
        print()

    def copy(self):
        #copy dense properties
        ret = dense()
        ret.data = self.data.copy()
        ret.statistic = self.statistic
        ret.format = self.format
        ret.encoder = self.encoder
        return ret
    
    def __add__(self, other):
        if(self.shape!=other.shape
            or self.statistic!=other.statistic
             or self.format!=other.format
              or self.encoder!=other.encoder):
            error("Error[dense.+]: Inconsistent object properties")
            
        ret = self.copy()
        ret.data = ret.data+other.data
        return ret
        
    def __sub__(self, other):
        if(self.shape!=other.shape
            or self.statistic!=other.statistic
             or self.format!=other.format
              or self.encoder!=other.encoder):
            error("Error[dense.-]: Inconsistent object properties")
            
        ret = self.copy()
        ret.data = ret.data-other.data
        return ret
        
    def __mul__(self, other):
        ret = self.copy()
        ret.data = ret.data*other
        return ret
        
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
            to_calc += (ret.statistic[i]==-1),
            if(ret.statistic[i]==hybrid_symbol):
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
        for i in range(ret.ndim):
            d = ret.shape[i]
            if ret.statistic[i] in fermi_type :
                dat = dat.take(indices=np.array([param.encoder(i) for i in range(d)]),axis=i)

            if time.time()-s0 > 2 :
                show_progress(i,kmax,process_name = "switch_encoder",ratio = False,color="cyan",time=time.time()-s00)
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

    def join_legs(self,string_inp,make_format='standard',intermediate_stat=(-1,1),save_memory=False):
        return join_legs(self,string_inp,make_format,intermediate_stat,save_memory)

    def split_legs(self,string_inp,final_stat,final_shape,intermediate_stat=(-1,1),save_memory=False):
        return split_legs(self,string_inp,final_stat,final_shape,intermediate_stat,save_memory)

    def hconjugate(self,input_string,save_memory=False):
        return hconjugate(self,input_string,save_memory)

    def svd(self,string_inp,cutoff=None,save_memory=False):
        return svd(self,string_inp,cutoff,save_memory)

    def eig(self,string_inp,cutoff=None,debug_mode=False,save_memory=False):
        return eig(self,string_inp,cutoff,debug_mode,save_memory)

####################################################
##            Sparse Grassmann arrays             ##
####################################################

class sparse:
    def __init__(self, data=None, encoder = "canonical", format = "standard", statistic = None):
    
        #copy sparse properties
        self.data = None
        self.statistic = None
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
            self.statistic = data.statistic
            self.format = data.format
            self.encoder = data.encoder
            default = False
        elif(type(data)==sparse):
            #copy sparse properties
            self.data = data.data.copy()
            self.statistic = data.statistic
            self.format = data.format
            self.encoder = data.encoder
            default = False
        elif data==None:
            "nothing to see here"
        else:
            error("Error[sparse]: Invalid initialized data")
            
        
        if statistic != None:
            self.statistic = make_tuple(statistic)
        
        if not default and not skip_power_of_two_check:
            for i,dim in enumerate(self.data.shape):
                if self.statistic[i] in fermi_type and dim != int(2**math.floor(np.log2(dim))):
                    error("Error[sparse]: Some of the fermionic tensor shapes are not a power of two.\n               Have you added the <statistic> argument when calling this function?")
               
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
        print(indent+"   statistic:",self.statistic)
        print(indent+"      format:",self.format)
        print(indent+"     encoder:",self.encoder)
        print(indent+"      memory:",memory_display(sys.getsizeof(self.data)+sys.getsizeof(self)))
        print(indent+"        norm:",self.norm)
        print(indent+"     entries:")

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
        print(indent+"   statistic:",self.statistic)
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
        ret.statistic = self.statistic
        ret.format = self.format
        ret.encoder = self.encoder
        return ret
    
    def __add__(self, other):
        if(self.shape!=other.shape
            or self.statistic!=other.statistic
             or self.format!=other.format
              or self.encoder!=other.encoder):
            error("Error[sparse.+]: Inconsistent object properties")
            
        ret = self.copy()
        ret.data = ret.data+other.data
        return ret
        
    def __sub__(self, other):
        if(self.shape!=other.shape
            or self.statistic!=other.statistic
             or self.format!=other.format
              or self.encoder!=other.encoder):
            error("Error[sparse.-]: Inconsistent object properties")
            
        ret = self.copy()
        ret.data = ret.data-other.data
        return ret
        
    def __mul__(self, other):
        ret = self.copy()
        ret.data = ret.data*other
        return ret
        
    def __rmul__(self, other):
        return self*other
        
    #def __len__(self):
    #    return len(self.data)
    
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

    def join_legs(self,string_inp,make_format='standard',intermediate_stat=(-1,1),save_memory=False):
        return join_legs(self,string_inp,make_format,intermediate_stat,save_memory)

    def split_legs(self,string_inp,final_stat,final_shape,intermediate_stat=(-1,1),save_memory=False):
        return split_legs(self,string_inp,final_stat,final_shape,intermediate_stat,save_memory)

    def hconjugate(self,input_string,save_memory=False):
        return hconjugate(self,input_string,save_memory)

    def svd(self,string_inp,cutoff=None,save_memory=False):
        return svd(self,string_inp,cutoff,save_memory)

    def eig(self,string_inp,cutoff=None,debug_mode=False,save_memory=False):
        return eig(self,string_inp,cutoff,debug_mode,save_memory)

####################################################
##       Parity Calculation (internal tools)      ##
####################################################

def absolute_parity(permutation, individual_parity):
    """
    Compute the absolute_parity of a permutation, assuming that some elements always commute
    with every other element.
    
    Parameters:
    permutation (list[int]): A list object.
    individual_parity (list[int]): A list of object's grassmann parity.
    
    Returns:
    int: 1 if the permutation is even, -1 if the permutation is odd.
    """
    
    def get_commutative_elements(permutation, individual_parity):
        """
        Return a set of commutine elements (individual_parity = even)
        """
        if(len(permutation)!=len(individual_parity)):
            error("Error[absolute_parity.get_commutative_elements]: Inconsistent array sizes!")
            
        else:
            commutative_elements = [x for i,x in enumerate(permutation) if (-1)**individual_parity[i]==1]
            return set(commutative_elements)
            
    
    commutative_elements = get_commutative_elements(permutation, individual_parity)
    n = len(permutation)
    noncommutative_elements = [x for x in permutation if x not in commutative_elements]
    inversions = 0
    for i in range(len(noncommutative_elements)):
        for j in range(i+1, len(noncommutative_elements)):
            if noncommutative_elements[i] > noncommutative_elements[j]:
                inversions += 1
    return (-1)**inversions

def relative_parity_int(permutation1, permutation2, individual_parity1):
    """
    Compute the relative_parity from permuting permutation1 to permutation2 with the individual parity given by individual_parity1 of the list permutation1
    
    Parameters:
    permutation1 (list): first list
    permutation2 (list): target list
    individual_parity1 (list): grassmann parity of the first list
    
    Returns:
    int: relative parity of the permutation
    """
    
    def permute_c(a, b, c):
        """
        Permute the elements of c according to the permutation that maps a to b.
        """
        x = list(np.argsort(a))
        xc = [ c[i] for i in x ]
        bxc = [ xc[i] for i in b ]
        return bxc

    def get_noncommutative_elements(permutation, individual_parity):
        """
        Return a LIST of commutine elements (individual_parity = even)
        """
        if(len(permutation)!=len(individual_parity)):
            error("Error[relative_parity_int.get_noncommutative_elements]: Inconsistent array sizes!")
            
        else:
            noncommutative_elements = [x for i,x in enumerate(permutation) if (-1)**individual_parity[i]==-1]
            return noncommutative_elements

    individual_parity2 = permute_c(permutation1, permutation2, individual_parity1)
    
    noncommutative_elements1 = get_noncommutative_elements(permutation1,individual_parity1)
    noncommutative_elements2 = get_noncommutative_elements(permutation2,individual_parity2)
    
    if(sorted(noncommutative_elements1) != sorted(noncommutative_elements2)):
        error("Error[relative_parity_int]: Inconsistent grassmann-odd indices!")
        
    
    absolute_parity1 = absolute_parity(permutation1, individual_parity1)
    absolute_parity2 = absolute_parity(permutation2, individual_parity2)
    return absolute_parity1*absolute_parity2
    
def relative_parity_single_input(string, individual_parity1):
    """
    the string version of relative_parity_int
    the sign factor version of single input einsum
    """
    [string1,string2] = list(string.split("->"))
    unique_chars = list(set(string1+string2))
    char_to_int = {char: i for i, char in enumerate(unique_chars)}
    permutation1 = [char_to_int[char] for char in string1]
    permutation2 = [char_to_int[char] for char in string2]
    return relative_parity_int(permutation1, permutation2, individual_parity1)

def relative_parity(string, individual_parity):
    """
    Imagine doing Grassmann version of Einsum.
    This function returns the sign factor of that sum.
    You have to enter the parity of the input indices in [individual_parity]
    The grassmann indices must not be added or removed!
    Use a different function to integrate them out!
    Examples:
    >> relative_parity( "abCdE,aXdYb->XCEY", [0,0,1,0,1,0,1,0,1,0] )
    >> 1
    
    >> relative_parity( "abCdE,aXYb->CXEY", [0,0,1,0,1,0,1,1,0] )
    >> -1
    
    """
    [string_input,string_output] = list(string.split("->"))
    string_list = list(string_input.split(","))
    
    join_string = ""
    for i in range(len(string_list)):
        join_string = join_string + string_list[i]
        
    if(len(join_string)!=len(individual_parity)):
        error("Error[relative_parity]: The number of input list and parity list are not consistent!")
        
        
    #remove the summed indices
    def remove_duplicates(list1, list2):
        new_list1 = []
        new_list2 = []
        for i, val in enumerate(list1):
            if list1.count(val)==1 :
                new_list1.append(val)
                new_list2.append(list2[i])
        return new_list1, new_list2
        
    join_string_list,individual_parity = remove_duplicates(list(join_string), individual_parity)
    join_string = ''.join(join_string_list)+"->"+string_output
    return relative_parity_single_input(join_string, individual_parity)

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

def einsum(*args,format="standard",encoder="canonical",debug_mode=False):

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #                     Important variables and its meanings
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #
    #       instruction_string = the whole input text with spaces removed
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
    instruction_string = args[0].replace(" ","")

    instruction_string = denumerate(instruction_string)

    has_output = instruction_string.count("->") > 0
    if instruction_string.count("->") > 1 :
        error("Error[einsum]: Only one arrow is allowed in the input string!")

    if has_output:
        input_string, output_string = instruction_string.split("->")
        if output_string.count(",") > 0 :
            error("Error[einsum]: Output string must not contain commas (',') !")
    else:
        input_string = instruction_string

    summand = input_string.replace(",","")

    obj_index_list = input_string.split(",")
    nobj = len(obj_index_list)

    obj_list = make_list(args[1:1+nobj])
    stats_list = sum([ make_list(obj.statistic) for obj in obj_list ],[])
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
            error("Error[einsum]: The contracted indices have inconsistent statistic!")
    
    if debug_mode :
        print()
        print("input:",instruction_string)
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
        individual_parity_list = []
        sgn_list = []

        k=1
        kmax = getsize(S1_shape)

        s0 = time.time()
        s00 = s0
        progress_space() # << Don't remove this. This is for the show_progress!

        for element in iterator:
            coords = iterator.multi_index
            dupped_coords = use_copy_map(copy_map,coords)
            
            individual_parity = [ param.gparity(i)%2 for i in dupped_coords ]

            if individual_parity not in individual_parity_list:
                individual_parity_list += individual_parity,
                sgn = relative_parity(S1_sgn_computation_string,individual_parity)
                sgn_list += sgn,
            else:
                index = individual_parity_list.index(individual_parity)
                sgn = sgn_list[index]

            S1[coords] = sgn

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
            S2i = [ param.sgn(j) for j in range(S2_dimlist[i]) ]
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
            individual_parity_list = []
            sgn_list = []
            for element in iterator:
                coords = iterator.multi_index

                individual_parity = [ param.gparity(i)%2 for i in coords ]

                if individual_parity not in individual_parity_list:
                    individual_parity_list += individual_parity,
                    sgn = relative_parity(S3_sgn_computation_string,individual_parity)
                    sgn_list += sgn,
                else:
                    index = individual_parity_list.index(individual_parity)
                    sgn = sgn_list[index]

                S3[coords] = sgn

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
        

        instruction_all_indices = instruction_string.replace(",","")
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

            
    ret = oe.contract(*tuple([einsum_string]+einsum_obj_list))

    if has_output :
        return this_type(ret,statistic=final_stats).force_encoder(this_encoder).force_format(this_format)
    else:
        if this_type == sparse :
            if type(ret.data)==memoryview:
                return ret
            else:
                return ret.data[0]
        else:
            return np.array(ret).flatten()[0]

####################################################
##                     Reshape                    ##
####################################################

def join_legs(InpObj,string_inp,make_format='standard',intermediate_stat=(-1,1),save_memory=False):

    process_name = "join_legs"
    process_length = 6
    process_color="green"
    step = 1
    s00 = time.time()
    progress_space() # << Don't remove this. This is for the show_progress!

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
    XObj_stats = Obj.statistic
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
    group_info, sorted_group_info = get_group_info(string_inp, Obj.statistic, Obj.shape)
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
    Obj.statistic = sorted_stat
    
    step = show_progress(step,process_length,process_name+" "+"<"+current_memory_display()+">",color=process_color,time=time.time()-s00)
    
    #===============================================================================#
    #   Step 2: Join fermionic indices with np.reshape                              #
    #===============================================================================#
    
    new_stats, new_shape, final_stats, final_shape = get_intermediate_info(sorted_group_info,intermediate_stat)
    #intermediate_tensor
    
    Obj.data = np.reshape(Obj.data,new_shape)
    Obj.statistic = new_stats
    
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
    Obj.statistic = final_stats
    
    clear_progress()
    tab_up()

    return Obj
    
def split_legs(InpObj,string_inp,final_stat,final_shape,intermediate_stat=(-1,1),save_memory=False):

    process_name = "split_legs"
    process_length = 6
    process_color="green"
    step = 1
    s00 = time.time()
    progress_space() # << Don't remove this. This is for the show_progress!

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
    XObj_stats = Obj.statistic
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
    Obj.statistic = new_stats
    
    step = show_progress(step,process_length,process_name+" "+"<"+current_memory_display()+">",color=process_color,time=time.time()-s00)
    #===============================================================================#
    #   Step 4: Switch encoder                                                      #
    #===============================================================================#
    
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
    Obj.statistic = new_stats
    
    step = show_progress(step,process_length,process_name+" "+"<"+current_memory_display()+">",color=process_color,time=time.time()-s00)
    #===============================================================================#
    #   Step 1: Move bosonic indices to the left of each group                      #
    #===============================================================================#
    
    unsorted_string = ''.join( [ indices for [indices, stats, shape] in group_info ] )
    sorted_string = ''.join( [ indices for [indices, stats, shape] in sorted_group_info ] )
    npeinsum_string = sorted_string+sign_factors_list[0]+"->"+unsorted_string
    npeinsum_obj = [Obj.data] + sign_factors_list[1]
    Obj.data = oe.contract(*make_tuple( [npeinsum_string] + npeinsum_obj ))
    Obj.statistic = final_stat
    
    clear_progress()
    tab_up()

    if this_format=='matrix':
        return Obj.switch_format(save_memory=True)
    
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
    group_info = [] # [string text, statistic, shape]
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
##          Trim the Grassmann-odd parts          ##
##               (for testing only)               ##
####################################################

def trim_grassmann_odd(Obj):
    objtype=type(Obj)

    if(Obj.encoder == 'canonical'):
        Obj = Obj.switch_encoder(save_memory=True)
        
    if(objtype==dense):
        Obj = sparse(Obj)
    C = Obj.coords
    s0 = time.time()
    s00 = s0
    progress_space() # << Don't remove this. This is for the show_progress!

    for i in range(Obj.nnz):
        fcoords = [ ind for j,ind in enumerate(C[i]) if (Obj.statistic[j] in fermi_type)]
        if(sum(fcoords)%2 == 1):
            Obj.data.data[i] = 0

        if time.time()-s0 > 0.5 :
            show_progress(i,Obj.nnz,"trim_grassmann_odd",time=time.time()-s00)

    clear_progress()
    tab_up()


    if(objtype==dense):
        Obj = dense(Obj)

    if(Obj_input.encoder == 'canonical'):
        Obj = Obj.switch_encoder(save_memory=True)

    return Obj

def is_grassmann_even(Obj):
    if Obj.encoder == 'canonical':
        Obj = Obj.switch_encoder(save_memory=True)
    if type(Obj) == dense:
        Obj = sparse(Obj)

    C = Obj.coords
    for x in C:
        parity = sum([ ind for i,ind in enumerate(x) if Obj.statistic[i] in fermi_type ])
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
        if np.abs(s/Λ[0]) > numer_cutoff:
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

    string = denumerate(string)

    
    Obj = InpObj.copy()
    this_type = type(Obj)
    this_format = Obj.format
    this_encoder = Obj.encoder
    XObj_stats = Obj.statistic
    XObj_shape = Obj.shape

    if save_memory :
        del InpObj.data
        del InpObj
        gc.collect()
        
        
    if(this_type==sparse):
        Obj = dense(Obj)
    if(this_type not in [dense,sparse]):
        error("Error[svd]: Object type must only be dense or sparse!")
        
    # check if Obj.statistic or final_statistic is weird or not
    for stat in Obj.statistic:
        if(stat not in allowed_stat):
            error("Error[svd]: The input object contains illegal statistic. (0, 1, -1, or "+hybrid_symbol+" only)")

    if string.count("(")==string.count(")") and string.count("(")>0:
        string = string.replace(" ","")
        string = string.replace(")("," ")
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
    stats_left  = Obj.statistic[:n_left]
    shape_right = Obj.shape[n_left:]
    stats_right = Obj.statistic[n_left:]

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

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #:::::      STEP 2 - BLOCK SVD (MAKE SURE IT'S PARITY-PRESERVING!)                          :::::
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    step = show_progress(step,process_length,process_name+" "+"<"+current_memory_display()+">",color=process_color,time=time.time()-s00) #3
    if Obj.statistic[0]==0 or Obj.statistic[1]==0:
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
    
    if Obj.statistic[0]==0:
        U = dense(U,encoder="parity-preserving",format="matrix",statistic=(0,0))
        Λ = dense(Λ,encoder="parity-preserving",format="matrix",statistic=(0,0))
        V = dense(V,encoder="parity-preserving",format="matrix",statistic=(0,Obj.statistic[1]))
        Λstatleft = 0
        Λstatright = 0
    elif Obj.statistic[1]==0:
        U = dense(U,encoder="parity-preserving",format="matrix",statistic=(Obj.statistic[0],0))
        Λ = dense(Λ,encoder="parity-preserving",format="matrix",statistic=(0,0))
        V = dense(V,encoder="parity-preserving",format="matrix",statistic=(0,0))
        Λstatleft = 0
        Λstatright = 0
    else:
        U = dense(U,encoder="parity-preserving",format="matrix",statistic=(Obj.statistic[0],1))
        Λ = dense(Λ,encoder="parity-preserving",format="matrix",statistic=(-1,1))
        V = dense(V,encoder="parity-preserving",format="matrix",statistic=(-1,Obj.statistic[1]))
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
    U = U.split_legs(Uind,Ustats,Ushape,save_memory=True)
    Λ = Λ.switch_encoder(save_memory=True)
    V = V.split_legs(Vind,Vstats,Vshape,save_memory=True)
    
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
        if np.abs(s/Λ[0]) > numer_cutoff:
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

    string = denumerate(string)

    
    Obj = InpObj.copy()
    this_type = type(Obj)
    this_format = Obj.format
    this_encoder = Obj.encoder
    XObj_stats = Obj.statistic
    XObj_shape = Obj.shape

    if save_memory :
        del InpObj.data
        del InpObj
        gc.collect()
        
        
    if(this_type==sparse):
        Obj = dense(Obj)
    if(this_type not in [dense,sparse]):
        error("Error[eig]: Object type must only be dense or sparse!")
        
    # check if Obj.statistic or final_statistic is weird or not
    for stat in Obj.statistic:
        if(stat not in allowed_stat):
            error("Error[eig]: The input object contains illegal statistic. (0, 1, -1, or "+hybrid_symbol+" only)")

    if string.count("(")==string.count(")") and string.count("(")>0:
        string = string.replace(" ","")
        string = string.replace(")("," ")
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
    stats_left  = Obj.statistic[:n_left]
    shape_right = Obj.shape[n_left:]
    stats_right = Obj.statistic[n_left:]

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

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #:::::      STEP 2 - BLOCK SVD (MAKE SURE IT'S PARITY-PRESERVING!)                          :::::
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    step = show_progress(step,process_length,process_name+" "+"<"+current_memory_display()+">",color=process_color,time=time.time()-s00) #3
    if Obj.statistic[0]==0 or Obj.statistic[1]==0:
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
    
    if Obj.statistic[0]==0:
        U = dense(U,encoder="parity-preserving",format="matrix",statistic=(0,0))
        Λ = dense(Λ,encoder="parity-preserving",format="matrix",statistic=(0,0))
        V = dense(V,encoder="parity-preserving",format="matrix",statistic=(0,Obj.statistic[1]))
        Λstatleft = 0
        Λstatright = 0
    elif Obj.statistic[1]==0:
        U = dense(U,encoder="parity-preserving",format="matrix",statistic=(Obj.statistic[0],0))
        Λ = dense(Λ,encoder="parity-preserving",format="matrix",statistic=(0,0))
        V = dense(V,encoder="parity-preserving",format="matrix",statistic=(0,0))
        Λstatleft = 0
        Λstatright = 0
    else:
        U = dense(U,encoder="parity-preserving",format="matrix",statistic=(Obj.statistic[0],1))
        Λ = dense(Λ,encoder="parity-preserving",format="matrix",statistic=(-1,1))
        V = dense(V,encoder="parity-preserving",format="matrix",statistic=(-1,Obj.statistic[1]))
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
    U = U.split_legs(Uind,Ustats,Ushape,save_memory=True)
    Λ = Λ.switch_encoder(save_memory=True)
    V = V.split_legs(Vind,Vstats,Vshape,save_memory=True)
    
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
##                   Conjugation                  ##
####################################################

def hconjugate(InpObj,string,save_memory=False):

    process_name = "hconjugate"
    process_length = 6
    process_color="yellow"
    step = 1
    s00 = time.time()
    progress_space() # << Don't remove this. This is for the show_progress!

    string = denumerate(string)
    
    if string.count("(")==string.count(")") and string.count("(")>0:
        string = string.replace(" ","")
        string = string.replace(")("," ")
        if string.count("(")>1 or string.count(")")<1:
            error("Error[hconjugate]: Parentheses don't match")
        string = string.replace(")","")
        string = string.replace("(","")


    # the string is of the form aaaa|bbb

    Obj = InpObj.copy()
    this_type = type(Obj)
    this_format = Obj.format
    this_encoder = Obj.encoder
    XObj_stats = Obj.statistic
    XObj_shape = Obj.shape

    if save_memory :
        del InpObj.data
        del InpObj
        gc.collect()

    #if this_type==sparse :
    #    Obj = dense(Obj)
    if this_type not in [dense,sparse] :
        error("Error[hconjugate]: Object type must only be dense or sparse!")
        
    # check if Obj.statistic or final_statistic is weird or not
    for stat in Obj.statistic:
        if(stat not in allowed_stat):
            error("Error[hconjugate]: The input object contains illegal statistic. (0, 1, -1, or "+hybrid_symbol+" only)")
            
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
    stats_left  = Obj.statistic[:n_left]
    shape_right = Obj.shape[n_left:]
    stats_right = Obj.statistic[n_left:]
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
    Obj = Obj.join_legs(join_legs_string_input,"matrix",save_memory=True)
    
    step = show_progress(step,process_length,process_name+" "+"<"+current_memory_display()+">",color=process_color,time=time.time()-s00)
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #:::::      STEP 2 - Perform Hermitian Conjugation                                          :::::
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    Obj.data = np.conjugate(oe.contract('ij->ji',Obj.data))
    
    new_stat = [1,1]
    
    if Obj.statistic[0] in fermi_type :
        new_stat[1] = -Obj.statistic[0]
    else:
        new_stat[1] = Obj.statistic[0]
        
    if Obj.statistic[1] in fermi_type :
        new_stat[0] = -Obj.statistic[1]
    else:
        new_stat[0] = Obj.statistic[1]
    
    new_stat = make_tuple(new_stat)
    
    Obj.statistic = new_stat
    
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
    Obj = Obj.split_legs(new_ind,new_stats,new_shape,save_memory=True)
    
    #if this_type==sparse :
    #    Obj = sparse(Obj)
    if this_format!=Obj.format :
        Obj = Obj.switch_format(save_memory=True)
    if this_encoder!=Obj.encoder :
        Obj = Obj.switch_encoder(save_memory=True)

    clear_progress()
    tab_up()

    return Obj

####################################################
##                    Utilities                   ##
####################################################

def random(shape,statistic,tensor_format=dense,dtype=float,skip_trimming=False):
    X = np.random.rand(*shape)
    if dtype == complex :
        X = complex(1,0)*X + complex(0,1)*np.random.rand(*shape)
    A = dense(X, statistic = statistic)
    if not skip_trimming:
        A = trim_grassmann_odd(A)
    if tensor_format==sparse:
        A = sparse(A)
        A = A.remove_zeros()
    return A

def power(T,p):
    this_type = type(T)
    this_format = T.format
    this_encoder = T.encoder
    T = dense(T).force_format("matrix").force_encoder("canonical")
    T.data = np.power(T.data,p)
    T = T.force_format(this_format).force_encoder(this_encoder)
    if this_type==sparse :
        T = sparse(T)
    return T

def sqrt(T):
    return power(T,0.5)
