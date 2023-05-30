import numpy as np
import math
from grassmanntn import param
import sparse as sp
import opt_einsum as oe
import time
import sys
import gc

hybrid_symbol = "*"
separator_list = ("|",":",";",",","."," ")

skip_parity_blocking_check = True
allowed_stat = (0,1,-1,hybrid_symbol)
fermi_type = (1,-1)
bose_type = (0,hybrid_symbol)
encoder_type = ("canonical","parity-preserving")
format_type = ("standard","matrix")
numer_cutoff = 1.0e-14
numer_display_cutoff = 1000*numer_cutoff
char_list = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z","A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]

'''
        USAGE ADVICES
        - Do not do einsum with hybrid indices (already prevent in the code)
        - hconjugate and join/split opertions are not commutative!
        - switch encoder before switching format!

'''

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
    print(text)
    exit()

def get_char(string):
    for char in char_list:
        if char not in string:
            return char
    error("Error[get_char]: Running out of index character!")    

def show_progress(step_inp,total_inp,random_text = ""):

    step = int(np.floor(32*step_inp/total_inp))
    total = 32

    if step > total:
        #gtn.error("Error[show_progress]: <step> cannot be larger than <total>!")
        step = total
    print("\r",end="")

    random_text = random_text + " "
    if(len(random_text)>2*total):
        random_text = " "

    progress_number = " "+str(step_inp)+"/"+str(total_inp)

    if len(random_text) > 2*step :
        random_text = random_text[(len(random_text)-2*step):]
    styled_random_text = "\u001b[1;37;44m"+random_text+"\u001b[0;0m"

    if 2*step-len(random_text)+len(random_text)+len(progress_number) > 2*total :
        progress_number = progress_number[:(2*total-2*step)]
    styled_number = "\u001b[1;34;47m"+progress_number+"\u001b[0;0m"

    filled_bar = "\u001b[0;;44m \u001b[0;0m"
    blank_bar = "\u001b[0;;47m \u001b[0;0m"


    n_filled = 2*step-len(random_text)
    n_blank  = 2*total-n_filled-len(random_text)-len(progress_number)

    total = n_filled+len(random_text)+len(progress_number)+n_blank

    print("   progress: ",end="")
    for i in range(n_filled):
        print(filled_bar,end="")
    
    print(styled_random_text,end="")
    print(styled_number,end="")

    for i in range(n_blank):
        print(blank_bar,end="")
    return step_inp+1

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

def clear_progress():
    print("\r",end="")
    for i in range(80):
        print(" ",end="")
    print("\r",end="")
    return 1

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
        return str(raw_memory)+" B"
    elif raw_memory<2**20:
        return str(raw_memory/(2**10))+" KiB"
    elif raw_memory<2**30:
        return str(raw_memory/(2**20))+" MiB"
    elif raw_memory<2**40:
        return str(raw_memory/(2**30))+" GiB"
    else:
        return str(raw_memory/(2**40))+" TiB"

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
            
        if not default:
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
        
    def switch_format(self):
        # multiply sign factor sigma[i] to every conjugated indices i

        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        # WORK IN THE FOLLOWING CONDITIONS      :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        # (WILL CONVERT AUTOMATICALLY)          :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #   - canonical encoder                 :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

        ret = self.copy()
        if(self.encoder=='parity-preserving'):
            ret = ret.switch_encoder().copy()
        iterator = np.nditer(ret.data, flags=['multi_index'])
        for element in iterator:
            coords = iterator.multi_index
            sgn_value = 1
            for i,ind in enumerate(coords):
                if(ret.statistic[i]==-1):
                    sgn_value *= param.sgn(ind)
                if(ret.statistic[i]==hybrid_symbol):
                    error("Error[switch_format]: Cannot switch format with a hybrid index.\n                      Split them into bosonic and fermionic ones first!")
                    
            ret[coords] *= sgn_value

        if(ret.format=='standard'):
            ret.format = 'matrix'
        elif(ret.format=='matrix'):
            ret.format = 'standard'
        else:
            error("Error[switch_format]: unknown format")
            
        if(self.encoder=='parity-preserving'):
            ret = ret.switch_encoder()
        return ret

    def switch_encoder(self):
        ret = self.copy()
        iterator = np.nditer(self.data, flags=['multi_index'])
        for element in iterator:
            coords = iterator.multi_index
            new_coords = []
            for i,ind in enumerate(coords):
                if(self.statistic[i] in fermi_type):
                    new_coords += [param.encoder(ind)]
                else:
                    new_coords += [ind]
            new_coords = tuple(new_coords)
            ret.data[coords] = self.data[new_coords]
        if(ret.encoder=='canonical'):
            ret.encoder='parity-preserving'
        else:
            ret.encoder='canonical'
        return ret

    def force_encoder(self,target="canonical"):
        if target not in encoder_type:
            error("Error[dense.force_encoder]: Unrecognized target encoder.")
        if target != self.encoder :
            return self.switch_encoder()
        else :
            return self.copy()

    def force_format(self,target="standard"):
        if target not in format_type:
            error("Error[dense.force_format]: Unrecognized target format.")
        if target != self.format :
            return self.switch_format()
        else :
            return self.copy()

    def join_legs(self,string_inp,make_format='standard',intermediate_stat=(-1,1)):
        return join_legs(self,string_inp,make_format,intermediate_stat)

    def split_legs(self,string_inp,final_stat,final_shape,intermediate_stat=(-1,1)):
        return split_legs(self,string_inp,final_stat,final_shape,intermediate_stat)

    def hconjugate(self,input_string):
        return hconjugate(self,input_string)

    def svd(self,string_inp,cutoff=None):
        return svd(self,string_inp,cutoff)

    def eig(self,string_inp,cutoff=None):
        return eig(self,string_inp,cutoff)

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
        
        if not default:
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

    def switch_format(self):
        #multiply sign factor sigma[i] to every conjugated indices i

        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        # WORK IN THE FOLLOWING CONDITIONS      :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        # (WILL CONVERT AUTOMATICALLY)          :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #   - canonical encoder                 :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

        ret = self.copy()
        if(self.encoder=='parity-preserving'):
            ret = ret.switch_encoder()
        C = self.coords
        for elem in range(self.nnz):
            coords = C[elem]
            sgn_value = 1
            for i,ind in enumerate(coords):
                if(ret.statistic[i]==-1):
                    sgn_value *= param.sgn(ind)
                if(ret.statistic[i]==hybrid_symbol):
                    error("Error[switch_format]: Cannot switch format with a hybrid index.\n                      Split them into bosonic and fermionic ones first!")
                    
            ret.data.data[elem] *= sgn_value

        if(ret.format=='standard'):
            ret.format = 'matrix'
        elif(ret.format=='matrix'):
            ret.format = 'standard'
        else:
            error("Error[switch_format]: unknown format")
            

        if(self.encoder=='parity-preserving'):
            ret = ret.switch_encoder()
        return ret

    def switch_encoder(self):
        ret = self.copy()
        C = self.coords
        for elem in range(self.nnz):
            coords = C[elem]

            new_coords = []
            for i,ind in enumerate(coords):
                if(self.statistic[i] in fermi_type):
                    new_coords += [param.encoder(ind)]
                else:
                    new_coords += [ind]
            new_coords = tuple(new_coords)

            for ind in range(len(new_coords)):
                ret.data.coords[ind][elem] = new_coords[ind]
        if(ret.encoder=='canonical'):
            ret.encoder='parity-preserving'
        else:
            ret.encoder='canonical'
        return ret

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

    def join_legs(self,string_inp,make_format='standard',intermediate_stat=(-1,1)):
        return join_legs(self,string_inp,make_format,intermediate_stat)

    def split_legs(self,string_inp,final_stat,final_shape,intermediate_stat=(-1,1)):
        return split_legs(self,string_inp,final_stat,final_shape,intermediate_stat)

    def hconjugate(self,input_string):
        return hconjugate(self,input_string)

    def svd(self,string_inp,cutoff=None):
        return svd(self,string_inp,cutoff)

    def eig(self,string_inp,cutoff=None):
        return eig(self,string_inp,cutoff)

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

def einsum(*args,format="standard",encoder="canonical",debug_mode=False):

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # WORK IN THE FOLLOWING CONDITIONS      :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # (WILL CONVERT AUTOMATICALLY)          :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #   - canonical encoder                 :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #   - standard format                   :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #                     Step 0: Error detection
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    input_string = args[0].replace(" ","")

    n_separator = input_string.count("->")
    result = ""
    if n_separator == 0:
        summand=input_string
    elif n_separator==1:
        [summand,result] = input_string.split("->")
    else:
        error("Error[einsum]: There can be no more than one '->' separator.")
    n_obj = summand.count(",")+1
    obj_list = [ obj for obj in args[1:n_obj+1] ]
    stat_list = sum([ make_list(obj.statistic) for obj in args[1:n_obj+1] ],[])
    dim_list = sum([ make_list(obj.shape) for obj in args[1:n_obj+1] ],[])

    for stat in stat_list:
        if stat == hybrid_symbol:
            error("Error[einsum]: Split the hybrid indices before performing einsum.")

    # make sure that all objects are in standard format

    for i in range(len(obj_list)):
        #obj_list[i].info("*")
        if obj_list[i].encoder != 'canonical':
            obj_list[i] = obj_list[i].switch_encoder().copy()
        if obj_list[i].format != 'standard':
            obj_list[i] = obj_list[i].switch_format().copy()

    #force all objects to be either dense or sparse
    this_type = type(obj_list[0])
    for obj in obj_list:
        if type(obj) != this_type:
            error("Error[einsum]: The summands must be either all dense or all sparse.")

    summand_with_comma = summand
    while summand.count(",")>0:
        summand = summand.replace(",","")

    # get some information about the indices
    # [ f=<char>, [lf,rf], [index locations in lf], [statistics] ]
    index_info = []
    for i,char in enumerate(summand):

        # skip if bosonic
        if stat_list[i] in bose_type:
            continue

        lf = summand.count(char)
        rf = result.count(char)

        if lf >2 or rf > 1 or (lf+rf)%2==1 :
            error("Error[einsum]: Inconsistent index statistics. (lf >2 or rf > 1 or (lf+rf)%2==1)")

        if lf==1:
            continue

        # add a new entry if it is a new character
        if summand[:i].count(char)==0 :
            if n_separator==0:
                index_info += [ [ char, [i] , [stat_list[i]] ] ]
            else:
                index_info += [ [ char, [i] , [stat_list[i]] ] ]
        else:
            for k in range(len(index_info)):
                if(index_info[k][0]==char):
                    index_info[k][1] += [i]
                    index_info[k][2] += [stat_list[i]]

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

    for elem in index_info:
        if len(elem[2])==2 and is_statwise_inconsistent(elem[2][0],elem[2][1]):
            error("Error[einsum]: The contracted indices have inconsistent statistic!")

    # count fermions number
    fermion_num_left = 0
    fermion_num_right = 0
    for i,char in enumerate(summand):
        if stat_list[i] in fermi_type:
            fermion_num_left+=1
    for char in result:
        i = summand.index(char)
        if stat_list[i] in fermi_type:
            fermion_num_right+=1

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #                     Step 1: Strings for sign factor computation
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    if debug_mode :
        t0 = time.time()
        print()
        print("Step 1: Strings for sign factor computation",end="")

    fsummand = ""
    fstat_list = []
    fdim_list = []
    for i,char in enumerate(summand):
        if stat_list[i] in fermi_type:
            fsummand += char
            fstat_list += [stat_list[i]]
            fdim_list += [dim_list[i]]

    xsummand = fsummand
    summed_pairs = [] # This is a list of the index character of the summed pairs
    for i,char in enumerate(fsummand):
        if fstat_list[i]==-1 and fsummand.count(char)==2:
            new_char = get_char(xsummand+summand)
            xsummand = xsummand[:i]+new_char+xsummand[i+1:]
            summed_pairs += [ [new_char,char] ]
    xstat_list = fstat_list.copy()
    xdim_list = fdim_list.copy()

    fsorted = sorted(fsummand)
    xsorted = ""
    for i,char in enumerate(fsorted):
        if xsorted[:i].count(char)>0:
            for [k2,k] in summed_pairs:
                if k==char :
                    xsorted+=k2
        else:
            xsorted+=char

    xsorted_stat_list = reordering(xsummand,xsorted,fstat_list)
    xsorted_dim_list = reordering(xsummand,xsorted,fdim_list)

    str_step1 = xsummand+"->"+xsorted
    str_sgn1 = xsummand

    if fermion_num_right>0 :
        ysorted = xsorted
        for pair in summed_pairs:
            ysorted = ysorted.replace(pair[0],"")
            ysorted = ysorted.replace(pair[1],"")

        yresult = ""
        # remove bosonic indices
        for char in result:
            index = summand.index(char)
            if stat_list[index] in fermi_type:
                yresult+=char

        ysorted_dim_list = []
        for char in ysorted:
            ysorted_dim_list += [xsorted_dim_list[xsorted.index(char)]]

        str_step2 = ysorted+"->"+yresult
        str_sgn2 = ysorted
        #print(str_step2)

        xstr_sgn2 = str_sgn2
        for [f2,f1] in summed_pairs:
            xstr_sgn2 = xstr_sgn2.replace(f2,f1)

    xstr_sgn1 = str_sgn1
    for [f2,f1] in summed_pairs:
        xstr_sgn1 = xstr_sgn1.replace(f2,"")

    xstr_sf = ""
    sf_dim_list = []
    for [f2,f1] in summed_pairs:
        xstr_sf += f1
        sf_dim_list += [xsorted_dim_list[xsorted.index(f1)]]

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #                     Step 2: Get the sign tensors
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    if debug_mode :
        print(" (",time.time()-t0,"s )...")
        t0 = time.time()
        print("Step 2: Get the sign tensors",end="")
    # S1

    # this is a little complicate...
    # let's remove the index that is to be summed later for faster computation
    xdim_list_reduced = []
    to_duplicate_list = [] # [duplicated_index, index_to_be_duplicated_from]
    for i,char in enumerate(xsummand):
        to_skip = False
        original_char = ""
        for [f2,f1] in summed_pairs:
            if f2==char:
                to_skip = True
                original_char = f1
                break
        if to_skip:
            to_duplicate_list += [ [i, xsummand.index(original_char) ] ]
            continue
        else:
            xdim_list_reduced += [ dim_list[ summand.index(char) ] ]
    xdim_list_reduced = make_tuple(xdim_list_reduced)
    
    if debug_mode :
        print(" (",time.time()-t0,"s )...")
        t0 = time.time()
        print("    S1:",end="")
    
    #==========================================
    S1 = np.zeros(xdim_list_reduced,dtype=int)
    iterator = np.nditer(S1, flags=['multi_index'])
    individual_parity_list = []
    sgn_list = []
    for element in iterator:
        coords = iterator.multi_index
        dupped_coords = list(coords)
        
        for [f1,f2] in to_duplicate_list:
            dupped_coords.insert(f1,0)

        for [f1,f2] in to_duplicate_list:
            dupped_coords[f1] = dupped_coords[f2]

        individual_parity = tuple([ param.gparity(i)%2 for i in dupped_coords ])
        
        if individual_parity not in individual_parity_list:
            individual_parity_list += [individual_parity]
            sgn1 = relative_parity(str_step1,individual_parity)
            sgn_list += [sgn1]
        else:
            index = individual_parity_list.index(individual_parity)
            sgn1 = sgn_list[index]

        S1[coords] = sgn1
    
    if debug_mode :
        print(" (",time.time()-t0,"s )...")
        t0 = time.time()
        print("    S2:",end="")

    # S2
    individual_parity_list = []
    sgn_list = []
    if fermion_num_right>0 :
        S2 = np.zeros(ysorted_dim_list,dtype=int)
        iterator = np.nditer(S2, flags=['multi_index'])

        for element in iterator:
            coords = iterator.multi_index
            individual_parity = tuple([ param.gparity(i) for i in coords ])
            if individual_parity not in individual_parity_list:
                individual_parity_list += [individual_parity]
                sgn2 = relative_parity(str_step2,individual_parity)
                sgn_list += [sgn2]
            else:
                index = individual_parity_list.index(individual_parity)
                sgn2 = sgn_list[index]
            S2[coords] = sgn2

    #dense(S2).switch_encoder().display("S2:"+str_step2)

    if debug_mode :
        print(" (",time.time()-t0,"s )...")
        t0 = time.time()
        print("    SF:",end="")
    # SF
    SF = np.zeros(sf_dim_list,dtype=int)
    iterator = np.nditer(SF, flags=['multi_index'])
    for element in iterator:
        coords = iterator.multi_index
        sgnf = 1
        for ind in coords:
            sgnf *= param.sgn(ind)
        SF[coords] = sgnf
    #dense(SF).switch_encoder().display("SF")

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #                     Step 3: Do the summation
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    if debug_mode :
        print(" (",time.time()-t0,"s )...")
        t0 = time.time()
        print("Step 3: Do the summation",end="")

    if this_type == dense:
        einsum_string1 = summand_with_comma
        einsum_obj_list = obj_list.copy()
        if fermion_num_right > 1:
            if len(xstr_sgn1)>0 :
                einsum_string1 += ","+xstr_sgn1
                einsum_obj_list += [S1]
            if len(xstr_sgn2)>0 :
                einsum_string1 += ","+xstr_sgn2
                einsum_obj_list += [S2]
            if len(xstr_sf)>0 :
                einsum_string1 += ","+xstr_sf
                einsum_obj_list += [SF]
        else:
            if len(xstr_sgn1)>0 :
                einsum_string1 += ","+xstr_sgn1
                einsum_obj_list += [S1]
            if len(xstr_sf)>0 :
                einsum_string1 += ","+xstr_sf
                einsum_obj_list += [SF]
        einsum_string = einsum_string1
        if len(result)>0 :
            einsum_string += "->"+result

        for i,obj in enumerate(einsum_obj_list):
            if type(obj)==dense :
                einsum_obj_list[i] = obj.data.copy()


        ret = oe.contract(*tuple([einsum_string]+einsum_obj_list))

        if len(result)>0:
            ret_stat = []
            for char in result:
                ret_stat += [ stat_list[ summand.index(char) ] ]
            ret = dense(ret,statistic=make_tuple(ret_stat))
        else:
            #ret = dense(ret)
            #ret.statistic = make_tuple(0)
            if type(ret)==np.ndarray :
                return ret.flatten()[0]
            else :
                return ret

        if format=="matrix":
            ret = ret.switch_format().copy()
        if encoder=="parity-preserving":
            ret = ret.switch_encoder().copy()

    elif this_type == sparse:

        # the pre-processed summand string
        einsum_string1 = summand_with_comma
        einsum_obj_list = obj_list.copy()
        if fermion_num_right > 1:
            if len(xstr_sgn1)>0 :
                einsum_string1 += ","+xstr_sgn1
                einsum_obj_list += [S1]
            if len(xstr_sgn2)>0 :
                einsum_string1 += ","+xstr_sgn2
                einsum_obj_list += [S2]
            if len(xstr_sf)>0 :
                einsum_string1 += ","+xstr_sf
                einsum_obj_list += [SF]
        else:
            if len(xstr_sgn1)>0 :
                einsum_string1 += ","+xstr_sgn1
                einsum_obj_list += [S1]
            if len(xstr_sf)>0 :
                einsum_string1 += ","+xstr_sf
                einsum_obj_list += [SF]

        # generate a list containing information about the summing vertices
        summed_pairs2 = []
        for i,char in enumerate(einsum_string1):

            if char == ",":
                einsum_string2 += ","
                continue

            nchar = einsum_string1.count(char)

            if nchar>2:
                char_replaced = [char]
                for j in range(nchar-1):
                    char_replaced += [get_char(einsum_string1+''.join(char_replaced))]

                einsum_string2 = einsum_string1
                for rchar in char_replaced[1:]:
                    einsum_string2 = einsum_string2.replace(char,rchar,1)

                summed_pairs2 += [ char_replaced ]

                einsum_string1 = einsum_string2

        for i in range(len(summed_pairs2)):
            summed_pairs2[i] = ''.join(summed_pairs2[i])
            einsum_string1 += ","+summed_pairs2[i]

        # generate a list of vertices
        einsum_string = einsum_string1
        vertex_shape_list = []
        for x in summed_pairs2:
            if(len(x)==1):
                continue
            dim = dim_list[summand.index(x[0])]
            shape = tuple( [ dim for i in range(len(x)) ] )

            coords = [ [] for i in range(len(x)) ]
            value = []
            for index in range(dim):
                for i in range(len(x)):
                    coords[i] += [index]
                value += [1]

            vertex = sp.COO(coords,value,shape)

            vertex_shape_list += [vertex.copy()]

        einsum_obj_list += vertex_shape_list

        for i,obj in enumerate(einsum_obj_list):
            if(type(obj)==sparse):
                einsum_obj_list[i] = obj.data
            if(type(obj)==np.array or type(obj)==np.ndarray):
                einsum_obj_list[i] = sp.COO.from_numpy(obj)

        if len(result)>0:
            einsum_string += "->"+result

        ret = oe.contract(*tuple([einsum_string]+einsum_obj_list))

        if len(result)>0:
            ret_stat = []
            for char in result:
                ret_stat += [ stat_list[ summand.index(char) ] ]
            ret = sparse(ret,statistic=make_tuple(ret_stat))
        else:
            #coords = [0]
            #value = make_list(ret)
            #ret = sparse(sp.COO([[0]],value,make_tuple(1)),statistic=make_tuple(0))
            if type(ret)==sp.COO :
                return make_list(ret.data)[0]
            else :
                return ret

        if format=="matrix":
            ret = ret.switch_format().copy()
        if encoder=="parity-preserving":
            ret = ret.switch_encoder().copy()

    if debug_mode :
        print(" (",time.time()-t0,"s )...")
        t0 = time.time()
        print("Step 4: Finish")
        print()

    return ret

####################################################
##                     Reshape                    ##
####################################################

display_stage = []

def join_legs(XGobj,string_inp,make_format='standard',intermediate_stat=(-1,1)):
    intermediate_stat = make_tuple(intermediate_stat)
    
    # Always output the parity-preserving encoder
    #===============================================================================#
    #   Step 0: Preconditioning the initial Object to standard & canonical          #
    #===============================================================================#
    Obj = XGobj.copy()
    this_type = type(XGobj)
    this_format = XGobj.format
    this_encoder = XGobj.encoder
    if this_type   == sparse:
        Obj = dense(Obj)
    if this_format == 'matrix':
        #force convert to standard
        Obj = Obj.switch_format()
    if this_encoder == 'parity-preserving':
        #force convert to standard
        Obj = Obj.switch_encoder()
        
    if 0 in display_stage:
        Obj.display("After stage 0")
    
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
    
    #print(Obj.statistic)
    #print(sorted_stat)
    
    #print(npeinsum_string)
    #for obj in npeinsum_obj:
    #    print(obj.shape)
    
    #sorted tensor
    Obj.data = oe.contract(*make_tuple( [npeinsum_string] + npeinsum_obj ))
    Obj.statistic = sorted_stat
    
    #Obj.display()
    if 1 in display_stage:
        Obj.display("After stage 1")
    
    #===============================================================================#
    #   Step 2: Join fermionic indices with np.reshape                              #
    #===============================================================================#
    
    new_stats, new_shape, final_stats, final_shape = get_intermediate_info(sorted_group_info,intermediate_stat)
    #intermediate_tensor
    
    Obj.data = np.reshape(Obj.data,new_shape)
    Obj.statistic = new_stats
    
    if 2 in display_stage:
        Obj.display("After stage 2")
    
    
    #===============================================================================#
    #   Step 3: Switch format if make_format='matrix'                               #
    #===============================================================================#
    
    if make_format == 'matrix':
        Obj = Obj.switch_format()
        if 3 in display_stage:
            Obj.display("After stage 3")
    else:
        if 3 in display_stage:
            print("Step 3 is skipped.")
    
    #===============================================================================#
    #   Step 4: Switch encoder                                                      #
    #===============================================================================#
    
    Obj = Obj.switch_encoder()
    
    if 4 in display_stage:
        Obj.display("After stage 4")
    
    #===============================================================================#
    #   Step 5: Merge bosons and fermions                                           #
    #===============================================================================#
    
    Obj.data = np.reshape(Obj.data,final_shape)
    Obj.statistic = final_stats
    
    if 5 in display_stage:
        Obj.display("After stage 5")
    
    return Obj
    
def split_legs(XGobj,string_inp,final_stat,final_shape,intermediate_stat=(-1,1)):
    intermediate_stat = make_tuple(intermediate_stat)
    final_stat = make_tuple(final_stat)
    final_shape = make_tuple(final_shape)

    #===============================================================================#
    #   Step 0: Preparation                                                         #
    #===============================================================================#
    
    Obj = XGobj.copy()
    this_type = type(XGobj)
    this_format = XGobj.format
    this_encoder = XGobj.encoder
    
    if 5 in display_stage:
        Obj.display("Before stage 5")
    
    #===============================================================================#
    #   Step 5: Split bosons and fermions                                           #
    #===============================================================================#
    
    group_info, sorted_group_info = get_group_info(string_inp, final_stat, final_shape)
    new_stats, new_shape, _, _ = get_intermediate_info(sorted_group_info,intermediate_stat)
    sign_factors_list = get_grouping_sign_factors(sorted_group_info, intermediate_stat)
    
    Obj.data = np.reshape(Obj.data,new_shape)
    Obj.statistic = new_stats
    
    if 4 in display_stage:
        Obj.display("Before stage 4")
    
    #===============================================================================#
    #   Step 4: Switch encoder                                                      #
    #===============================================================================#
    
    Obj = Obj.switch_encoder()
    
    if this_format == 'matrix':
        if 3 in display_stage:
            Obj.display("Before stage 3")
    else:
        if 3 in display_stage:
                print("Step 3 is skipped.")
                
    #===============================================================================#
    #   Step 3: Switch format if this_format='matrix'                               #
    #===============================================================================#
    
    if this_format == 'matrix':
        Obj = Obj.switch_format()
        
    if 2 in display_stage:
        Obj.display("Before stage 2")
    
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
    
    if 1 in display_stage:
        Obj.display("Before stage 1")
    
    
    #===============================================================================#
    #   Step 1: Move bosonic indices to the left of each group                      #
    #===============================================================================#
    
    unsorted_string = ''.join( [ indices for [indices, stats, shape] in group_info ] )
    sorted_string = ''.join( [ indices for [indices, stats, shape] in sorted_group_info ] )
    npeinsum_string = sorted_string+sign_factors_list[0]+"->"+unsorted_string
    npeinsum_obj = [Obj.data] + sign_factors_list[1]
    Obj.data = oe.contract(*make_tuple( [npeinsum_string] + npeinsum_obj ))
    Obj.statistic = final_stat
    
    #Obj.display()
    if 0 in display_stage:
        Obj.display("Before stage 0")
    
    if this_format=='matrix':
        return Obj.switch_format()
    
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

def trim_grassmann_odd(Obj_input):
    objtype=type(Obj_input)

    Obj = Obj_input.copy()

    if(Obj.encoder == 'canonical'):
        Obj = Obj.switch_encoder()
        

    if(objtype==dense):
        Obj = sparse(Obj)
    C = Obj.coords
    for i in range(Obj.nnz):
        fcoords = [ ind for j,ind in enumerate(C[i]) if (Obj.statistic[j] in fermi_type)]
        if(sum(fcoords)%2 == 1):
            Obj.data.data[i] = 0
    if(objtype==dense):
        Obj = dense(Obj)

    if(Obj_input.encoder == 'canonical'):
        Obj = Obj.switch_encoder()

    return Obj

def is_grassmann_even(Obj_input):
    Obj = Obj_input.copy()
    if Obj.encoder == 'canonical':
        Obj = Obj.switch_encoder()
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
    if np.abs(Λ[0]) < numer_cutoff:
        error("Error[SortedSVD]: np.linalg.svd() returns zero singular value vector!")

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
        Λx = np.diag(np.pad(Λx,(0,padding_dimension),'constant',constant_values=(0,0)       ))
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

display_stage_svd = []

def svd(XGobj,string,cutoff=None):

    # the string is of the form aaaa|bbb

    this_type = type(XGobj)
    this_format = XGobj.format
    this_encoder = XGobj.encoder
    Obj = XGobj.copy()
    if(this_type==sparse):
        Obj = dense(Obj)
    if(this_type not in [dense,sparse]):
        error("Error[svd]: Object type must only be dense or sparse!")
        
    # check if XGobj.statistic or final_statistic is weird or not
    for stat in XGobj.statistic:
        if(stat not in allowed_stat):
            error("Error[svd]: The input object contains illegal statistic. (0, 1, -1, or "+hybrid_symbol+" only)")
            
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
    
    if 0 in display_stage_svd :
        Obj.display("After stage 0")

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
    Obj = Obj.join_legs(join_legs_string_input,"matrix",intermediate_stat=intermediate_stat)

    if 1 in display_stage_svd :
        Obj.display("After stage 1")
        
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #:::::      STEP 2 - BLOCK SVD (MAKE SURE IT'S PARITY-PRESERVING!)                          :::::
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    if Obj.statistic[0]==0 or Obj.statistic[1]==0:
        U, Λ, V = SortedSVD(Obj.data,cutoff)
        Λ = np.diag(Λ)
    else:
        U, Λ, V = BlockSVD(Obj.data,cutoff)

    if 2 in display_stage_svd :
        cU = oe.contract('ij->ji',np.conjugate(U))
        cV = oe.contract('ij->ji',np.conjugate(V))
        dense(oe.contract('ij,jk->ik',cU,U)).display("cUU (stage 2)")
        dense(oe.contract('ij,jk->ik',V,cV)).display("VcV (stage 2)")
        dense(U).display("U (stage 2)")
        dense(V).display("U (stage 2)")
        
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #:::::      STEP 3 - RECONSTRUCT U, Λ, and V AS GRASSMANN TENSORS                           :::::
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

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

    if 3 in display_stage_svd :
        U.display("U (stage 3)")
        Λ.display("Λ (stage 3)")
        V.display("V (stage 3)")
        
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
            Vstats+=[XGobj.statistic[i-1]]
            Vshape+=[XGobj.shape[i-1]]
        else:
            Uind+=char
            Ustats+=[XGobj.statistic[i]]
            Ushape+=[XGobj.shape[i]]

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
    
    U = U.split_legs(Uind,Ustats,Ushape)
    Λ = Λ.switch_encoder()
    V = V.split_legs(Vind,Vstats,Vshape)
    
    if 4 in display_stage_svd :
        U.display("U (stage 4)")
        Λ.display("Λ (stage 4)")
        V.display("V (stage 4)")
        
    if(this_format == 'standard'):
        U = U.switch_format()
        Λ = Λ.switch_format()
        V = V.switch_format()

    if(this_encoder == 'parity-preserving'):
        U = U.switch_encoder()
        Λ = Λ.switch_encoder()
        V = V.switch_encoder()

    if(this_type==sparse):
        U = sparse(U)
        Λ = sparse(Λ)
        V = sparse(V)

    return U, Λ, V

####################################################
##            Eigen value decomposition           ##
####################################################

def SortedEig(M,cutoff=None):
    Λ, U = np.linalg.eig(M)

    idx = np.abs(Λ).argsort()[::-1]   
    Λ = Λ[idx]
    U = U[:,idx]
    
    nnz = 0
    if np.abs(Λ[0]) < numer_cutoff:
        for s in Λ:
            print(s)
        error("Error[SortedEig]: np.linalg.eig() returns zero eigenvalue vector!")
        
    for i,s in enumerate(Λ):
        if np.abs(s/Λ[0]) > numer_cutoff:
            nnz+=1
        else:
            break

    if cutoff!=None and cutoff < nnz:
        nnz = cutoff

    Λ = Λ[:nnz]
    U = U[:,:nnz]
    
    return Λ, U

# I = cUU
def BlockEig(Obj,cutoff=None):
    
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

    ΛE, UE = SortedEig(ME,halfcutoff)
    ΛO, UO = SortedEig(MO,halfcutoff)

    d = max(len(ΛE),len(ΛO))
    d = int(2**math.ceil(np.log2(d)))

    def padding(Ux, Λx, padding_dimension):
        Ux = np.pad(Ux,((0,0),(0,padding_dimension)),'constant',constant_values=((0,0),(0,0)))
        Λx = np.diag(np.pad(Λx,(0,padding_dimension),'constant',constant_values=(0,0)       ))
        return Λx, Ux

    ΛE, UE = padding(UE, ΛE, d-len(ΛE))
    ΛO, UO = padding(UO, ΛO, d-len(ΛO))

    #print()
    #print("--------------------")
    #for i,s in enumerate(np.diag(ΛE)):
    #    print(" even:",clean_format(s))
    #print()
    #for i,s in enumerate(np.diag(ΛO)):
    #    print("  odd:",clean_format(s))
    #print("--------------------")
    #print()

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
    

    return Λ, U

display_stage_eig = []

def eig(XGobj,string,cutoff=None):

    # the string is of the form aaaa|bbb

    this_type = type(XGobj)
    this_format = XGobj.format
    this_encoder = XGobj.encoder
    Obj = XGobj.copy()
    if(this_type==sparse):
        Obj = dense(Obj)
    if(this_type not in [dense,sparse]):
        error("Error[svd]: Object type must only be dense or sparse!")
        
    # check if XGobj.statistic or final_statistic is weird or not
    for stat in XGobj.statistic:
        if(stat not in allowed_stat):
            error("Error[svd]: The input object contains illegal statistic. (0, 1, -1, or "+hybrid_symbol+" only)")
            
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
    
    if 0 in display_stage_eig :
        Obj.display("After stage 0")

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
    Obj = Obj.join_legs(join_legs_string_input,"matrix",intermediate_stat=intermediate_stat)

    if 1 in display_stage_eig :
        Obj.display("After stage 1")
        
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #:::::      STEP 2 - BLOCK SVD (MAKE SURE IT'S PARITY-PRESERVING!)                          :::::
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    if Obj.statistic[0]==0 or Obj.statistic[1]==0:
        Λ, U = SortedEig(Obj.data,cutoff)
        Λ = np.diag(Λ)
    else:
        Λ, U = BlockEig(Obj.data,cutoff)

    if 2 in display_stage_eig :
        cU = oe.contract('ij->ji',np.conjugate(U))
        dense(oe.contract('ij,jk->ik',cU,U)).display("cUU (stage 2)")
        dense(oe.contract('ij,jk->ik',V,cV)).display("VcV (stage 2)")
        dense(U).display("U (stage 2)")
        dense(V).display("U (stage 2)")
        
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #:::::      STEP 3 - RECONSTRUCT U, Λ, and V AS GRASSMANN TENSORS                           :::::
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    #the first way is to form the tensor first, then split
    Λstatleft = -1
    Λstatright = +1
    if Obj.statistic[0]==0:
        U = dense(U,encoder="parity-preserving",format="matrix",statistic=(0,0))
        Λ = dense(Λ,encoder="parity-preserving",format="matrix",statistic=(0,0))
        Λstatleft = 0
        Λstatright = 0
    elif Obj.statistic[1]==0:
        U = dense(U,encoder="parity-preserving",format="matrix",statistic=(Obj.statistic[0],0))
        Λ = dense(Λ,encoder="parity-preserving",format="matrix",statistic=(0,0))
        Λstatleft = 0
        Λstatright = 0
    else:
        U = dense(U,encoder="parity-preserving",format="matrix",statistic=(Obj.statistic[0],1))
        Λ = dense(Λ,encoder="parity-preserving",format="matrix",statistic=(-1,1))
    dΛ = Λ.shape[0]

    if 3 in display_stage_eig :
        U.display("U (stage 3)")
        Λ.display("Λ (stage 3)")
        
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #:::::      STEP 4 - Split the legs                                                         :::::
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    # count the number of indices in the two groups
    Uind = ""
    Ustats = []
    Ushape = []
    partition_found = False
    for i,char in enumerate(string):
        if char in separator_list :
            partition_found = True
            continue
        if not partition_found :
            Uind+=char
            Ustats+=[XGobj.statistic[i]]
            Ushape+=[XGobj.shape[i]]

    new_ind1 = ""
    for char in char_list:
        if char not in Uind:
            new_ind1 = char
            break

    Uind   = "("+Uind+")" + new_ind1
    Ustats = tuple(Ustats + [-Λstatleft])
    Ushape = tuple(Ushape + [dΛ])
    
    U = U.split_legs(Uind,Ustats,Ushape)
    Λ = Λ.switch_encoder()
    
    if 4 in display_stage_eig :
        U.display("U (stage 4)")
        Λ.display("Λ (stage 4)")
        
    if(this_format == 'matrix'):
        U = U.switch_format()
        Λ = Λ.switch_format()

    if(this_encoder == 'parity-preserving'):
        U = U.switch_encoder()
        Λ = Λ.switch_encoder()

    if(this_type==sparse):
        U = sparse(U)
        Λ = sparse(Λ)

    return Λ, U


####################################################
##                   Conjugation                  ##
####################################################

display_stage_hconjugate = []

def hconjugate(XGobj,string):

    # the string is of the form aaaa|bbb

    this_type = type(XGobj)
    this_format = XGobj.format
    this_encoder = XGobj.encoder
    Obj = XGobj.copy()

    if this_type==sparse :
        Obj = dense(Obj)
    if this_type not in [dense,sparse] :
        error("Error[svd]: Object type must only be dense or sparse!")
        
    # check if XGobj.statistic or final_statistic is weird or not
    for stat in XGobj.statistic:
        if(stat not in allowed_stat):
            error("Error[svd]: The input object contains illegal statistic. (0, 1, -1, or "+hybrid_symbol+" only)")
            
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
    
    if 0 in display_stage_hconjugate :
        Obj.display("After stage 0")

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
            
    Obj = Obj.join_legs(join_legs_string_input,"matrix")
    
    if 1 in display_stage_hconjugate :
        Obj.display("After stage 1")
        
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
    
    if 2 in display_stage_hconjugate :
        Obj.display("After stage 2")
        
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
            Vstats+=[XGobj.statistic[i-1]]
            Vshape+=[XGobj.shape[i-1]]
        else:
            Uind+=char
            Ustats+=[XGobj.statistic[i]]
            Ushape+=[XGobj.shape[i]]
    
    new_ind = "("+Vind+")("+Uind+")"
    new_stats = Vstats + Ustats
    new_shape = make_tuple(Vshape + Ushape)
    for i in range(len(new_stats)):
        if new_stats[i] in fermi_type :
            new_stats[i]*=-1
    new_stats = make_tuple(new_stats)
    
    Obj = Obj.split_legs(new_ind,new_stats,new_shape)
    
    if 3 in display_stage_hconjugate :
        Obj.display("After stage 3")
        
    if this_type==sparse :
        Obj = sparse(Obj)
    if this_format!=Obj.format :
        Obj = Obj.switch_format()
    if this_encoder!=Obj.encoder :
        Obj = Obj.switch_encoder()

    return Obj


####################################################
##                    Utilities                   ##
####################################################

def random(shape,statistic,tensor_format=dense,dtype=float):
    X = np.random.rand(*shape)
    if dtype == complex :
        X = complex(1,0)*X + complex(0,1)*np.random.rand(*shape)
    A = dense(X, statistic = statistic)
    A = trim_grassmann_odd(A)
    if tensor_format==sparse :
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

####################################################
##                       TRG                      ##
####################################################

def trg(T,dcut=16):

    # mandatory properties of T:
    #    - shape = (nx,ny,nx,ny)
    #    - statistic = (1,1,-1,-1)

    if [T.shape[0],T.shape[1]] != [T.shape[2],T.shape[3]] :
        error("Error[trg]: The shape must be of the form (m,n,m,n)!")

    if make_list(T.statistic) != [1,1,-1,-1] :
        error("Error[trg]: The statistic must be (1,1,-1,-1)!")

    #===============================================================================#
    #   Step 1: Rearrange the tensor legs in two ways                               #
    #===============================================================================#
    
    T1 = einsum('ijkl->jkli',T)
    T2 = einsum('ijkl->klij',T)

    U1,S1,V1 = T1.svd('ab cd')
    U2,S2,V2 = T2.svd('ab cd')

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
    
    sqrtS = sqrt(S1)
    U1 = einsum('abx,xc->abc',U1,sqrtS)
    V1 = einsum('ax,xbc->abc',sqrtS,V1)

    sqrtS = sqrt(S2)
    U2 = einsum('abx,xc->abc',U2,sqrtS)
    V2 = einsum('ax,xbc->abc',sqrtS,V2)

    
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
    
    VV = einsum('kwz,lxw->lxzk',V1,V2);
    UU = einsum('yxi,zyj->jzxi',U1,U2);
    T2 = einsum('lxzk,jzxi->ijkl',VV,UU);

    tr1 = einsum('ijkl,klij',T,T);
    tr2 = einsum('ijij',T2);
    err = np.abs(tr1-tr2)
    print("Error:",err)
    
    Tnorm = T2.norm
    T2 = dense(T2)
    T2.data = T2.data/Tnorm
    if type(T) == sparse :
        T2 = sparse(T2)
    
    return T2, Tnorm

####################################################
##                     2D ATRG                    ##
####################################################

def atrg2dy(T1,T2,dcut=16,intermediate_dcut=None,iternum=None):

    process_name = "atrg2d"
    if iternum != None:
        process_name = process_name+"["+str(iternum)+"]"
    process_length = 6
    step = 1

    if intermediate_dcut==None:
        intermediate_dcut=dcut

    T1 = einsum("ijkl->li jk",T1)
    T2 = einsum("ijkl->li jk",T2)

    U1, S1, V1 = T1.svd("li jk",intermediate_dcut)
    U2, S2, V2 = T2.svd("li jk",intermediate_dcut)

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
    step = show_progress(step,process_length,process_name)
    A = V1.copy()
    B = einsum("lia,ab->lib",U1,S1)
    C = einsum("ab,bjk->ajk",S2,V2)
    D = U2.copy()

    M = einsum("ajk,jib->aibk",C,B)

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

    step = show_progress(step,process_length,process_name)
    U, S, V = M.svd("ai bk",intermediate_dcut)

    sqrtS = sqrt(S)
    Y = einsum('abx,xc->abc',U,sqrtS)
    X = einsum('ax,xbc->abc',sqrtS,V)

    del U,S,V,sqrtS
    gc.collect()

    step = show_progress(step,process_length,process_name)
    Q1 = einsum('iax,xbj->ijab',D,Y)
    Q2 = einsum('kya,ylb->abkl',X,A)

    step = show_progress(step,process_length,process_name)
    Q = einsum('ijab,abkl->ijkl',Q1,Q2)
    step = show_progress(step,process_length,process_name)
    
    U,S,V = Q.svd("ij kl",dcut)

    sqrtS = sqrt(S)
    H = einsum('abx,xc->abc',U,sqrtS)
    G = einsum('ax,xbc->abc',sqrtS,V)

    del U,S,V,sqrtS
    gc.collect()

    step = show_progress(step,process_length,process_name)

    T = einsum('lai,kaj->ijkl',H,G)

    clear_progress()

    Tnorm = T.norm
    T.data = T.data/Tnorm

    return T, Tnorm

def atrg2dx(T1,T2,dcut=16,intermediate_dcut=None,iternum=None):
    T1 = einsum('ijkl->jilk',T1)
    T, Tnorm = atrg2dy(T1,T2,dcut,intermediate_dcut,iternum)
    T = einsum('jilk->ijkl',T)
    return T, Tnorm

####################################################
##                     3D ATRG                    ##
####################################################

def hotrg3dz(T1,T2,dcut=16,intermediate_dcut=None,iternum=None):

    process_name = "hotrg3d"
    if iternum != None:
        process_name = process_name+"["+str(iternum)+"]"
    process_length = 32
    step = 1

    if intermediate_dcut==None:
        intermediate_dcut=dcut

    step = show_progress(step,process_length,process_name)
    T1 = einsum('ijklmn->ikmn jl',T1)
    step = show_progress(step,process_length,process_name)
    X1,S1,Y1 = T1.svd('ikmn jl',intermediate_dcut)
    sqrtS = sqrt(S1)
    step = show_progress(step,process_length,process_name)
    X1 = einsum('ikmna,ab->ikbmn',X1,sqrtS)
    step = show_progress(step,process_length,process_name)
    Y1 = einsum('ab,bjl->ajl',sqrtS,Y1)

    step = show_progress(step,process_length,process_name)
    T1b = einsum('ikbmn,bjl->ikmn jl',X1,Y1)

    step = show_progress(step,process_length,process_name)
    T2 = einsum('ijklmn->ikmn jl',T2)
    step = show_progress(step,process_length,process_name)
    X2,S2,Y2 = T2.svd('ikmn jl',intermediate_dcut)
    sqrtS = sqrt(S2)
    step = show_progress(step,process_length,process_name)
    X2 = einsum('ikmna,ab->ikbmn',X2,sqrtS)
    step = show_progress(step,process_length,process_name)
    Y2 = einsum('ab,bjl->ajl',sqrtS,Y2)

    step = show_progress(step,process_length,process_name)
    Qpx = einsum('iIKmn,jJLmn->IKJLmn ij',X1,X2)
    step = show_progress(step,process_length,process_name)
    Qmx = einsum('IKJLmn ij->IJ iKjLmn',Qpx)

    step = show_progress(step,process_length,process_name)
    Qpy = einsum('KiI,LjJ->KILJ ij',Y1,Y2)
    step = show_progress(step,process_length,process_name)
    Qmy = einsum('KILJ ij->IJ KiLj',Qpy)

    del X1,S1,Y1,X2,S2,Y2,sqrtS
    gc.collect()

    step = show_progress(step,process_length,process_name)
    cQpx = Qpx.hconjugate('IKJLmn ij')
    step = show_progress(step,process_length,process_name)
    cQmx = Qmx.hconjugate('IJ iKjLmn')
    step = show_progress(step,process_length,process_name)
    cQpy = Qpy.hconjugate('KILJ ij')
    step = show_progress(step,process_length,process_name)
    cQmy = Qmy.hconjugate('IJ KiLj')


    step = show_progress(step,process_length,process_name)
    Mpx = einsum('ij abcdef,abcdef IJ-> ij IJ ',cQpx,Qpx)
    step = show_progress(step,process_length,process_name)
    Mmx = einsum('ij abcdef,abcdef IJ-> ij IJ ',Qmx,cQmx)
    step = show_progress(step,process_length,process_name)
    Mpy = einsum('ij abcd,abcd IJ-> ij IJ ',cQpy,Qpy)
    step = show_progress(step,process_length,process_name)
    Mmy = einsum('ij abcd,abcd IJ-> ij IJ ',Qmy,cQmy)

    del Qpx,Qmx,Qpy,Qmy,cQpx,cQmx,cQpy,cQmy
    gc.collect()

    step = show_progress(step,process_length,process_name)
    Upx, Spx, Vpx = Mpx.eig('ij IJ',dcut)
    step = show_progress(step,process_length,process_name)
    Umx, Smx, Vmx = Mmx.eig('ij IJ',dcut)
    step = show_progress(step,process_length,process_name)
    Upy, Spy, Vpy = Mpy.eig('ij IJ',dcut)
    step = show_progress(step,process_length,process_name)
    Umy, Smy, Vmy = Mmy.eig('ij IJ',dcut)
    
    del Mpx,Mmx,Mpy,Mmy
    gc.collect()

    if Spx.shape[0] < Smx.shape[0] :
        Ux = Upx.copy()
    else:
        Ux = Umx.copy()
    if Spy.shape[0] < Smy.shape[0] :
        Uy = Upy.copy()
    else:
        Uy = Umy.copy()

    clear_progress()

    return 1,2