import numpy as np
import math
from grassmannTN import param
import sparse as sp
import opt_einsum as oe
import time

hybrid_symbol = "*"
separator_list = ("|",":",";",",","."," ")

skip_parity_blocking_check = True
allowed_stat = (0,1,-1,hybrid_symbol)
fermi_type = (1,-1)
bose_type = (0,hybrid_symbol)
encoder_type = ("canonical","parity-preserving")
format_type = ("standard","matrix")

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

def error(text):
    print(text)
    exit()

def get_char(string):
    for char in char_list:
        if char not in string:
            return char
    error("Error[get_char]: Running out of index character!")    

def clean_format(number):
    if np.abs(np.real(number))>1.0e-14 and np.abs(np.imag(number))<1.0e-14:
        return round(np.real(number),13)
    elif np.abs(np.real(number))<1.0e-14 and np.abs(np.imag(number))>1.0e-14:
        return round(np.imag(number),13)*complex(0,1)
    elif np.abs(np.real(number))<1.0e-14 and np.abs(np.imag(number))<1.0e-14:
        return 0
    else:
        return round(np.real(number),13)+round(np.imag(number),13)*complex(0,1)

####################################################
##             Densed Grassmann Array             ##
####################################################

class dense:
    def __init__(self, data=None, bosonic = False, encoder = "canonical", format = "standard", statistic=None):
    
        
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
            self.format = 'standard'
            if bosonic:
                self.statistic = tuple([ 0 for i in range(len(self.shape))])
            else:
                self.statistic = tuple([ 1 for i in range(len(self.shape))])
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
            self.format = 'standard'
            if bosonic:
                self.statistic = tuple([ 0 for i in range(len(self.shape))])
            else:
                self.statistic = tuple([ 1 for i in range(len(self.shape))])
            default = False
        elif(data==None):
            "nothing to see here"
        else:
            error("Error[dense]: Invalid initialized data.")
            
        
        if statistic != None:
            self.statistic = statistic
            
        if not default:
            for i,dim in enumerate(self.data.shape):
                if self.statistic[i] in fermi_type and dim != int(2**math.floor(np.log2(dim))):
                    error("Caution[dense]: Some of the fermionic tensor shapes are not a power of two.\n                Have you added the <statistic> argument when calling this function?")
                
    
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

    def display(self,name=None):
        print()
        if name != None:
            print("        name:",name)
        print("  array type: dense")
        print("       shape:",self.shape)
        print("        size:",self.size)
        print("   statistic:",self.statistic)
        print("      format:",self.format)
        print("     encoder:",self.encoder)
        print("     entries:")
        iterator = np.nditer(self, flags=['multi_index'])
        for element in iterator:
            coords = iterator.multi_index
            if(np.abs(element.item())>1.0e-14):
                print(coords,clean_format(element.item()))
        print()

    def info(self,name=None):
        print()
        if name != None:
            print("        name:",name)
        print("  array type: dense")
        print("       shape:",self.shape)
        print("        size:",self.size)
        print("   statistic:",self.statistic)
        print("      format:",self.format)
        print("     encoder:",self.encoder)
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
                    sgn_value *= param.sgn[ind]
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
            ret = ret.switch_encoder().copy()
        return ret

    def switch_encoder(self):
        ret = self.copy()
        iterator = np.nditer(self.data, flags=['multi_index'])
        for element in iterator:
            coords = iterator.multi_index
            new_coords = []
            for i,ind in enumerate(coords):
                if(self.statistic[i] in fermi_type):
                    new_coords += [param.encoder[ind]]
                else:
                    new_coords += [ind]
            new_coords = tuple(new_coords)
            ret.data[coords] = self.data[new_coords]
        if(ret.encoder=='canonical'):
            ret.encoder='parity-preserving'
        else:
            ret.encoder='canonical'
        return ret

    def join_legs(self,string_inp,statistic=None,format='standard'):
        return join_legs(self,string_inp,statistic,format)

    def split_legs(self,string_inp,statistic,shape,format='standard'):
        return split_legs(self,string_inp,statistic,shape,format)

    def hconjugate(self,input_string):
        return hconjugate(self,input_string)
        
####################################################
##            Sparse Grassmann arrays             ##
####################################################

class sparse:
    def __init__(self, data=None, bosonic = False, encoder = "canonical", format = "standard", statistic = None):
    
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
            self.format = 'standard'
            if bosonic:
                self.statistic = tuple([ 0 for i in range(len(self.shape))])
            else:
                self.statistic = tuple([ 1 for i in range(len(self.shape))])
            default = False
        elif(type(data)==sp.COO):
            self.data  = data.copy()
            self.format = 'standard'
            if bosonic:
                self.statistic = tuple([ 0 for i in range(len(self.shape))])
            else:
                self.statistic = tuple([ 1 for i in range(len(self.shape))])
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
            self.statistic = statistic
        
        if not default:
            for i,dim in enumerate(self.data.shape):
                if self.statistic[i] in fermi_type and dim != int(2**math.floor(np.log2(dim))):
                    error("Caution[sparse]: Some of the fermionic tensor shapes are not a power of two.\n                 Have you added the <statistic> argument when calling this function?")
               
        
    
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

    def display(self, name=None):
        print()
        if name != None:
            print("        name:",name)
        print("  array type: sparse")
        print("       shape:",self.shape)
        print("        size:",self.size)
        print("   statistic:",self.statistic)
        print("      format:",self.format)
        print("     encoder:",self.encoder)
        print("   non-zeros:",self.nnz)
        print("     entries:")

        C = self.coords
        V = self.value
        for elem in range(self.nnz):
            print(C[elem],clean_format(V[elem]))
        print()

    def info(self,name=None):
        print()
        if name != None:
            print("        name:",name)
        print("  array type: sparse")
        print("       shape:",self.shape)
        print("        size:",self.size)
        print("   statistic:",self.statistic)
        print("      format:",self.format)
        print("     encoder:",self.encoder)
        print("   non-zeros:",self.nnz)
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
            if np.abs(ret.data.data[entry]) < 1.0e-15:
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
                    sgn_value *= param.sgn[ind]
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
                    new_coords += [param.encoder[ind]]
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

    def join_legs(self,string_inp,statistic=None,format='standard'):
        return join_legs(self,string_inp,statistic,format)

    def split_legs(self,string_inp,statistic,shape,format='standard'):
        return split_legs(self,string_inp,statistic,shape,format)

    def hconjugate(self):
        return hconjugate(self)
        
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

def einsum(*args,format="standard",encoder="canonical"):

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # WORK IN THE FOLLOWING CONDITIONS      :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # (WILL CONVERT AUTOMATICALLY)          :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #   - canonical encoder                 :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #   - standard format                   :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #                     Step 0: Error detection
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    input_string = args[0]

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
    
    S1 = np.zeros(xdim_list_reduced,dtype=int)
    iterator = np.nditer(S1, flags=['multi_index'])
    for element in iterator:
        coords = iterator.multi_index
        dupped_coords = list(coords)
        
        for [f1,f2] in to_duplicate_list:
            dupped_coords.insert(f1,0)

        for [f1,f2] in to_duplicate_list:
            dupped_coords[f1] = dupped_coords[f2]

        individual_parity = tuple([ param.gparity[i] for i in dupped_coords ])
        sgn1 = relative_parity(str_step1,individual_parity)
        S1[coords] = sgn1

    #dense(S1).switch_encoder().display("S1:"+str_step1)

    # S2
    if fermion_num_right>0 :
        S2 = np.zeros(ysorted_dim_list,dtype=int)
        iterator = np.nditer(S2, flags=['multi_index'])

        for element in iterator:
            coords = iterator.multi_index
            individual_parity = tuple([ param.gparity[i] for i in coords ])
            sgn2 = relative_parity(str_step2,individual_parity)
            S2[coords] = sgn2

    #dense(S2).switch_encoder().display("S2:"+str_step2)

    # SF
    SF = np.zeros(sf_dim_list,dtype=int)
    iterator = np.nditer(SF, flags=['multi_index'])
    for element in iterator:
        coords = iterator.multi_index
        sgnf = 1
        for ind in coords:
            sgnf *= param.sgn[ind]
        SF[coords] = sgnf
    #dense(SF).switch_encoder().display("SF")

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


        ret = np.einsum(*tuple([einsum_string]+einsum_obj_list))

        if len(result)>0:
            ret_stat = []
            for char in result:
                ret_stat += [ stat_list[ summand.index(char) ] ]
            ret = dense(ret,statistic=make_tuple(ret_stat))
        else:
            ret = dense(ret)
            ret.statistic = make_tuple(0)

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
            coords = [0]
            value = [ret.data[0]]
            ret = sparse(sp.COO([[0]],[ret.data[0]],make_tuple(1)))
            ret.statistic = make_tuple(0)

        if format=="matrix":
            ret = ret.switch_format().copy()
        if encoder=="parity-preserving":
            ret = ret.switch_encoder().copy()

    #ret.info("?")

    return ret

####################################################
##                     Reshape                    ##
####################################################

# object conditions checked
def join_legs(XGobj,string_inp,statistic=None,format='standard'):

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # WORK IN THE FOLLOWING CONDITIONS      :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # (WILL CONVERT AUTOMATICALLY)          :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #   - canonical encoder                 :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #   - standard format                   :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    # the string is of the form aaa(bb)cccc(ddd)eeee
    # The indices will then be grouped together as one index

    #i change the variable name around because I want the input variable name to be statistic
    final_statistic_inp = statistic

    string = string_inp
    if string.count(")") == 0 and string.count(")") == 0:
        nseparator = 0
        for separator in separator_list:
            nseparator += string.count(separator)
        if nseparator>0 :
            for separator in separator_list:
                string = string.replace(separator,")(")
            string = "("+string+")"


    if final_statistic_inp != None:
        final_statistic = make_tuple(final_statistic_inp)

    objtype = type(XGobj)
    Obj = XGobj.copy()
    if(objtype==sparse):
        Obj = dense(Obj)
    if(objtype not in [dense,sparse]):
        error("Error[join_legs]: Object type must only be dense or sparse!")
        
    # check if XGobj.statistic or final_statistic is weird or not
    for stat in XGobj.statistic:
        if(stat not in allowed_stat):
            error("Error[join_legs]: The input object contains illegal statistic. (0, 1, -1, or "+hybrid_symbol+" only)")
    
    if final_statistic_inp != None :
        for stat in final_statistic:
            if(stat not in allowed_stat):
                error("Error[join_legs]: The final statistic list contains illegal statistic. (0, 1, -1, or "+hybrid_symbol+" only)")
            
    if(XGobj.encoder == 'parity-preserving'):
        Obj = Obj.switch_encoder()
    if(XGobj.format == 'matrix'):
        Obj = Obj.switch_format()

    # The coding starts here ==============================================================================

    # prepare the information list of the groupings -------------------------------------------------------
    groups_info = [""] #[string,type]
    is_outside = True
    for char in string:
        current_pos = len(groups_info)-1
        if char != "(" and char != ")" and not is_outside:
            groups_info[current_pos] += char
        elif char != "(" and char != ")" and is_outside:
            groups_info += [char]
        elif char == "(" and is_outside :
            groups_info += [""]
            is_outside = False
        elif char == ")" and not is_outside :
            groups_info += [""]
            is_outside = True
        elif char == "(" and not is_outside:
            error("Error[join_legs]: No nested parenthesis!")
            
        elif char == ")" and is_outside:
            error("Error[join_legs]: Unmatched parenthesis!")
            
        #print("pending:",groups_info)
    for i,substring in enumerate(groups_info):
        if len(substring) == 0:
            groups_info[i] = ""
        elif len(substring) == 1:
            groups_info[i] = substring
    while groups_info.count("")>0:
        groups_info.remove("")

    n_groups = len(groups_info)
    if final_statistic_inp!=None and n_groups != len(final_statistic):
        error("Error[join_legs]: The number of final indices is inconsistent with the grouping!")

    # add entries about the dimensions and statistics
    index = 0
    for i,elem in enumerate(groups_info):
        shape = XGobj.shape[index:index+len(elem)]           # <================ change XGobj.shape to final_shape in split_legs
        statistic = XGobj.statistic[index:index+len(elem)]   # <======== change XGobj.statistic to final_statistic in split_legs
        groups_info[i] = [elem,shape,statistic]

        # check if is hybrid
        n_nonfermions = statistic.count(0)+statistic.count(hybrid_symbol)
        is_hybrid = (
            (n_nonfermions<len(statistic) and n_nonfermions>0)
            or (statistic.count(hybrid_symbol)>0)
            )
        if final_statistic_inp!=None and ((is_hybrid and final_statistic[i] != hybrid_symbol) or (not is_hybrid and final_statistic[i] == hybrid_symbol)):
            error("Error[join_legs]: The final statistic is inconsistent with object and the grouping!")
            
        index += len(elem)

    # initialize the final stats if necessary
    if final_statistic_inp == None :
        final_statistic = []
        for elem in groups_info:
            nbose = elem[2].count(0)+elem[2].count(hybrid_symbol)
            nfermi = elem[2].count(-1)+elem[2].count(1)
            if nbose>0 and nfermi>0:
                final_statistic += [hybrid_symbol]
            elif nbose>0 and nfermi==0:
                final_statistic += [0]
            elif nbose==0 and nfermi>0:
                final_statistic += [1]
        final_statistic = make_tuple(final_statistic)

    # prepare the reordered group info  -------------------------------------------------------------------

    reorderd_groups_info = []
    for [ind_list,shape,stat_list] in groups_info:
        new_ind_list = ""
        new_shape = []
        new_stat_list = []
        for i,stat in enumerate(stat_list):
            if(stat in fermi_type):
                new_ind_list   += ind_list[i]
                new_shape += [shape[i]]
                new_stat_list  += [stat_list[i]]
            else:
                new_ind_list   = ind_list[i]+new_ind_list
                new_shape = [shape[i]]+new_shape
                new_stat_list  = [stat_list[i]]+new_stat_list
        reorderd_groups_info += [ [new_ind_list,new_shape,new_stat_list] ]

    # add sign factor for merging mixed statistic  --------------------------------------------------------

    sgn_str  = ""
    sgn_list = []
    for j,[ind_list,shape,stat_list] in enumerate(groups_info):
        #print(stat_list," vs ",final_statistic[j])
        for i,stat in enumerate(stat_list):
            if stat == 1 and final_statistic[j]==-1: #<========== This is different from join_legs
                sgn = [ (-1)**param.gparity[k] for k in range(shape[i]) ]
                sgn_str += ind_list[i]
                sgn_list += [sgn]
        #print(ind_list,"---multiply--->",''.join(sgn_str))

    # input string for einsum -----------------------------------------------------------------------------
    str_einsum_input  = ""
    for group in groups_info:
        str_einsum_input += group[0]
    for char in sgn_str:
        str_einsum_input += ","+char

    # output string for einsum ----------------------------------------------------------------------------
    str_einsum_output = ""
    for group in reorderd_groups_info:
        str_einsum_output += group[0]

    # reordered statistic (unused in split_legs) ----------------------------------------------------------
    reordered_stat  = []
    for group in reorderd_groups_info:
        reordered_stat  += group[2]

    reordered_stat = tuple(reordered_stat)

    # You can copy the above for split_legs as well =======================================================

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #:::::      STEP 1 - REARRANGE BOSONIC INDICES TO THE RIGHT                                       :::::
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    #Obj.display("++++")
    #for sgn in sgn_list:
    #    print(sgn)

    einsum_input = tuple([str_einsum_input+"->"+str_einsum_output] + [Obj.data] + sgn_list )

    Obj.data = np.einsum(*einsum_input)
    Obj.statistic = reordered_stat

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #:::::      STEP 2 - JOIN THE FERMIONIC INDICES WITH CANONICAL ENCODER (DONT SWITCH)              :::::
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


    #Obj.display("----")
    
    # compute the new shape -------------------------------------------------------------------------------
    new_shape = []
    new_stat_list = []
    for j,[ind,shape,stat_list] in enumerate(reorderd_groups_info):
        boson_dim = 1
        fermi_dim = 1
        boson_count = 0
        fermi_count = 0
        for i,stat in enumerate(stat_list):
            if stat in bose_type:
                boson_count += 1
                boson_dim *= shape[i]
            else:
                fermi_count += 1
                fermi_dim *= shape[i]
        if(boson_count>0):
            new_shape += [boson_dim]
        if(fermi_count>0):
            new_shape += [fermi_dim]

        if(boson_count>0 and fermi_count>0):
            new_stat_list += [0,1] #the fermion is assign a +1 stat automatically for the hybrid case
        elif(boson_count>0 and fermi_count==0):
            new_stat_list += [0]
        elif(boson_count==0 and fermi_count>0):
            new_stat_list += [final_statistic[j]] #if it is just fermion, the final_statistic can be applied here.

    new_shape = tuple(new_shape)
    new_stat_list = tuple(new_stat_list)

    Obj.data = np.reshape(Obj.data,new_shape)
    Obj.statistic = new_stat_list

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #:::::      STEP 3 - JOIN ALL THE INDICES WITH PARITY-PRESERVING ENCODER                          :::::
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    if format == 'matrix' :
        Obj = Obj.switch_format()

    # compute the new shape -------------------------------------------------------------------------------
    new_shape = []
    for [ind,shape,stat_list] in reorderd_groups_info:
        dim = 1
        for subdim in shape:
            dim *= subdim
        new_shape += [dim]
    new_shape = tuple(new_shape)

    Obj = Obj.switch_encoder()  # <======================= SWITCH TO PARITY-PRESERVING ENCODER
    Obj.data = np.reshape(Obj.data,new_shape)
    Obj.statistic = final_statistic
    Obj = Obj.switch_encoder()  # <======================= SWITCH BACK TO CANONICAL ENCODER
    
    # revert to the original conditions ===================================================================

    if(XGobj.encoder == 'parity-preserving'):
        Obj = Obj.switch_encoder()

    if(objtype==sparse):
        Obj = sparse(Obj)

    return Obj

def split_legs(XGobj,string_inp,statistic,shape,format='standard'):

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # WORK IN THE FOLLOWING CONDITIONS      :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # (WILL CONVERT AUTOMATICALLY)          :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #   - canonical encoder                 :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    # the string is of the form aaa(bb)cccc(ddd)eeee
    # The indices will then be grouped together as one index

    final_statistic_inp = statistic
    final_shape_inp = shape

    string = string_inp
    if string.count(")") == 0 and string.count(")") == 0:
        nseparator = 0
        for separator in separator_list:
            nseparator += string.count(separator)
        if nseparator>0 :
            for separator in separator_list:
                string = string.replace(separator,")(")
            string = "("+string+")"


    final_statistic = list(final_statistic_inp)
    final_shape     = list(final_shape_inp)

    objtype = type(XGobj)
    Obj = XGobj.copy()
    if(objtype==sparse):
        Obj = dense(Obj)
    if(objtype not in [dense,sparse]):
        error("Error[split_legs]: Object type must only be dense or sparse!")
        
    # check if XGobj.statistic or final_statistic is weird or not
    for stat in XGobj.statistic:
        if(stat not in allowed_stat):
            error("Error[split_legs]: The input object contains illegal statistic. (0, 1, -1, or "+hybrid_symbol+" only)")
            
    for stat in final_statistic:
        if(stat not in allowed_stat):
            error("Error[split_legs]: The final statistic list contains illegal statistic. (0, 1, -1, or "+hybrid_symbol+" only)")
            
    if(XGobj.encoder == 'parity-preserving'):
        Obj = Obj.switch_encoder()

    # The coding starts here ==============================================================================

    # prepare the information list of the groupings -------------------------------------------------------
    groups_info = [""] #[string,type]
    is_outside = True
    for char in string:
        current_pos = len(groups_info)-1
        if char != "(" and char != ")" and not is_outside:
            groups_info[current_pos] += char
        elif char != "(" and char != ")" and is_outside:
            groups_info += [char]
        elif char == "(" and is_outside :
            groups_info += [""]
            is_outside = False
        elif char == ")" and not is_outside :
            groups_info += [""]
            is_outside = True
        elif char == "(" and not is_outside:
            error("Error[split_legs]: No nested parenthesis!")
            
        elif char == ")" and is_outside:
            error("Error[split_legs]: Unmatched parenthesis!")
            
    for i,substring in enumerate(groups_info):
        if len(substring) == 0:
            groups_info[i] = ""
        elif len(substring) == 1:
            groups_info[i] = substring
    while groups_info.count("")>0:
        groups_info.remove("")

    string_no_brackets = string
    while string_no_brackets.count("(")>0 or string_no_brackets.count(")")>0 :
        string_no_brackets = string_no_brackets.replace("(","")
        string_no_brackets = string_no_brackets.replace(")","")

    n_groups = len(string_no_brackets)
    if n_groups != len(final_statistic):
        error("Error[split_legs]: The number of final indices is inconsistent with the grouping!")
        
    # add entries about the dimensions and statistics
    index = 0
    for i,elem in enumerate(groups_info):
        shape = final_shape[index:index+len(elem)]           # <================ change XGobj.shape to final_shape in split_legs
        statistic = final_statistic[index:index+len(elem)]   # <======== change XGobj.statistic to final_statistic in split_legs
        groups_info[i] = [elem,shape,statistic]

        # check if is hybrid
        n_nonfermions = statistic.count(0)+statistic.count(hybrid_symbol)
        is_hybrid = (
            (n_nonfermions<len(statistic) and n_nonfermions>0)
            or (statistic.count(hybrid_symbol)>0)
            )
        if (is_hybrid and XGobj.statistic[i] != hybrid_symbol) or (not is_hybrid and XGobj.statistic[i] == hybrid_symbol):
            error("Error[split_legs]: The final statistic is inconsistent with object and the grouping!")
            
        index += len(elem)

    # prepare the reordered group info  -------------------------------------------------------------------

    reorderd_groups_info = []
    for [ind_list,shape,stat_list] in groups_info:
        new_ind_list = ""
        new_shape = []
        new_stat_list = []
        for i,stat in enumerate(stat_list):
            if(stat in fermi_type):
                new_ind_list   += ind_list[i]
                new_shape += [shape[i]]
                new_stat_list  += [stat_list[i]]
            else:
                new_ind_list   = ind_list[i]+new_ind_list
                new_shape = [shape[i]]+new_shape
                new_stat_list  = [stat_list[i]]+new_stat_list
        reorderd_groups_info += [ [new_ind_list,new_shape,new_stat_list] ]

    # add sign factor for merging mixed statistic  --------------------------------------------------------

    sgn_str  = ""
    sgn_list = []
    for j,[ind_list,shape,stat_list] in enumerate(groups_info):
        #print(stat_list," vs ",XGobj.statistic[j])
        for i,stat in enumerate(stat_list):
            if stat == 1 and XGobj.statistic[j]==-1: #<========== This is different from join_legs
                sgn = [ (-1)**param.gparity[k] for k in range(shape[i]) ]
                sgn_str += ind_list[i]
                sgn_list += [sgn]
        #print(ind_list,"<---multiply---",''.join(sgn_str))

    # input string for einsum -----------------------------------------------------------------------------
    str_einsum_output  = ""
    for group in groups_info:
        str_einsum_output += group[0]

    # output string for einsum ----------------------------------------------------------------------------
    str_einsum_input = ""
    for group in reorderd_groups_info:
        str_einsum_input += group[0]
    for char in sgn_str:
        str_einsum_input += ","+char

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #:::::      STEP 1 - SPLIT THE BOSONIC AND FERMIONIC INDICES WITH PARITY-PRESERVING ENCODER       :::::
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


    # compute the new shape -------------------------------------------------------------------------------
    new_shape = []
    new_stat_list = []
    for j,[ind,shape,stat_list] in enumerate(reorderd_groups_info):
        boson_dim = 1
        fermi_dim = 1
        boson_count = 0
        fermi_count = 0
        for i,stat in enumerate(stat_list):
            if stat in bose_type:
                boson_count += 1
                boson_dim *= shape[i]
            else:
                fermi_count += 1
                fermi_dim *= shape[i]
        if(boson_count>0):
            new_shape += [boson_dim]
        if(fermi_count>0):
            new_shape += [fermi_dim]

        if(boson_count>0 and fermi_count>0):
            new_stat_list += [0,1] #the fermion is assign a +1 stat automatically for the hybrid case
        elif(boson_count>0 and fermi_count==0):
            new_stat_list += [0]
        elif(boson_count==0 and fermi_count>0):
            new_stat_list += [XGobj.statistic[j]] #if it is just fermion, the XGobj.statistic can be applied here.

    new_shape = tuple(new_shape)
    new_stat_list = tuple(new_stat_list)

    Obj = Obj.switch_encoder()  # <======================= SWITCH TO PARITY-PRESERVING ENCODER
    Obj.data = np.reshape(Obj.data,new_shape)
    Obj.statistic = new_stat_list
    Obj = Obj.switch_encoder()  # <======================= SWITCH BACK TO CANONICAL ENCODER
    
    if XGobj.format == 'matrix':
        Obj = Obj.switch_format()

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #:::::      STEP 2 - SPLIT THE FERMIONIC INDICES WITH CANONICAL ENCODER (DONT SWITCH)             :::::
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    # compute the new shape -------------------------------------------------------------------------------
    new_shape = []
    new_stat_list = []
    for [ind,shape,stat_list] in reorderd_groups_info:
        new_shape += shape
        new_stat_list += stat_list

    new_shape = tuple(new_shape)
    new_stat_list = tuple(new_stat_list)

    Obj.data = np.reshape(Obj.data,new_shape)
    Obj.statistic = new_stat_list

    #Obj.display("----")
    
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #:::::      STEP 3 - REARRANGE BOSONIC INDICES TO THE RIGHT                                       :::::
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    einsum_input = tuple([str_einsum_input+"->"+str_einsum_output] + [Obj.data] + sgn_list )

    Obj.data = np.einsum(*einsum_input)
    Obj.statistic = final_statistic_inp

    #Obj.display("++++")
    #for sgn in sgn_list:
    #    print(sgn)

    # revert to the original conditions ===================================================================

    if(XGobj.encoder == 'parity-preserving'):
        Obj = Obj.switch_encoder()

    if format == 'matrix' :
        Obj = Obj.switch_format()

    if(objtype==sparse):
        Obj = sparse(Obj)

    return Obj

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
##                      SVDs                      ##
####################################################

def BlockSVD(Obj):
    
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

    UE, ΛE, VE = np.linalg.svd(ME, full_matrices=False, compute_uv=True)
    UO, ΛO, VO = np.linalg.svd(MO, full_matrices=False, compute_uv=True)

    d = max(len(ΛE),len(ΛO))
    d = int(2**math.ceil(np.log2(d)))

    def padding(Ux, Λx, Vx, padding_dimension):
        Ux = np.pad(Ux,((0,0),(0,padding_dimension)),'constant',constant_values=((0,0),(0,0)))
        Λx = np.diag(np.pad(Λx,(0,padding_dimension),'constant',constant_values=(0,0)       ))
        Vx = np.pad(Vx,((0,padding_dimension),(0,0)),'constant',constant_values=((0,0),(0,0)))
        return Ux, Λx, Vx

    UE, ΛE, VE = padding(UE, ΛE, VE, d)
    UO, ΛO, VO = padding(UO, ΛO, VO, d)

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

def SVD(XGobj,string):

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # WORK IN THE FOLLOWING CONDITIONS      :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # (WILL CONVERT AUTOMATICALLY)          :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #   - parity-preserving encoder         :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #   - matrix format                     :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    # the string is of the form aaaa|bbb

    objtype = type(XGobj)
    Obj = XGobj.copy()
    if(objtype==sparse):
        Obj = dense(Obj)
    if(objtype not in [dense,sparse]):
        error("Error[SVD]: Object type must only be dense or sparse!")
        
    # check if XGobj.statistic or final_statistic is weird or not
    for stat in XGobj.statistic:
        if(stat not in allowed_stat):
            error("Error[SVD]: The input object contains illegal statistic. (0, 1, -1, or "+hybrid_symbol+" only)")
            
    if(XGobj.encoder == 'canonical'):
        Obj = Obj.switch_encoder()
    if(XGobj.format == 'standard'):
        Obj = Obj.switch_format()

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

        error("Error[SVD]: The input string must contain one and only one partition "+partition_string+" in it.")
        

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #:::::      STEP 1 - JOIN LEGS BAESD ON THE GROUPINGS                                             :::::
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

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
    Obj0 = Obj.copy()
    Obj = Obj.join_legs(join_legs_string_input,(get_stat(stats_left,-1),get_stat(stats_right,+1)),"matrix")
    
    Test = Obj.split_legs(join_legs_string_input,Obj0.statistic,Obj0.shape,"matrix")
    #Obj0.display("Obj0")
    #Obj.display("Obj")
    #Test.display("Test")
    #exit()
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #:::::      STEP 2 - BLOCK SVD (MAKE SURE IT'S PARITY-PRESERVING!)                                :::::
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    U, Λ, V = BlockSVD(Obj.data)

    #dense(np.einsum('ij,jk,kl->il',U, Λ, V)-Obj.data).display("zero test")

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #:::::      STEP 3 - RECONSTRUCT U, Λ, and V AS GRASSMANN TENSORS                                 :::::
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    #the first way is to form the tensor first, then split
    
    U = dense(U,encoder="parity-preserving")
    Λ = dense(Λ,encoder="parity-preserving")
    V = dense(V,encoder="parity-preserving")

    U.statistic = (Obj.statistic[0],1)
    Λ.statistic = (-1,1)
    V.statistic = (-1,Obj.statistic[1])
    dΛ = Λ.shape[0]

    U.format = "matrix"
    Λ.format = "matrix"
    V.format = "matrix"

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
    Ustats = tuple(Ustats + [+1])
    Vstats = tuple([-1] + Vstats)
    Ushape = tuple(Ushape + [dΛ])
    Vshape = tuple([dΛ] + Vshape)

    #U.display("before spliting legs")

    U = U.split_legs(Uind,Ustats,Ushape,'matrix')
    V = V.split_legs(Vind,Vstats,Vshape,'matrix')

    #U.display("after spliting legs")

    # revert to the original conditions ===================================================================

    if(XGobj.format == 'standard'):
        U = U.switch_format()
        Λ = Λ.switch_format()
        V = V.switch_format()

    if(XGobj.encoder == 'canonical'):
        U = U.switch_encoder()
        Λ = Λ.switch_encoder()
        V = V.switch_encoder()

    if(objtype==sparse):
        U = sparse(U)
        Λ = sparse(Λ)
        V = sparse(V)

    return U, Λ, V

####################################################
##                   Conjugation                  ##
####################################################

def hconjugate(Obj,string_input):

    # string is of the form <indices1|indices2>
    string = string_input
    for separator in separator_list:
        string = string.replace(separator,"|")
    if string.count("|") > 1:
        error("Error[hconjugate]: The number of separator can only be 0 or 1 only.")

    if Obj.statistic.count(hybrid_symbol)>0:
        error("Error[hconjugate]: Split the hybrid indices before performing the conjugate.")

    # get stat_left/right and shape_left/right
    n = 0
    if string.count("|") == 0:
        stats = Obj.statistic
        shape = Obj.shape
    elif string.count("|") == 1:
        n = string.index("|")
        stats_left  = Obj.statistic[:n]
        stats_right = Obj.statistic[n:]
        shape_left  = Obj.shape[:n]
        shape_right = Obj.shape[n:]

    hObj = Obj.copy()

    this_type = type(Obj)
    this_format = Obj.format
    this_encoder = Obj.encoder

    if this_type == sparse:
        hObj = dense(hObj)
    if this_encoder == 'parity-preserving':
        hObj = hObj.switch_encoder()

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #:::::      STEP 1 - JOIN LEGS BAESD ON THE GROUPINGS                                             :::::
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

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

    shape_left  = hObj.shape[:n_left]
    stats_left  = hObj.statistic[:n_left]
    shape_right = hObj.shape[n_left:]
    stats_right = hObj.statistic[n_left:]

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

    #hObj.switch_encoder().display("before joining legs")
    #hObj0pj = hObj.copy()
    hObj = hObj.join_legs(join_legs_string_input,(get_stat(stats_left,-1),get_stat(stats_right,+1)),format='matrix')
    #hObj0 = hObj.copy()
    
    #hObj.switch_encoder().display("after joining legs")

    #hObj.switch_encoder().display("after joining legs")

    M = hObj.data
    hM = np.conjugate(np.einsum('ij->ji',M))

    #dense(M).display("raw form")

    #dense(np.einsum('ij,jk->ik',M,hM)).display("UcU (raw form)")
    #dense(np.einsum('ij,jk->ik',hM,M)).display("cUU (raw form)")

    new_stats_joined = []
    for stat in hObj.statistic :
        if stat in fermi_type :
            new_stats_joined = [-stat]+new_stats_joined
        else:
            new_stats_joined = [stat]+new_stats_joined
    hObj.data = hM
    hObj.statistic = make_tuple(new_stats_joined)

    new_shape = make_tuple(make_list(shape_right)+make_list(shape_left))
    new_stats = make_list(stats_right)+make_list(stats_left)
    for i in range(Obj.ndim):
        if new_stats[i] in fermi_type:
            new_stats[i] *= -1
    new_stats = make_tuple(new_stats)

    [left_string,right_string] = string.split("|")
    join_legs_string_output = "("+ right_string+")("+left_string+")"

    hObj = hObj.split_legs(join_legs_string_output,new_stats,new_shape,format=this_format)

    #hObj0 = hObj0.split_legs(join_legs_string_input,hObj0pj.statistic,hObj0pj.shape)

    #hObj0.switch_encoder().display("after spliting legs")

    #print(new_stats,hObj0pj.statistic)
    #print(new_shape,hObj0pj.shape)
    #exit()
    return hObj
