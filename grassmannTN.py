import numpy as np
import math
from grassmannTN import param
import sparse as sp
import opt_einsum as oe
import time

hybrid_symbol = "*"
svd_partition = ("|",":",";",","," ")

skip_parity_blocking_check = True
allowed_stat = (0,1,-1,hybrid_symbol)
fermi_type = (1,-1)
bose_type = (0,hybrid_symbol)
encoder_type = ("canonical","parity-preserving")
format_type = ("standard","matrix")

char_list = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z","A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]

'''


        USAGE ADVICES
        - Split the legs before using einsum!
          Reason: The ordering of the indices is obscure in the joined form.
                  The missing sign factor is highly likely to happen!
        - Manually reverse the splitted leg's ordering with einsum after the conjugatation.
          Reason: The ordering of legs after they are joined is kept frozen until split.

        - Conjugate and split/join operations are not commutative!

'''

####################################################
##             Densed Grassmann Array             ##
####################################################

class DGarray:
    def __init__(self, data=None, bosonic = False, encoder = "canonical", format = "standard", statistic=None):
    
        
        #copy DGarray properties
        self.data = None
        self.statistic = None
        self.format = format
        self.encoder = encoder
    
        default = True
    
        if(encoder not in encoder_type):
            print("Error[DGarray]: Unknown encoder.")
            exit()
        if(format not in format_type):
            print("Error[DGarray]: Unknown format.")
            exit()

        if(type(data)==np.array or type(data)==np.ndarray or type(data)==list):
            self.data = np.array(data)
            self.format = 'standard'
            if bosonic:
                self.statistic = tuple([ 0 for i in range(len(self.shape))])
            else:
                self.statistic = tuple([ 1 for i in range(len(self.shape))])
            default = False
        elif(type(data)==SGarray):
            #copy DGarray properties
            self.data = data.data.todense()
            self.statistic = data.statistic
            self.format = data.format
            self.encoder = data.encoder
            default = False
        elif(type(data)==DGarray):
            #copy DGarray properties
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
            print("Error[DGarray]: Invalid initialized data.")
            exit()
        
        if statistic != None:
            self.statistic = statistic
            
        if not default:
            for i,dim in enumerate(self.data.shape):
                if self.statistic[i] in fermi_type and dim != int(2**math.floor(np.log2(dim))):
                    print("Caution[DGarray]: Some of the fermionic tensor shapes are not a power of two.")
                
    
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

    def display(self):
        print()
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
                print(coords,element.item(),)
        print()

    def info(self):
        print()
        print("  array type: dense")
        print("       shape:",self.shape)
        print("        size:",self.size)
        print("   statistic:",self.statistic)
        print("      format:",self.format)
        print("     encoder:",self.encoder)
        print()

    def copy(self):
        #copy DGarray properties
        ret = DGarray()
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
            print("Error[DGarray.+]: Inconsistent object properties")
            exit()
        ret = self.copy()
        ret.data = ret.data+other.data
        return ret
        
    def __sub__(self, other):
        if(self.shape!=other.shape
            or self.statistic!=other.statistic
             or self.format!=other.format
              or self.encoder!=other.encoder):
            print("Error[DGarray.-]: Inconsistent object properties")
            exit()
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
            ret = ret.switch_encoder()
        iterator = np.nditer(ret.data, flags=['multi_index'])
        for element in iterator:
            coords = iterator.multi_index
            sgn_value = 1
            for i,ind in enumerate(coords):
                if(ret.statistic[i]==-1):
                    sgn_value *= param.sgn[ind]
                if(ret.statistic[i]==hybrid_symbol):
                    print("Error[switch_format]: Cannot switch format with a hybrid index.")
                    print("                      Split them into bosonic and fermionic ones first!")
                    exit()
            ret[coords] *= sgn_value

        if(ret.format=='standard'):
            ret.format = 'matrix'
        elif(ret.format=='matrix'):
            ret.format = 'standard'
        else:
            print("Error[switch_format]: unknown format")
            exit()

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
        
####################################################
##            Sparse Grassmann arrays             ##
####################################################

class SGarray:
    def __init__(self, data=None, bosonic = False, encoder = "canonical", format = "standard", statistic = None):
    
        #copy SGarray properties
        self.data = None
        self.statistic = None
        self.format = format
        self.encoder = encoder

        default = True
    
        if(encoder not in encoder_type):
            print("Error[SGarray]: Unknown encoder.")
            exit()
        if(format not in format_type):
            print("Error[SGarray]: Unknown format.")
            exit()

        
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
        elif(type(data)==DGarray):
            #copy SGarray properties
            self.data  = sp.COO.from_numpy(data.data)
            self.statistic = data.statistic
            self.format = data.format
            self.encoder = data.encoder
            default = False
        elif(type(data)==SGarray):
            #copy SGarray properties
            self.data = data.data.copy()
            self.statistic = data.statistic
            self.format = data.format
            self.encoder = data.encoder
            default = False
        elif data==None:
            "nothing to see here"
        else:
            print("Error[SGarray]: Invalid initialized data")
            exit()
        
        if statistic != None:
            self.statistic = statistic
        
        if not default:
            for i,dim in enumerate(self.data.shape):
                if self.statistic[i] in fermi_type and dim != int(2**math.floor(np.log2(dim))):
                    print("Caution[SGarray]: Some of the fermionic tensor shapes are not a power of two.")
                
    
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

    def display(self):
        print()
        print("  array type: sparse")
        print("       shape:",self.shape)
        print("        size:",self.size)
        print("   statistic:",self.statistic)
        print("      format:",self.format)
        print("     encoder:",self.encoder)
        print("   non-zeros:",self.nnz)
        print("     entries:")
        print()

        C = self.coords
        V = self.value
        for elem in range(self.nnz):
            print(C[elem],V[elem])

    def info(self):
        print()
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
        #copy SGarray properties
        ret = SGarray()
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
            print("Error[SGarray.+]: Inconsistent object properties")
            exit()
        ret = self.copy()
        ret.data = ret.data+other.data
        return ret
        
    def __sub__(self, other):
        if(self.shape!=other.shape
            or self.statistic!=other.statistic
             or self.format!=other.format
              or self.encoder!=other.encoder):
            print("Error[SGarray.-]: Inconsistent object properties")
            exit()
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
                    print("Error[switch_format]: Cannot switch format with a hybrid index.")
                    print("                      Split them into bosonic and fermionic ones first!")
                    exit()
            ret.data.data[elem] *= sgn_value

        if(ret.format=='standard'):
            ret.format = 'matrix'
        elif(ret.format=='matrix'):
            ret.format = 'standard'
        else:
            print("Error[switch_format]: unknown format")
            exit()

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
            print("Error[absolute_parity.get_commutative_elements]: Inconsistent array sizes!")
            exit()
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
            print("Error[relative_parity_int.get_noncommutative_elements]: Inconsistent array sizes!")
            exit()
        else:
            noncommutative_elements = [x for i,x in enumerate(permutation) if (-1)**individual_parity[i]==-1]
            return noncommutative_elements

    individual_parity2 = permute_c(permutation1, permutation2, individual_parity1)
    
    noncommutative_elements1 = get_noncommutative_elements(permutation1,individual_parity1)
    noncommutative_elements2 = get_noncommutative_elements(permutation2,individual_parity2)
    
    if(sorted(noncommutative_elements1) != sorted(noncommutative_elements2)):
        print("Error[relative_parity_int]: Inconsistent grassmann-odd indices!")
        exit()
    
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
        print("Error[relative_parity]: The number of input list and parity list are not consistent!")
        exit()
        
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

####################################################
##                     Einsums                    ##
####################################################

def assigning_sgn(sgn_inp,string_for_sign_computation,format):
    
    print(string_for_sign_computation)
    
    # sgn_inp is a blank sign matrix
    # string_for_sign_computation is used for sign computation (duh)
    # summed_list shows which symbols are always the same

    sgn = sgn_inp.copy()
    iterator = np.nditer(sgn, flags=['multi_index'])
    for element in iterator:
        coords = iterator.multi_index
        if(format=="standard"):

            #----- to do ----------------------------------------#
            # Properly compute the individual parity             #
            #----------------------------------------------------#
            
            # compute individual parity
            # edit this part
            parity_from_indices = [param.gparity[i] for i in coords]
            
            sgn_value = relative_parity(string_for_sign_computation,parity_from_indices)
            #this is the sign from basic permutation
            
            #print(string_for_sign_computation,parity_from_indices," ",sgn_value)
            sgn[coords] = sgn_value
        else:
            sgn[coords] = 1
    return sgn

# object conditions checked
# Unified version
def einsum(*args):

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # WORK IN THE FOLLOWING CONDITIONS      :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # (WILL CONVERT AUTOMATICALLY)          :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #   - canonical encoder                 :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    string = args[0]

    string_input = ""
    string_output = ""
    if(string.count("->")==1):
        [string_input,string_output] = list(string.split("->"))
    else:
        string_input = string

    string_list = list(string_input.split(","))
    number_of_objects = len(string_list)
    type_list = [ type(obj) for obj in args[1:1+number_of_objects]]
    encoder_list = [ obj.encoder for obj in args[1:1+number_of_objects]]
    format_list = [ obj.format for obj in args[1:1+number_of_objects]]
    obj_list = [ obj for obj in args[1:1+number_of_objects]]

    this_type = type_list[0]
    for i in range(1,number_of_objects):
        if type_list[i]!=this_type :
            print("Error[einsum]: Objects must be of the same type!")
            exit()

    if(this_type!=DGarray and this_type!=SGarray):
        print("Error[einsum]: Objects must be either DGarray or SGarray!")
        exit()

    this_encoder = encoder_list[0]
    for i in range(1,number_of_objects):
        if encoder_list[i]!=this_encoder :
            print("Error[einsum]: Objects must be under the same encoder!")
            exit()

    this_format = format_list[0]
    for i in range(1,number_of_objects):
        if format_list[i]!=this_format :
            print("Error[einsum]: The format of all Objects must be the same!")
            exit()

    #just to be sure, convert everything to the canonical encoder
    for i,obj in enumerate(obj_list):
        if obj.encoder == 'parity-preserving':
            obj_list[i] = obj.switch_encoder()

    newargs = tuple([args[0]]+obj_list)

    if(this_type==DGarray):
        ret = deinsum(*newargs)
    else:
        ret = seinsum(*newargs)

    if(this_encoder=='parity-preserving'):
        ret = ret.switch_encoder()

    return ret

# Dense version
def deinsum(*args):
    #----------------------------------------------------#
    # The same usage as np.einsum                        #
    # but with sign factor applied.                      #
    # Fermionic indices CAN be summed here.              #
    #----------------------------------------------------#
    
    #----------------------------------------------------#
    # Remove basic errors                                #
    # 1) must contain only one or no instances of '->'.  #
    # 2) the result must not contain ','.                #
    # 3) statistics are not 0, 1, or -1.                 #
    # 4) indices and statistic must be consistent.       #
    #----------------------------------------------------#
    
    string = args[0]
    # detecting error 1)
    if(string.count("->")>1):
        print("Error[deinsum]: The string contains more than one instances of '->'!")
        exit()

    string_input = ""
    string_output = ""
    if(string.count("->")==1):
        [string_input,string_output] = list(string.split("->"))
    else:
        string_input = string
    
    # detecting error 2)
    if(string_output.count(",")!=0):
        print("Error[deinsum]: The string_output contains ','.")
        exit()
    
    string_list = list(string_input.split(","))
    number_of_objects = len(string_list)


    DGarray_list = [ DGarray(obj) for obj in args[1:1+number_of_objects]]

    this_format = DGarray_list[0].format
    
    def return_list(stat):
        if stat==1 or stat==-1 or stat==hybrid_symbol or len(stat)==1:
            if(np.isscalar(stat)):
                return [stat]
            else:
                return list(stat)
        else:
            return list(stat)
    statistic_list = tuple(sum([ return_list(obj.statistic) for obj in DGarray_list ],[]))
    
    # detecting error 3)
    for stat in statistic_list:
        if(stat not in allowed_stat):
            print("Error[deinsum]: Statistics must be 0, +1, -1, or "+hybrid_symbol+".")
            exit()
    
    # detecting error 4)
    joined_string_input = list(string_input.replace(",",""))
    ind_stat_dict = [ [a,b] for a,b in zip(joined_string_input,statistic_list)]
    for i, element1 in enumerate(ind_stat_dict):
        for j in range(i+1,len(ind_stat_dict)):
            element2 = ind_stat_dict[j]
            if(
                #indices are the same
                element1[0]==element2[0] and
                (
                    #one stat is 0 but the other is not (bosonic case)
                    (element1[1]== 0 and element2[1]!= 0)
                    or
                    #one stat is +1 but the other stat is not -1
                    (element1[1]==1 and element2[1]!=-1)
                    or
                    #one stat is -1 but the other stat is not +1
                    (element1[1]==-1 and element2[1]!=1)
                    or
                    #one stat is hybrid but the other stat is not hybrid
                    (element1[1]==hybrid_symbol and element2[1]!=hybrid_symbol)
                    )
                ):
                print("Error[deinsum]: Inconsistent index and statistic.")
                exit()
    
    ind_stat_dict = {a:b for a,b in ind_stat_dict}
    output_statistic_list = tuple([ ind_stat_dict[ind] for ind in string_output ])
    
        
    #Input = "aaaaa,aaa,aaaaa->bbbb",A1,A2,A3,[i,j,k,..]
    #string = "aaaaa,aaa,aaaaa->bbbb"
    #string_input = "aaaaa,aaa,aaaaa"
    #joined_string_input = ["a","a",...,"a"]
    #string_output = "bbbb"
    #string_list = [aaaaa,aaa,aaaaa]
    #DGarray_list = [A1,A2,A3]
    #statistic_list = [i,j,k,..]
    
    #----------------------------------------------------#
    # Replace the conjugated indices by another char.    #
    # Add the pair [ind,ind-bar] at the end of           #
    # the output.                                        #
    #----------------------------------------------------#
    
    
    #summed indices is always paired by the conditions above
    #first add "-" in the conjugated variable
        
    replaced_joined_string_input = joined_string_input.copy()
    summed_list = []
    sign_sum_factor_string = []
    for i,elem in enumerate(replaced_joined_string_input):
        if(statistic_list[i]==-1 and replaced_joined_string_input.count(elem)==2):
            for char in char_list:
                if(char not in replaced_joined_string_input):
                    replaced_joined_string_input[i]=char
                    break
            summed_list += [(elem,replaced_joined_string_input[i])]
            sign_sum_factor_string += [elem]
    sign_sum_factor_string = ''.join(sign_sum_factor_string)

    replaced_input = replaced_joined_string_input.copy()
    nitems = len(string_list)
    inserted_loc = 0
    for i in range(nitems):
        replaced_input.insert(inserted_loc+len(string_list[i]),",")
        inserted_loc += 1 + len(string_list[i])
    replaced_input = replaced_input[:-1]
    replaced_input = ''.join(replaced_input)
    
    replaced_output = string_output
    for pair in summed_list:
        replaced_output += pair[0] + pair[1]
        
    replaced_string = '->'.join([replaced_input,replaced_output])
    
    #get shape list
    shape_list = []
    for elem in DGarray_list:
        shape_list += list(elem.shape)
    
    sgn_sum_shape = [ size for i,size in enumerate(shape_list) if (
                                                statistic_list[i]==-1
                                                and joined_string_input.count(joined_string_input[i])==2
                                                )]

    def get_Gindex_string(string, statistic_list):
        """
        return the string indices for DGeinsum
        """
        [string_input,string_output] = list(string.split("->"))
        string_list = list(string_input.split(","))
        join_string = "".join(string_list)
        if(len(join_string)!=len(statistic_list)):
            print("the number of input list and parity list are not consistent!")
            exit()
        #remove the summed indices
        def remove_bosonic(list1, list2):
            new_list1 = []
            for i, val in enumerate(list1):
                if list2[i] in fermi_type:
                    new_list1.append(val)
            return new_list1
        join_string_list = remove_bosonic(list(join_string), statistic_list)
        return ''.join(join_string_list)
        
    # get the string for einsum input
    sign_factor_string = get_Gindex_string(replaced_string,statistic_list)
    #remove the 'newly added' indices
    removed_list = [ sign_factor_string.index(pair[1]) for pair in summed_list]
    for pair in summed_list:
        sign_factor_string = sign_factor_string.replace(pair[1],"")

    delta_string = ''.join([ ","+x+xbar for (x,xbar) in summed_list])
    
    def get_Gindex_shape(DGarray_list, statistic_list):
        """
        return shape of (permutation) sign factor tensor
        """
        DGarray_number = len(DGarray_list)
        shape_list = []
        for obj in DGarray_list:
            shape_list += list(obj.shape)
        Gindex_shape = [ x for i,x in enumerate(shape_list) if statistic_list[i] in fermi_type]
        return Gindex_shape

    # prepare the (permutation) sign factor tensor
    sgn_shape = get_Gindex_shape(DGarray_list, statistic_list)
    sgn_shape = [ x for i,x in enumerate(sgn_shape) if i not in removed_list ]
    sgn = np.zeros(sgn_shape,dtype=int)
    
    if(len(sgn_shape)>0 and len(sgn_sum_shape)>0):
        string_input_with_sgn = ",".join([replaced_input,sign_factor_string,sign_sum_factor_string])
    elif(len(sgn_shape)>0 and len(sgn_sum_shape)==0):
        string_input_with_sgn = ",".join([replaced_input,sign_factor_string])
    elif(len(sgn_shape)==0 and len(sgn_sum_shape)>0):
        string_input_with_sgn = ",".join([replaced_input,sign_sum_factor_string])
    else:
        string_input_with_sgn = replaced_input

    string_input_with_sgn += delta_string
    string_with_sgn = "->".join([string_input_with_sgn,string_output])

    def remove_bosonic_right(string):
        [string_input,string_output] = list(string.split("->"))
        string_input_list = list(string_input)
        string_output_list = list(string_output)
        new_string_output = []
        for val in string_output_list:
            if val in string_input_list:
                new_string_output += [val]
        return '->'.join([string_input,''.join(new_string_output)])
    string_for_sign_computation = "->".join([sign_factor_string,replaced_output])
    string_for_sign_computation = remove_bosonic_right(string_for_sign_computation)
    
    # ---------------------------- THIS IS TIME CONSUMING ----------------------------
    # assign the value to the sign factor tensor
    sgn = assigning_sgn(sgn,string_for_sign_computation,format)
    # --------------------------------------------------------------------------------

    # prepare the (sum) sign factor tensor
    sgn_sum = np.zeros(sgn_sum_shape,dtype=int)
    iterator = np.nditer(sgn_sum, flags=['multi_index'])
    for element in iterator:
        coords = iterator.multi_index
        sgn_value = 1
        for i in coords:
            sgn_value *= param.sgn[i]
        sgn_sum[coords] = sgn_value
    
    #also add the kronecker delta to sum the fermions
    delta_list = []
    for size in sgn_sum_shape:
        delta_list += [np.identity(size,dtype=int)]
    
    if(len(sgn_shape)>0 and len(sgn_sum_shape)>0):
        DGarray_list_with_sgn = DGarray_list + [sgn,sgn_sum] + delta_list
    elif(len(sgn_shape)>0 and len(sgn_sum_shape)==0):
        DGarray_list_with_sgn = DGarray_list + [sgn] + delta_list
    elif(len(sgn_shape)==0 and len(sgn_sum_shape)>0):
        DGarray_list_with_sgn = DGarray_list + [sgn_sum] + delta_list
    else:
        DGarray_list_with_sgn = DGarray_list + delta_list

    if(string_output==""):
        string_with_sgn = string_with_sgn.replace("->","")

    #X = DGarray(np.array([np.einsum(*tuple([string_with_sgn]+DGarray_list_with_sgn))]))
    X = DGarray(np.einsum(*tuple([string_with_sgn]+DGarray_list_with_sgn)))
    X.statistic = output_statistic_list
    X.encoder = DGarray_list[0].encoder
    X.format = DGarray_list[0].format

    return X

# Sparse version
def seinsum(*args):
    #----------------------------------------------------#
    # The same usage as np.einsum                        #
    # but with sign factor applied.                      #
    # Fermionic indices CAN be summed here.              #
    #----------------------------------------------------#
    
    #----------------------------------------------------#
    # Remove basic errors                                #
    # 1) must contain only one or no instances of '->'.  #
    # 2) the result must not contain ','.                #
    # 3) statistics are not 0, 1, or -1.                 #
    # 4) indices and statistic must be consistent.       #
    #----------------------------------------------------#
    
    string = args[0]
    # detecting error 1)
    if(string.count("->")>1):
        print("Error[seinsum]: The string contains more than one instances of '->'!")
        exit()

    string_input = ""
    string_output = ""
    if(string.count("->")==1):
        [string_input,string_output] = list(string.split("->"))
    else:
        string_input = string

    # detecting error 2)
    if(string_output.count(",")!=0):
        print("Error[seinsum]: The string_output contains ','.")
        exit()
    
    string_list = list(string_input.split(","))
    number_of_objects = len(string_list)


    SGarray_list = [ SGarray(obj) for obj in args[1:1+number_of_objects]]

    this_format = SGarray_list[0].format

    def return_list(stat):
        if stat==1 or stat==-1 or stat==hybrid_symbol or len(stat)==1:
            if(np.isscalar(stat)):
                return [stat]
            else:
                return list(stat)
        else:
            return list(stat)
    statistic_list = tuple(sum([ return_list(obj.statistic) for obj in SGarray_list ],[]))
    
    # detecting error 3)
    for stat in statistic_list:
        if(stat not in allowed_stat):
            print("Error[seinsum]: Statistics must be 0, +1, -1, or"+hybrid_symbol+".")
            exit()
    
    # detecting error 4)
    joined_string_input = list(string_input.replace(",",""))
    ind_stat_dict = [ [a,b] for a,b in zip(joined_string_input,statistic_list)]
    for i, element1 in enumerate(ind_stat_dict):
        for j in range(i+1,len(ind_stat_dict)):
            element2 = ind_stat_dict[j]
            if(
                #indices are the same
                element1[0]==element2[0] and
                (
                    #one stat is 0 but the other is not (bosonic case)
                    (element1[1]== 0 and element2[1]!= 0)
                    or
                    #one stat is +1 but the other stat is not -1
                    (element1[1]==1 and element2[1]!=-1)
                    or
                    #one stat is -1 but the other stat is not +1
                    (element1[1]==-1 and element2[1]!=1)
                    or
                    #one stat is hybrid but the other stat is not hybrid
                    (element1[1]==hybrid_symbol and element2[1]!=hybrid_symbol)
                    )
                ):
                print("Error[seinsum]: Inconsistent index and statistic.")
                exit()
    
    ind_stat_dict = {a:b for a,b in ind_stat_dict}
    output_statistic_list = tuple([ ind_stat_dict[ind] for ind in string_output ])
    
        
    #Input = "aaaaa,aaa,aaaaa->bbbb",A1,A2,A3,[i,j,k,..]
    #string = "aaaaa,aaa,aaaaa->bbbb"
    #string_input = "aaaaa,aaa,aaaaa"
    #joined_string_input = ["a","a",...,"a"]
    #string_output = "bbbb"
    #string_list = [aaaaa,aaa,aaaaa]
    #SGarray_list = [A1,A2,A3]
    #statistic_list = [i,j,k,..]
    
    #----------------------------------------------------#
    # Replace the conjugated indices by another char.    #
    # Add the pair [ind,ind-bar] at the end of           #
    # the output.                                        #
    #----------------------------------------------------#
    
    
    #summed indices is always paired by the conditions above
    #first add "-" in the conjugated variable
    
    replaced_joined_string_input = joined_string_input.copy()
    summed_list = []
    sign_sum_factor_string = []
    for i,elem in enumerate(replaced_joined_string_input):
        if(statistic_list[i]==-1 and replaced_joined_string_input.count(elem)==2):
            for char in char_list:
                if(char not in replaced_joined_string_input):
                    replaced_joined_string_input[i]=char
                    break
            summed_list += [(elem,replaced_joined_string_input[i])]
            sign_sum_factor_string += [elem]
    sign_sum_factor_string = ''.join(sign_sum_factor_string)

    replaced_input = replaced_joined_string_input.copy()
    nitems = len(string_list)
    inserted_loc = 0
    for i in range(nitems):
        replaced_input.insert(inserted_loc+len(string_list[i]),",")
        inserted_loc += 1 + len(string_list[i])
    replaced_input = replaced_input[:-1]
    replaced_input = ''.join(replaced_input)
    
    replaced_output = string_output
    for pair in summed_list:
        replaced_output += pair[0] + pair[1]
        
    replaced_string = '->'.join([replaced_input,replaced_output])
    
    #get shape list
    shape_list = []
    for elem in SGarray_list:
        shape_list += list(elem.shape)
    
    sgn_sum_shape = [ size for i,size in enumerate(shape_list) if (
                                                statistic_list[i]==-1
                                                and joined_string_input.count(joined_string_input[i])==2
                                                )]

    def get_Gindex_string(string, statistic_list):
        """
        return the string indices for DGeinsum
        """
        [string_input,string_output] = list(string.split("->"))
        string_list = list(string_input.split(","))
        join_string = "".join(string_list)
        if(len(join_string)!=len(statistic_list)):
            print("the number of input list and parity list are not consistent!")
            exit()
        #remove the summed indices
        def remove_bosonic(list1, list2):
            new_list1 = []
            for i, val in enumerate(list1):
                if list2[i] in fermi_type:
                    new_list1.append(val)
            return new_list1
        join_string_list = remove_bosonic(list(join_string), statistic_list)
        return ''.join(join_string_list)
        
    # get the string for einsum input
    sign_factor_string = get_Gindex_string(replaced_string,statistic_list)
    #remove the 'newly added' indices
    removed_list = [ sign_factor_string.index(pair[1]) for pair in summed_list]
    for pair in summed_list:
        sign_factor_string = sign_factor_string.replace(pair[1],"")

    delta_string = ''.join([ ","+x+xbar for (x,xbar) in summed_list])
    
    def get_Gindex_shape(DGarray_list, statistic_list):
        """
        return shape of (permutation) sign factor tensor
        """
        DGarray_number = len(DGarray_list)
        shape_list = []
        for obj in DGarray_list:
            shape_list += list(obj.shape)
        Gindex_shape = [ x for i,x in enumerate(shape_list) if statistic_list[i] in fermi_type]
        return Gindex_shape

    # prepare the (permutation) sign factor tensor
    sgn_shape = get_Gindex_shape(SGarray_list, statistic_list)
    sgn_shape = [ x for i,x in enumerate(sgn_shape) if i not in removed_list ]
    sgn = np.zeros(sgn_shape,dtype=int)
    
    if(len(sgn_shape)>0 and len(sgn_sum_shape)>0):
        string_input_with_sgn = ",".join([replaced_input,sign_factor_string,sign_sum_factor_string])
    elif(len(sgn_shape)>0 and len(sgn_sum_shape)==0):
        string_input_with_sgn = ",".join([replaced_input,sign_factor_string])
    elif(len(sgn_shape)==0 and len(sgn_sum_shape)>0):
        string_input_with_sgn = ",".join([replaced_input,sign_sum_factor_string])
    else:
        string_input_with_sgn = replaced_input

    string_input_with_sgn += delta_string
    string_with_sgn = "->".join([string_input_with_sgn,string_output])

    def remove_bosonic_right(string):
        [string_input,string_output] = list(string.split("->"))
        string_input_list = list(string_input)
        string_output_list = list(string_output)
        new_string_output = []
        for val in string_output_list:
            if val in string_input_list:
                new_string_output += [val]
        return '->'.join([string_input,''.join(new_string_output)])
    string_for_sign_computation = "->".join([sign_factor_string,replaced_output])
    string_for_sign_computation = remove_bosonic_right(string_for_sign_computation)
    
    # ---------------------------- THIS IS TIME CONSUMING ----------------------------
    # assign the value to the sign factor tensor
    sgn = assigning_sgn(sgn,string_for_sign_computation,format)
    # --------------------------------------------------------------------------------
    
    sgn = SGarray(sgn,bosonic=True)

    # prepare the (sum) sign factor tensor
    sgn_sum = np.zeros(sgn_sum_shape,dtype=int)
    iterator = np.nditer(sgn_sum, flags=['multi_index'])
    for element in iterator:
        coords = iterator.multi_index
        sgn_value = 1
        for i in coords:
            sgn_value *= param.sgn[i]
        sgn_sum[coords] = sgn_value
    sgn_sum = SGarray(sgn_sum,bosonic=True)

    #also add the kronecker delta to sum the fermions
    delta_list = []
    for size in sgn_sum_shape:
        delta_list += [SGarray(np.identity(size,dtype=int),bosonic=True)]

    if(len(sgn_shape)>0 and len(sgn_sum_shape)>0):
        SGarray_list_with_sgn = SGarray_list + [sgn,sgn_sum] + delta_list
    elif(len(sgn_shape)>0 and len(sgn_sum_shape)==0):
        SGarray_list_with_sgn = SGarray_list + [sgn] + delta_list
    elif(len(sgn_shape)==0 and len(sgn_sum_shape)>0):
        SGarray_list_with_sgn = SGarray_list + [sgn_sum] + delta_list
    else:
        SGarray_list_with_sgn = SGarray_list + delta_list

    #opt_einsum.contract do not support multi-indices summation
    #delta nodes must be added appropriately!

    def add_delta_nodes(string_array_list, SGarray_list):
        str_length_list = [len(string) for string in string_array_list.split(",")]
        shape_list = []
        for obj in SGarray_list:
            shape_list += list(obj.shape)
        merged_input = string_array_list.replace(",","")

        key_list = []
        for i in range(len(merged_input)):
            if(merged_input[i] in merged_input[:i]):
                #if the character exists previously
                this_char = merged_input[i]
                #replace this character with a new character
                replaced_char=""
                for char in char_list:
                    if(char not in merged_input):
                        replaced_char=char
                        merged_input = merged_input[:i]+char+merged_input[i+1:]
                        break

                #add this character to the key list
                is_not_added = True
                for j,[key,key_num,key_char,key_shape] in enumerate(key_list):
                    if key==this_char:
                        key_list[j][1] += 1
                        key_list[j][2] = key_list[j][2]+replaced_char
                        is_not_added = False
                if is_not_added :
                    key_list += [ [this_char,2,this_char+replaced_char,shape_list[merged_input.index(this_char)]] ]

        updated_string_array_list = merged_input
        add_index = -1
        for comma_index in str_length_list:
            add_index += comma_index+1
            updated_string_array_list = updated_string_array_list[:add_index] + "," + updated_string_array_list[add_index:]

        updated_string_array_list += key_list[0][2]
        for i in range(1,len(key_list)):
            updated_string_array_list += "," + key_list[i][2]

        delta_nodes = []
        for key_entry in key_list:
            dim = key_entry[3]
            node_shape = [dim]*key_entry[1]
            node = np.zeros(node_shape,dtype=int)
            for i in range(dim):
                ind = tuple([i]*key_entry[1])
                node[ind] = 1
            delta_nodes += [SGarray(node,bosonic=True)]

        return updated_string_array_list,delta_nodes

    temp_str , temp_obj = add_delta_nodes(string_input_with_sgn,SGarray_list_with_sgn)

    string_input_with_sgn = temp_str
    SGarray_list_with_sgn += temp_obj

    string_with_sgn = "->".join([string_input_with_sgn,string_output])
    #may be we should convert everything to sp.COO

    if(string_output==""):
        string_with_sgn = string_with_sgn.replace("->","")

    obj_list = []
    for obj in SGarray_list_with_sgn:
        obj_list += [obj.data]

    X = oe.contract(*tuple([string_with_sgn]+obj_list))
    #X = DGarray(np.array([X.todense()]))

    X = SGarray(X)
    X.statistic = output_statistic_list
    X.encoder = SGarray_list[0].encoder
    X.format = SGarray_list[0].format

    return X

####################################################
##                   Conjugation                  ##
####################################################

# object conditions checked
def hconjugate_old(Obj_input):

    # Hermitian conjugation
    # Note if you split legs of a conjugated tensor
    # You have to manually reverse the leg orientation.

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # WORK IN THE FOLLOWING CONDITIONS      :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # (WILL CONVERT AUTOMATICALLY)          :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #   - canonical encoder                 :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #   - matrix format                     :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    Obj = Obj_input.copy()

    if(type(Obj_input)==DGarray):
        Obj = SGarray(Obj)
    if(Obj_input.format=='standard'):
        Obj = Obj.switch_format()
    if(Obj_input.encoder=='parity-preserving'):
        Obj = Obj.switch_encoder()

    V = Obj.value.copy()
    C = Obj.coords.copy()
    #construct a totally new COO object
    new_coords_system = []
    for axis in range(Obj.ndim):
        new_coords_system += [[]]

    conj_data = []
    for i in range(Obj.nnz):
        new_coords = []
        for ind in C[i]:
            new_coords = [ind]+new_coords
        new_coords = tuple(new_coords)

        for axis in range(Obj.ndim):
            new_coords_system[axis] += [new_coords[axis]]
        conj_data += [np.conjugate(V[i])]

    newshape = []
    for dim in Obj.shape:
        newshape = [dim] + newshape
    newshape = tuple(newshape)

    newstat = []
    for stat in Obj.statistic:
        if(stat==hybrid_symbol):
            newstat = [stat] + newstat
        else:
            newstat = [-stat] + newstat
    newstat = tuple(newstat)

    newCOO = sp.COO(new_coords_system, conj_data, shape=newshape)

    Obj.data = newCOO.copy()
    Obj.statistic = newstat
    
    if(type(Obj_input)==DGarray):
        Obj = DGarray(Obj)
    if(Obj_input.format=='standard'):
        Obj = Obj.switch_format()
    if(Obj_input.encoder=='parity-preserving'):
        Obj = Obj.switch_encoder()

    return Obj

def hconjugate(XGobj,string):

    if(XGobj.format!='matrix'):
        print("error[hconjugate]: Cannot perform the conjugate in the standard format.")
        print("                   Convert to the matrix format first!")
        exit()

    # the string is of the form aaaa|bbb

    objtype = type(XGobj)
    Obj = XGobj.copy()
    if(objtype==SGarray):
        Obj = DGarray(Obj)
    if(objtype not in [DGarray,SGarray]):
        print("error[hconjugate]: Object type must only be DGarray or SGarray!")
        exit()
    # check if XGobj.statistic or final_statistic is weird or not
    for stat in XGobj.statistic:
        if(stat not in allowed_stat):
            print("error[hconjugate]: The input object contains illegal statistic. (0, 1, -1, or "+hybrid_symbol+" only)")
            exit()

    partition_count = 0
    for partition in svd_partition:
        partition_count += string.count(partition)
    if(partition_count!=1):
        partition_string = ""
        for i, partition in enumerate(svd_partition):
            if(i==0):
                partition_string += "( "
            elif(i==len(svd_partition)-1):
                partition_string += ", or "
            else:
                partition_string += ", "

            partition_string += "'"+partition+"'"

            if(i==len(svd_partition)-1):
                partition_string += " )"

        print("error[hconjugate]: The input string must contain one and only one partition "+partition_string+" in it.")
        exit()

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #:::::      STEP 1 - JOIN LEGS BAESD ON THE GROUPINGS                                             :::::
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    # count the number of indices in the two groups
    n_left = 0
    n_right = 0
    partition_found = False
    string_left = ""
    string_right = ""
    for char in string:
        if char in svd_partition :
            partition_found = True
            continue
        if(partition_found):
            n_right+=1
            string_right+=char
        else:
            n_left+=1
            string_left+=char

    join_legs_string_input = string
    for partition in svd_partition:
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
    def get_stat(vector):
        boson_count = 0
        fermi_count = 0
        contain_hybrid = False
        for elem in vector:
            if elem in bose_type :
                boson_count += 1
            if elem in fermi_type :
                fermi_count += 1
            if elem == hybrid_symbol:
                contain_hybrid = True
                break

        if(boson_count==0 and fermi_count>0):
            return 1
        elif(boson_count>0 and fermi_count==0):
            if(contain_hybrid):
                return hybrid_symbol
            else:
                return 0
        elif(boson_count>0 and fermi_count>0):
            return hybrid_symbol

    Obj = join_legs(Obj,join_legs_string_input,(get_stat(stats_left),get_stat(stats_right)))

    Objdat = Obj.data
    Obj_entries_type = type(Objdat[0,0])
    
    hObjdat = np.zeros([Obj.shape[1],Obj.shape[0]] , dtype = Obj_entries_type)
    for i in range(Obj.shape[0]):
        for j in range(Obj.shape[1]):
            hObjdat[j,i] = np.conjugate(Objdat[i,j])

    Obj.data = hObjdat
    Obj.statistic = (Obj.statistic[1],Obj.statistic[0])

    new_string = "("+string_right+")("+string_left+")"
    new_shape = tuple(list(shape_right)+list(shape_left))
    new_stats = list(stats_right)+list(stats_left)

    for i,stat in enumerate(new_stats):
        if stat in fermi_type:
            new_stats[i] *= -1
    new_stats = tuple(new_stats)

    Obj = split_legs(Obj,new_string,new_stats,new_shape)

    if(objtype==SGarray):
        Obj = SGarray(Obj)
    
    return Obj


####################################################
##                     Reshape                    ##
####################################################

# object conditions checked
def join_legs(XGobj,string,final_statistic_inp):

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # WORK IN THE FOLLOWING CONDITIONS      :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # (WILL CONVERT AUTOMATICALLY)          :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #   - canonical encoder                 :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    # the string is of the form aaa(bb)cccc(ddd)eeee
    # The indices will then be grouped together as one index
    
    if np.isscalar(final_statistic_inp):
        final_statistic = tuple([final_statistic_inp])
    else:
        final_statistic = list(final_statistic_inp)

    objtype = type(XGobj)
    Obj = XGobj.copy()
    if(objtype==SGarray):
        Obj = DGarray(Obj)
    if(objtype not in [DGarray,SGarray]):
        print("error[join_legs]: Object type must only be DGarray or SGarray!")
        exit()
    # check if XGobj.statistic or final_statistic is weird or not
    for stat in XGobj.statistic:
        if(stat not in allowed_stat):
            print("error[join_legs]: The input object contains illegal statistic. (0, 1, -1, or "+hybrid_symbol+" only)")
            exit()
    for stat in final_statistic:
        if(stat not in allowed_stat):
            print("error[join_legs]: The final statistic list contains illegal statistic. (0, 1, -1, or "+hybrid_symbol+" only)")
            exit()
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
            print("error[join_legs]: No nested parenthesis!")
            exit()
        elif char == ")" and is_outside:
            print("error[join_legs]: Unmatched parenthesis!")
            exit()
        #print("pending:",groups_info)
    for i,substring in enumerate(groups_info):
        if len(substring) == 0:
            groups_info[i] = ""
        elif len(substring) == 1:
            groups_info[i] = substring
    while groups_info.count("")>0:
        groups_info.remove("")

    n_groups = len(groups_info)
    if n_groups != len(final_statistic):
        print("error[join_legs]: The number of final indices is inconsistent with the grouping!")
        exit()
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
        if (is_hybrid and final_statistic[i] != hybrid_symbol) or (not is_hybrid and final_statistic[i] == hybrid_symbol):
            print("error[join_legs]: The final statistic is inconsistent with object and the grouping!")
            exit()
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
        for i,stat in enumerate(stat_list):
            if stat == 1 and final_statistic[j]==-1:
                sgn = [ (-1)**param.gparity[k] for k in range(shape[i]) ]
                sgn_str += ind_list[i]
                sgn_list += [sgn]

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

    einsum_input = tuple([str_einsum_input+"->"+str_einsum_output] + [Obj.data] + sgn_list )

    Obj.data = np.einsum(*einsum_input)
    Obj.statistic = reordered_stat

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #:::::      STEP 2 - JOIN THE FERMIONIC INDICES WITH CANONICAL ENCODER (DONT SWITCH)              :::::
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
            new_stat_list += [final_statistic[j]] #if it is just fermion, the final_statistic can be applied here.

    new_shape = tuple(new_shape)
    new_stat_list = tuple(new_stat_list)

    Obj.data = np.reshape(Obj.data,new_shape)
    Obj.statistic = new_stat_list

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #:::::      STEP 3 - JOIN ALL THE INDICES WITH PARITY-PRESERVING ENCODER                          :::::
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


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

    if(objtype==SGarray):
        Obj = SGarray(Obj)

    return Obj

def split_legs(XGobj,string,final_statistic_inp,final_shape_inp):

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # WORK IN THE FOLLOWING CONDITIONS      :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # (WILL CONVERT AUTOMATICALLY)          :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #   - canonical encoder                 :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    # the string is of the form aaa(bb)cccc(ddd)eeee
    # The indices will then be grouped together as one index

    final_statistic = list(final_statistic_inp)
    final_shape     = list(final_shape_inp)

    objtype = type(XGobj)
    Obj = XGobj.copy()
    if(objtype==SGarray):
        Obj = DGarray(Obj)
    if(objtype not in [DGarray,SGarray]):
        print("error[split_legs]: Object type must only be DGarray or SGarray!")
        exit()
    # check if XGobj.statistic or final_statistic is weird or not
    for stat in XGobj.statistic:
        if(stat not in allowed_stat):
            print("error[split_legs]: The input object contains illegal statistic. (0, 1, -1, or "+hybrid_symbol+" only)")
            exit()
    for stat in final_statistic:
        if(stat not in allowed_stat):
            print("error[split_legs]: The final statistic list contains illegal statistic. (0, 1, -1, or "+hybrid_symbol+" only)")
            exit()
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
            print("error[split_legs]: No nested parenthesis!")
            exit()
        elif char == ")" and is_outside:
            print("error[split_legs]: Unmatched parenthesis!")
            exit()
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
        print("error[split_legs]: The number of final indices is inconsistent with the grouping!")
        exit()
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
            print("error[split_legs]: The final statistic is inconsistent with object and the grouping!")
            exit()
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
        for i,stat in enumerate(stat_list):
            if stat == 1 and XGobj.statistic[j]==-1:
                sgn = [ (-1)**param.gparity[k] for k in range(shape[i]) ]
                sgn_str += ind_list[i]
                sgn_list += [sgn]

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

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #:::::      STEP 3 - REARRANGE BOSONIC INDICES TO THE RIGHT                                       :::::
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    einsum_input = tuple([str_einsum_input+"->"+str_einsum_output] + [Obj.data] + sgn_list )

    Obj.data = np.einsum(*einsum_input)
    Obj.statistic = final_statistic_inp

    # revert to the original conditions ===================================================================

    if(XGobj.encoder == 'parity-preserving'):
        Obj = Obj.switch_encoder()

    if(objtype==SGarray):
        Obj = SGarray(Obj)

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
        

    if(objtype==DGarray):
        Obj = SGarray(Obj)
    C = Obj.coords
    for i in range(Obj.nnz):
        fcoords = [ ind for j,ind in enumerate(C[i]) if (Obj.statistic[j] in fermi_type)]
        if(sum(fcoords)%2 == 1):
            Obj.data.data[i] = 0
    if(objtype==DGarray):
        Obj = DGarray(Obj)

    if(Obj_input.encoder == 'canonical'):
        Obj = Obj.switch_encoder()

    return Obj

def is_grassmann_even(Obj_input):
    Obj = Obj_input.copy()
    if Obj.encoder == 'canonical':
        Obj = Obj.switch_encoder()
    if type(Obj) == DGarray:
        Obj = SGarray(Obj)

    C = Obj.coords
    for x in C:
        parity = sum([ ind for i,ind in enumerate(x) if Obj.statistic[i] in fermi_type ])
        if parity%2!=0 :
            return False
    return True

####################################################
##                      SVDs                      ##
####################################################

# cU * U = I
# cV * V = I
# V * cV = I

def BlockSVD(Obj):
    
    # performing an svd of a matrix block by block

    if(type(Obj)!=np.array and type(Obj)!=np.ndarray):
        print("Error[BlockSVD]: An input must be of type numpy.array or numpy.ndarray only!")
        exit()

    if(Obj.ndim!=2):
        print("Error[BlockSVD]: An input must be a matrix only!")
        exit()

    m = Obj.shape[0]
    n = Obj.shape[1]

    if(m%2!=0 and n%2!=0):
        print("Error[BlockSVD]: The matrix dimensions must be even!")
        exit()


    if(m==0 and n==0):
        print("Error[BlockSVD]: The matrix dimensions must be at least 2!")
        exit()

    parity_norm = 0
    for i in range(m):
        for j in range(n):
            if (i+j)%2!=0:
                parity_norm += np.linalg.norm(Obj[i,j])
    if( (not skip_parity_blocking_check) and parity_norm/(m*n/2)>1.0e-14):
        print("Error[BlockSVD]: This matrix is not constructed from a Grassmann-even tensor.")
        print("                 (Or that one of the indices are non-fermionic.)")
        exit()

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

    '''
    M = UO.copy()
    cM = np.einsum('ij->ji',np.conjugate(M))

    print(
            np.einsum('ik,kj->ij',cM,M)
        )
    print()

    print(
            np.einsum('ik,kj->ij',M,cM)
        )

    exit()
    '''

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
    if(objtype==SGarray):
        Obj = DGarray(Obj)
    if(objtype not in [DGarray,SGarray]):
        print("error[SVD]: Object type must only be DGarray or SGarray!")
        exit()
    # check if XGobj.statistic or final_statistic is weird or not
    for stat in XGobj.statistic:
        if(stat not in allowed_stat):
            print("error[SVD]: The input object contains illegal statistic. (0, 1, -1, or "+hybrid_symbol+" only)")
            exit()
    if(XGobj.encoder == 'canonical'):
        Obj = Obj.switch_encoder()
    if(XGobj.format == 'standard'):
        Obj = Obj.switch_format()

    partition_count = 0
    for partition in svd_partition:
        partition_count += string.count(partition)
    if(partition_count!=1):
        partition_string = ""
        for i, partition in enumerate(svd_partition):
            if(i==0):
                partition_string += "( "
            elif(i==len(svd_partition)-1):
                partition_string += ", or "
            else:
                partition_string += ", "

            partition_string += "'"+partition+"'"

            if(i==len(svd_partition)-1):
                partition_string += " )"

        print("error[SVD]: The input string must contain one and only one partition "+partition_string+" in it.")
        exit()

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #:::::      STEP 1 - JOIN LEGS BAESD ON THE GROUPINGS                                             :::::
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    # count the number of indices in the two groups
    n_left = 0
    n_right = 0
    partition_found = False
    for char in string:
        if char in svd_partition :
            partition_found = True
            continue
        if(partition_found):
            n_right+=1
        else:
            n_left+=1

    join_legs_string_input = string
    for partition in svd_partition:
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
    def get_stat(vector):
        boson_count = 0
        fermi_count = 0
        for elem in vector:
            if elem in bose_type :
                boson_count += 1
            if elem in fermi_type :
                fermi_count += 1

        if(boson_count==0 and fermi_count>0):
            return 1
        elif(boson_count>0 and fermi_count==0):
            return 0
        elif(boson_count>0 and fermi_count>0):
            return hybrid_symbol

    Obj = join_legs(Obj,join_legs_string_input,(get_stat(stats_left),get_stat(stats_right)))

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #:::::      STEP 2 - BLOCK SVD (MAKE SURE IT'S PARITY-PRESERVING!)                                :::::
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    U, Λ, V = BlockSVD(Obj.data)

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #:::::      STEP 3 - RECONSTRUCT U, Λ, and V AS GRASSMANN TENSORS                                 :::::
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    U = DGarray(U,encoder="parity-preserving")
    Λ = DGarray(Λ,encoder="parity-preserving")
    V = DGarray(V,encoder="parity-preserving")

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
        if char in svd_partition :
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

    U = split_legs(U,Uind,Ustats,Ushape)
    V = split_legs(V,Vind,Vstats,Vshape)

    # revert to the original conditions ===================================================================

    if(XGobj.encoder == 'canonical'):
        U = U.switch_encoder()
        Λ = Λ.switch_encoder()
        V = V.switch_encoder()
    if(XGobj.format == 'standard'):
        U = U.switch_format()
        Λ = Λ.switch_format()
        V = V.switch_format()

    if(objtype==SGarray):
        U = SGarray(U)
        Λ = SGarray(Λ)
        V = SGarray(V)

    return U, Λ, V


####################################################
##                   Isometries                   ##
####################################################

