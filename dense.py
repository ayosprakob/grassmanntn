import numpy as np
import math
from gTN import sparse as sgTN
from gTN import param


info=["class:DGarray","func:einsum","func:relative_parity"]

####################################################
##               Parity Calculation               ##
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
            print("error[absolute_parity.get_commutative_elements]: Inconsistent array sizes!")
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
            print("error[relative_parity_int.get_noncommutative_elements]: Inconsistent array sizes!")
            exit()
        else:
            noncommutative_elements = [x for i,x in enumerate(permutation) if (-1)**individual_parity[i]==-1]
            return noncommutative_elements

    individual_parity2 = permute_c(permutation1, permutation2, individual_parity1)
    
    noncommutative_elements1 = get_noncommutative_elements(permutation1,individual_parity1)
    noncommutative_elements2 = get_noncommutative_elements(permutation2,individual_parity2)
    
    if(sorted(noncommutative_elements1) != sorted(noncommutative_elements2)):
        print("error[relative_parity_int]: Inconsistent grassmann-odd indices!")
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
        print("error[relative_parity]: The number of input list and parity list are not consistent!")
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
##             Densed Grassmann Array             ##
####################################################


class DGarray:
    def __init__(self, data=[], bosonic = False):
    
        self.array = None
        self.shape = None
        self.size  = None
        self.ndim  = None
        self.is_decorated = False
        
        if(type(data)==np.array or type(data)==np.ndarray or type(data)==list):
            self.array = np.array(data)
            self.shape = self.shape_method()
            self.size  = self.size_method()
            self.ndim  = self.ndim_method()
            self.is_decorated = False
            if bosonic:
                self.statistic = tuple([ 0 for i in range(len(self.shape))])
            else:
                self.statistic = tuple([ 1 for i in range(len(self.shape))])
        elif(type(data)==DGarray):
            converted = data.copy()
            #copy DGarray properties
            self.array = converted.array
            self.shape = converted.shape
            self.size  = converted.size
            self.ndim  = converted.ndim
            self.statistic = converted.statistic
            self.is_decorated = converted.is_decorated
        elif(type(data)==sgTN.SGarray):
            converted = sgTN.make_DGarray(data)
            #copy DGarray properties
            self.array = converted.array
            self.shape = converted.shape
            self.size  = converted.size
            self.ndim  = converted.ndim
            self.statistic = converted.statistic
            self.is_decorated = converted.is_decorated
        else:
            print("error[DGarray]: Invalid initialized data")
            exit()
    
    def __getitem__(self, index):
        return self.array[index]
    
    def __setitem__(self, index, value):
        self.array[index] = value
        
    def copy(self):
        #copy DGarray properties
        ret = DGarray()
        ret.array = self.array.copy()
        ret.shape = self.shape
        ret.size  = self.size
        ret.ndim  = self.ndim
        ret.statistic = self.statistic
        ret.is_decorated = self.is_decorated
        return ret
    
    def __add__(self, other):
        if(self.shape!=other.shape or self.statistic!=other.statistic):
            print("error[DGarray.+]: Inconsistent shape or statistic")
            exit()
        ret = self.copy()
        ret.array = ret.array+other.array
        return ret
        
    def __sub__(self, other):
        if(self.shape!=other.shape or self.statistic!=other.statistic):
            print("error[DGarray.-]: Inconsistent shape or statistic")
            exit()
        ret = self.copy()
        ret.array = ret.array-other.array
        return ret
        
    def __mul__(self, other):
        ret = self.copy()
        ret.array = ret.array*other
        return ret
        
    def __rmul__(self, other):
        return self*other
        
    def __len__(self):
        return len(self.array)
    
    def __str__(self):
        return str(self.array)
    
    def __repr__(self):
        return repr(self.array)
    
    def shape_method(self):
        return (self.array.shape)
    
    def size_method(self):
        return (self.array.size)
    
    def ndim_method(self):
        return (self.array.ndim)
    
    def set_statistic(self,statistic_list):
        if(len(self.shape)!=len(statistic_list)):
            print("error[DGarray.set_statistic]: statistic_list and object shape are not consistent!")
            exit()
        self.statistic = statistic_list
        
    def decorated(self):
        #multiply sign factor sigma[i] to every conjugated indices i
        ret = self.copy()
        iterator = np.nditer(ret.array, flags=['multi_index'])
        for element in iterator:
            coords = iterator.multi_index
            sgn_value = 1
            for i,ind in enumerate(coords):
                if(ret.statistic[i]==-1):
                    sgn_value *= param.sgn[ind]
            ret[coords] *= sgn_value
        ret.is_decorated = not self.is_decorated
        return ret
        

#-------------------------------------------------------#
# For fermionic sum, the strategy is to write           #
# another function to check wether the fermion          #
# is summed or not first.                               #
# If not, use beinsum.                                  #
# Else, write a function to reorder the output string   #
# as [original output]+[integrated pair1]+[pair2]+...   #
# with beinsum.                                         #
# Then run another function to execute the sum.         #
#-------------------------------------------------------#

def einsum(*args):
    #----------------------------------------------------#
    # The same usage as np.einsum                        #
    # but with sign factor applied.                      #
    # Fermionic indices CAN be summed here.              #
    #----------------------------------------------------#
    
    #----------------------------------------------------#
    # Remove basic errors                                #
    # 1) must contain only one instance of '->'.         #
    # 2) the result must not contain ','.                #
    # 3) statistics are not 0, 1, or -1.                 #
    # 4) indices and statistic must be consistent.       #
    #----------------------------------------------------#
    
    string = args[0]
    # detecting error 1)
    if(string.count("->")!=1):
        print("error[einsum]: The string contains no or more than one instance of '->'.")
        exit()
    
    [string_input,string_output] = list(string.split("->"))
    
    # detecting error 2)
    if(string_output.count(",")!=0):
        print("error[einsum]: The string_output contains ','.")
        exit()
    
    string_list = list(string_input.split(","))
    number_of_object = len(string_list)
    DGarray_list = [ DGarray(obj) for obj in args[1:1+number_of_object]]
    statistic_list = tuple(sum([ list(obj.statistic) for obj in DGarray_list ],[]))
    
    # detecting error 3)
    for stat in statistic_list:
        if(stat!=0 and stat!=1 and stat!=-1):
            print("error[einsum]: Statistics must be 0, +1, or -1.")
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
                    )
                ):
                print("error[einsum]: Inconsistent index and statistic.")
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
    
    char_list = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z","A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
    
    replaced_joined_string_input = joined_string_input.copy()
    summed_list = []
    sign_sum_factor_string = []
    for i,elem in enumerate(replaced_joined_string_input):
        if(statistic_list[i]==-1 and replaced_joined_string_input.count(elem)==2):
            for char in char_list:
                if(char not in replaced_joined_string_input):
                    replaced_joined_string_input[i]=char
                    break
            summed_list = summed_list+[(elem,replaced_joined_string_input[i])]
            sign_sum_factor_string = sign_sum_factor_string + [elem]
    sign_sum_factor_string = ''.join(sign_sum_factor_string)
    
    replaced_input = replaced_joined_string_input.copy()
    nitems = len(string_list)
    inserted_loc = 0
    for i in range(nitems):
        replaced_input.insert(inserted_loc+len(string_list[i]),",")
        inserted_loc = inserted_loc + 1 + len(string_list[i])
    replaced_input = replaced_input[:-1]
    replaced_input = ''.join(replaced_input)
    
    replaced_output = string_output
    for pair in summed_list:
        replaced_output = replaced_output + pair[0] + pair[1]
        
    replaced_string = '->'.join([replaced_input,replaced_output])
    
    #get shape list
    shape_list = []
    for elem in DGarray_list:
        shape_list = shape_list + list(elem.shape)
    
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
                if list2[i]==1 or list2[i]==-1:
                    new_list1.append(val)
            return new_list1
        join_string_list = remove_bosonic(list(join_string), statistic_list)
        return ''.join(join_string_list)
        
    # get the string for einsum input
    sign_factor_string = get_Gindex_string(replaced_string,statistic_list)
    delta_string = ''.join([ ","+x+xbar for (x,xbar) in summed_list])
    string_input_with_sgn = ",".join([replaced_input,sign_factor_string,sign_sum_factor_string])
    string_input_with_sgn = string_input_with_sgn+delta_string
    string_with_sgn = "->".join([string_input_with_sgn,string_output])
    
    def get_Gindex_shape(DGarray_list, statistic_list):
        """
        return shape of (permutation) sign factor tensor
        """
        DGarray_number = len(DGarray_list)
        shape_list = []
        for obj in DGarray_list:
            shape_list = shape_list + list(obj.shape)
        Gindex_shape = [ x for i,x in enumerate(shape_list) if statistic_list[i]==1 or statistic_list[i]==-1]
        return Gindex_shape
    # prepare the (permutation) sign factor tensor
    sgn_shape = get_Gindex_shape(DGarray_list, statistic_list)
    sgn = np.zeros(sgn_shape,dtype=int)
    
    def remove_bosonic_right(string):
        [string_input,string_output] = list(string.split("->"))
        string_input_list = list(string_input)
        string_output_list = list(string_output)
        new_string_output = []
        for val in string_output_list:
            if val in string_input_list:
                new_string_output = new_string_output + [val]
        return '->'.join([string_input,''.join(new_string_output)])
    # assign the value to the sign factor tensor
    iterator = np.nditer(sgn, flags=['multi_index'])
    string_for_sign_computation = "->".join([sign_factor_string,replaced_output])
    string_for_sign_computation = remove_bosonic_right(string_for_sign_computation)
    for element in iterator:
        coords = iterator.multi_index
        
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
        delta_list = delta_list + [np.identity(size,dtype=int)]
    
    DGarray_list_with_sgn = DGarray_list + [sgn,sgn_sum] + delta_list
    
    X = DGarray(np.einsum(*tuple([string_with_sgn]+DGarray_list_with_sgn)))
    X.statistic = output_statistic_list
    return X

