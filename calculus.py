import numpy as np
from numpy import linalg as la
import math
from grassmannTN import grassmannTN as gTN
from grassmannTN import param
import sparse as sp
import opt_einsum as oe

hybrid_symbol = "*"
svd_partition = "|"

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
##                     Einsums                    ##
####################################################

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
        print("error[einsum]: The string contains more than one instances of '->'!")
        exit()

    string_input = ""
    string_output = ""
    if(string.count("->")==1):
        [string_input,string_output] = list(string.split("->"))
    else:
        string_input = string
    
    # detecting error 2)
    if(string_output.count(",")!=0):
        print("error[einsum]: The string_output contains ','.")
        exit()
    
    string_list = list(string_input.split(","))
    number_of_objects = len(string_list)


    DGarray_list = [ gTN.DGarray(obj) for obj in args[1:1+number_of_objects]]

    def return_list(stat):
        if stat==1 or stat==-1 or stat==hybrid_symbol or len(stat)==1:
            return [stat]
        else:
            return list(stat)
    statistic_list = tuple(sum([ return_list(obj.statistic) for obj in DGarray_list ],[]))
    
    # detecting error 3)
    for stat in statistic_list:
        if(stat!=0 and stat!=1 and stat!=-1 and stat!=hybrid_symbol):
            print("error[einsum]: Statistics must be 0, +1, -1, or"+hybrid_symbol+".")
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
                if list2[i]==1 or list2[i]==-1:
                    new_list1.append(val)
            return new_list1
        join_string_list = remove_bosonic(list(join_string), statistic_list)
        return ''.join(join_string_list)
        
    # get the string for einsum input
    sign_factor_string = get_Gindex_string(replaced_string,statistic_list)
    delta_string = ''.join([ ","+x+xbar for (x,xbar) in summed_list])
    
    def get_Gindex_shape(DGarray_list, statistic_list):
        """
        return shape of (permutation) sign factor tensor
        """
        DGarray_number = len(DGarray_list)
        shape_list = []
        for obj in DGarray_list:
            shape_list += list(obj.shape)
        Gindex_shape = [ x for i,x in enumerate(shape_list) if statistic_list[i]==1 or statistic_list[i]==-1]
        return Gindex_shape
    # prepare the (permutation) sign factor tensor
    sgn_shape = get_Gindex_shape(DGarray_list, statistic_list)
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

    #X = gTN.DGarray(np.array([np.einsum(*tuple([string_with_sgn]+DGarray_list_with_sgn))]))
    X = gTN.DGarray(np.einsum(*tuple([string_with_sgn]+DGarray_list_with_sgn)))
    X.statistic = output_statistic_list
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
        print("error[einsum]: The string contains more than one instances of '->'!")
        exit()

    string_input = ""
    string_output = ""
    if(string.count("->")==1):
        [string_input,string_output] = list(string.split("->"))
    else:
        string_input = string

    # detecting error 2)
    if(string_output.count(",")!=0):
        print("error[einsum]: The string_output contains ','.")
        exit()
    
    string_list = list(string_input.split(","))
    number_of_objects = len(string_list)


    SGarray_list = [ gTN.SGarray(obj) for obj in args[1:1+number_of_objects]]

    def return_list(stat):
        if stat==1 or stat==-1 or stat==hybrid_symbol or len(stat)==1:
            return [stat]
        else:
            return list(stat)
    statistic_list = tuple(sum([ return_list(obj.statistic) for obj in SGarray_list ],[]))
    
    # detecting error 3)
    for stat in statistic_list:
        if(stat!=0 and stat!=1 and stat!=-1 and stat!=hybrid_symbol):
            print("error[einsum]: Statistics must be 0, +1, -1, or"+hybrid_symbol+".")
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
    #SGarray_list = [A1,A2,A3]
    #statistic_list = [i,j,k,..]
    
    #----------------------------------------------------#
    # Replace the conjugated indices by another char.    #
    # Add the pair [ind,ind-bar] at the end of           #
    # the output.                                        #
    #----------------------------------------------------#
    
    
    #summed indices is always paired by the conditions above
    #first add "-" in the conjugated variable
    
    char_list = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z","A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","α","β","Γ","γ","Δ","δ","ε","ζ","η","Θ","θ","ι","κ","Λ","λ","μ","ν","Ξ","ξ","Π","π","ρ","Σ","σ","ς","τ","υ","Φ","φ","χ","Ψ","ψ","Ω","ω"]
    
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
            sign_sum_factor_string = sign_sum_factor_string + [elem]
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
                if list2[i]==1 or list2[i]==-1:
                    new_list1.append(val)
            return new_list1
        join_string_list = remove_bosonic(list(join_string), statistic_list)
        return ''.join(join_string_list)
        
    # get the string for einsum input
    sign_factor_string = get_Gindex_string(replaced_string,statistic_list)
    delta_string = ''.join([ ","+x+xbar for (x,xbar) in summed_list])

    def get_Gindex_shape(SGarray_list, statistic_list):
        """
        return shape of (permutation) sign factor tensor
        """
        DGarray_number = len(SGarray_list)
        shape_list = []
        for obj in SGarray_list:
            shape_list = shape_list + list(obj.shape)
        Gindex_shape = [ x for i,x in enumerate(shape_list) if statistic_list[i]==1 or statistic_list[i]==-1]
        return Gindex_shape
    # prepare the (permutation) sign factor tensor
    sgn_shape = get_Gindex_shape(SGarray_list, statistic_list)
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
    sgn = gTN.SGarray(sgn,bosonic=True)

    # prepare the (sum) sign factor tensor
    sgn_sum = np.zeros(sgn_sum_shape,dtype=int)
    iterator = np.nditer(sgn_sum, flags=['multi_index'])
    for element in iterator:
        coords = iterator.multi_index
        sgn_value = 1
        for i in coords:
            sgn_value *= param.sgn[i]
        sgn_sum[coords] = sgn_value
    sgn_sum = gTN.SGarray(sgn_sum,bosonic=True)

    #also add the kronecker delta to sum the fermions
    delta_list = []
    for size in sgn_sum_shape:
        delta_list += [gTN.SGarray(np.identity(size,dtype=int),bosonic=True)]

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
            delta_nodes += [gTN.SGarray(node,bosonic=True)]

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
    #X = gTN.DGarray(np.array([X.todense()]))

    X = gTN.SGarray(X)
    X.statistic = output_statistic_list
    return X

# Unified version
def einsum(*args):

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

    this_type = type_list[0]
    for i in range(1,number_of_objects):
        if type_list[i]!=this_type :
            print("error[einsum]: Objects must be of the same type!")
            exit()

    if(this_type!=gTN.DGarray and this_type!=gTN.SGarray):
        print("error[einsum]: Objects must be either gTN.DGarray or gTN.SGarray!")
        exit()

    if(this_type==gTN.DGarray):
        return deinsum(*args)
    else:
        return seinsum(*args)


####################################################
##                     Reshape                    ##
####################################################

def merge_legs(string,XGobj,final_statistic_inp):

    # the string is of the form aaa(bb)cccc(ddd)eeee
    # The indices will then be grouped together as one index

    final_statistic = list(final_statistic_inp)

    objtype = type(XGobj)
    Obj = XGobj.copy()
    if(objtype==gTN.SGarray):
        Obj = gTN.DGarray(Obj)
    if(objtype!=gTN.DGarray and objtype!=gTN.SGarray):
        print("error[merge_legs]: Object type must only be DGarray or SGarray!")
        exit()

    # check if XGobj.statistic or final_statistic is weird or not
    for stat in XGobj.statistic:
        if(stat!=0 and stat!=1 and stat!=-1 and stat!=hybrid_symbol):
            print("error[merge_legs]: The input object contains illegal statistic. (0, 1, -1, or "+hybrid_symbol+" only)")
            exit()
    for stat in final_statistic:
        if(stat!=0 and stat!=1 and stat!=-1 and stat!=hybrid_symbol):
            print("error[merge_legs]: The final statistic list contains illegal statistic. (0, 1, -1, or "+hybrid_symbol+" only)")
            exit()

    string_listing = [["",""]] #[string,type]
    is_outside = True
    for char in string:
        current_pos = len(string_listing)-1

        if char != "(" and char != ")" and not is_outside:
            string_listing[current_pos][0] += char

        elif char != "(" and char != ")" and is_outside:
            string_listing += [[char,""]]

        elif char == "(" and is_outside :
            string_listing += [["","grouped"]]
            is_outside = False

        elif char == ")" and not is_outside :
            string_listing += [["",""]]
            is_outside = True

        elif char == "(" and not is_outside:
            print("error[merge_legs]: No nested parenthesis!")
            exit()

        elif char == ")" and is_outside:
            print("error[merge_legs]: Unmatched parenthesis!")
            exit()

    for i,[substring,grouping_type] in enumerate(string_listing):
        if len(substring) == 0:
            string_listing[i] = ["",""]
        elif len(substring) == 1:
            string_listing[i] = [substring,""]

    while string_listing.count(["",""])>0:
        string_listing.remove(["",""])


    n_groups = len(string_listing)
    if n_groups != len(final_statistic):
        print("error[merge_legs]: The number of final indices is inconsistent with the grouping!")
        exit()

    # add entries about the dimensions and statistics
    index = 0
    for i,elem in enumerate(string_listing):
        shape = XGobj.shape[index:index+len(elem[0])]
        statistic = XGobj.statistic[index:index+len(elem[0])]
        string_listing[i] = [elem,shape,statistic]
        
        # check if is hybrid

        n_nonfermions = statistic.count(0)+statistic.count(hybrid_symbol)
        is_hybrid = (
            (n_nonfermions<len(statistic) and n_nonfermions>0)
            or (statistic.count(hybrid_symbol)>0)
            )

        if (is_hybrid and final_statistic[i] != "*") or (not is_hybrid and final_statistic[i] == "*"):
            print("error[merge_legs]: The final statistic is inconsistent with object and the grouping!")
            exit()

        index += len(elem[0])

    # compute the sign factors
    print()
    sgn_list = []
    sgn_string_list = []
    for i,elem in enumerate(string_listing):
        if(final_statistic[i]!=1):
            # apply sign factor if final_statistic=+1 only
            continue
        #print(elem,"->",final_statistic[i])

        sgn_str = ""
        sgn_dim = []
        for i,stat in enumerate(elem[2]):
            if stat == -1:
                sgn_dim += [elem[1][i]]
                sgn_str += elem[0][0][i]
        sgn_string_list += sgn_str
        #print(sgn_dim)
        for obj in range(len(sgn_dim)):
            sgn_obj = np.array([ (-1)**param.gparity[i] for i in range(sgn_dim[obj]) ])
            sgn_list += [sgn_obj]
    #print(sgn_string_list)
    #for obj in sgn_list:
    #    print(obj)



    # apply the sign factors
    str_for_einsum = string
    while str_for_einsum.count("(")>0 or str_for_einsum.count(")")>0:
        str_for_einsum = str_for_einsum.replace("(","")
        str_for_einsum = str_for_einsum.replace(")","")

    str_for_einsum_temp = str_for_einsum
    for x in sgn_string_list:
        str_for_einsum += ","+x
    str_for_einsum = str_for_einsum + "->" + str_for_einsum_temp
    obj_list = [Obj.data]
    for x in sgn_list:
        obj_list += [x]


    Obj_array = np.einsum(*tuple([str_for_einsum]+obj_list))


    #reshape
    dim_list = []
    for elem in string_listing:
        prod_dim = 1
        for dim in elem[1]:
            prod_dim *= dim
        dim_list += [prod_dim]

    dim_list = tuple(dim_list)
    Obj_array = Obj_array.reshape(dim_list)

    Obj.data = Obj_array
    Obj.statistic = tuple(final_statistic)

    if objtype == gTN.SGarray:
        return gTN.SGarray(Obj)

    return Obj

def unmerge_legs(string,XGobj,final_statistic_inp,final_shape_inp):

    # the string is of the form aaa(bb)cccc(ddd)eeee
    # The indices will then be grouped together as one index

    final_statistic = list(final_statistic_inp)
    final_shape = list(final_shape_inp)


    #swap variables....

    # XGobj.statistic <----> final_statistic
    # XGobj.shape     <----> final_shape

    objtype = type(XGobj)
    Obj = XGobj.copy()
    if(objtype==gTN.SGarray):
        Obj = gTN.DGarray(Obj)
    if(objtype!=gTN.DGarray and objtype!=gTN.SGarray):
        print("error[unmerge_legs]: Object type must only be DGarray or SGarray!")
        exit()

    # check if final_statistic or XGobj.statistic is weird or not
    for stat in final_statistic:
        if(stat!=0 and stat!=1 and stat!=-1 and stat!=hybrid_symbol):
            print("error[unmerge_legs]: The input object contains illegal statistic. (0, 1, -1, or "+hybrid_symbol+" only)")
            exit()
    for stat in XGobj.statistic:
        if(stat!=0 and stat!=1 and stat!=-1 and stat!=hybrid_symbol):
            print("error[unmerge_legs]: The final statistic list contains illegal statistic. (0, 1, -1, or "+hybrid_symbol+" only)")
            exit()

    string_listing = [["",""]] #[string,type]
    is_outside = True
    for char in string:
        current_pos = len(string_listing)-1

        if char != "(" and char != ")" and not is_outside:
            string_listing[current_pos][0] += char

        elif char != "(" and char != ")" and is_outside:
            string_listing += [[char,""]]

        elif char == "(" and is_outside :
            string_listing += [["","grouped"]]
            is_outside = False

        elif char == ")" and not is_outside :
            string_listing += [["",""]]
            is_outside = True

        elif char == "(" and not is_outside:
            print("error[unmerge_legs]: No nested parenthesis!")
            exit()

        elif char == ")" and is_outside:
            print("error[unmerge_legs]: Unmatched parenthesis!")
            exit()

    for i,[substring,grouping_type] in enumerate(string_listing):
        if len(substring) == 0:
            string_listing[i] = ["",""]
        elif len(substring) == 1:
            string_listing[i] = [substring,""]

    while string_listing.count(["",""])>0:
        string_listing.remove(["",""])


    n_groups = len(string_listing)
    if n_groups != len(XGobj.statistic):
        print("error[unmerge_legs]: The number of final indices is inconsistent with the grouping!")
        exit()

    # add entries about the dimensions and statistics
    index = 0
    for i,elem in enumerate(string_listing):
        shape = final_shape[index:index+len(elem[0])]
        statistic = final_statistic[index:index+len(elem[0])]
        string_listing[i] = [elem,shape,statistic]
        
        # check if is hybrid

        n_nonfermions = statistic.count(0)+statistic.count(hybrid_symbol)
        is_hybrid = (
            (n_nonfermions<len(statistic) and n_nonfermions>0)
            or (statistic.count(hybrid_symbol)>0)
            )

        if (is_hybrid and XGobj.statistic[i] != "*") or (not is_hybrid and XGobj.statistic[i] == "*"):
            print("error[unmerge_legs]: The final statistic is inconsistent with object and the grouping!")
            exit()

        index += len(elem[0])

    # compute the sign factors
    print()
    sgn_list = []
    sgn_string_list = []
    for i,elem in enumerate(string_listing):
        if(XGobj.statistic[i]!=1):
            # apply sign factor if XGobj.statistic=+1 only
            continue
        #print(elem,"->",XGobj.statistic[i])

        sgn_str = ""
        sgn_dim = []
        for i,stat in enumerate(elem[2]):
            if stat == -1:
                sgn_dim += [elem[1][i]]
                sgn_str += elem[0][0][i]
        sgn_string_list += sgn_str
        #print(sgn_dim)
        for obj in range(len(sgn_dim)):
            sgn_obj = np.array([ (-1)**param.gparity[i] for i in range(sgn_dim[obj]) ])
            sgn_list += [sgn_obj]
    #print(sgn_string_list)
    #for obj in sgn_list:
    #    print(obj)


    #reshape first then apply the sign factor
    Obj_array = Obj.data.reshape(final_shape)

    # apply the sign factors
    str_for_einsum = string
    while str_for_einsum.count("(")>0 or str_for_einsum.count(")")>0:
        str_for_einsum = str_for_einsum.replace("(","")
        str_for_einsum = str_for_einsum.replace(")","")

    str_for_einsum_temp = str_for_einsum
    for x in sgn_string_list:
        str_for_einsum += ","+x
    str_for_einsum = str_for_einsum + "->" + str_for_einsum_temp
    obj_list = [Obj_array]
    for x in sgn_list:
        obj_list += [x]

    Obj_array = np.einsum(*tuple([str_for_einsum]+obj_list))

    Obj.data = Obj_array
    Obj.statistic = tuple(final_statistic)

    if objtype == gTN.SGarray:
        return gTN.SGarray(Obj)

    return Obj


####################################################
##                      SVDs                      ##
####################################################

def SVD(string,XGobj):

    # input string is of the form 'aaaaaa,bbbbbb' to tell which location do we perform the SVD

    objtype = type(XGobj)
    Obj = XGobj.copy()
    if(objtype==gTN.SGarray):
        Obj = gTN.DGarray(Obj)
    if(objtype!=gTN.DGarray and objtype!=gTN.SGarray):
        print("error[SVD]: Object type must only be DGarray or SGarray!")
        exit()

    if(string.count(svd_partition)!=1):
        print("error[SVD]: There must be one and only one partition("+svd_partition+") in the index string.")
        exit()

    [str_a,str_b] = string.split(svd_partition)
    partition_loc = len(str_a)
    shape_a = Obj.shape[:partition_loc]
    shape_b = Obj.shape[partition_loc:]
    stat_a = Obj.statistic[:partition_loc]
    stat_b = Obj.statistic[partition_loc:]

    #First is to move bosonic variables to the right of each partition
    str2_a = ""
    shape2_a = []
    stat2_a = []
    for i in range(len(stat_a)):
        if(stat_a[i]==0 or stat_a[i]==hybrid_symbol):
            str2_a = str_a[i]+str2_a
            shape2_a = [shape_a[i]]+shape2_a
            stat2_a = [stat_a[i]]+stat2_a
        else:
            str2_a += str_a[i]
            shape2_a += [shape_a[i]]
            stat2_a += [stat_a[i]]

    str2_b = ""
    shape2_b = []
    stat2_b = []
    for i in range(len(stat_b)):
        if(stat_b[i]==0 or stat_b[i]==hybrid_symbol):
            str2_b = str_b[i]+str2_b
            shape2_b = [shape_b[i]]+shape2_b
            stat2_b = [stat_b[i]]+stat2_b
        else:
            str2_b += str_b[i]
            shape2_b += [shape_b[i]]
            stat2_b += [stat_b[i]]

    #rearrange
    str_for_rearrange = str_a+str_b+"->"+str2_a+str2_b
    Obj2 = einsum(str_for_rearrange,Obj)
    
    reshape_string = "("+str2_a+")("+str2_b+")"

    #determine the reshaped statistic
    n_nonfermions_a = stat2_a.count(0)+stat2_a.count(hybrid_symbol)
    n_nonfermions_b = stat2_b.count(0)+stat2_b.count(hybrid_symbol)
    is_hybrid_a = n_nonfermions_a < len(stat2_a) and n_nonfermions_a>0
    is_hybrid_b = n_nonfermions_b < len(stat2_b) and n_nonfermions_b>0
    new_stat_a = -1
    new_stat_b = -1
    if(is_hybrid_a):
        new_stat_a = hybrid_symbol
    if(is_hybrid_b):
        new_stat_b = hybrid_symbol
    new_stat = tuple([new_stat_a,new_stat_b])

    encoder_switched = False
    if(Obj.encoding == 'canonical'):
        Obj = Obj.switch_encoder()
        encoder_switched = True

    reshaped_Obj2 = merge_legs(reshape_string,Obj2,new_stat)

    # In this function, we don't decorate the matrices.
    # It turns out that doing the SVD directly gives the same result
    # (up to some sign factors that has nothing to do with the fermions)
    M = reshaped_Obj2.data
    U, Λ, V = la.svd(M, full_matrices=False)
    # ij -> ik,k,kj
    def get_mindim(x):
        return int(2**math.ceil(np.log2(x)))

    d_lambda = get_mindim(len(Λ))

    U_padded = np.pad(U,((0,0),(0,d_lambda)),'constant',constant_values=((0,0),(0,0)))
    Λ_padded = np.pad(Λ,(0,d_lambda),        'constant',constant_values=(0,0)        )
    V_padded = np.pad(V,((0,d_lambda),(0,0)),'constant',constant_values=((0,0),(0,0)))

    char_list = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z","A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
    new_char = ""
    for char in char_list:
        if char not in str_a+str_b:
            new_char = char
            break

    str2_U = str2_a+new_char
    str_Λ  = new_char
    str2_V = new_char+str2_b
    str_U = str_a+new_char
    str_V = new_char+str_b

    print(str2_U,str_Λ,str2_V)
    print(str_U,str_Λ,str_V)


    # construct new tensors ---------------------------


    # DON'T FORGET TO REVERT THE ENCODER AND INDICES (str and str2, etc)
    if(encoder_switched):
        Obj = Obj.switch_encoder()

    return XGobj



####################################################
##                   Isometries                   ##
####################################################
