import numpy as np
import math
import sparse as sp
import opt_einsum as oe
from gTN import dense as dgTN
from gTN import param

#-----------------------------------------------------#
# Turn np.array <dA> to sp.COO <sA>                   #
#   sA = sp.COO.from_numpy(dA)                        #
#                                                     #
# Turn sp.COO <sA> to np.array <dA>                   #
#   dA = sA.todense()                                 #
#                                                     #
# Einsum (return as dense)                            #
#   dA = oe.contract('input->output',operator_list)   #
#   **the sum is only possible with an index pair.    #
#-----------------------------------------------------#

'''



a = sp.COO.from_numpy(A)
b = sp.COO.from_numpy(B)

c = sp.COO.from_numpy(oe.contract('MNaQ,bNPMa->QbP',A,B))

print(c.shape)

'''


####################################################
##                 Sparse arrays                  ##
####################################################

class SGarrayElement:
    
    def __init__(self,coords,value):
        self.coords = coords
        self.value  = value
        
    def __add__(self, other):
        ret.array = ret.array+other.array
        return ret
    
    def getlist(self):
        return [self.coords,self.value]

class SGarray:
    def __init__(self,input_data=[],bosonic=False):
    
        self.data = None
        self.shape = None
        self.statistic = None
        self.is_decorated = False
        
        if(type(input_data)==list):
            self.data = input_data
            self.shape = None
            self.statistic = None
            self.is_decorated = False
        elif(type(input_data)==dgTN.DGarray):
            temp = make_SGarray(input_data)
            #copy SGarray properties
            self.data = temp.data.copy()
            self.shape = temp.shape
            self.statistic = temp.statistic
            self.is_decorated = temp.is_decorated
        elif(type(input_data)==np.array):
            temp = dgTN.DGarray(input_data,bosonic=bosonic)
            temp = SGarray(temp)
            #copy SGarray properties
            self.data = temp.data.copy()
            self.shape = temp.shape
            self.statistic = temp.statistic
            self.is_decorated = temp.is_decorated
         
    def __getitem__(self, index):
        return self.data[index]
    
    def __setitem__(self, index, value):
        self.data[index] = value

    def __len__(self):
        return len(self.data)
    
    def __str__(self):
        return str(self.data)
    
    def __repr__(self):
        return repr(self.data)
        
    def copy(self):
        #copy SGarray properties
        ret = SGarray()
        ret.data = self.data.copy()
        ret.shape = self.shape
        ret.statistic = self.statistic
        ret.is_decorated = self.is_decorated
        return ret
    
    def getlist(self):
        return [ elem.getlist() for elem in self ]
    
    def sorted_and_clean(self,cutoff=1.0e-15):
        #copy SGarray properties
        temp = sorted([ element.getlist() for element in self if np.abs(element.value)>cutoff])
        ret = SGarray([ SGarrayElement(coords,value) for [coords,value] in temp ])
        ret.shape = self.shape
        ret.statistic = self.statistic
        ret.is_decorated = self.is_decorated
        return ret
    
    def __add__(self, other):
        if(self.shape!=other.shape or self.statistic!=other.statistic):
            print("objects' shape or statistics are not consistent!")
        
        self_list = sorted(self.getlist())
        other_list = sorted(other.getlist())
        ret = self_list.copy()
        
        def coordlist(elem_list):
            return [ coords for [coords,value] in elem_list]
            
        for [other_coords,other_value] in other_list:
            ret_coord = coordlist(ret)
            if other_coords in ret_coord:
                index = ret_coord.index(other_coords)
                ret[index][1] += other_value
            else:
                ret.append([other_coords,other_value])
        
        #copy SGarray properties
        ret = SGarray([ SGarrayElement(elem[0],elem[1]) for elem in ret])
        ret.shape = self.shape
        ret.statistic = self.statistic
        ret.is_decorated = self.is_decorated
        
        return ret
    
    def __sub__(self, other):
        if(self.shape!=other.shape or self.statistic!=other.statistic):
            print("objects' shape or statistics are not consistent!")
        
        self_list = sorted(self.getlist())
        other_list = sorted(other.getlist())
        ret = self_list.copy()
        
        def coordlist(elem_list):
            return [ coords for [coords,value] in elem_list]
            
        for [other_coords,other_value] in other_list:
            ret_coord = coordlist(ret)
            if other_coords in ret_coord:
                index = ret_coord.index(other_coords)
                ret[index][1] += -other_value
            else:
                ret.append([other_coords,-other_value])
        
        #copy SGarray properties
        ret = SGarray([ SGarrayElement(elem[0],elem[1]) for elem in ret])
        ret.shape = self.shape
        ret.statistic = self.statistic
        ret.is_decorated = self.is_decorated
        
        return ret
    
    def __mul__(self, other):
        ret = self.copy()
        for elem in ret:
            elem.value *= other
        return ret
        
    def __rmul__(self, other):
        return self*other
        
    def decorated(self):
        #multiply sign factor sigma[i] to every conjugated indices i
        ret = self.copy()
        for element in ret.data:
            coords = element.coords
            sgn_value = 1
            for i,ind in enumerate(coords):
                if(ret.statistic[i]==-1):
                    sgn_value *= param.sgn[ind]
            element.value *= sgn_value
        ret.is_decorated = not self.is_decorated
        return ret
    
    def rearrange(self,string=""):
        # rearrange the indices based on the given string
        # The input string must be of the form 'abcd->badc'
        
        #check if the string is bugged or not
        if(string.count("->")!=1):
            print("error[SGarray.rearrange]: The input string must be of the form 'abcd->badc'.")
            exit()
        
        [before,after] = list(string.split("->"))
        
        sorted_before = ''.join(sorted(list(before)))
        sorted_after = ''.join(sorted(list(after)))
        
        if(sorted_before!=sorted_after):
            print("error[SGarray.rearrange]: Make sure that the rearrangement instruction is correct.")
            exit()
        
        #now make the dictionary
        dictionary = []
        for i,char in enumerate(before):
            #print(i,after.index(char))
            dictionary.append([after.index(char),i])
        dictionary = sorted(dictionary)
        
        
        temp = self.copy()
        for element in temp:
            parity_list = [ param.gparity[ind] for ind in element.coords ]
            sgn = dgTN.relative_parity(string,parity_list)
            element.coords = tuple([ element.coords[f] for [i,f] in dictionary])
            element.value *= sgn
            
        #swap shape and statistic as well
        #copy SGarray properties (this line is for easy search)
        temp.shape = tuple([ self.shape[f] for [i,f] in dictionary])
        temp.statistic = tuple([ self.statistic[f] for [i,f] in dictionary])
        
        return temp
        
    def reshape(self,string=""):
        # The input string must be of the form 'abcdr->ab,cde'
        # Plan this carefully
            
        return 1
        
        
def make_SGarray(array_object):
    #transform array or DGarray to SGarray
    iterator = np.nditer(array_object, flags=['multi_index'])
    ret = SGarray([ SGarrayElement(iterator.multi_index,element.item()) for element in iterator])
    ret.shape = array_object.shape
    ret.is_decorated = array_object.is_decorated
    if(type(array_object)==dgTN.DGarray):
        ret.statistic = array_object.statistic
    return ret

def make_DGarray(SGarray_object):
    #transform SGarray to DGarray
    if(len(SGarray_object)==0):
        ret = dgTN.DGarray(np.zeros(list(SGarray_object.shape),dtype=float))
    else:
        ret = dgTN.DGarray(np.zeros(list(SGarray_object.shape),dtype=type(SGarray_object[0])))
    ret.statistic = SGarray_object.statistic
    ret.is_decorated = SGarray_object.is_decorated
    for element in SGarray_object:
        ret[element.coords] = element.value
    return ret

