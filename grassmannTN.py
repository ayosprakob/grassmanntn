import numpy as np
import math
from grassmannTN import param
import sparse as sp

####################################################
##             Densed Grassmann Array             ##
####################################################


class DGarray:
    def __init__(self, data=[], bosonic = False, encoding = "canonical"):
    
        #copy DGarray properties
        self.data = None
        self.statistic = None
        self.is_decorated = False
        self.encoding = encoding

        if(encoding!='canonical' and encoding!='parity-preserving'):
            print("error[DGarray]: Encoding must be either 'canonical' or 'parity-preserving' only.")
            exit()

        if(type(data)==np.array or type(data)==np.ndarray or type(data)==list):
            self.data = np.array(data)
            self.is_decorated = False
            if bosonic:
                self.statistic = tuple([ 0 for i in range(len(self.shape))])
            else:
                self.statistic = tuple([ 1 for i in range(len(self.shape))])
        elif(type(data)==SGarray):
            #copy DGarray properties
            self.data = data.data.todense()
            self.statistic = data.statistic
            self.is_decorated = data.is_decorated
            self.encoding = data.encoding
        elif(type(data)==DGarray):
            #copy DGarray properties
            self.data = data.data.copy()
            self.statistic = data.statistic
            self.is_decorated = data.is_decorated
            self.encoding = data.encoding
        elif(np.isscalar(data)):
            self.data = np.array(list([data]))
            self.is_decorated = False
            if bosonic:
                self.statistic = tuple([ 0 for i in range(len(self.shape))])
            else:
                self.statistic = tuple([ 1 for i in range(len(self.shape))])
        else:
            print("error[DGarray]: Invalid initialized data")
            exit()
    
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

    def display(self):
        print("  array type: dense")
        print("       shape:",self.shape)
        print("        size:",self.size)
        print("   statistic:",self.statistic)
        print("   decorated:",self.is_decorated)
        print("    encoding:",self.encoding)
        print("     entries:")
        iterator = np.nditer(self, flags=['multi_index'])
        for element in iterator:
            coords = iterator.multi_index
            print(coords,element.item())

    def copy(self):
        #copy DGarray properties
        ret = DGarray()
        ret.data = self.data.copy()
        ret.statistic = self.statistic
        ret.is_decorated = self.is_decorated
        ret.encoding = self.encoding
        return ret
    
    def __add__(self, other):
        if(self.shape!=other.shape or self.statistic!=other.statistic):
            print("error[DGarray.+]: Inconsistent shape or statistic")
            exit()
        ret = self.copy()
        ret.data = ret.data+other.data
        return ret
        
    def __sub__(self, other):
        if(self.shape!=other.shape or self.statistic!=other.statistic):
            print("error[DGarray.-]: Inconsistent shape or statistic")
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
        
    def decorated(self):
        #multiply sign factor sigma[i] to every conjugated indices i
        ret = self.copy()
        iterator = np.nditer(ret.data, flags=['multi_index'])
        for element in iterator:
            coords = iterator.multi_index
            sgn_value = 1
            for i,ind in enumerate(coords):
                if(ret.statistic[i]==-1):
                    sgn_value *= param.sgn[ind]
            ret[coords] *= sgn_value
        ret.is_decorated = not self.is_decorated
        return ret

    def switch_encoder(self):
        ret = self.copy()
        iterator = np.nditer(self.data, flags=['multi_index'])
        for element in iterator:
            coords = iterator.multi_index
            new_coords = []
            for i,ind in enumerate(coords):
                if(self.statistic[i]==1 or self.statistic[i]==-1):
                    new_coords += [param.encoder[ind]]
                else:
                    new_coords += [ind]
            new_coords = tuple(new_coords)
            ret.data[coords] = self.data[new_coords]
        if(ret.encoding=='canonical'):
            ret.encoding='parity-preserving'
        else:
            ret.encoding='canonical'
        return ret
        
####################################################
##            Sparse Grassmann arrays             ##
####################################################


class SGarray:
    def __init__(self, data=[], bosonic = False, encoding = "canonical"):
    
        #copy SGarray properties
        self.data = None
        self.statistic = None
        self.is_decorated = False
        self.encoding = encoding

        if(encoding!='canonical' and encoding!='parity-preserving'):
            print("error[SGarray]: Encoding must be either 'canonical' or 'parity-preserving' only.")
            exit()
        
        if(type(data)==np.array or type(data)==np.ndarray or type(data)==list):
            self.data  = sp.COO.from_numpy(np.array(data))
            self.is_decorated = False
            if bosonic:
                self.statistic = tuple([ 0 for i in range(len(self.shape))])
            else:
                self.statistic = tuple([ 1 for i in range(len(self.shape))])
        elif(type(data)==sp.COO):
            self.data  = data.copy()
            self.is_decorated = False
            if bosonic:
                self.statistic = tuple([ 0 for i in range(len(self.shape))])
            else:
                self.statistic = tuple([ 1 for i in range(len(self.shape))])
        elif(type(data)==DGarray):
            #copy SGarray properties
            self.data  = sp.COO.from_numpy(data.data)
            self.statistic = data.statistic
            self.is_decorated = data.is_decorated
            self.encoding = data.encoding
        elif(type(data)==SGarray):
            #copy SGarray properties
            self.data = data.data.copy()
            self.statistic = data.statistic
            self.is_decorated = data.is_decorated
            self.encoding = data.encoding
        else:
            print("error[SGarray]: Invalid initialized data")
            exit()
    
    #def __getitem__(self, index):
    #    return self.coords[index], self.value[index]
    
    #def __setitem__(self, index, coords_value):
    #    coords, value = coords_value
    #    self.data.data[index] = value
    #    for axis in range(len(self.data.shape)):
    #        self.data.coords = 

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

    def display(self):
        print("  array type: sparse")
        print("       shape:",self.shape)
        print("        size:",self.size)
        print("   statistic:",self.statistic)
        print("   decorated:",self.is_decorated)
        print("    encoding:",self.encoding)
        print("   non-zeros:",self.nnz)
        print("     entries:")
        for elem in range(self.nnz):
            print(self.coords[elem],self.value[elem])

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
        ret.is_decorated = self.is_decorated
        ret.encoding = self.encoding
        return ret
    
    def __add__(self, other):
        if(self.shape!=other.shape or self.statistic!=other.statistic):
            print("error[SGarray.+]: Inconsistent shape or statistic")
            exit()
        ret = self.copy()
        ret.data = ret.data+other.data
        return ret
        
    def __sub__(self, other):
        if(self.shape!=other.shape or self.statistic!=other.statistic):
            print("error[SGarray.-]: Inconsistent shape or statistic")
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
    
    def decorated(self):
        #multiply sign factor sigma[i] to every conjugated indices i
        ret = self.copy()
        for elem in range(self.nnz):
            coords = self.coords[elem]
            sgn_value = 1
            for i,ind in enumerate(coords):
                if(ret.statistic[i]==-1):
                    sgn_value *= param.sgn[ind]
            ret.data.data[elem] *= sgn_value
        ret.is_decorated = not ret.is_decorated
        return ret

    def switch_encoder(self):
        ret = self.copy()
        for elem in range(self.nnz):
            coords = self.coords[elem]

            new_coords = []
            for i,ind in enumerate(coords):
                if(self.statistic[i]==1 or self.statistic[i]==-1):
                    new_coords += [param.encoder[ind]]
                else:
                    new_coords += [ind]
            new_coords = tuple(new_coords)

            for ind in range(len(new_coords)):
                ret.data.coords[ind][elem] = new_coords[ind]
        if(ret.encoding=='canonical'):
            ret.encoding='parity-preserving'
        else:
            ret.encoding='canonical'
        return ret
