
import numpy as np
import math
import sparse as sp
import grassmanntn as gtn
import sympy
import time
import sys

char_list = (
    "a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"
    ,"A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"
    ,"α","β","Γ","γ","Δ","δ","ε","ζ","η","Θ","θ","ι","κ","λ","μ","ν","Ξ","ξ","Π","π","ρ","Σ","σ","ς","τ","υ","Φ","ϕ","φ","χ","Ψ","ψ","Ω","ω"
    )


####################################################
##                 Error messages                 ##
####################################################

error_message = {
    "invalid multiplication": "Invalid grassmann number multiplication.",
    "invalid addition": "Invalid grassmann number addition.",
    "invalid division": "Grassmann numbers can only be divided by a numpy scalar or a sympy symbolic.",
    "invalid power": "The power must be integer only.",
    "invalid exponentiation": "The base must be of type numpy scalar or a sympy symbolic.",
    "invalid integral": "Invalid application of the Berezin integral.",
    "invalid basis": "The basis provided is not consistent with the object's basis.",
    "contact me": "You are not supposed to see this message. Please contact me via\nAtis Yosprakob <yosprakob2-at-gmail.com>"
}

####################################################
##                Random Utilities                ##
####################################################

def exterior(term):
    term_truncate = term.copy()
    while term_truncate.count("")>0:
        term_truncate.remove("")
    return "^".join(term_truncate)

def get_char(string):
    #generate a new character that is not inside <string>
    for char in char_list:
        if char not in string:
            return char
    error("Error[get_char]: Running out of index character!")    

def isxscalar(object):
    # is extended scalar
    return np.isscalar(object) or isinstance(object, sympy.Expr)

def iszero(object,cutoff=1.0e-15):
    if not isxscalar(object):
        return False
    elif np.isscalar(object):
        return np.abs(object)<cutoff
    else:
        return object==0

def show_progress(step_inp,total_inp,process_name = "", ratio=True, color="blue", time=0):

    if gtn.progress_bar_enabled:

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

    if gtn.progress_bar_enabled:
        print("\r",end="")
        for i in range(90):
            print(" ",end="")
        print("\r",end="")
        return 1

def progress_space():
    if gtn.progress_bar_enabled:
        print()

def tab_up():
    if gtn.progress_bar_enabled:
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

####################################################
##                grassmann_number                ##
####################################################

def set_ac(*args):
    # this should be used instead of grassmann_number() for convenience
    return set_anticommutative(*args)

def set_anticommutative(*args):
    # this should be used instead of grassmann_number() for convenience

    # convert args into a list first
    if type(args)==tuple and len(args)==1:
        if type(args[0])==str:
            arr = [args[0]]
        else:
            arr = list(args[0])
    else:
        arr = list(args)

    if len(arr)==1:
        return grassmann_number(np.array([0,1]),arr)
    else:
        ret = []
        for elem in arr:
            ret += [set_anticommutative(elem)]
        ret = np.array(ret)
        return ret

def is_generator(gnum):
    # check if it as the grassmann generator
    return (type(gnum)==grassmann_number
            and len(gnum.basis)==1
            and iszero(gnum.data[0]) 
            and iszero(gnum.data[1]-1) )

class grassmann_number:

    # TO DO:
    # __neg__
    # __add__
    # __radd__
    # __sub__
    # __rsub__
    # __mul__
    # __rmul__
    # __pow__
    # __rpow__
    # __truediv__
    # __str__
    # __repr__


    def __init__(self, data=None, generator=None):
        
        #properties
        self.data = data     #a dense vector of size 2^nbits
        self.generator = generator.copy()   #a list of symbols for each bit
        
        if int(math.log2(data.size)) != len(generator) :
            print("in <grassmann_number>: the number of bits of the input are not consistent.")
            exit()
            
    @property
    def size(self):
        return self.data.size

    @property
    def basis(self):
        # I do this for data protection
        # the generator should not be changed willy-nilly
        return self.generator.copy()

    @property
    def terms(self):
        ret = []
        for coord,val in enumerate(self.data):
            
            # obtain the bit representation of the coordinate
            binary = ('{0:0'+str(self.nbits)+'b}').format(coord)[::-1]

            # convert to an array of symbols
            basis = []
            for i,b in enumerate(binary):
                if b=="1" :
                    basis += [self.basis[i]]
            if basis == []:
                basis = ["1"]

            ret += [basis]

        return ret

    @property
    def nbits(self):
        return len(self.basis)

    def copy(self):
        cdata = self.data.copy()
        cgenerator = self.basis
        return grassmann_number(cdata,cgenerator)
    
    def compactify(self):
        # remove unused generators
        ret = 0
        for i,term in enumerate(self.terms):
            data = self.data[i]
            if iszero(data) :
                continue
            basis = 1
            for g in term:
                basis*=set_ac(g)
            ret += data*basis
        return ret

    def simplify(self):
        ret = self.copy()
        ret.data = np.array([ sympy.simplify(coeff) for coeff in self.data])
        return ret

    
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
        
    def __add__(self, other):

        ret = self.copy()

        if isxscalar(other):
            # adding the scalar
            if isinstance(other, sympy.Expr):
                ret.data = ret.data*sympy.simplify(1)
            ret.data[0] += other
            return ret
        elif type(other) == grassmann_number:

            if self.basis == other.basis :
                ret.data = self.data+other.data
                return ret
            else:
                newgenerator = list(set(list(self.basis)+list(other.basis)))
                newgenerator.sort()
                new_nbits = len(newgenerator)
                newdata  = list([0]*(2**new_nbits))

                self_terms  = self.terms
                other_terms = other.terms
                self_data   = self.data
                other_data  = other.data

                for ind, term in enumerate(self_terms):
                    data = self_data[ind]
                    coord = sum([ (ng in term)*(2**i) for i,ng in enumerate(newgenerator) ])
                    newdata[coord] += data

                for ind, term in enumerate(other_terms):
                    data = other_data[ind]
                    coord = sum([ (ng in term)*(2**i) for i,ng in enumerate(newgenerator) ])
                    newdata[coord] += data

                newdata = np.array(newdata)

                return grassmann_number(newdata,newgenerator)

        else:
            print(error_message["invalid addition"])
            exit()
        
    def __mul__(self, other):
        
        if isxscalar(other):
            # multiplying the scalar
            ret = self.copy()
            ret.data = self.data*other
            return ret
        elif type(other) == grassmann_number:

            newgenerator = list(set(list(self.basis)+list(other.basis)))
            newgenerator.sort()
            new_nbits = len(newgenerator)
            newdata  = list([0]*(2**new_nbits))

            self_terms  = self.terms
            other_terms = other.terms
            self_data   = self.data
            other_data  = other.data

            s0 = time.time()
            s00 = s0
            ind=0
            indmax = len(self_terms)*len(other_terms)
            progress_space()

            for ind1, term1 in enumerate(self_terms):
                data1 = self_data[ind1]
                for ind2, term2 in enumerate(other_terms):

                    if time.time()-s0 > 2 :
                        show_progress(ind,indmax,process_name = "multiplication",ratio = False,color="green",time=time.time()-s00)
                        s0 = time.time()
                    ind+=1

                    data2 = other_data[ind2]

                    # check if it is zero or not
                    iszero = False
                    for gen in term1:
                        iszero = iszero or (gen in term2 and gen!="1")

                    if iszero :
                        continue

                    unsorted_term = list(term1)+list(term2)
                    while unsorted_term.count("1")>0:
                        unsorted_term.remove("1")
                    sorted_term = unsorted_term.copy()
                    sorted_term.sort()

                    # now that we have two lists, time to compute the sign factor

                    # first create two associated strings
                    nchar = len(unsorted_term)
                    unsorted_char = ""
                    for i in range(nchar):
                        unsorted_char += get_char(unsorted_char)

                    sorted_char = "".join(reordering(unsorted_term,sorted_term,unsorted_char))

                    sign = relative_sign_redux(unsorted_char,sorted_char)

                    #print(unsorted_term,"-->",sorted_term)
                    #print(sign)
                    #print()

                    data = data1*data2*sign
                    term = sorted_term
                    coord = sum([ (ng in term)*(2**i) for i,ng in enumerate(newgenerator) ])
                    newdata[coord] += data

            clear_progress()
            tab_up()

            newdata = np.array(newdata)

            return grassmann_number(newdata,newgenerator)
        else:
            print(error_message["invalid multiplication"])
            exit()
        
    def __rmul__(self, other):
        if isxscalar(other):
            # multiplying the scalar
            ret = self.copy()
            ret.data = self.data*other
            return ret
        elif type(other) == grassmann_number:
            print(error_message["contact me"])
            exit()
        else:
            print(error_message["invalid multiplication"])
            exit()
    
    def __truediv__(self, other):
        
        if isxscalar(other):
            # dividing the scalar
            ret = self.copy()
            ret.data = self.data/other
            return ret
        else:
            print(error_message["invalid division"])
            exit()
            
    def __pow__(self,other):
        if type(other)!=int or other<0:
            print(error_message["invalid power"])
            exit()
        ret = 1
        for i in range(other):
            ret *= self
            if iszero(ret) :
                break
        
        return ret
    
    
    def __rpow__(self,other):
        if not isxscalar(other):
            print(error_message["invalid exponentiation"])
            exit()
        if isinstance(other, sympy.Expr):
            return exp(sympy.log(other)*self)
        else:
            return exp(np.log(other)*self)
        
    def __str__(self):
        ret = []
        for coord,val in enumerate(self.data):
            
            # obtain the bit representation of the coordinate
            binary = ('{0:0'+str(self.nbits)+'b}').format(coord)[::-1]

            # convert to an array of symbols
            generator = exterior([ self.basis[i] if b=="1" else "" for i,b in enumerate(binary) ])
            if iszero(val) :
                continue
            if generator == "":
                #generator = "1"
                ret += [str(val)]
            else:
                ret += ["("+str(val)+")*"+generator]
        ret = " + ".join(ret)
        if ret=="":
            ret="0"

        if len(ret)>100:
            ret = ret.replace(" + ","\n + ")

        return ret
    
    def __repr__(self):
        return self.__str__()
    
    def is_even(self):
        
        for coord,val in enumerate(self.data):
            
            if iszero(val) :
                continue

            # obtain the bit representation of the coordinate
            binary = ('{0:0'+str(self.nbits)+'b}').format(coord)[::-1]

            # count
            parity = 0
            for i,b in enumerate(binary):
                if b=="1" :
                    parity+=1

            if parity%2==1 :
                return False

        return True

    def is_odd(self):
        
        for coord,val in enumerate(self.data):
            
            if iszero(val) :
                continue

            # obtain the bit representation of the coordinate
            binary = ('{0:0'+str(self.nbits)+'b}').format(coord)[::-1]

            # count
            parity = 0
            for i,b in enumerate(binary):
                if b=="1" :
                    parity+=1

            if parity%2==0 :
                return False

        return True

    def is_grassmann(self):
        
        for coord,val in enumerate(self.data):
            
            if iszero(val) :
                continue

            # obtain the bit representation of the coordinate
            binary = ('{0:0'+str(self.nbits)+'b}').format(coord)[::-1]

            # count
            parity = 0
            for i,b in enumerate(binary):
                if b=="1" :
                    parity+=1

            if parity>0 :
                return True

        return False

    def get_coeff(self,basis=None):
        # get a list of nonzero coefficients (still partially symbolic!)

        if basis==None:
            basis = self.basis

        l1 = list(self.basis)
        l2 = list(basis)
        l1.sort()
        l2.sort()

        if l1 != l2:
            print(error_message["invalid basis"])
            exit()

        coeff = []
        all_coord = [ [] for b in basis]
        for coord,val in enumerate(self.data):
            
            if iszero(val) :
                continue

            # obtain the bit representation of the coordinate
            binary = ('{0:0'+str(self.nbits)+'b}').format(coord)[::-1]
            parity = [ 1 if c=='1' else 0 for c in binary]
            reordered_parity = reordering(self.basis,basis,parity)

            # convert the basis to ints:
            obj1 = [ i for i in range(len(basis))]
            obj2 = [ self.basis.index(c) for c in basis]

            coeff += [val*relative_sign_int(obj1,obj2,parity)]

            for loc, p in enumerate(reordered_parity):
                all_coord[loc] += [p]

        coeff = np.array(coeff)

        return coeff, all_coord

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

def reordering(lista,listb,mylist):

    rC = []
    for b in listb:
        
        # locate the location of b in the original list
        index = lista.index(b)

        # add mylist's element at this index to the list
        rC += [mylist[index]]

    return rC

def relative_sign_redux(string1, string2):
    """
    give two strings, and it will compute the relative sign
    """
    combined_string = string1+"->"+string2
    parity = [1]*len(string1)
    return relative_sign_single_input(combined_string,parity)


####################################################
##                Berezin Calculus                ##
####################################################

def d(varlist=[], reverse=False):
    return berezin_measure(varlist,reverse)

class berezin_measure:
    def __init__(self, varlist=[], reverse=False):
        
        #properties
        self.varlist = []
        
        if type(varlist)==grassmann_number:
            #check if it is a single number
            if is_generator(varlist):
                self.varlist += [varlist]
        elif type(varlist)==list or type(varlist)==np.ndarray:
            
            npvarlist = np.array(varlist)
            if reverse:
                npvarlist = npvarlist[::-1]
            
            all_are_singlevar = True
            for gnum in npvarlist:
                all_are_singlevar = all_are_singlevar and is_generator(gnum)
            if all_are_singlevar:
                self.varlist += list(npvarlist)
    
    def __str__(self):
        display=[]
        for dg in self.varlist:
            generator = dg.basis
            if generator == ["1"]:
                continue
            this_display = ""
            if "1" in generator:
                this_display += generator[1]
            else:
                this_display += generator[0]
            display += ["∫"+this_display+" "]
        display = "".join(display)
        return display
    
    def __repr__(self):
        return self.__str__()
    
    def __mul__(self,other):
        
        # self*other
        # allowed cases: other = grassmann_number or scalar
    
        if (type(other)!=grassmann_number
                and type(other)!=berezin_measure
                and not isxscalar(other)):
            print(error_message["invalid integral"])
            exit()
        
        if type(other)==berezin_measure :
            new_varlist = list(self.varlist)+list(other.varlist)
            return berezin_measure(new_varlist)
        

        # below is the computation of the d on grassmann numbers

        integrand = other
        
        #if the d is repeating, return zero
        for dx in self.varlist:
            if self.varlist.count(dx)>1:
                return 0
        
        ret = 0
        data = integrand.data
        indmax = len(integrand.terms)


        s0 = time.time()
        s00 = s0
        progress_space()
        
        for ind, term in enumerate(integrand.terms):

            if time.time()-s0 > 2 :
                show_progress(ind,indmax,process_name = "Berezin integral",ratio = False,color="blue",time=time.time()-s00)
                s0 = time.time()

            coeff = data[ind]
            if coeff == 0:
                continue
            symblist = [ var.basis[0] for var in self.varlist ]

            #check if this term is zero
            iszero = False
            for var in symblist:
                iszero = iszero or (var not in term)

            if iszero:
                continue

            nd = len(symblist)
            sign=1
            for id in range(len(symblist)):

                iloc = nd-id-1
                measure = symblist[iloc]
                sign*=(-1)**term.index(measure)

                term.remove(measure)
                #print(symblist[:iloc],term,sign*coeff)

            this_term = sign*coeff
            for symb in term:
                this_term*=set_ac(symb)
            ret+=this_term

        clear_progress()
        tab_up()

        return ret
            
####################################################
##              Arithmetic Functions              ##
####################################################

def exp(gnum):
    
    # Extract the non-Grassmann part first!

    nbits = gnum.nbits
    
    C = gnum.data[0]
    X = gnum-C
    #compute exp(C) * exp(X)
    
    ret = 1    # the first term of the expansion
    Xp=1       # X^p
    pfact=1    # p!
    
    for pm1 in range(nbits):
        p = pm1+1
        pfact*=p
        Xp=Xp*X
        ret=ret+Xp/pfact
    if isinstance(C, sympy.Expr):
        ret = ret*sympy.exp(C)
    else:
        ret = ret*np.exp(C)
    
    return ret