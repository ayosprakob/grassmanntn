# grassmanntn
This is a package that aims to simplify the coding involving Grassmann tensor network.
I try to make the coding as convenient as when you use other standard packages like numpy or scipy.
All the sign factors are computed automatically without explicit user inputs.

> **Note**
> The documentation page is under construction.

## Prerequisites

- [numpy](https://numpy.org/doc/stable/index.html)
- [sparse](https://sparse.pydata.org/en/stable/)
- [opt_einsum](https://optimized-einsum.readthedocs.io/en/stable/)

## Installation
(I'm not used to this. I plan to make installation easier in the future.)

Download `__init__.py`, `grassmanntn.py`, and `param.py` and put them in a folder called `grassmanntn`. Put this folder in the same directory as your Python project.

### Example: initialization
To initialize the tensor, you can import it from either a `numpy.array()` or `sparse.COO()` object. You can also choose your Grassmann tensor to be stored in a `dense` or `sparse` format:

```
import numpy as np
import grassmanntn as gtn

A_coeff = np.arange(4096).reshape(8,8,8,8)             # defining an 8x8x8x8 coefficient matrix
A_stats = (1,1,-1,-1)                                  # -1 is a conjugated index,
                                                       # +1 is a non-conjugated index.
                                                       # 0 is a bosonic index
A_dense  = gtn.dense(data=A_coeff, statistic=A_stats)  # defining a dense grassmann tensor
A_sparse = gtn.sparse(data=A_coeff, statistic=A_stats) # defining a sparse grassmann tensor
```
The two formats can be switched at any time:
```
A_dense = gtn.dense(A_sparse)
```
Every function in this package is agnostic to the tensor format.

### Example: contraction
Typically, when two tensors are contracted, appropriate sign factor must be multiplied due to the Grassmann anti-commutativity; e.g.,
for $`\mathcal{A}_{\bar\psi\phi}=\sum_{I,J}A_{IJ}\bar\psi^I\phi^J`$ and $`\mathcal{B}_{\eta\bar\phi}=\sum_{I,J}B_{IJ}\eta^I\bar\phi^J`$, we have
```math
\mathcal{C}_{\bar\psi\eta}=\int d\bar\phi d\phi e^{-\bar\phi\cdot\phi}\mathcal{A}_{\bar\psi\phi}\mathcal{B}_{\eta\bar\phi}=\sum_{I,J,K}(A_{IJ}B_{KJ}s_{JK})\bar\psi^I\eta^K
```
where, for $`I=(i_1,i_2,\cdots i_m)\in\{0,1\}^m`$, $`J=(j_1,j_2,\cdots j_n)\in\{0,1\}^n`$, and $`K=(k_1,k_2,\cdots k_l)\in\{0,1\}^l`$, we have to introduce the sign factor
```math
s_{JK}=\left(\prod_{a < b}(-)^{j_aj_b}\right)\times\left(\prod_{a,b}(-)^{j_ak_b}\right).
```

In the coding, we have to write the sign factor down explicitly---hardcoded. This is not a good idea if your algorithm involves a lot of contractions with many indices.

In this package, the contraction can be as simple as writing
```
C = gtn.einsum('IJ,KJ->IK',A,B)
```
where `A`, `B`, and `C` belong to a class of Grassmann tensor. A Grassmann tensor object contains every necessary information for the computation; e.g., the coefficient tensor and the statistic of the indices.

### Example: tensor decompositions
Tensor decomposition can also be done in an equally simple way. For example, if we want to decompose $`\mathcal{A}_{\psi_1\psi_2\bar\psi_3\bar\psi_4}`$ into $`\mathcal{X}_{\psi_1\psi_2\eta}`$ and $`\mathcal{Y}_{\bar\eta\bar\psi_3\bar\psi_4}`$, this can be done with a few lines of code:
```
U, Λ, V = A_dense.svd('ij,kl')
X = gtn.einsum( 'ija,ab -> ijb', U, gtn.sqrt(Λ))
Y = gtn.einsum( 'ab,bkl -> akl', gtn.sqrt(Λ), V)
```
You can also use the method `.eig()` instead of `.svd()` if your tensor is a Hermitian Grassmann matrix.

### Example: Hermitian conjugation
Hermitian conjugate of a Grassmann tensor can be computed using the `.hconjugate()` method
```
cU = U.hconjugate('ij,a')
```
Of course, you need to identify the index separation first (in this case, it is `ij` and `a`). Hermitian conjugate is only well-defined when the indices are separated into two groups, corresponding to the two matrix indices.
After the conjugation, the indices are now arranged as `aij`.

### Example: tensor renormalization group
I also wrote some built-in functions for tensor coarse-graining such as the Grassmann TRG or Grassmann ATRG methods.
```
T_trg = gtn.trg(T,dcut=64)                  # TRG method
T_atrgx = gtn.atrg2dx(T1,T2,dcut=64)        # anisotropic TRG along the x-axis
T_atrgy = gtn.atrg2dy(T1,T2,dcut=64)        # anisotropic TRG along the y-axis
```
