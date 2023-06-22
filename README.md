# grassmanntn
This is a package that aims to simplify the coding involving Grassmann tensor network.
I try to make the coding as convenient as when you use other standard packages like numpy or scipy.
All the sign factors are computed automatically without explicit user inputs.

### Examples
Typically, when two tensors are contracted, appropriate sign factor must be multiplied due to the Grassmann anti-commutativity; e.g.,
$$\mathcal{A}_{\bar\psi\phi}$$
