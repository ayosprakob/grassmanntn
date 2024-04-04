# GrassmannTN
This is a package that aims to simplify the coding involving Grassmann tensor network.
I try to make the coding as convenient as when you use other standard packages like numpy or scipy.
All the sign factors are computed automatically without explicit user inputs.

## Useful links
- [documentation](https://ayosprakob.github.io/grassmanntn/)
- [PyPI](https://pypi.org/project/grassmanntn/)
- [full paper](https://scipost.org/SciPostPhysCodeb.20/pdf)
- [arXiv preprint](https://doi.org/10.48550/arXiv.2309.07557)

## Prerequisites

- [numpy](https://numpy.org/doc/stable/index.html)
- [sparse](https://sparse.pydata.org/en/stable/)
- [opt_einsum](https://optimized-einsum.readthedocs.io/en/stable/)
- [sympy](https://www.sympy.org/)

## Installation
```
pip install grassmanntn
```

Once the package is installed, download [example.py](https://github.com/ayosprakob/grassmanntn/blob/main/example.py) and try running it with
```
python3 example.py --show_progress
```
An example code of a one-flavor two-dimensional $`\mathbb{Z}_2`$ gauge theory should be able to run.

## Examples
Please see [documentation](https://ayosprakob.github.io/grassmanntn/).
