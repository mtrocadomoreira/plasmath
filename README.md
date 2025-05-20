
# Plasmath

Plasmath is a Python module for plasma physics calculations and unit handling. It provides a set of functions and utilities to work with plasma-related quantities, normalization, and various plasma profiles.

## Features

- Unit handling using [the `pint` library](https://pint.readthedocs.io/en/stable/)
- Plasma physics calculations (e.g., plasma frequency, skin depth)
- Beam-related calculations (e.g., beam frequency, betatron frequency)
- Normalization functions for various plasma quantities
- Profile generation functions (Gaussian, cylindrical Gaussian, cosine) with symbolic variables using [the `sympy` library](https://www.sympy.org/en/index.html)
- Utility functions for numerical simulations

## Installation

> [!IMPORTANT] 
> Plasmath hasn't been released as a package yet. 
> Please note that it is still undergoing active development and may therefore lack documentation and experience frequent breaking changes. 


Plasmath can be installed using the following command:
```bash
pip install git+https://github.com/mtrocadomoreira/plasmath.git
```

# Usage

Here are some examples of how to use Plasmath:


```python
import plasmath as plm

# Get the unit registry 
# (see pint documentation for more information)
u = plasmath.get_ureg()

# Calculate plasma frequency
n0 = 1e18 * u.cm**-3
omega_p = plasmath.plasma_frequency(n0)
print(f"Plasma frequency: {omega_p}")

# Calculate plasma skin depth
skin_depth = plasmath.plasma_skin_depth(n0)
print(f"Plasma skin depth: {skin_depth}")

# Normalize a quantity
length = 1 * u.mm
normalized_length = plasmath.norm(length, n0)
print(f"Normalized length: {normalized_length}")

# Generate a Gaussian profile
profile, var = plasmath.profile_gauss(sig=0.1*u.mm, xc=0.0, var='x')
print(f"Gaussian profile: {profile}")
```

# Main functions

* `plasma_frequency(n0)`: Calculate plasma frequency
* `plasma_skin_depth(n0)`: Calculate plasma skin depth
* `beam_frequency(nb0, Mb, qb)`: Calculate beam frequency
* `betatron_frequency(nb0, Mb, qb, gamma)`: Calculate betatron frequency
* `norm(quant, n0, ...)`: Normalize plasma quantities
* `profile_gauss(sig, xc, var)`: Generate Gaussian profile
* `profile_gauss_rcyl(sig, var)`: Generate cylindrical Gaussian profile
* `profile_cos(sig, xc, var, sigcut)`: Generate cosine profile

# Contributing

Contributions to Plasmath are welcome! Please submit pull requests or open issues on the project's GitHub repository.

# License

[Add license information here]
