# Mann‑Inexact Solver for Nonlinear Volterra Integral Equations

This repository contains a high‑performance C++ implementation of the adaptive Mann‑inexact solver with Anderson acceleration for nonlinear Volterra integral equations with one‑sided Lipschitz kernels, as described in the paper:

> Edem, I. D., Udoaka, O. G., Akpakwa, J. I., [Collaborator]. (2026).  
> *A Computationally Efficient Fixed‑Point Framework for Nonlinear Volterra Integral Equations...*

## Dependencies
- C++17 compiler (g++ 7+ or clang 5+)
- [Eigen3](https://eigen.tuxfamily.org/) (header‑only)
- (Optional) MPI and OpenMP for parallel execution

## Building
```bash
mkdir build && cd build
cmake .. -DEIGEN3_INCLUDE_DIR=/path/to/eigen
make
