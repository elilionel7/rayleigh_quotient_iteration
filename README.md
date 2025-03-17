# Rayleigh Quotient Iteration Project

This project implements Rayleigh Quotient Iteration to compute eigenfunctions and eigenvalues for the Laplace operator on a circular domain with Dirichlet boundary conditions, using a spectral grid.

## Key Files

- **`exmp1.py`**: Main script to run the iteration. Sets up the grid, computes eigenfunctions, and prints results (eigenvalues, L2 errors, orthogonality).
- **`rayleigh_operator.py`**: Defines the `RayleighOperator` class, handling the operator construction, QR solve, and iteration logic.
- **`grid_data2.py`**: Generates the spectral grid and preconditioner data.
- **`plot_eigfunc.py`**: Provides plotting functionality for eigenfunctions.

## Prerequisites

- Python 3.x
- Required libraries: `numpy`, `scipy`

Install dependencies:

```bash
pip install -r requirements.txt
```

Run file

```run
python exmp.py
```
