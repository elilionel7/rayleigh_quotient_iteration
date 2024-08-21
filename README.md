<!-- README.md -->
# README

## PDE
 solving the Laplace equation \(-\Delta u = 0\) with boundary conditions \(u = 1\) 


## Scripts

### 1. `exmp1_v1.py`

- **Description**: 
  - This script solves the Laplace equation \(-\Delta u = 0\) with boundary condition \(u = 1\) using the `qr_solve2` method from the original code.


### 2. `exmp1_v2.py`

- **Description**: 
  - This script also solves the Laplace equation \(-\Delta u = 0\) with boundary condition \(u = 1\), but it uses the Rayleigh Quotient Iteration method with qr_solve2 to approx the solution of the pde. 

### 3. `rayleigh_operator.py`

- **Description**: 
  - This script defines the `RayleighOperator` class, inherits the `operator_data` class from the original code. It has the `rayleigh_quotient_iteration` method, which is used to solve PDEs by iteratively solving the pde through Rayleigh Quotient Iteration by calling the qr_solve2 to update u.


1. **Run `exmp1_v1.py`**: 
   - Command: `python exmp1_v1.py`

2. **Run `exmp1_v2.py`**:
   - Command: `python exmp1_v2.py`

