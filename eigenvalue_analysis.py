import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import LinearOperator, eigs
from scipy.special import jv, jn_zeros
from rayleigh_operator import RayleighOperator
from grid_data2 import grid_data

j0 = jn_zeros(0, 1)[0]  

def exact_eigenfunc(x, y, mode=1):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    if mode == 1:
        j0_k = jn_zeros(0, 1)[0]
        return jv(0, j0_k * r / 0.95)
    elif mode == 2:
        j1_k = jn_zeros(1, 1)[0]
        return jv(1, j1_k * r / 0.95) * np.cos(theta)
    elif mode == 3:
        j1_k = jn_zeros(1, 1)[0]
        return jv(1, j1_k * r / 0.95) * np.sin(theta)


def plot_boundary_check(u, gdata, title="Boundary Check"):
    u_grid = np.abs(u.reshape(gdata.m, gdata.m))
    mask = ~gdata.flag
    plt.figure()
    plt.contourf(gdata.x1, gdata.x2, u_grid * mask, levels=50)
    plt.colorbar()
    plt.title(title)
    plt.show()


def compute_eigenvalues(operator, gdata, k=3):
    size_interior = gdata.k

    def matvec(u_interior):
        u_full = np.zeros((gdata.m, gdata.m))
        u_full[gdata.flag] = u_interior

        Au_full_grid = operator.lap(u_full) + operator.lap(u_full.T).T
        Au_interior = Au_full_grid[gdata.flag]

        return Au_interior

    A = LinearOperator((size_interior, size_interior), matvec=matvec)

    try:
        eigenvalues, eigenvectors = eigs(A, k=k, which='SM', tol=1e-6)
        return eigenvalues.real, eigenvectors.real
    except Exception as e:
        print(f"eigs failed: {e}")
        return np.array([]), np.array([])
def compute_l2_error(u_computed, u_exact, operator):
    u_diff = u_computed - u_exact
    return np.sqrt(operator.inner_product(u_diff, u_diff))

def plot_eigenfunctions(u_computed, u_exact, gdata, mode, eigenvalue, l2_error):
    fig = plt.figure(figsize=(12, 5))
    
    ax1 = fig.add_subplot(121, projection='3d')
    u_comp_grid = u_computed.reshape(gdata.m, gdata.m)
    u_comp_grid[~gdata.flag] = 0  # enforce boundary explicitly
    print(f"Boundary values norm (Computed Mode {mode}):", 
          np.linalg.norm(u_comp_grid[~gdata.flag]))
    surf1 = ax1.plot_surface(gdata.x1, gdata.x2, u_comp_grid.real, cmap='viridis')
    ax1.set_title(f'Computed Eigenfunction (Mode {mode}, λ={eigenvalue.real:.4f})')
    
    ax2 = fig.add_subplot(122, projection='3d')
    u_exact_grid = u_exact.reshape(gdata.m, gdata.m)
    u_exact_grid[~gdata.flag] = 0
    print(f"Boundary values norm (Exact Mode {mode}):", 
          np.linalg.norm(u_exact_grid[~gdata.flag]))
    surf2 = ax2.plot_surface(gdata.x1, gdata.x2, u_exact_grid.real, cmap='viridis')
    ax2.set_title(f'Exact Bessel Eigenfunction (Mode {mode}, L2 Error={l2_error.real:.2e})')
    
    plt.tight_layout()
    plt.show()

def main():
    pts = 50
    R = 0.95
    l = 2
    num_eigfuncs = 3
    nr_quad = 20
    ntheta_quad = 60

    gdata = grid_data(
        pts,
        [lambda x: R * np.cos(x), lambda x: R * np.sin(x)],
        l,
        precond=True,
        order='spectral',
        nr_quad=nr_quad,
        ntheta_quad=ntheta_quad
    )
    operator = RayleighOperator(gdata, l=l)

    print("Computing eigenvalues with scipy.sparse.linalg.eigs...")
    eigenvalues_eigs, eigenvectors_eigs = compute_eigenvalues(operator, gdata, k=num_eigfuncs)

    print("\nRunning Rayleigh quotient iteration...")
    eigfuncs = []
    eigenvalues_rq = []
    for i in range(num_eigfuncs):
        mode = i + 1
        u0 = exact_eigenfunc(gdata.x1, gdata.x2, mode=mode)
        u0 = u0.flatten()
        u0 /= np.linalg.norm(u0)
        u, lambdaU, iterations = operator.rq_int_iter_eig(l, u0=u0, eigenfunctions=eigfuncs, tol=1e-6, max_iter=50)
        eigfuncs.append(u)
        eigenvalues_rq.append(lambdaU)
        print(f"Eigenfunction {i+1}: Eigenvalue={lambdaU:.4f}, Iterations={iterations}")

    print("\nComparing with exact Bessel solutions...")
    for i in range(num_eigfuncs):
        mode = i + 1
        u_computed = eigfuncs[i]
        eigenvalue = eigenvalues_rq[i]
        u_exact = exact_eigenfunc(gdata.x1, gdata.x2, mode=mode)
        j_zeros = jn_zeros(0, 1) if mode == 1 else jn_zeros(1, 1)
        exact_eigenvalue = (j_zeros[0] / R)**2
        l2_error = compute_l2_error(u_computed, u_exact.flatten(), operator)
        # plot_boundary_check(u_computed, gdata, title=f"Boundary Magnitude - Mode {mode}")


        plot_eigenfunctions(u_computed, u_exact, gdata, mode, eigenvalue, l2_error)
        relative_error, _ = operator.verify_eigenfunction(u_computed, l, eigenvalue)

        print(f"Mode {mode}:")
        print(f"  Computed Eigenvalue (RQ): {eigenvalue:.4f}")
        print(f"  Exact Eigenvalue: {exact_eigenvalue:.4f}")
        print(f"  L2 Error: {l2_error:.2e}")
        print(f"  Verification Relative Error: {relative_error:.2e}")
        if eigenvalues_eigs.size > 0:
            print(f"  eigs Eigenvalue: {eigenvalues_eigs[i]:.4f}")

if __name__ == "__main__":
    main()

