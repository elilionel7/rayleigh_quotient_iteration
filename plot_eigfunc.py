import matplotlib.pyplot as plt
import numpy as np

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