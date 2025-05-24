import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_eigenfunctions(u_computed, u_analytic, gdata, filename=None):
    
    m = gdata.m
    u_comp_grid = u_computed.reshape(m, m)
    u_ana_grid = u_analytic.reshape(m, m)
    
    fig = plt.figure(figsize=(16, 6))
    
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(gdata.x1, gdata.x2, u_comp_grid,
                           cmap='viridis', rstride=1, cstride=1)
    ax1.set_title("Computed Eigenfunction")
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
    
    # Plot analytic solution
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(gdata.x1, gdata.x2, u_ana_grid,
                           cmap='viridis', rstride=1, cstride=1)
    ax2.set_title("Analytic Eigenfunction")
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, bbox_inches='tight', dpi=300)
    else:
        plt.show()
    plt.close()


def plot_Au_surfaces(Au_list, gdata_list, pts_list, filename=None):
    
    n = len(Au_list)
    fig = plt.figure(figsize=(6 * n, 5))
    
    for i, (Au, gdata, pts) in enumerate(zip(Au_list, gdata_list, pts_list)):
        Au_full_grid = np.zeros_like(gdata.x1)
        
        Au_full_grid[gdata.flag] = Au[:gdata.k]

        ax = fig.add_subplot(1, n, i+1, projection='3d')
        surf = ax.plot_surface(gdata.x1, gdata.x2, Au_full_grid, cmap='viridis', rstride=1, cstride=1)
        ax.set_title(f"Au, pts={pts}")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    
    plt.tight_layout()
    if filename:
        plt.savefig(filename, bbox_inches='tight', dpi=300)
    else:
        plt.show()
    plt.close()
