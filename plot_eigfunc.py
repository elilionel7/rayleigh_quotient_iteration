import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_eigenfunctions(u_computed, u_analytic, gdata, filename=None):
    
    m = gdata.m
    u_comp_grid = u_computed.reshape(m, m)
    u_ana_grid = u_analytic.reshape(m, m)
    
    # Create figure
    fig = plt.figure(figsize=(16, 6))
    
    # Plot computed solution
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
    
    # Save or show
    if filename:
        plt.savefig(filename, bbox_inches='tight', dpi=300)
    else:
        plt.show()
    plt.close()
