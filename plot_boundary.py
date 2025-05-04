import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import LinearOperator, eigs
from scipy.special import jn_zeros
from rayleigh_operator import RayleighOperator
from grid_data2 import grid_data

def plot_boundary_check(u, gdata, title="Boundary Check"):
    u_grid = np.abs(u.reshape(gdata.m, gdata.m))
    mask = ~gdata.flag
    plt.figure()
    plt.contourf(gdata.x1, gdata.x2, u_grid * mask, levels=50)
    plt.colorbar()
    plt.title(title)
    plt.show()
