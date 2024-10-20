# griddata.py

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "SEEM_Chebyshev_master"))

from SEEM_Chebyshev_master.grid_data2 import grid_data
import numpy as np


class GridDataWithWeights(grid_data):
    def __init__(
        self, m, bdry, l, precond=True, order="cubic", density=1, bc="dirichlet"
    ):
        super().__init__(m, bdry, l, precond, order, density, bc)

        self.nr_quad = 4
        self.ntheta_quad = 6
        
        # Radial nodes and weights
        self.r_quad, self.w_r_quad = self.radial_nodes_and_weights(self.nr_quad, r0=0.0, r1=1.0)

        # Angular nodes and weights
        self.theta_quad = self.get_trapezoidal_nodes(self.ntheta_quad, a=0.0, b=2 * np.pi)
        self.w_theta_quad = self.trapezoidal_weights(self.ntheta_quad)

        # 2D quadrature grid
        self.R_quad, self.Theta_quad = np.meshgrid(self.r_quad, self.theta_quad, indexing='ij')
        self.x_nodes_quad = self.R_quad * np.cos(self.Theta_quad)
        self.y_nodes_quad = self.R_quad * np.sin(self.Theta_quad)

        # Flatten quadrature nodes
        self.eval_xi = self.x_nodes_quad.flatten()
        self.eval_yi = self.y_nodes_quad.flatten()
                
        # quadrature weights
        self.w2d_quad = np.outer(self.w_r_quad, self.w_theta_quad)
        self.weights = self.w2d_quad.flatten()

        # Adjust weights for Jacobian 
        self.weights *= self.R_quad.flatten()

        def trapezoidal_weights(self, n, a=0.0, b=2 * np.pi):
    
            h = (b - a) / (n - 1)
            w = np.full(n, h)
            w[0] = w[-1] = h / 2
            return w


        # def radial_nodes_and_weights(nr, r0=0.0, r1=1.0):
        #     r = np.linspace(r0, r1, nr)
        #     w_r = np.full(nr, (r1 - r0) / (nr - 1))
        #     w_r[0] = w_r[-1] = w_r[0] / 2
        #     return r, w_r  #gauss instead

        def radial_nodes_and_weights(self, nr, r0=0.0, r1= 0.95):
            nodes, weights = leggauss(nr)

            r = 0.5 * (nodes + 1) * (r1 - r0) + r0  
            w_r = 0.5 * (r1 - r0) * weights

            return r, w_r


        def get_trapezoidal_nodes(self, n, a=0.0, b=2 * np.pi):
            h = (b - a) / (n - 1)
            x = np.linspace(a, b, n)
            return x


