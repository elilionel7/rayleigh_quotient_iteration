import numpy as np;
from matplotlib import pyplot as plt
from delta import dirac, dvdirac
from scipy.sparse import vstack as sparseVstack
from CSSEM import get_h
import scipy

class grid_data:
    def __init__(self,m,bdry,l,precond=True,order='cubic',density=1,bc='dirichlet'):
        self.m = m
#        Physical Grid
        self.x = np.cos((2*np.arange(self.m)+1)*np.pi/(2*self.m))
        self.x1, self.x2 = np.meshgrid(self.x,self.x,indexing='ij')
#        Frequency Grid
        self.fs = np.arange(self.m)*1.0
        self.fx, self.fy = np.meshgrid(self.fs,self.fs,indexing='ij')
#        Define the boundary. Currently implemented for star-shaped domains.
#        self.b, self.flag = self.createbdry(bound)
        bdry2 = [lambda x:np.arccos(bdry[0](x)), lambda x: np.arccos(bdry[1](x))];
        self.b, self.bn, self.g_grid  = createbdry(bdry2,density*np.pi/self.m*2)
        self.flag = self.createflag(bdry)
        self.p = np.shape(self.b)[0]
        self.k = np.sum(self.flag)
#        Interpolation Operator
        self.order=order
        dirichlet = []
        neumann = []
        for i in range(self.p):
            dirichlet.append(dirac(self.b[i,:],self.x,order=order))
            neumann.append(dvdirac(self.b[i,:].tolist(),self.bn[i,:].tolist(),self.x))
        dirichlet = tuple(dirichlet)
        neumann = tuple(neumann)
        if order == 'cubic':
            self.dirichlet = sparseVstack(dirichlet).tocsr()
        elif order == 'spectral':
            self.dirichlet = np.vstack(dirichlet)
            self.neumann = np.vstack(neumann)
        if bc == 'dirichlet':
            self.xx = self.dirichlet
        elif bc == 'neumann':
            self.xx = self.neumann
        elif bc == 'robin':
            self.xx = self.dirichlet + self.neumann
        self.xxT = self.xx.transpose()
        if precond == True:
    #        Create the preconditioner.
            self.flag2 = self.Flag()
            if bc == 'dirichlet':
                h = get_h()[l-2]
            else:
                h = get_h()[l-1]
    #        Preconditioner
            self.bn = np.arccos(self.b)
    #        No flips.
            self.distancex = np.absolute(np.outer(np.ones(self.p),self.bn[:,0]) \
                                     -np.outer(self.bn[:,0],np.ones(self.p)))
            self.distancey = np.absolute(np.outer(np.ones(self.p),self.bn[:,1]) \
                                     -np.outer(self.bn[:,1],np.ones(self.p)))
            self.B = h.ev(self.distancex,self.distancey)
    #        x-Flip.
            self.distancex = np.absolute(np.outer(np.ones(self.p),self.bn[:,0]) \
                                     -np.outer(-self.bn[:,0],np.ones(self.p)))
            self.distancey = np.absolute(np.outer(np.ones(self.p),self.bn[:,1]) \
                                     -np.outer(self.bn[:,1],np.ones(self.p)))
            self.B = self.B + h.ev(self.distancex,self.distancey)
    #        y-Flip.
            self.distancex = np.absolute(np.outer(np.ones(self.p),self.bn[:,0]) \
                                     -np.outer(self.bn[:,0],np.ones(self.p)))
            self.distancey = np.absolute(np.outer(np.ones(self.p),self.bn[:,1]) \
                                     -np.outer(-self.bn[:,1],np.ones(self.p)))
            self.B = self.B + h.ev(self.distancex,self.distancey)
    #        xy-Flip.
            self.distancex = np.absolute(np.outer(np.ones(self.p),self.bn[:,0]) \
                                     -np.outer(-self.bn[:,0],np.ones(self.p)))
            self.distancey = np.absolute(np.outer(np.ones(self.p),self.bn[:,1]) \
                                     -np.outer(-self.bn[:,1],np.ones(self.p)))
            self.B = self.B + h.ev(self.distancex,self.distancey)
            self.B = self.B/(2*self.m)**2
            self.B = scipy.linalg.lu_factor(self.B)
    def Flag(self):
        flag = self.flag.astype(int)
        w = np.zeros((self.m,self.m))
        w = np.roll(flag,-1,0) + np.roll(flag,1,0) + np.roll(flag,-1,1) + np.roll(flag,1,1)
        w[~self.flag] = 0
        return w
    def contour(self,w):
        plt.figure()
        plt.subplot(111)
        plt.contourf(self.x1,self.x2,w,50)
        plt.colorbar()
        return 
    def createflag(self,f):
#        Calculate the interior.
        r = self.x1**2 + self.x2**2
        theta = np.arctan2(self.x2,self.x1)
        flag = r <= f[0](theta)**2 + f[1](theta)**2
        return flag
    def FD(self,w1):
    #    Applies finite difference to the interior.
        v1 = np.zeros((self.m,self.m))
        v1[self.flag] = w1
        v = self.m**2/(2*np.pi)**2 * (self.flag2*v1 - np.roll(v1,-1,0) - np.roll(v1,1,0) \
                       - np.roll(v1,-1,1) - np.roll(v1,1,1))
        return v[self.flag]
    
def createbdry(f,dx):
#        Find the length of the boundary curve.
    m = 8192
    k = np.arange(m)/m
    B = np.transpose([f[0](2*np.pi*k), f[1](2*np.pi*k)])
    BTfreq = np.fft.fftfreq(m,1/float(m))
    BT = np.zeros((m,2))
    BT[:,0] = np.real(np.fft.ifft(np.fft.fft(B[:,0])*1j*BTfreq))
    BT[:,1] = np.real(np.fft.ifft(np.fft.fft(B[:,1])*1j*BTfreq))
    BT = BT*2*np.pi/m
    df = np.linalg.norm(BT,axis=1)
    l = np.sum(df)
    # Find normal vectors.
    BN = np.zeros((m,2))
    BN[:,0] = BT[:,1]/df
    BN[:,1] = -BT[:,0]/df
    # Place 1/dx points per unit length.
    n = int(l/dx)+1
    dx = l/n
    b = np.array(B[0,:]).T
    g_grid = 0
    bn = np.array(BN[0,:]).T
    i = 1
    j = 0
    p = 0
    while i < n:
        p = p + .5*(df[j]+np.roll(df,-1)[j])
        j = j + 1
        if p >= i * dx:
            b = np.vstack((b,B[j,:]))
            bn = np.vstack((bn,BN[j,:]))
            g_grid = np.hstack((g_grid,2*np.pi*j/m))
            i = i + 1
    b = np.cos(b)
    bn = np.vstack((bn[:,0]/-np.sqrt(1-b[:,0]**2),bn[:,1]/-np.sqrt(1-b[:,1]**2))).T
    bn_norm = np.outer(np.linalg.norm(bn,axis=1),np.array([1,1]))
    bn = bn/bn_norm
    return b, bn, g_grid

