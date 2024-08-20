import numpy as np
from scipy.fftpack import dct as dct
from scipy.fftpack import idct as idct
from scipy.fftpack import dctn as dctn
from scipy.fftpack import idctn as idctn
from scipy.fftpack import dst as dst
from scipy.fftpack import idst as idst
from scipy.sparse.linalg import LinearOperator
import scipy

class operator_data:
    def __init__(self,gdata,l,precond=False):
        self.gdata = gdata
        self.l = l
        self.MM = LinearOperator((self.gdata.k+self.gdata.p,self.gdata.k+self.gdata.p),matvec = self.M)
        if precond == True:
            self.PP = LinearOperator((self.gdata.k+self.gdata.p,self.gdata.k+self.gdata.p),matvec = self.precond)
    def ker(self,w,l=None):
        if l == None:
            l = self.l
        w = np.reshape(w,(self.gdata.m,self.gdata.m))
        w = np.real(idctn(dctn(w)*(1+self.gdata.fx**2 + self.gdata.fy**2)**-l))/(2*self.gdata.m)**2
        return w.flatten()
    def lap(self,w):
        w1 = np.copy(w)
        w1 = dct(w1,axis=0)*-self.gdata.fx
        w1 = np.roll(w1,-1,axis=0)
        w1 = idst(w1,axis=0)/(2*self.gdata.m)
        w1 = w1 * self.gdata.x1/(1-self.gdata.x1**2)**(3/2)
        w2 = np.copy(w)
        w2 = idct(dct(w2,axis=0)*-self.gdata.fx**2,axis=0)/(2*self.gdata.m)
        w2 = w2 * -1/(1-self.gdata.x1**2)
        w = w2
        w = w1 + w2
        return w
    def lapt(self,w):
        w1 = np.copy(w)
        w1 = w1 * self.gdata.x1/(1-self.gdata.x1**2)**(3/2)
        w1 = dst(w1,axis=0)/(2*(self.gdata.m))
        w1 = np.roll(w1,1,axis=0)
        w1 = idct(w1*-self.gdata.fx,axis=0)
        w2 = np.copy(w)
        w2 = w2 * -1/(1-self.gdata.x1**2)
        w2 = dct(w2,axis=0)/(2*(self.gdata.m))
        w2 = w2 * -self.gdata.fx**2
        w2 = idct(w2,axis=0)
        z = w1 + w2
        return z
    def C(self,w):
    #    Evaluate at b.
        b = self.gdata.xx.dot(w)
    #    Take Chebyshev derivative
        w = np.reshape(w,(self.gdata.m,self.gdata.m))
        z = self.lap(w) + np.transpose(self.lap(np.transpose(w)))
        z = z[self.gdata.flag]
        return np.hstack((z,b))
    def Ct(self,w):
        z = np.zeros((self.gdata.m,self.gdata.m))
        z[self.gdata.flag] = w[:self.gdata.k]
        z = self.lapt(z) + self.lapt(z.T).T
        z = z.flatten()
        z += self.gdata.xxT.dot(w[-self.gdata.p:])
#        for j in np.arange(-self.gdata.p,0):
#            z = z + w[j]*self.gdata.xx[j,:]
        return z
    def M(self,w):
        w = self.Ct(w)
        w = self.ker(w)
        w = self.C(w)
        return w
    def transform(self,w):
        u = self.Ct(w)
        u = self.ker(u)
        return u
    def precond(self,w):
        w1 = w[:self.gdata.k]
        if self.l == 3:
            w1 = w1 + self.gdata.FD(w1)
        if self.l == 4:
            w1 = w1 + self.gdata.FD(2*w1 + self.gdata.FD(w1))
        if self.l == 5:
            w1 = w1 + self.gdata.FD(3*w1 + self.gdata.FD(3*w1 + self.gdata.FD(w1)))
        w2 = w[-self.gdata.p:]
        w2 = scipy.linalg.lu_solve(self.gdata.B,w2)
        return np.hstack((w1,w2))
    def solve(self,interior,boundary,tol=1e-8):
        rhs1 = interior(self.gdata.x1[self.gdata.flag],self.gdata.x2[self.gdata.flag])
        rhs2 = boundary(self.gdata.b[:,0],self.gdata.b[:,1])
        rhs = np.hstack((rhs1,rhs2))
        u,it = PCG(self.MM,self.PP,rhs,tol)
        u = self.transform(u)
        return u,it
    def vec_solve(self,rhs,tol=1e-8):
        u,it = PCG(self.MM,self.PP,rhs,tol)
        u = self.transform(u)
        return u,it
    def make_m_matrix(self):
        m = np.zeros((self.gdata.k+self.gdata.p,self.gdata.k+self.gdata.p))
        for i in np.arange(self.gdata.k+self.gdata.p):
            z = np.zeros(self.gdata.k+self.gdata.p)
            z[i] = 1
            z = self.M(z)
            m[:,i] = z
        return m
    def qr_solve(self,interior,boundary,l):
        rhs1 = interior(self.gdata.x1[self.gdata.flag],self.gdata.x2[self.gdata.flag]);
        rhs2 = boundary(self.gdata.b[:,0],self.gdata.b[:,1]);
        rhs = np.hstack((rhs1,rhs2));
        rct = self.make_rct_matrix(l);
        cond = np.linalg.cond(rct);
        Q,R = np.linalg.qr(rct);
        u = scipy.linalg.solve_triangular(R.T,rhs,lower=True); 
        u = np.dot(Q,u);
        u = np.reshape(u,(self.gdata.m,self.gdata.m));
        u = self.ker(u,l).flatten();
        return u,cond;
    def qr_solve2(self,interior,boundary,l):
        rhs1 = interior(self.gdata.x1[self.gdata.flag],self.gdata.x2[self.gdata.flag]);
        rhs2 = boundary(self.gdata.g_grid)
        rhs = np.hstack((rhs1,rhs2));
        rct = self.make_rct_matrix(l);
        cond = np.linalg.cond(rct);
        Q,R = np.linalg.qr(rct);
        u = scipy.linalg.solve_triangular(R.T,rhs,lower=True); 
        u = np.dot(Q,u);
        u = np.reshape(u,(self.gdata.m,self.gdata.m));
        u = self.ker(u,l).flatten();
        return u,cond;
    def make_rct_matrix(self,l):
        m = np.zeros((self.gdata.m**2,self.gdata.k+self.gdata.p));
        for i in range(0,self.gdata.k+self.gdata.p):
            z = np.zeros(self.gdata.k+self.gdata.p);
            z[i] = 1;
            z = self.ker(self.Ct(z),l).flatten();
            m[:,i] = z;
        return m;
    

def PCG(MM,PP,rhs,tol=1e-8):
    tol = np.linalg.norm(rhs)*tol
    it = 0
    u = np.zeros_like(rhs)
    r = rhs
    z = PP*r
    p = z
    while np.sqrt(np.dot(r,r)) > tol and it < np.size(rhs):
        rz = np.dot(r,z)
        Mp = MM*p
        alpha = rz/np.dot(p,Mp)
        u = u + alpha*p
        r = r - alpha*Mp
        z = PP*r
        beta = np.dot(r,z)/rz
        p = z + beta*p
        it = it + 1
    return u,it
