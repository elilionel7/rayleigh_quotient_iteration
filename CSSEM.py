''' Functions for CSSEM.'''

import numpy as np
import scipy
import scipy.interpolate

def get_h():
    try:
        h = np.load('hfunct.npy')
        return h
    except:
        m = 2048
        h = []
        x = 2*np.pi*np.mgrid[0:m:1]/float(m)
        fs=np.fft.fftfreq(m,1/float(m))
        fx,fy = np.meshgrid(fs,fs)
        for i in [2,3,4,5]:
            aa = np.real(np.fft.ifft2(m**2*np.ones((m,m))*(1+fx**2 + fy**2)**-i))
            h.append(scipy.interpolate.RectBivariateSpline(x[::16],x[::16],aa[::16,::16],kx=5,ky=5))
        np.save('hfunct.npy',h)
        return h




