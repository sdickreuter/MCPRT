import numpy as np
from numba import njit, jitclass, float64, int64, void, boolean, uint64, complex128, prange, deferred_type

import cmath
from scipy import interpolate

from utils import *
import matplotlib.pyplot as plt
import scipy.integrate as integrate

alpha = np.linspace(-np.pi/2,np.pi/2)
intensity = (np.cos(alpha))**2

@njit
def calcInt(alpha1,alpha2):
    #return 2*(np.cos(alpha))**2/(np.pi)
    #return integrate.quad(lambda x: 2*np.cos(x)**2/np.pi,alpha1,alpha2)
    return ( -alpha1 - np.sin(alpha1)*np.cos(alpha1) + alpha2 + np.sin(alpha2)*np.cos(alpha2))/np.pi

print(calcInt(-np.pi/2,np.pi/2))

#plt.polar(alpha, intensity*1.0)
#plt.polar(alpha, intensity*2.0)
#plt.show()

