import locale

locale.setlocale(locale.LC_NUMERIC, 'C')

#import matplotlib as mpl
#mpl.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from numba import jit, jitclass, float32, int32, void, boolean
import cmath
from mcprt3 import *
import progress
from scipy.signal import savgol_filter
from scipy import interpolate


@njit
def transmitted_k(k, normal, n1, n2):
    alpha = angle_between(k, normal)
    beta = cmath.asin( (n1/n2) * np.sin(alpha) ).real
    if n1 > n2:
        beta = -beta
    if np.abs(beta) < np.pi/2:
        transmitted = rotate_vector(normal, beta)
    else:
        transmitted = np.array([np.NaN,np.NaN])
    return transmitted


k = np.array([ 9.52054888e-01, -2.37309269e-03])
normal = np.array([ 1.0, 0.0])

print(transmitted_k(k,normal,1.0,1.5))
print(transmitted_k(k,normal,1.5,1.0))

print(' ')
k = np.array([ 1.0, 0.0])
normal = np.array([ 1.0, 0.0])

print(transmitted_k(k,normal,1.0,1.5))
print(transmitted_k(k,normal,1.5,1.0))