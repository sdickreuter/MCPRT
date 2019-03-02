import locale

locale.setlocale(locale.LC_NUMERIC, 'C')

#import matplotlib as mpl
#mpl.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from numba import jit, jitclass, float32, int32, void, boolean
import cmath
from mcprt4 import *
import progress
from scipy.signal import savgol_filter
from scipy import interpolate
import time


#E0 = np.sqrt(2) * np.sqrt(1 + np.cos(alpha) / (2 * dipole.wavelength)) * np.exp(
#            1j * np.dot(dipole.k * surf.n2, r) + 1j * dipole.phase) / np.sqrt(length(r))

wavelength = 0.3
k = unit_vector(np.array([1.0,1.0])) * 2 * np.pi / wavelength
r = np.linspace(0,1,100)
r = np.array([r,r]).transpose()

a = np.zeros(r.shape[0],dtype=np.complex)
for i in range(r.shape[0]):
    a[i] = np.exp(1j*np.dot(k, r[i,:]))

plt.plot(np.angle(a))
plt.show()

r1 = np.array([0.0,1.0])
r2 = np.array([0.0,-1.0])
a1 = angle_between(k,r1)
a2 = angle_between(k,r2)

alphas = np.linspace(-np.pi,np.pi,100)
plt.plot(alphas,np.sqrt(1 + np.cos(alphas)))
plt.show()