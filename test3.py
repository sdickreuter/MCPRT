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

c = 1.0#2.998e8  # m/s
wl = 1.0#0.00001

num = 10
ys = np.linspace(-2, 2, num)
xs = np.repeat(1.0, num)
s = Surface(np.vstack((xs, ys)).T, reflectivity=0.0, transmittance=1.0, n1=1.5, n2=1.0)


d = Dipole(np.array([0.0,0.0]), np.array([1.0,0.0]), np.array([1.0,0.0]), wl)

interact_dipole_with_surface(d, s)


plt.plot(s.points[:, 0], s.points[:, 1], 'bx')
n = s.normals / np.sqrt(s.normals[:, 0].max() ** 2 + s.normals[:,1].max() ** 2)
plt.quiver(s.points[:, 0], s.points[:, 1], n[:, 0], n[:, 1])
plt.title("normals")
plt.show()

plt.plot(s.points[:, 0], s.points[:, 1], 'rx')
k = s.ks / np.sqrt(s.ks[:, 0].max() ** 2 + s.ks[:,1].max() ** 2)
plt.quiver(s.points[:, 0], s.points[:, 1], k[:, 0], k[:, 1])
plt.title("k-vectors")
plt.show()

plt.plot(s.points[:, 0], s.points[:, 1], 'gx')
p = s.phasors / np.sqrt(s.phasors[:, 0].max() ** 2 + s.phasors[:,1].max() ** 2)
plt.quiver(s.points[:, 0], s.points[:, 1], p[:, 0], p[:, 1])
plt.title("phasors")
plt.show()


dipoles = generate_dipoles_from_surface(s)
for d in dipoles:
    plt.plot(d.r[0], d.r[1], 'bx')
    k = d.k / np.sqrt(np.sum(d.k** 2))
    plt.quiver(d.r[0], d.r[1], k[0], k[1])

    k2 = ( d.r - np.array([0.0,0.0]) )
    k2 = unit_vector(k2)
    plt.quiver(d.r[0], d.r[1], k2[0], k2[1],color='g')

plt.plot(0.0, 0.0, 'bx')
plt.title("dipoles k")
plt.show()