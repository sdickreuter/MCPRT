import locale

locale.setlocale(locale.LC_NUMERIC, 'C')

import matplotlib as mpl

mpl.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from numba import jit, jitclass, float32, int32, void, boolean
import cmath
from mcprt import *
import progress


n = 1
hits = np.zeros(n, dtype=np.float64)
rs = np.zeros((n, 2), dtype=np.float64)
ks = np.zeros((n, 2), dtype=np.float64)
ts = np.zeros(n, dtype=np.float64)
ps = np.zeros(n, dtype=np.float64)
ks[:,0] = 1.0
ks[:,1] = 0.5
ks = unit_vector(ks)
a = Wavelets(rs,ks,ts,0.1,ps,mode=modes['ray'])

num = 100
# ys = np.linspace(-0.5, 0.5, num)
# xs = np.repeat(1.0, num)
# screen = Surface(np.vstack((xs, ys)).T, reflectivity=0.0, transmittance=1.0, n1=1.0, n2=2.0)

p0 = np.array([0.0, 0.0])
points = Lense._gen_concave_points(None,p0,1.0,1.0,num)
screen = Surface(points, reflectivity=0.0, transmittance=1.0, n1=1.0, n2=1.0)


screen.flip_normals()


prob = []
angle1 = []
angle2 = []
for i in range(screen.points.shape[0]-1):
    point1 = screen.points[i,:]
    point2 = screen.points[i+1,:]

    prob.append(a.calc_probability_of_wavelet(0,point1,point2))

    v1 = np.subtract(point1, a.r[0, :])
    v2 = np.subtract(point2, a.r[0, :])
    angle1.append(angle_between(v1, a.k[0, :]))
    angle2.append(angle_between(v2, a.k[0, :]))

plt.plot(prob,'b.')
plt.show()
plt.close()

plt.plot(angle1)
plt.plot(angle2)
plt.show()
plt.close()

b = screen.interact_with_all_wavelets(a)

