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

plotit = False

num = 30#128#1024#2048

lense1 = HyperbolicLense(x=0.0, y=0,f=2.0,height=0.5, num=num)
lense1._make_surfaces(flipped=True)

num = 200#500
ys = np.linspace(-wl*10, wl*10, num)
#xs = np.repeat(lense2.x+lense2._calc_f_back(), num)
xs = np.repeat(lense1.back.points[:,0].max()+lense1.f, num)
print('focus x: '+str(lense1.back.points[:,0].max()+lense1.f))
#xs = np.repeat(15.70, num)
print('screen x: '+ str(xs[0]))
screen = Surface(np.vstack((xs, ys)).T, reflectivity=0.0, transmittance=1.0, n1=1.0, n2=1.0)
#screen.flip_normals()

ys = np.linspace(lense1.front.midpoints[:,1].min(),lense1.front.midpoints[:,1].max(),30)
dipoles = []
for y in ys:
    dipoles.append(Dipole(np.array([-1.0,y]), np.array([1.0,0.0]), np.array([1.0,0.0]), wl))

onlense1_front = interact_dipoles_with_surface(dipoles, lense1.front)
plot_all(lense1.front,onlense1_front,"lense 1 front")
print(len(onlense1_front))


onlense1_back = interact_dipoles_with_surface(onlense1_front, lense1.back)
plot_all(lense1.back,onlense1_back,"lense 1 back")
print(len(onlense1_back))


onscreen = interact_dipoles_with_surface(onlense1_back, screen)
print(len(onscreen))

#plot_all(screen,onscreen,"screen")
plt.plot(screen.points[:, 0], screen.points[:, 1], 'bx')
n = screen.phasors / np.sqrt(screen.phasors[:, 0].max() ** 2 + screen.phasors[:, 1].max() ** 2)
for i in range(n.shape[0]):
    n[i, :] *= length(screen.phasors[i, :])
plt.quiver(screen.points[:, 0], screen.points[:, 1], n[:, 0], n[:, 1])
plt.title("screen normals")
plt.show()
y = np.sqrt(screen.phasors[:,0]**2+screen.phasors[:,1]**2)
plt.plot(y)
plt.show()