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
lense2 = HyperbolicLense(x=0.0, y=0,f=2.0,height=0.5, num=num)
lense2._make_surfaces(flipped=True)

lense1.shift(dx=lense1.f+lense1.front.points[:,0].min()+0.475)
lense2.shift(dx=lense1.back.points[:,0].max()+lense1.f)

alpha_max = np.abs(angle_between(np.array([1,0],dtype=np.float64),lense1.front.points[0,:]))#np.pi/10)
print("alpha_max: "+str(alpha_max)+'  '+str(alpha_max*180/np.pi))

d = -0.040#-0.041

num = 200#500
ys = np.linspace(-wl*10, wl*10, num)
#xs = np.repeat(lense2.x+lense2._calc_f_back(), num)
xs = np.repeat(lense2.back.points[:,0].max()+lense2.f, num)+d#-0.0405#+0.0001#-0.015
print('focus x: '+str(lense2.back.points[:,0].max()+lense2.f))
#xs = np.repeat(15.70, num)
print('screen x: '+ str(xs[0]))
screen = Surface(np.vstack((xs, ys)).T, reflectivity=0.0, transmittance=1.0, n1=1.0, n2=1.0)
#screen.flip_normals()



d = Dipole(np.array([0.0,0.0]), np.array([1.0,0.0]), np.array([1.0,0.0]), wl)

interact_dipole_with_surface(d, lense1.front)
onlense1_front = generate_dipoles_from_surface(lense1.front)
plot_all(lense1.front,onlense1_front,"lense 1 front")
#print(lense1.front.ks)

interact_dipoles_with_surface(onlense1_front, lense1.back)
onlense1_back = generate_dipoles_from_surface(lense1.back)
plot_all(lense1.back,onlense1_back,"lense 1 back")

interact_dipoles_with_surface(onlense1_back, lense2.front)
onlense2_front = generate_dipoles_from_surface(lense2.front)
plot_all(lense2.front,onlense2_front,"lense 2 front")

interact_dipoles_with_surface(onlense2_front, lense2.back)
onlense2_back = generate_dipoles_from_surface(lense2.back)
plot_all(lense2.back,onlense2_back,"lense 2 back")

interact_dipoles_with_surface(onlense2_back, screen)
onscreen = generate_dipoles_from_surface(screen)
#plot_all(screen,onscreen,"screen")
# plt.plot(screen.points[:, 0], screen.points[:, 1], 'bx')
# n = screen.normals / np.sqrt(screen.normals[:, 0].max() ** 2 + screen.normals[:, 1].max() ** 2)
# for i in range(n.shape[0]):
#     n[i, :] *= length(screen.phasors[i, :])
# plt.quiver(screen.points[:, 0], screen.points[:, 1], n[:, 0], n[:, 1])
# plt.title("screen normals")
# plt.show()
y = np.sqrt(screen.phasors[:,0]**2+screen.phasors[:,1]**2)
plt.plot(y)
plt.show()