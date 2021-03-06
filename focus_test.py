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

#c = 2.998e8  # m/s
wl = 500e-9

plotit = False

num = 1024#2048

lense1 = HyperbolicLense(x=0.0, y=0,f=2.0,height=0.1, num=num)
lense1._make_surfaces(flipped=True)

num = 100#500
ys = np.linspace(-wl*10, wl*10, num)
#xs = np.repeat(lense2.x+lense2._calc_f_back(), num)
xs = np.repeat(lense1.back.points[:,0].max()+lense1.f, num)
print('focus x: '+str(lense1.back.points[:,0].max()+lense1.f))
#xs = np.repeat(15.70, num)
print('screen x: '+ str(xs[0]))
screen = Surface(np.vstack((xs, ys)).T, reflectivity=0.0, transmittance=1.0, n1=1.0, n2=1.0)
#screen.flip_normals()


num=200
xs = np.linspace(lense1.back.points[:,0].max(), lense1.back.points[:,0].max()+lense1.f*1.5, num)
ys = np.repeat(0, num)
screen2 = Surface(np.vstack((xs, ys)).T, reflectivity=0.0, transmittance=1.0, n1=1.0, n2=1.0)


ys = np.linspace(lense1.front.midpoints[:,1].min(),lense1.front.midpoints[:,1].max(),100)
dipoles = []
for y in ys:
    dipoles.append(Dipole(np.array([-1.0,y]), np.array([1.0,0.0]), 1.0, 0.0, wl))

print('Start interact with lense1.front')
onlense1_front = interact_dipoles_with_surface(dipoles, lense1.front)
print(len(onlense1_front))
plot_screen_Eabs(lense1.front)
plot_dipoles(onlense1_front)
#plot_dipoles_phase(onlense1_front)

print('Start interact with lense1.back')
onlense1_back = interact_dipoles_with_surface(onlense1_front, lense1.back)
print(len(onlense1_back))
plot_screen_Eabs(lense1.back)
plot_dipoles(onlense1_back)
#plot_dipoles_phase(onlense1_back)

print('Start interact with screen')
interact_dipoles_with_surface(onlense1_back, screen)
plot_screen_Eabs(screen)
# #plot_all(screen,onscreen,"screen")
# fig, ax1 = plt.subplots()
# intensity = np.square(np.real(screen.phasors))
# ax1.plot(screen.midpoints[:,1],intensity, 'b-')
# ax2 = ax1.twinx()
# ax2.plot(screen.midpoints[:,1],np.angle(screen.phasors), 'r-')
# plt.show()


print('Start interact with screen2')
interact_dipoles_with_surface(onlense1_back, screen2)
plot_screen_Eabs(screen2)
# fig, ax1 = plt.subplots()
# intensity = np.square(np.real(screen2.phasors.copy()))
# ax1.plot(screen2.midpoints[:,0],intensity, 'b-')
# #ax2 = ax1.twinx()
# #ax2.plot(screen2.midpoints[:,0],np.angle(screen2.phasors), 'r-')
# plt.show()

# num = 50
# dxs = np.linspace(-0.1,0.1,5)
#
# fig, ax1 = plt.subplots()
#
# for dx in dxs:
#     ys = np.linspace(-wl*2, wl*2, num)
#     xs = np.repeat(lense1.back.points[:,0].max()+lense1.f, num)+dx
#     screen = Surface(np.vstack((xs, ys)).T, reflectivity=0.0, transmittance=1.0, n1=1.0, n2=1.0)
#     interact_dipoles_with_screen_verbose(onlense1_back, screen)
#     intensity = np.square(np.real(screen.phasors.copy()))
#     ax1.plot(screen.midpoints[:,1],intensity,label=str(np.round(dx,2)))
#
# plt.legend()
# plt.show()

num = 100
dxs = np.linspace(-wl*50,-wl*50,100)

dat = np.zeros((len(dxs),num-1))

for i in range(len(dxs)):
    dx = dxs[i]
    ys = np.linspace(-wl*10, wl*10, num)
    xs = np.repeat(lense1.back.points[:,0].max()+lense1.f, num)+dx
    screen = Surface(np.vstack((xs, ys)).T, reflectivity=0.0, transmittance=1.0, n1=1.0, n2=1.0)
    print("interact "+str(i))
    interact_dipoles_with_surface(onlense1_back, screen)
    dat[i, :] = np.sqrt(np.sum(np.square(np.abs(screen.E)),axis=1))
    #dat[i, :] = np.angle(screen.E[:,0])


#plt.matshow(dat)
xmax = lense1.back.points[:,0].max()+lense1.f + dxs.max()
xmin = lense1.back.points[:,0].max()+lense1.f + dxs.min()

#plt.imshow(dat,aspect="equal",extent=(xmax,xmin,ys.max(),ys.min()))
plt.matshow(dat)
plt.show()

