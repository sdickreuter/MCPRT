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


num = 100
ys = np.linspace(-0.1, 0.1, num)
xs = np.repeat(1.0, num)
screen = Surface(np.vstack((xs, ys)).T, reflectivity=0.0, transmittance=1.0, n1=1.0, n2=1.5)


#ys = np.array([1.0,-1.0])
y1 = np.linspace(-10e-6,10e-6,200)

#ys = np.hstack((y1+1e-4,y1-1e-4))
#ys = np.array([0.0])
ys = np.linspace(-0.1,0.1,200)
dipoles = []
for y in ys:
    dipoles.append(Dipole(np.array([-1.0,y]), np.array([1.0,0.0]), 1.0, 0.0, wl))


print('Start interact with screen')
onscreen = interact_dipoles_with_surface(dipoles, screen)
plot_screen_E(screen)
plot_screen_Eabs(screen)
#plot_screen_Evec(screen)
#plot_screen_Eimag(screen)
# #plot_all(screen,onscreen,"screen")
# fig, ax1 = plt.subplots()
# intensity = np.square(np.real(screen.phasors))
# ax1.plot(screen.midpoints[:,1],intensity, 'b-')
# ax2 = ax1.twinx()
# ax2.plot(screen.midpoints[:,1],np.angle(screen.phasors), 'r-')
# plt.show()

plot_dipoles_phase(onscreen)
plot_dipoles(onscreen)

# num = 100
# dxs = np.linspace(-wl*50,-wl*50,100)
#
# dat = np.zeros((len(dxs),num-1))
#
# for i in range(len(dxs)):
#     dx = dxs[i]
#     ys = np.linspace(-wl*10, wl*10, num)
#     xs = np.repeat(lense1.back.points[:,0].max()+lense1.f, num)+dx
#     screen = Surface(np.vstack((xs, ys)).T, reflectivity=0.0, transmittance=1.0, n1=1.0, n2=1.0)
#     print("interact "+str(i))
#     interact_dipoles_with_screen(onlense1_back, screen)
#     dat[i, :] = np.sqrt(np.sum(np.square(np.real(screen.E)),axis=1))
#     #dat[i, :] = np.angle(screen.E[:,0])
#
#
# #plt.matshow(dat)
# xmax = lense1.back.points[:,0].max()+lense1.f + dxs.max()
# xmin = lense1.back.points[:,0].max()+lense1.f + dxs.min()
#
# #plt.imshow(dat,aspect="equal",extent=(xmax,xmin,ys.max(),ys.min()))
# plt.matshow(dat)
# plt.show()

