import locale

locale.setlocale(locale.LC_NUMERIC, 'C')

#import matplotlib as mpl
#mpl.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from numba import jit, jitclass, float32, int32, void, boolean
import cmath
from mcprt import *
import progress

c = 2.998e8  # m/s
wl = 0.001#0.00001

plotit = True

iterations = 1000#200


num = 1000
ys = np.linspace(-1.0, 1.0, num)
xs = np.repeat(10.0, num)
screen = Surface(np.vstack((xs, ys)).T, reflectivity=0.0, transmittance=1.0, n1=1.0, n2=1.0)
screen.flip_normals()

n = 10000
hits = np.zeros(n, dtype=np.float64)
rs = np.zeros((n, 2), dtype=np.float64)
ks = np.zeros((n, 2), dtype=np.float64)
ts = np.zeros(n, dtype=np.float64)
ps = np.zeros(n, dtype=np.float64)
ks[:,0] = 1.0
ks[:,1] = 0.0
ks = unit_vector(ks)
#rs[:,1] = 0.1
rs[:,1] = np.linspace(-0.01,0.01,n)+0.1
a = Wavelets(rs,ks,ts,wl,ps,mode=modes['spherical'])

n = 10000
hits = np.zeros(n, dtype=np.float64)
rs = np.zeros((n, 2), dtype=np.float64)
ks = np.zeros((n, 2), dtype=np.float64)
ts = np.zeros(n, dtype=np.float64)
ps = np.zeros(n, dtype=np.float64)
ks[:,0] = 1.0
ks[:,1] = 0.0
ks = unit_vector(ks)
#rs[:,1] = -0.1
rs[:,1] = np.linspace(-0.01,0.01,n)-0.1
b = Wavelets(rs,ks,ts,wl,ps,mode=modes['spherical'])



prog = progress.Progress(max=iterations)
for i in range(iterations):

    onscreen = screen.interact_with_all_wavelets(a)
    screen.add_phase_from_wavelets(onscreen)
    # plt.plot(screen.midpoints[onscreen.surface_index,1],onscreen.phases,'b.')
    # plt.show()
    onscreen = screen.interact_with_all_wavelets(b)
    screen.add_phase_from_wavelets(onscreen)
    # plt.plot(screen.midpoints[onscreen.surface_index,1],onscreen.phases,'b.')
    # plt.show()

    print(str(i) + " count on screen1: " + str(screen.count))
    prog.next()
    print(str(np.round(prog.percent,1))+'%  ' + str(prog.eta_td))

intensity = np.abs(screen.phasor) ** 2  # np.cos(screen.phase)**2

plt.plot(screen.midpoints[:, 1], intensity)
plt.xlabel("position on screen / m")
plt.ylabel("intensities / a.u.")
plt.savefig("twoslit_onscreen.png", dpi=600)
plt.show()
plt.close()

# plt.plot(screen.midpoints[:, 1], screen.phasor.real)
# plt.plot(screen.midpoints[:, 1], screen.phasor.imag)
# plt.xlabel("position on screen / m")
# plt.ylabel("phase / a.u.")
# #plt.savefig("dipole_" + str(int(np.round(theta * 180 / np.pi))) + "_theta_" + str(d) + "_onscreen.png", dpi=600)
# plt.show()
# plt.close()
#
# plt.plot(screen.midpoints[:, 1], screen.hits)
# plt.xlabel("position on screen / m")
# plt.ylabel("number of hits")
# #plt.savefig("dipole_" + str(int(np.round(theta * 180 / np.pi))) + "_theta_" + str(d) + "_onscreen_hits.png", dpi=600)
# plt.show()
# plt.close()

np.savetxt("twoslit_onscreen.csv",np.vstack([screen.midpoints[:,1],intensity]).T)
