import locale

locale.setlocale(locale.LC_NUMERIC, 'C')

#import matplotlib as mpl
#mpl.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from numba import jit, jitclass, float32, int32, void, boolean
import cmath
from mcprt2 import *
import progress
from scipy.signal import savgol_filter
from scipy import interpolate

c = 1.0#2.998e8  # m/s
wl = 0.0001#0.00001

plotit = False

iterations = 100#500#200

theta = 0#np.pi/2

num = 30#128#1024#2048

lense1 = HyperbolicLense(x=0.0, y=0,f=2.0,height=0.5, num=num)
lense2 = HyperbolicLense(x=0.0, y=0,f=2.0,height=0.5, num=num)
lense2._make_surfaces(flipped=True)

lense1.shift(dx=lense1.f+lense1.front.points[:,0].min()+0.475)
lense2.shift(dx=lense1.back.points[:,0].max()+lense1.f)

alpha_max = np.abs(angle_between(np.array([1,0],dtype=np.float64),lense1.front.points[0,:]))#np.pi/10)
print("alpha_max: "+str(alpha_max)+'  '+str(alpha_max*180/np.pi))

d = -0.041

num = 200#500
ys = np.linspace(-wl*10, wl*10, num)
#xs = np.repeat(lense2.x+lense2._calc_f_back(), num)
xs = np.repeat(lense2.back.points[:,0].max()+lense2.f, num)+d#-0.0405#+0.0001#-0.015
print('focus x: '+str(lense2.back.points[:,0].max()+lense2.f))
#xs = np.repeat(15.70, num)
print('screen x: '+ str(xs[0]))
screen = Surface(np.vstack((xs, ys)).T, reflectivity=0.0, transmittance=1.0, n1=1.0, n2=1.0)
screen.flip_normals()


# for i in range(100):
#     num = 1000
#     dipole = make_dipole(wl,theta, alpha_max,num,'ray')
#     print(i)
#

num = 30#10000#50000
dipole = make_dipole(wl,theta, alpha_max,num,'ray')

lense1.front.interact_with_all_wavelets(dipole)
onlense1_front = generate_wavelets_from_surface(lense1.front,num,wl)

lense1.back.interact_with_all_wavelets(onlense1_front)
onlense1_back = generate_wavelets_from_surface(lense1.back,num,wl)

lense2.front.interact_with_all_wavelets(onlense1_back)
onlense2_front = generate_wavelets_from_surface(lense2.front,num,wl)

lense2.back.interact_with_all_wavelets(onlense2_front)
onlense2_back = generate_wavelets_from_surface(lense2.back,num,wl)

onlense2_back.mode = modes['gaussian']
screen.interact_with_all_wavelets(onlense2_back)
onscreen = generate_wavelets_from_surface(screen,num,wl)

plot_all(lense1.front,onlense1_front,"lense 1 front")
plot_all(lense1.back,onlense1_back,"lense 1 back")
plot_all(lense2.front,onlense2_front,"lense 2 front")
plot_all(lense2.back,onlense2_back,"lense 2 back")

#
# print("onscreen: " + str(onscreen.n))
# #screen.add_phase_from_wavelets(onscreen)
#
# # x = []
# # y = []
# # for i in range(len(onlense1_front.surface_index)):
# #     x.append(lense1.front.midpoints[onlense1_front.surface_index[i],1])
# #     y.append(onlense1_front.t0[i])
# #
# # plt.plot(x,y,'b.')
# # plt.show()
# # plt.close()
# #
# # x = []
# # y = []
# # for i in range(len(onscreen.surface_index)):
# #     x.append(screen.midpoints[onscreen.surface_index[i],1])
# #     y.append(onscreen.t0[i])
# #
# # plt.plot(x,y,'b.')
# # plt.show()
# # plt.close()
#
#
if plotit:

    # plt.figure(figsize=(10,3))
    #
    # divider = 50#200
    # plt.plot(lense1.front.points[:, 0], lense1.front.points[:, 1])
    # for i in range(lense1.front.midpoints.shape[0]):
    #     plt.arrow(lense1.front.midpoints[i, 0], lense1.front.midpoints[i, 1], lense1.front.normals[i, 0] / (divider), lense1.front.normals[i, 1] / (divider))
    #
    # plt.plot(lense1.back.points[:, 0], lense1.back.points[:, 1])
    # for i in range(lense1.back.midpoints.shape[0]):
    #     plt.arrow(lense1.back.midpoints[i, 0], lense1.back.midpoints[i, 1], lense1.back.normals[i, 0] / (divider), lense1.back.normals[i, 1] / (divider))
    #
    # plt.plot(lense2.front.points[:, 0], lense2.front.points[:, 1])
    # for i in range(lense2.front.midpoints.shape[0]):
    #     plt.arrow(lense2.front.midpoints[i, 0], lense2.front.midpoints[i, 1], lense2.front.normals[i, 0] / (divider), lense2.front.normals[i, 1] / (divider))
    #
    # plt.plot(lense2.back.points[:, 0], lense2.back.points[:, 1])
    # for i in range(lense2.back.midpoints.shape[0]):
    #     plt.arrow(lense2.back.midpoints[i, 0], lense2.back.midpoints[i, 1], lense2.back.normals[i, 0] / (divider), lense2.back.normals[i, 1] / (divider))
    #
    #
    # plt.plot(screen.points[:, 0], screen.points[:, 1])
    # plt.show()
    # plt.close()


    divider = 300#200
    plt.plot(dipole.r[:, 0], dipole.r[:, 1])
    for i in range(dipole.r.shape[0]):
        #plt.plot(dipole.r[i, 0], dipole.r[i, 1], "bo")
        plt.arrow(dipole.r[i, 0], dipole.r[i, 1], dipole.k[i, 0] / (divider*10), dipole.k[i, 1] / (divider*10))
    plt.plot(lense1.front.points[:, 0], lense1.front.points[:, 1])
    for i in range(onlense1_front.r.shape[0]):
        #plt.plot(onlense1_front.r[i, 0], onlense1_front.r[i, 1], "bo")
        plt.arrow(onlense1_front.r[i, 0], onlense1_front.r[i, 1], onlense1_front.k[i, 0] / (divider*10), onlense1_front.k[i, 1] / (divider*10))
    plt.plot(lense1.back.points[:, 0], lense1.back.points[:, 1])
    for i in range(onlense1_back.r.shape[0]):
        #plt.plot(onlense1_back.r[i, 0], onlense1_back.r[i, 1], "bo")
        plt.arrow(onlense1_back.r[i, 0], onlense1_back.r[i, 1], onlense1_back.k[i, 0] / divider, onlense1_back.k[i, 1] / divider)
    plt.plot(lense2.front.points[:, 0], lense2.front.points[:, 1])
    for i in range(onlense2_front.r.shape[0]):
        #plt.plot(onlense2_front.r[i, 0], onlense2_front.r[i, 1], "bo")
        plt.arrow(onlense2_front.r[i, 0], onlense2_front.r[i, 1], onlense2_front.k[i, 0] / (divider*10),
                  onlense2_front.k[i, 1] / (divider*10))
    plt.plot(lense2.back.points[:, 0], lense2.back.points[:, 1])
    for i in range(onlense2_back.r.shape[0]):
        #plt.plot(onlense2_back.r[i, 0], onlense2_back.r[i, 1], "bo")
        plt.arrow(onlense2_back.r[i, 0], onlense2_back.r[i, 1], onlense2_back.k[i, 0] / divider,
                  onlense2_back.k[i, 1] / divider)
    plt.plot(screen.points[:, 0], screen.points[:, 1]/(wl*50))
    # for i in range(onscreen.r.shape[0]):
    #     plt.plot(onscreen.r[i, 0], onscreen.r[i, 1], "bo")
    #     plt.arrow(onscreen.r[i, 0], onscreen.r[i, 1], onscreen.k[i, 0] / divider, onscreen.k[i, 1] / divider)
    #plt.savefig("dipole_" + str(int(np.round(theta * 180 / np.pi))) + "_theta_setup.svg", dpi=600)
    plt.show()
    plt.close()
#
#
# screen.clear()
#

lense1.front.clear()
lense1.back.clear()
lense2.front.clear()
lense2.back.clear()
screen.clear()

prog = progress.Progress(max=iterations)
num = 10000#50000
dipole = make_dipole(wl,theta, alpha_max,num,'ray')

lense1.front.interact_with_all_wavelets(dipole)
onlense1_front = generate_wavelets_from_surface(lense1.front,num,wl)

lense1.back.interact_with_all_wavelets(onlense1_front)
onlense1_back = generate_wavelets_from_surface(lense1.back,num,wl)

lense2.front.interact_with_all_wavelets(onlense1_back)
onlense2_front = generate_wavelets_from_surface(lense2.front,num,wl)

lense2.back.interact_with_all_wavelets(onlense2_front)

for i in range(iterations):
    onlense2_back = generate_wavelets_from_surface(lense2.back, num*10, wl)
    onlense2_back.mode = modes['gaussian']
    screen.interact_with_all_wavelets(onlense2_back)


    # plt.plot(screen.midpoints[onscreen.surface_index, 1],onscreen.phases,'b.')
    # plt.xlabel("position on screen / m")
    # plt.ylabel("t / a.u.")
    # plt.show()
    # plt.close()

    print(str(i) + " count on screen1: " + str(screen.count))
    prog.next()
    print(str(np.round(prog.percent,1))+'%  ' + str(prog.eta_td))


#print(screen.field.shape)
#intensities = screen.field[:,0]**2#np.sum(screen.field ** 2,axis=1)
print(screen.phasor)
intensity = np.abs(screen.phasor)**2# np.cos(screen.phase)**2

plt.plot(screen.midpoints[:,1],intensity)
plt.xlabel("position on screen / m")
plt.ylabel("intensities / a.u.")
#plt.savefig("dipole_"+str(int(np.round(theta*180/np.pi)))+"_theta_"+str(d)+"_onscreen.png", dpi=600)
plt.show()
plt.close()

# plt.plot(screen.midpoints[:, 1], screen.phasor.real)
# plt.plot(screen.midpoints[:, 1], screen.phasor.imag)
# plt.xlabel("position on screen / m")
# plt.ylabel("phase / a.u.")
# # plt.savefig("dipole_" + str(int(np.round(theta * 180 / np.pi))) + "_theta_" + str(d) + "_onscreen.png", dpi=600)
# plt.show()
# plt.close()


plt.plot(screen.midpoints[:,1],screen.hits)
plt.xlabel("position on screen / m")
plt.ylabel("number of hits")
#plt.savefig("dipole_"+str(int(np.round(theta*180/np.pi)))+"_theta_"+str(d)+"_onscreen_hits.png", dpi=600)
plt.show()
plt.close()


#np.savetxt("dipole_"+str(int(np.round(theta*180/np.pi)))+"_theta_"+str(d)+"_onscreen.csv",np.vstack([screen.midpoints[:,1],intensities]).T)
#np.savetxt("dipole_"+str(int(np.round(theta*180/np.pi)))+"_theta_"+str(d)+"_onscreen_hits.csv",np.vstack([screen.midpoints[:,1],screen.hits]).T)

