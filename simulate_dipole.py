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


plotit = True

iterations = 10000

theta = np.pi/2

num = 100

lense1 = Lense(x=0.0, y=0,r1=np.inf,r2=2.0,height=0.5, num=num)
lense2 = Lense(x=0.0, y=0,r1=-2.0,r2=np.inf,height=0.5, num=num)

lense1.shift(dx=lense1._calc_f_front()+lense1.front.points[:,0].min())
lense2.shift(dx=lense1.back.points[:,0].max()+lense1._calc_f_back()+lense2._calc_f_front())

# lense1.shift(dx=lense1._calc_f_front()+lense1.front.points[:,0].min())
# lense2.shift(dx=lense1.back.points[:,0].max()+lense1._calc_f_back()/2)

alpha_max = np.abs(angle_between(np.array([1,0],dtype=np.float64),lense1.front.points[0,:]))#np.pi/10)
print("alpha_max: "+str(alpha_max)+'  '+str(alpha_max*180/np.pi))


# dipole = make_dipole(theta, np.pi/2,100000)
# angles = np.zeros(dipole.k.shape[0])
# for i in range(len(angles)):
#     if dipole.k[i,1] < 0:
#         angles[i] = -angle_between(dipole.k[i,:],np.array([1,0])).real
#     else:
#         angles[i] = angle_between(dipole.k[i,:],np.array([1,0])).real
#
# plt.hist(angles,bins=100)
# plt.axvline(x=alpha_max,color='k',linestyle='--')
# plt.axvline(x=-alpha_max,color='k',linestyle='--')
# plt.xlabel("angle to optical axis / rad")
# plt.ylabel("occurance")
# plt.savefig("dipole_"+str(int(np.round(theta*180/np.pi)))+"_theta_source_histogramm.png", dpi=600)
# #plt.show()
# plt.close()
#
# dipole = make_dipole(theta, alpha_max,100000)
# angles = np.zeros(dipole.k.shape[0])
# for i in range(len(angles)):
#     if dipole.k[i,1] < 0:
#         angles[i] = -angle_between(dipole.k[i,:],np.array([1,0])).real
#     else:
#         angles[i] = angle_between(dipole.k[i,:],np.array([1,0])).real
#
# plt.hist(angles,bins=100)
# plt.xlabel("angle to optical axis / rad")
# plt.ylabel("occurance")
# plt.savefig("dipole_"+str(int(np.round(theta*180/np.pi)))+"_theta_source_histogramm_alphamax.png", dpi=600)
# #plt.show()
# plt.close()


num = 100
ys = np.linspace(-0.35, 0.35, num)
#xs = np.repeat(lense2.x+lense2._calc_f_back(), num)
xs = np.repeat(lense2.back.points[:,0].max()+lense2._calc_f_back(), num)
print('focus x: '+str(lense2.back.points[:,0].max()+lense2._calc_f_back()))
#xs = np.repeat(15.70, num)
print('screen x: '+ str(xs[0]))
screen = Surface(np.vstack((xs, ys)).T, reflectivity=0.0, transmittance=1.0, n1=1.0, n2=1.0)
screen.flip_normals()


num = 200

dipole = make_dipole(0.03,theta, alpha_max,num,'ray')
print("dipole: " + str(dipole.n))

onlense1_front = lense1.front.interact_with_all_wavelets(dipole)
print("onlense1 front: " + str(onlense1_front.n))
onlense1_back = lense1.back.interact_with_all_wavelets(onlense1_front)
print("onlense1 back: " + str(onlense1_back.n))

onlense2_front = lense2.front.interact_with_all_wavelets(onlense1_back)
print("onlense2 front: " + str(onlense2_front.n))
onlense2_back = lense2.back.interact_with_all_wavelets(onlense2_front)
print("onlense2 back: " + str(onlense2_back.n))


onlense2_back.mode = modes['gaussian']
onscreen = screen.interact_with_all_wavelets(onlense2_back)

print("onscreen: " + str(onscreen.n))
screen.add_field_from_wavelets(onscreen)

# x = []
# y = []
# for i in range(len(onlense1_front.surface_index)):
#     x.append(lense1.front.midpoints[onlense1_front.surface_index[i],1])
#     y.append(onlense1_front.t0[i])
#
# plt.plot(x,y,'b.')
# plt.show()
# plt.close()

# x = []
# y = []
# for i in range(len(onscreen.surface_index)):
#     x.append(screen.midpoints[onscreen.surface_index[i],1])
#     y.append(onscreen.t0[i])
#
# plt.plot(x,y,'b.')
# plt.show()
# plt.close()


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

    # plt.plot(onlense1_back.t0)
    # plt.show()
    # plt.plot(onlense2_back.t0)
    # plt.show()
    # plt.plot(onscreen.t0)
    # plt.show()
    plt.figure(figsize=(10,3))

    divider = 50#200
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
    plt.plot(screen.points[:, 0], screen.points[:, 1])
    # for i in range(onscreen.r.shape[0]):
    #     plt.plot(onscreen.r[i, 0], onscreen.r[i, 1], "bo")
    #     plt.arrow(onscreen.r[i, 0], onscreen.r[i, 1], onscreen.k[i, 0] / divider, onscreen.k[i, 1] / divider)
    plt.savefig("dipole_" + str(int(np.round(theta * 180 / np.pi))) + "_theta_setup.svg", dpi=600)
    plt.show()
    plt.close()


screen.clear()

prog = progress.Progress(max=iterations)
num=1000
for i in range(iterations):

    dipole = make_dipole(0.03,theta, alpha_max, num,mode='ray')

    onlense1_front = lense1.front.interact_with_all_wavelets(dipole)
    #onlense1_front.mode = modes['gaussian']

    # for j in range(int(iterations/100)):
    #     dipole = make_dipole(theta, alpha_max, num)
    #     onlense1_front.append_wavelets(lense1.front.interact_with_all_wavelets(dipole))

    onlense1_back = lense1.back.interact_with_all_wavelets(onlense1_front)
    onlense2_front = lense2.front.interact_with_all_wavelets(onlense1_back)
    onlense2_back = lense2.back.interact_with_all_wavelets(onlense2_front)

    onlense2_back = lense2.back.interact_with_all_wavelets(onlense2_front)
    onlense2_back.mode = modes['gaussian']
    onscreen = screen.interact_with_all_wavelets(onlense2_back)

    screen.add_field_from_wavelets(onscreen)

    print(str(i) + " count on screen1: " + str(screen.count))
    prog.next()
    print(str(np.round(prog.percent,1))+'%  ' + str(prog.eta_td))


plt.plot(screen.midpoints[:,1],screen.field ** 2)
plt.xlabel("position on screen / m")
plt.ylabel("intensity / a.u.")
plt.savefig("dipole_"+str(int(np.round(theta*180/np.pi)))+"_theta_onscreen.png", dpi=600)
plt.show()
plt.close()


plt.plot(screen.midpoints[:,1],screen.hits)
plt.xlabel("position on screen / m")
plt.ylabel("number of hits")
plt.savefig("dipole_"+str(int(np.round(theta*180/np.pi)))+"_theta_onscreen_hits.png", dpi=600)
plt.show()
plt.close()


np.savetxt("dipole_"+str(int(np.round(theta*180/np.pi)))+"_theta_onscreen.csv",np.vstack((screen.midpoints[:,1],screen.field ** 2)))

np.savetxt("dipole_"+str(int(np.round(theta*180/np.pi)))+"_theta_onscreen_hits.csv",np.vstack((screen.midpoints[:,1],screen.hits)))

