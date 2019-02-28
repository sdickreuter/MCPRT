import locale

locale.setlocale(locale.LC_NUMERIC, 'C')

import numpy as np
from numba import njit, jitclass, float64, int64, void, boolean, uint64, complex128, prange, deferred_type

import cmath
from scipy import interpolate

from utils import *

#c = 2.998e8  # m/s
c = 1.0  # m/s


spec_Dipole = [
    ('r', float64[:]),
    ('k', float64[:]),
    ('phase', float64),
    ('wavelength', float64),

]

Dipole_type = deferred_type()

@jitclass(spec_Dipole)
class Dipole(object):

    def __init__(self, r, k, phase, wavelength):
        self.r = r
        self.k = unit_vector(k)
        self.phase = phase
        self.k = k * 2 * np.pi / wavelength
        self.wavelength = wavelength

Dipole_type.define(Dipole.class_type.instance_type)



spec_DipoleSet = [
    ('r', float64[:,:]),
    ('k', float64[:,:]),
    ('phase', float64[:]),
    ('wavelength', float64),

]

DipoleSet_type = deferred_type()

@jitclass(spec_DipoleSet)
class DipoleSet(object):

    def __init__(self, r, k, phase, wavelength):
        self.r = r
        self.k = unit_vector(k)
        self.phase = phase
        self.k = k * 2 * np.pi / wavelength
        self.wavelength = wavelength

DipoleSet_type.define(DipoleSet.class_type.instance_type)



spec_Surface = [
    ('points', float64[:,:]),
    ('midpoints', float64[:, :]),
    ('normals', float64[:, :]),
    ('E', complex128[:,:]),
    ('ks', float64[:, :]),
    ('counts', int64[:]),
    ('n1', float64),
    ('n2', float64),
    ('reflectivity', float64),
    ('transmittance', float64),
]

Surface_type = deferred_type()

@jitclass(spec_Surface)
class Surface(object):

    def __init__(self, points, reflectivity, transmittance, n1=1.0, n2=1.0):
        self.points = points
        self.reflectivity = reflectivity
        self.transmittance = transmittance
        self.n1 = n1
        self.n2 = n2
        self.midpoints = np.zeros((self.points.shape[0] - 1, 2))
        self.ks = np.zeros((self.points.shape[0] - 1, 2))
        self.normals = np.zeros((self.points.shape[0] - 1, 2))
        for i in range(self.points.shape[0] - 1):
            self.midpoints[i] = 0.5 * np.add(self.points[i], self.points[i + 1])
            normal = rotate_vector(np.subtract(self.points[i], self.points[i + 1]), -np.pi / 2)
            normal = unit_vector(normal)
            self.normals[i] = normal

        self.E = np.zeros((self.midpoints.shape[0],2), dtype=np.complex128)
        self.counts = np.zeros(self.midpoints.shape[0], dtype=np.int64)

    def _update_midpoints(self):
        for i in range(self.points.shape[0]-1):
            self.midpoints[i] = 0.5 * np.add(self.points[i], self.points[i + 1])

    def _update_normals(self):
        for i in range(self.points.shape[0]-1):
            normal = rotate_vector(np.subtract(self.points[i], self.points[i + 1]), np.pi / 2)
            normal = unit_vector(normal)
            self.normals[i] = normal

    def flip_normals(self):
        for i in range(self.normals.shape[0]):
            self.normals[i] =  rotate_vector(self.normals[i], np.pi )

Surface_type.define(Surface.class_type.instance_type)


@njit
def reflected_k(vector, normal):
    reflected = vector - (2 * (np.dot(normal, vector)) * normal)
    return reflected

@njit
def transmitted_k(k, normal, n1, n2):
    alpha = angle_between(k, normal)
    beta = cmath.asin( (n1/n2) * np.sin(alpha) ).real
    if n1 > n2:
        beta = -beta
    if np.abs(beta) < np.pi/2:
        transmitted = rotate_vector(normal, beta)
    else:
        transmitted = np.array([np.NaN,np.NaN])
    return transmitted


@njit
def calcInt(alpha1,alpha2):
    #return 2*(np.cos(alpha))**2/(np.pi)
    #return integrate.quad(lambda x: 2*np.cos(x)**2/np.pi,alpha1,alpha2)
    return ( -alpha1 - np.sin(alpha1)*np.cos(alpha1) + alpha2 + np.sin(alpha2)*np.cos(alpha2))/np.pi

@njit
def interact_dipole_with_surface(dipole, surf):
    new_dipoles = []
    f = (c / surf.n1) / dipole.wavelength
    #f = c/ dipole.wavelength#(c / surf.n1) / dipole.wavelength
    phasor = np.zeros((2))
    for i in range(surf.midpoints.shape[0]):
        r = surf.midpoints[i, :] - dipole.r
        alpha = angle_between(dipole.k, r)
        E0 = np.sqrt(2) * np.sqrt(1+np.cos(alpha)/(2*dipole.wavelength))*np.exp(1j*dipole.k*r+1j*dipole.phase)/np.sqrt(length(r))
        surf.E[i, :] += E0*unit_vector(rotate_vector(r,np.pi/2))

        trans_k = transmitted_k(rotate_vector(np.real(surf.E[i, :]), -np.pi / 2), surf.normals[i, :], surf.n1, surf.n2)
        trans_k = unit_vector(trans_k)
        #phase = 2 * np.pi * f * length(v) / (c / surf.n1) + dipole.phase
        if not np.isnan(trans_k[0]):
            new_dipoles.append(Dipole(surf.midpoints[i, :],
                                      trans_k,
                                      0.0,
                                      dipole.wavelength))
    return new_dipoles

@njit
def interact_dipoles_with_surface(dipoles, surf):
    new_dipoles = []
    for i in range(surf.midpoints.shape[0]):
        for dipole in dipoles:
            #interact_dipole_with_surface(d,surf)
            #f = (c / surf.n1) / dipole.wavelength
            #f = c / d.wavelength  # (c / surf.n1) / dipole.wavelength
            r = surf.midpoints[i, :] - dipole.r
            alpha = angle_between(dipole.k, r)
            E0 = np.sqrt(2) * np.sqrt(1 + np.cos(alpha) / (2 * dipole.wavelength)) * np.exp(
                1j * dipole.k * r + 1j * dipole.phase) / np.sqrt(length(r))
            surf.E[i, :] += E0 * unit_vector(rotate_vector(r, np.pi / 2))


            # phase = 2 * np.pi * f * length(v) / (c / surf.n1) + dipole.phase

        trans_k = transmitted_k(rotate_vector(np.real(surf.E[i, :]), np.pi / 2), surf.normals[i, :], surf.n1, surf.n2)
        trans_k = unit_vector(trans_k)
        if not np.isnan(trans_k[0]):
            new_dipoles.append(Dipole(surf.midpoints[i, :],
                                          trans_k,
                                          0.0,
                                      dipoles[0].wavelength))
    return new_dipoles


@njit
def interact_dipoles_with_screen(dipoles, surf):
    for i in range(surf.midpoints.shape[0]):
        for dipole in dipoles:
            #interact_dipole_with_surface(d,surf)
            #f = (c / surf.n1) / dipole.wavelength
            #f = c / d.wavelength  # (c / surf.n1) / dipole.wavelength
            r = surf.midpoints[i, :] - dipole.r
            alpha = angle_between(dipole.k, r)
            E0 = np.sqrt(2) * np.sqrt((1 + np.cos(alpha)) / (2 * dipole.wavelength)) * np.exp(
                1j * dipole.k * r + 1j * dipole.phase) / np.sqrt(length(r))
            surf.E[i, :] += E0 * unit_vector(rotate_vector(r, np.pi / 2))


            # phase = 2 * np.pi * f * length(v) / (c / surf.n1) + dipole.phase




import time

@jit
def interact_dipoles_with_surface_verbose(dipoles, surf, n = 10):
    dipoles = np.array_split(dipoles, n)
    new_dipoles = []
    for i in range(len(dipoles)):
        start_time = time.time()
        new_dipoles += interact_dipoles_with_surface(list(dipoles[i]), surf)
        print('ETA ' + str(np.round((n - i) * (time.time() - start_time),1))+' s')
    return new_dipoles


@jit
def interact_dipoles_with_screen_verbose(dipoles, surf, n = 10):
    dipoles = np.array_split(dipoles, n)
    for i in range(len(dipoles)):
        start_time = time.time()
        interact_dipoles_with_screen(list(dipoles[i]), surf)
        print('ETA ' + str(np.round((n - i) * (time.time() - start_time),1))+' s')


class Lense(object):

    def __init__(self, x, y, r1, r2, height,reflectivity=0.0,transmittance=1.0, n1=1.0, n2=1.5, num=128):
        self.n1 = n1
        self.n2 = n2
        self.r1 = r1
        self.r2 = r2
        self.x = x
        self.y = y
        self.height = height

        points1 = self._generate_lens_points(self.r1,self.height,num)

        points2 = self._generate_lens_points(self.r2,self.height,num)

        self.front = Surface(points1, reflectivity=reflectivity, transmittance=transmittance, n1=n1, n2=n2)
        self.back = Surface(points2, reflectivity=reflectivity, transmittance=transmittance, n1=n2, n2=n1)

        self.front.points[:,0] -= self.height / 5
        self.back.points[:, 0] += self.height / 5

        self.front.points[:,0] += x
        self.back.points[:, 0] += x
        self.front.points[:,1] += y
        self.back.points[:, 1] += y

        self.front._update_midpoints()
        self.front._update_normals()
        self.back._update_midpoints()
        self.back._update_normals()

        if self.r1 is not np.inf:
            self.front.flip_normals()
        if self.r2 is not np.inf:
            self.back.flip_normals()

        #self.front.flip_normals()
        self.d = self.back.points[:, 0].max() - self.front.points[:, 0].min()


        self.f = self._calc_f()
        print('d: '+str(self.d))
        print('f: ' + str(self.f))

    def _gen_concave_points(self, p0, r, height, num):
        if height > np.abs(r):
            height = np.abs(r)
        v = np.array([r, 0])
        theta_max = np.arcsin(height / r)
        thetas = np.linspace(theta_max, -theta_max, num)
        points = np.zeros((num, 2))
        for i, theta in enumerate(thetas):
            # R = gen_rotation_matrix(theta)
            # buf = np.dot(R, v)
            buf = rotate_vector(v,theta)
            points[i] = p0 + buf
        return points

    def _generate_lens_points(self,r,height, num):

        if (r is not np.inf) and (r is not -np.inf):
            p0 = np.array([0.0, 0.0])
            points1 = self._gen_concave_points(p0, r, height, num)
            if r > 0:
                points1[:, 0] -= points1[:, 0].min()
            else:
                points1[:, 0] -= points1[:, 0].max()
        else:
            points1 = np.zeros((num, 2))
            points1[:, 1] = np.linspace(self.height, -self.height, num)

        return points1

    def _calc_f(self):
        return np.abs(1/ ( (self.n2-self.n1)/self.n1 * ( 1 / self.r1 - 1/ self.r2) + ((self.n2-self.n1)**2*self.d)/(self.n2*self.n1*self.r1*self.r2) ))

    def _calc_f_back(self):
        h1 = -(self.f*(self.n2-1)*self.d)/( self.r2*self.n2)
        return np.abs((self.f-h1) - (-self.front.points[:,0].min()+self.x))

    def _calc_f_front(self):
        h2 = -(self.f*(self.n2-1)*self.d)/( self.r1*self.n2)
        return np.abs((self.f-h2) - (self.back.points[:,0].max()-self.x))


    def shift(self, dx = 0, dy = 0):
        self.x += dx
        self.y += dy
        self.front.points[:, 0] += dx
        self.back.points[:, 0] += dx
        self.front.points[:, 1] += dy
        self.back.points[:, 1] += dy
        self.front._update_midpoints()
        self.back._update_midpoints()


class HyperbolicLense(object):

    def __init__(self, x, y, f, height,reflectivity=0.0,transmittance=1.0, n1=1.0, n2=1.5, num=128):
        self.n1 = n1
        self.n2 = n2
        self.f = f
        self.x = x
        self.y = y
        self.height = height
        self.reflectivity = reflectivity
        self.transmittance = transmittance
        self.num = num

        self._make_surfaces()


    def _make_surfaces(self,flipped = False):

        if flipped:
            points2 = self._gen_hyperbolic_points()
            for i in range(self.front.points.shape[0]):
                points2[i, :] = rotate_vector(self.front.points[i,:], np.pi)
            points2 = np.flipud(points2)
            points1 = self._generate_line()
        else:
            points1 = self._gen_hyperbolic_points()
            points2 = self._generate_line()

        self.front = Surface(points1, reflectivity=self.reflectivity, transmittance=self.transmittance, n1=self.n1, n2=self.n2)
        self.back = Surface(points2, reflectivity=self.reflectivity, transmittance=self.transmittance, n1=self.n2, n2=self.n1)

        self.front.points[:,0] -= self.height / 5
        self.back.points[:, 0] += self.height / 5

        self.front.points[:,0] += self.x
        self.back.points[:, 0] += self.x
        self.front.points[:,1] += self.y
        self.back.points[:, 1] += self.y

        self.front._update_midpoints()
        self.front._update_normals()
        self.back._update_midpoints()
        self.back._update_normals()

        #self.front.flip_normals()
        #self.back.flip_normals()
        self.d = self.back.points[:, 0].max() - self.front.points[:, 0].min()



    # from :
    # https://www.google.de/url?sa=t&rct=j&q=&esrc=s&source=web&cd=6&cad=rja&uact=8&ved=2ahUKEwij4-Ly6M7cAhVRKuwKHW6QDCcQFjAFegQICRAC&url=https%3A%2F%2Fwww.ssl.berkeley.edu%2F~mlampton%2FHyperbolicLens.pdf&usg=AOvVaw2TZIPNb1xqk1ap_BvLLGZb
    def _gen_hyperbolic_points(self):
        C = 1/((self.n2-1)*self.f)
        S = 1-self.n2**2

        y = np.linspace(self.height, -self.height, self.num)
        x = np.zeros(self.num)
        x = C*(y**2) / (1 + np.sqrt( 1 - S*(C**2)*(y**2)) )

        x = x - x.max()
        points = np.zeros((self.num, 2))
        points[:, 0] = x
        points[:, 1] = y
        return points

    def _generate_line(self):

        points1 = np.zeros((self.num, 2))
        points1[:, 1] = np.linspace(self.height, -self.height, self.num)

        return points1


    def shift(self, dx = 0, dy = 0):
        self.x += dx
        self.y += dy
        self.front.points[:, 0] += dx
        self.back.points[:, 0] += dx
        self.front.points[:, 1] += dy
        self.back.points[:, 1] += dy
        self.front._update_midpoints()
        self.back._update_midpoints()


import matplotlib.pyplot as plt

def plot_all(surf, dipoles,title=''):
    plt.plot(surf.points[:, 0], surf.points[:, 1], 'bx')
    n = surf.normals / np.sqrt(surf.normals[:, 0].max() ** 2 + surf.normals[:, 1].max() ** 2)
    plt.quiver(surf.points[:, 0], surf.points[:, 1], n[:, 0], n[:, 1])
    plt.title(title + " normals")
    plt.show()

    # plt.plot(surf.points[:, 0], surf.points[:, 1], 'rx')
    # k = surf.ks / np.sqrt(surf.ks[:, 0].max() ** 2 + surf.ks[:, 1].max() ** 2)
    # plt.quiver(surf.points[:, 0], surf.points[:, 1], k[:, 0], k[:, 1])
    # plt.title(title + " k-vectors")
    # plt.show()

    #plt.plot(surf.points[:, 0], surf.points[:, 1], 'gx')
    #p =  rotate_vector(np.array([0.0,1.0],dtype=np.float64),np.angle(surf.phasors)) * np.abs(surf.phasors)
    #plt.quiver(surf.points[:, 0], surf.points[:, 1], p[:, 0], p[:, 1])
    fig, ax1 = plt.subplots()
    ax1.plot(np.abs(surf.phasors),'b-')
    ax2 = ax1.twinx()
    ax2.plot(np.angle(surf.phasors),'r-')
    plt.title(title + " phasors")
    plt.show()

    for d in dipoles:
        plt.plot(d.r[0], d.r[1], 'bx')
        k = d.k / np.sqrt(np.sum(d.k** 2))
        plt.quiver(d.r[0], d.r[1], k[0], k[1])

    plt.title(title + " dipoles k")
    plt.show()


def plot_E(surf):
    fig, ax1 = plt.subplots()
    # ax1.plot(np.abs(surf.phasors) ** 2, 'b-')
    ax1.plot(np.sqrt(np.sum(np.square(np.real(surf.E)),axis=1)), 'b-')
    plt.show()
    plt.figure()
    plt.scatter(surf.midpoints[:,0],surf.midpoints[:,1])
    plt.quiver(surf.midpoints[:,0],surf.midpoints[:,1], np.real(surf.E[:,0])/1,np.real(surf.E[:,1])/1)
    plt.show()



def plot_dipoles(dipoles):
    plt.figure()
    for d in dipoles:
        plt.scatter(d.r[0],d.r[1])
        k = unit_vector(d.k)
        plt.arrow(d.r[0], d.r[1], k[0]/1, k[1]/1)

    plt.show()

def print_dipoles(dipoles):
    for d in dipoles:
        print("r " + str(np.round(d.r, 1)) + " | k " + str(np.round(d.k, 1)) + " | ph " + str(np.round(d.phase, 1)))