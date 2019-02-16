import locale

locale.setlocale(locale.LC_NUMERIC, 'C')

import numpy as np
from numba import njit, jit, jitclass, float64, int64, void, boolean, uint64, complex128, prange
import cmath
from scipy import interpolate

#c = 2.998e8  # m/s
c = 1.0  # m/s

modes = {
    "spherical": 1,
    "gaussian": 2,
    "ray": 3,
}

@njit
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

@njit
def rotate_vector(vector, theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.zeros((2, 2))
    R[0, 0] = c
    R[1, 0] = -s
    R[0, 1] = s
    R[1, 1] = c
    #return np.matmul(R, vector)
    return R @ vector

# @jit(nopython=True)
# def gen_rotation_matrix(theta):
#     c, s = np.cos(theta), np.sin(theta)
#     R = np.array([[c, -s], [s, c]])
#     return R

@njit
def weightedChoice(weights):
    """
    Return a random item from objects, with the weighting defined by weights
    (which must sum to 1).
    From: http://stackoverflow.com/a/10803136
    """
    cs = np.cumsum(weights)  # An array of the weights, cumulatively summed.
    idx = np.sum(cs < np.random.rand())  # Find the index of the first weight over a random value.
    return idx

@njit
def determinant(v, w):
    return v[0] * w[1] - v[1] * w[0]

@jit('double(double[:], double[:])',nopython=True)
def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    v = np.dot(v1_u, v2_u)

    if determinant(v1,v2) > 0:
        return np.real(cmath.acos(v))
    else:
        return np.real(-cmath.acos(v))

# calculation of line intersection from
# https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines-in-python
@njit
def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0] * p2[1] - p2[0] * p1[1])
    return A, B, -C

@njit
def intersection(L1, L2):
    D = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    x = Dx / D
    y = Dy / D
    return x, y

@njit
def get_intersect(A, B, C, D):
    a1 = B[1] - A[1]
    b1 = A[0] - B[0]
    c1 = a1 * (A[0]) + b1 * (A[1])

    a2 = D[1] - C[1]
    b2 = C[0] - D[0]
    c2 = a2 * (C[0]) + b2 * (C[1])

    determinant = a1 * b2 - a2 * b1

    if determinant == 0:
        return np.inf, np.inf
    else :
        x = (b2 * c1 - b1 * c2) / determinant
        y = (a1 * c2 - a2 * c1) / determinant
        return x,y


spec_Wavelets = [
    ('r', float64[:,:]),
    ('k', float64[:,:]),
    ('t0', float64[:]),
    ('phases', float64[:]),
    ('intensities', float64[:]),
    ('surface_index', int64[:]),
    ('wavelength', float64),
    #('f', float64),
    ('mode', uint64),
    ('n', int64)
]


@jitclass(spec_Wavelets)
class Wavelets(object):
    def __init__(self, r, k, t0, wavelength, intensities, phases, mode):
        self.r = r
        self.n = r.shape[0]
        self.k = np.zeros((self.n,2))
        for i in range(self.n):
            self.k[i,:] = ((k[i,:] / np.linalg.norm(k[i,:])) * 2 * np.pi) / wavelength
        self.t0 = t0
        self.phases = phases
        self.wavelength = wavelength
        #self.f = c / wavelength
        self.mode = mode
        self.intensities = intensities
        self.surface_index = np.zeros(self.n,dtype=np.int64)

#@njit(void(ExposurePoints.class_type.instance_type, ContourPoints.class_type.instance_type), parallel=False)
#def calc_forces(epoints, cpoints):


    def calc_optical_path_of_wavelet(self, index, point, n=1.0):
        f = (c/n) / self.wavelength
        #return np.sqrt(np.sum(np.square(np.subtract(self.r[index,:], point)))) / (c/n) + self.t0[index]
        #self.phases[index] += np.dot(np.subtract(self.r[index,:], point),self.k[index,:])*n
        #self.phases[index] += 2 * cmath.pi * f *np.sqrt(np.sum(np.square(np.subtract(self.r[index,:], point)))) / (c/n)
        #return 2 * cmath.pi * f *np.sqrt(np.sum(np.square(np.subtract(self.r[index,:], point)))) / (c/n)
        return np.sqrt(np.sum(np.square(np.subtract(self.r[index,:], point))))*n

    def calc_t(self, index, point, n=1.0):
        #t = np.sqrt(np.sum(np.square(np.subtract(self.r[index,:], point))))*n/c
        t = np.sqrt(np.sum(np.square(np.subtract(self.r[index,:], point)))) / (c/n)
        #self.t0[index] += t
        return t


    def calc_field(self, points, t, n=1.0):
        f = (c/n) / self.wavelength
        field = np.zeros(points.shape[0],dtype=np.float64)
        for j in range(points.shape[0]):
            for i in range(self.n):
                r = self.r[i,:] - points[j,:]
                field[j] += 1 / r * cmath.exp(1j * ( ( np.dot(self.k[i, :],r)
                                                     - 2 * cmath.pi * f * (t - self.t0[i]) + self.phases[i])))
        return np.real(field)

    def field_at_r(self,t,index):
        n = 1.0
        f = (c/n) / self.wavelength
        field = np.real(cmath.exp(1j *( -2 * cmath.pi * f * (t - self.t0[index]) + self.phases[index])))
        return field#*rotate_vector(self.k[index,:],np.pi/2) / np.linalg.norm(self.k[index,:])

    def phasor_at_r(self,index):
        n = 1.0
        f = (c/n) / self.wavelength
        phase = 2 * cmath.pi * f * self.t0[index] + self.phases[index]
        #phase = phase % (np.pi * 2)
        #print(cmath.exp(1j*phase))
        return  cmath.exp(1j*phase)#.real, cmath.exp(1j*phase).imag  #*rotate_vector(self.k[index,:],np.pi/2) / np.linalg.norm(self.k[index,:])

    def calc_probability_of_wavelet(self,index, point1, point2):
        v1 = np.subtract(point1,self.r[index, :])
        v2 = np.subtract(point2,self.r[index, :])

        if self.mode == 1: # spherical
            phi2 = angle_between(v1, v2)
            probability = np.abs((phi2) / (np.pi))

        elif self.mode == 2: # gaussian
            phi1 = angle_between(v1, self.k[index, :])
            phi2 = angle_between(v2, self.k[index, :])
            if phi2 > phi1:
                probability = np.real( 1 / (4 * np.pi) * (
                            cmath.sin(2 * phi1 + np.pi) - 2 * phi1 - cmath.sin(2 * phi2 + np.pi) + 2 * phi2))
            else:
                probability = np.real( 1 / (4 * np.pi) * (
                            cmath.sin(2 * phi2 + np.pi) - 2 * phi2 - cmath.sin(2 * phi1 + np.pi) + 2 * phi1))
            # if phi1 == phi2:
            #     probability = np.real( 1 / (8 * np.pi))

        elif self.mode == 3: #ray
            if self.is_between(self.k[index, :],v1,v2):
                probability = 1.0
            else:
                probability = 0.0

        return probability



    def angle_between_clockwise(self, v1, v2):
        """ Returns the angle in radians between vectors 'v1' and 'v2'::
        """
        def determinant(v, w):
            return v[0] * w[1] - v[1] * w[0]
        v1_u = v1 / np.linalg.norm(v1)
        v2_u = v2 / np.linalg.norm(v2)
        v = np.dot(v1_u, v2_u)
        if determinant(v1,v2) > 0:
            return cmath.acos(v)
        else:
            return 2*np.pi-cmath.acos(v)

    def angle_between_anticlockwise(self, v1, v2):
        """ Returns the angle in radians between vectors 'v1' and 'v2'::
        """
        def determinant(v, w):
            return v[0] * w[1] - v[1] * w[0]
        v1_u = v1 / np.linalg.norm(v1)
        v2_u = v2 / np.linalg.norm(v2)
        v = np.dot(v1_u, v2_u)
        if determinant(v1,v2) < 0:
            return cmath.acos(v)
        else:
            return 2*np.pi-cmath.acos(v)

    def is_between(self,k,v1,v2):
        def determinant(v, w):
            return v[0] * w[1] - v[1] * w[0]
        return (determinant(k,v1)>0) ^ (determinant(k,v2)>0)

    def append_wavelets(self, wavelets):
        self.r = np.concatenate((self.r, wavelets.r))
        self.k = np.concatenate((self.k, wavelets.k))
        self.n += wavelets.n
        self.t0 = np.concatenate((self.t0, wavelets.t0))
        self.phases = np.concatenate((self.phases, wavelets.phases))






spec_Surface = [
    ('points', float64[:, :]),
    ('reflectivity', float64),
    ('transmittance', float64),
    ('n1', float64),
    ('n2', float64),
    ('midpoints', float64[:, :]),
    ('intensities', float64[:]),
    ('ks', float64[:, :]),
    ('ts', float64[:]),
    ('field', float64[:,:]),
    ('phasor', complex128[:]),
    ('hits', int64[:]),
    ('normals', float64[:, :]),
    ('count', int64),
]

@jitclass(spec_Surface)
class Surface(object):
    def __init__(self, points, reflectivity, transmittance, n1=1.0, n2=1.0):
        self.points = points
        self.reflectivity = reflectivity
        self.transmittance = transmittance
        self.n1 = n1
        self.n2 = n2
        self.midpoints = np.zeros((self.points.shape[0]-1,2))
        self.intensities = np.zeros(self.points.shape[0]-1)
        self.ts = np.zeros(self.points.shape[0]-1)
        self.ks = np.zeros((self.points.shape[0]-1,2))
        self.normals = np.zeros((self.points.shape[0] - 1,2))
        for i in range(self.points.shape[0]-1):
            self.midpoints[i] = 0.5 * np.add(self.points[i], self.points[i + 1])
            normal = rotate_vector(np.subtract(self.points[i], self.points[i + 1]), np.pi / 2)
            normal = unit_vector(normal)
            self.normals[i] = normal

        self.field = np.zeros((self.midpoints.shape[0],2),np.float64)
        self.phasor = np.zeros(self.midpoints.shape[0],dtype=np.complex128)
        self.hits = np.zeros(self.midpoints.shape[0],dtype=np.int64)
        self.count = 0

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

    def reflected_k(self, vector, normal):
        reflected = vector - (2 * (np.dot(normal, vector)) * normal)
        return reflected

    def transmitted_k(self, k, normal):
        alpha = angle_between(k, normal)
        beta = cmath.asin( (self.n1/self.n2) * np.sin(alpha) ).real
        transmitted = rotate_vector(normal, beta)
        return transmitted

    def localize_wavelet(self, wavelets, index):
        probabilities = np.zeros(self.points.shape[0] - 1, dtype=np.float64)
        for i in range(self.points.shape[0] - 1):
            probabilities[i] = wavelets.calc_probability_of_wavelet(index,self.points[i], self.points[i + 1])

        if (wavelets.mode == 1) or (wavelets.mode == 2):
            # if np.random.rand() >= np.sum(probabilities):
            #     probabilities /= np.sum(probabilities)
            #     index = self.weightedChoice(probabilities)
            #     return self.midpoints[index], self.normals[index], True
            # else:
            #     return np.array([0.0, 0.0]), np.array([0.0, 0.0]), False
            probabilities /= np.sum(probabilities)
            surface_index = weightedChoice(probabilities)
            return surface_index, self.midpoints[surface_index,:],True

        elif wavelets.mode == 3:
            for j in range(len(probabilities)):
                if probabilities[j] > 0:
                    #x,y = intersection(line(wavelets.r[index,:],wavelets.r[index,:]+wavelets.k[index,:]),line(self.points[i],self.points[i+1]))
                    x,y = get_intersect(wavelets.r[index,:],wavelets.r[index,:]+wavelets.k[index,:],self.points[j],self.points[j+1])
                    return j, np.array([x,y]) ,True
            return -1, np.array([0.0,0.0]), False

    def interact_with_wavelet(self, wavelets, index):
        surface_index, point, hit = self.localize_wavelet(wavelets, index)
        absorbed = True
        if hit:
            if self.reflectivity > 0.0:
                if np.random.rand() <= self.reflectivity:
                    new_k = self.reflected_k(wavelets.k[index,:], self.normals[surface_index,:])
                    absorbed = False
            elif self.transmittance > 0.0:
                if np.random.rand() <= self.transmittance:
                    new_k = self.transmitted_k(wavelets.k[index,:], self.normals[surface_index,:])
                    absorbed = False

            if not absorbed:
                #wavelets.calc_optical_path_of_wavelet(index, self.midpoints[surface_index, :], self.n1)
                wavelets.calc_optical_path_of_wavelet(index, point, self.n1)
                #k = k / np.linalg.norm(k) * 2 * np.pi / wavelets.wavelength
                return surface_index, point, new_k, True

        return -1, np.array([0.0, 0.0]),np.array([0.0, 0.0]), False

    def add_field_from_wavelets(self,wavelets):
        self.count += wavelets.n

        for i in range(wavelets.n):
            j = wavelets.surface_index[i]
            self.field[j] += wavelets.field_at_r(i,1.0)#*np.sin(np.real(angle_between(self.rotate_vector(wavelets.k[i,:],np.pi/2),(self.points[j+1, :]-self.points[j, :]))))
            #self.field[j] += np.cos((wavelets.phase_at_r(i)-self.phase[j])/2)
            self.hits[j] += 1

    def add_phase_from_wavelets(self,wavelets):
        self.count += wavelets.n

        for i in range(wavelets.n):
            j = wavelets.surface_index[i]
            #self.phase[j] += ( (wavelets.phase_at_r(i)  ) + self.phase[j] )/2
            self.phasor[j] += wavelets.phasor_at_r(i)
            self.hits[j] += 1


    def interact_with_all_wavelets_rays(self, wavelets):

        for i in range(wavelets.n):
            surface_index, point, k, hit = self.interact_with_wavelet(wavelets,i)
            if hit:
                self.hits[surface_index] += 1
                self.intensities[surface_index] =  wavelets.intensities[i]
                self.ks[surface_index,:] += k

                self.ts[surface_index] += wavelets.calc_t(i,point,self.n1)+wavelets.t0[i]

                self.phasor[surface_index] +=  wavelets.phasor_at_r(i)
                self.count += 1
        #self.intensities /= self.hits
        self.ts /= self.hits
        for i in range(self.ks.shape[0]):
            self.ks[i,:] = ((self.ks[i,:] / np.linalg.norm(self.ks[i,:])) * 2 * np.pi) / wavelets.wavelength

    def interact_with_all_wavelets_other(self, wavelets):
        hits = np.zeros(wavelets.n, dtype=np.float64)
        rs = np.zeros((wavelets.n,2), dtype=np.float64)
        ks = np.zeros((wavelets.n,2), dtype=np.float64)
        ts = np.zeros(wavelets.n, dtype=np.float64)
        s_index = np.zeros(wavelets.n,dtype=np.int64)
        intensities = np.zeros(wavelets.n, dtype=np.float64)

        r = np.zeros(2)
        k = np.zeros(2)
        t = 0.0
        hit = False

        for i in range(wavelets.n):
            surface_index, point, k, hit = self.interact_with_wavelet(wavelets,i)
            intensities[i] += wavelets.intensities[i]
            hits[i] = hit
            rs[i] = point
            ks[i] = k
            ts[i] = t
            s_index[i] = surface_index
            self.phasor[surface_index] += wavelets.phasor_at_r(i)
            self.hits[surface_index] += 1

        self.ts /= self.hits
        self.count += np.sum(hits)
        #indices = (hits > 0)
        #intensities = np.ones(len(indices))
        #new_wavelets = Wavelets(rs[indices,:],ks[indices,:],ts[indices],wavelets.wavelength,intensities,wavelets.phases[indices],wavelets.mode)
        #new_wavelets.surface_index = s_index[indices]
        #return new_wavelets



    def clear(self):
        self.field = np.zeros((self.midpoints.shape[0], 2), np.float64)
        self.hits = np.zeros(self.midpoints.shape[0], dtype=np.int64)
        self.intensities = np.zeros(self.points.shape[0]-1)
        self.ts = np.zeros(self.points.shape[0]-1)
        self.ks = np.zeros((self.points.shape[0]-1,2))
        self.phasor = np.zeros(self.midpoints.shape[0], dtype=np.complex128)
        self.count = 0

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


    # def fliplr(self):
    #     self.front.points[:, 0] -= self.x
    #     self.back.points[:, 0] -= self.x
    #     self.front.points[:, 1] -= self.y
    #     self.back.points[:, 1] -= self.y
    #     for i in range(self.front.points.shape[0]):
    #         self.front.points[i, :] = rotate_vector(self.front.points[i,:], np.pi)
    #         self.back.points[i, :] = rotate_vector(self.back.points[i,:], np.pi)
    #     self.front.points[:, 0] += self.x
    #     self.back.points[:, 0] += self.x
    #     self.front.points[:, 1] += self.y
    #     self.back.points[:, 1] += self.y
    #     self.front._update_midpoints()
    #     self.front._update_normals()
    #     #self.front.flip_normals()
    #     self.back._update_midpoints()
    #     self.front._update_normals()
    #     buf = self.front
    #     self.front = self.back
    #     self.back = buf
    #     self.back.flip_normals()
    #     self.back.n1 = self.n2
    #     self.back.n2 = self.n1
    #     self.front.n1 = self.n1
    #     self.front.n2 = self.n2


# @jit
# def make_dipole(wavelength, theta,alpha_max,num, mode='ray'):
#     rs = np.zeros((num, 2))
#     ks = np.zeros((num, 2))
#     ks[:, 0] = np.repeat(1.0, num)
#     alphas = np.linspace(-alpha_max, alpha_max, num)
#     probabilities = (np.cos(theta - alphas) ** 2)
#     #probabilities = np.ones(num)
#     probabilities /= np.sum(probabilities)
#     b = np.zeros(num)
#
#     for i in range(ks.shape[0]):
#         index = weightedChoice(probabilities)
#         b[i] = alphas[index]
#         ks[i, :] = rotate_vector(ks[i, :], alphas[index])
#
#     phases = np.zeros((num))
#     for i in range(len(alphas)):
#         if theta-b[i]-np.pi/2 < 0:
#             phases[i] = np.pi
#
#     t0s = np.zeros((num))
#     return Wavelets(r=rs, k=ks, t0=t0s, wavelength=wavelength, phases=phases, mode=modes[mode])

@jit
def make_dipole(wavelength, theta,alpha_max,num, mode='ray'):
    rs = np.zeros((num, 2))
    ks = np.zeros((num, 2))
    ks[:, 0] = np.repeat(1.0, num)
    alphas = np.linspace(-alpha_max, alpha_max, num)
    probabilities = (np.cos(theta - alphas) ** 2)
    #probabilities = np.ones(num)
    probabilities /= np.sum(probabilities)
    b = np.zeros(num)

    for i in range(ks.shape[0]):
        ks[i, :] = rotate_vector(ks[i, :], alphas[i])

    phases = np.zeros((num))
    for i in range(len(alphas)):
        if theta-alphas[i]-np.pi/2 < 0:
            phases[i] = np.pi

    t0s = np.zeros((num))
    return Wavelets(r=rs, k=ks, t0=t0s, wavelength=wavelength, intensities=probabilities, phases=phases, mode=modes[mode])


def generate_wavelets_from_surface(surf: Surface,n: int,wavelength: float):
    indices = ((surf.intensities > 0) & (surf.hits > 0) & np.logical_not(np.isnan(surf.ts)))

    # #x = np.arange(0, surf.intensities[indices].shape[0], 1)
    # x = np.linspace(0, 1, surf.intensities[indices].shape[0])
    # f_x = interpolate.UnivariateSpline(x, surf.midpoints[indices,0], k=3, s=None)#len(x))*1e6)
    # f_y = interpolate.UnivariateSpline(x, surf.midpoints[indices, 1], k=3, s=None)#len(x)*1e6)
    # f_intensities = interpolate.UnivariateSpline(x, surf.intensities[indices], k=3, s=None)#len(x)*1e6)  # 0.0001)#len(x)*1e1)
    # f_ts = interpolate.UnivariateSpline(x, surf.ts[indices], k=3, s=None)#len(x)*1e6)  # 0.0001)#len(x)*1e1)
    # f_kx = interpolate.UnivariateSpline(x, surf.ks[indices,0], k=3, s=None)#len(x)*1e6)  # 0.0001)#len(x)*1e1)
    # f_ky = interpolate.UnivariateSpline(x, surf.ks[indices,1], k=3, s=None)#len(x)*1e6)  # 0.0001)#len(x)*1e1)
    # #x = np.linspace(0, surf.intensities[indices].shape[0], n)
    # #x = np.linspace(-1, surf.intensities[indices].shape[0], n)
    # x = np.linspace(0, 1, n)
    # phases = np.zeros(n)

    # #x = np.arange(0, surf.intensities[indices].shape[0], 1)
    # x = np.linspace(0, 1, surf.intensities[indices].shape[0])
    # t = np.linspace(0.1,0.9,5)
    # f_x = interpolate.LSQUnivariateSpline(x, surf.midpoints[indices,0],t,check_finite=True)
    # f_y = interpolate.LSQUnivariateSpline(x, surf.midpoints[indices, 1],t,check_finite=True)
    # f_intensities = interpolate.LSQUnivariateSpline(x, surf.intensities[indices],t,check_finite=True)
    # f_ts = interpolate.LSQUnivariateSpline(x, surf.ts[indices],t,check_finite=True)
    # f_kx = interpolate.LSQUnivariateSpline(x, surf.ks[indices,0],t,check_finite=True)
    # f_ky = interpolate.LSQUnivariateSpline(x, surf.ks[indices,1],t,check_finite=True)
    # #x = np.linspace(0, surf.intensities[indices].shape[0], n)
    # #x = np.linspace(-1, surf.intensities[indices].shape[0], n)
    # x = np.linspace(0, 1, n)
    # phases = np.zeros(n)

    #x = np.arange(0, surf.intensities[indices].shape[0], 1)
    x = np.linspace(0, 1, surf.intensities[indices].shape[0])
    f_x = interpolate.InterpolatedUnivariateSpline(x, surf.midpoints[indices,0], check_finite=True)
    f_y = interpolate.InterpolatedUnivariateSpline(x, surf.midpoints[indices, 1],check_finite=True)
    f_intensities = interpolate.InterpolatedUnivariateSpline(x, surf.intensities[indices],check_finite=True)
    f_ts = interpolate.InterpolatedUnivariateSpline(x, surf.ts[indices],check_finite=True)
    f_kx = interpolate.InterpolatedUnivariateSpline(x, surf.ks[indices,0],check_finite=True)
    f_ky = interpolate.InterpolatedUnivariateSpline(x, surf.ks[indices,1],check_finite=True)
    #x = np.linspace(0, surf.intensities[indices].shape[0], n)
    #x = np.linspace(-1, surf.intensities[indices].shape[0], n)
    x = np.linspace(0, 1, n)
    phases = np.zeros(n)

    #Wavelets(r,k,t0,...)
    return Wavelets(r=np.vstack((f_x(x),f_y(x))).T, k=np.vstack((f_kx(x),f_ky(x))).T, t0=f_ts(x), wavelength=wavelength, intensities=f_intensities(x), phases=phases, mode=modes['ray'])

import matplotlib.pyplot as plt

def plot_all(surf, wavelets,title=''):
    indices = ((surf.intensities > 0) & (surf.hits > 0) & np.logical_not(np.isnan(surf.ts)))
    plt.plot(surf.points[:, 0], surf.points[:, 1], 'bx')
    plt.plot(wavelets.r[:, 0], wavelets.r[:, 1], 'r.')
    k = wavelets.k / np.sqrt(wavelets.k[:, 0].max() ** 2 + wavelets.k[:,
                                                                       1].max() ** 2)  # np.linalg.norm(onlense1_front.k,axis=0)
    plt.quiver(wavelets.r[:, 0], wavelets.r[:, 1], k[:, 0], k[:, 1])
    plt.title(title)
    plt.show()
    plt.plot(surf.hits)
    plt.title(title+' hits')
    plt.show()
    plt.plot(wavelets.r[:, 1], wavelets.intensities)
    plt.plot(surf.midpoints[indices, 1], surf.intensities[indices])
    #print(np.array2string(surf.intensities, separator=','))
    plt.title(title+' intensity')
    plt.show()
    plt.plot(wavelets.r[:, 1], wavelets.t0)
    x = np.linspace(0, 1, surf.intensities[indices].shape[0])
    # f_ts = interpolate.UnivariateSpline(x, lense1.front.ts[indices], k=3, s=None)  # len(x)*1e6)  # 0.0001)#len(x)*1e1)
    # x = np.linspace(-1, lense1.front.intensities[indices].shape[0], 100)
    t = np.linspace(0.1, 0.9, 5)
    # f_ts = interpolate.LSQUnivariateSpline(x, lense1.front.ts[indices], t,check_finite=True)
    f_ts = interpolate.InterpolatedUnivariateSpline(x, surf.ts[indices], check_finite=True)
    x2 = np.linspace(0, 1, 100)
    x3 = np.linspace(surf.midpoints[indices, 1].min(), surf.midpoints[indices, 1].max(), 100)
    plt.plot(x3, f_ts(x2), 'rx')
    plt.plot(surf.midpoints[indices, 1], surf.ts[indices], 'b+')
    plt.title(title+' t0')
    plt.show()

#
# @njit
# def savitzky_golay(y, window_size, order, deriv=0, rate=1):
#     """Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
#     The Savitzky-Golay filter removes high frequency noise from data.
#     It has the advantage of preserving the original shape and
#     features of the signal better than other types of filtering
#     approaches, such as moving averages techniques.
#     modified, from ftp.princetoninstruments.com/public/Software/User/Written/Utilities/Python/savgol_filter.py
#     Parameters
#     ----------
#     y : array_like, shape (N,)
#         the values of the time history of the signal.
#     window_size : int
#         the length of the window. Must be an odd integer number.
#     order : int
#         the order of the polynomial used in the filtering.
#         Must be less then `window_size` - 1.
#     deriv: int
#         the order of the derivative to compute (default = 0 means only smoothing)
#     Returns
#     -------
#     ys : ndarray, shape (N)
#         the smoothed signal (or it's n-th derivative).
#     Notes
#     -----
#     The Savitzky-Golay is a type of low-pass filter, particularly
#     suited for smoothing noisy data. The main idea behind this
#     approach is to make for each point a least-square fit with a
#     polynomial of high order over a odd-sized window centered at
#     the point.
#     Examples
#     --------
#     t = np.linspace(-4, 4, 500)
#     y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
#     ysg = savitzky_golay(y, window_size=31, order=4)
#     import matplotlib.pyplot as plt
#     plt.plot(t, y, label='Noisy signal')
#     plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
#     plt.plot(t, ysg, 'r', label='Filtered signal')
#     plt.legend()
#     plt.show()
#     References
#     ----------
#     .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
#        Data by Simplified Least Squares Procedures. Analytical
#        Chemistry, 1964, 36 (8), pp 1627-1639.
#     .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
#        W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
#        Cambridge University Press ISBN-13: 9780521880688
#     """
#     import numpy as np
#     from math import factorial
#
#     try:
#         window_size = np.abs(np.int(window_size))
#         order = np.abs(np.int(order))
#     except ValueError, msg:
#         raise ValueError("window_size and order have to be of type int")
#     if window_size % 2 != 1 or window_size < 1:
#         raise TypeError("window_size size must be a positive odd number")
#     if window_size < order + 2:
#         raise TypeError("window_size is too small for the polynomials order")
#     order_range = range(order+1)
#     half_window = (window_size -1) // 2
#     # precompute coefficients
#     b = np.mat([[k**i for i in order_range] for k in range(-half_window,
#                                                             half_window+1)])
#     m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
#     # pad the signal at the extremes with
#     # values taken from the signal itself
#     firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
#     lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
#     y = np.concatenate((firstvals, y, lastvals))
#     return np.convolve( m[::-1], y, mode='valid')