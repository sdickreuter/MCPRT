import locale

locale.setlocale(locale.LC_NUMERIC, 'C')

import numpy as np
from numba import njit, jit, jitclass, float64, int64, void, boolean, uint64, complex128, prange
import cmath
from scipy import interpolate


@njit
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

@njit
def rotate_vector(vector, theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.zeros((2, 2),dtype=np.float64)
    R[0, 0] = c
    R[1, 0] = -s
    R[0, 1] = s
    R[1, 1] = c
    #return np.matmul(R, vector)
    return R @ vector

@njit
def rotate_complex(c, theta):
    rotated = rotate_vector(np.array([c.real,c.imag]),theta)
    return rotated[0]+1j*rotated[1]



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

@njit
def length(v):
    return np.sqrt(np.sum(np.square(v)))

@njit
def angle(v):
    return np.sqrt(np.sum(np.square(v)))


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




