import vec3.vec3
import numpy as np
from scipy.linalg import null_space, cholesky, inv

def calc_w(vp1, vp2, vp3):
    x1, y1 = vp1
    x2, y2 = vp2
    x3, y3 = vp3
    A = np.array([[(x1*x2)+(y1*y2), x2+x1, y1+y2, 1],
                   [(x1*x3)+(y1*y3), x3+x1, y1+y3, 1],
                   [(x3*x2)+(y3*y2), x2+x3, y3+y2, 1]])
    w = null_space(A)
    w /= w[3]
    return np.array([[w[0],  0,   w[1]],
                     [0,    w[0], w[2]],
                     [w[1], w[2], w[3]]], dtype=np.float32)


def project_to_stereomap(p, r, w, h):
    sphere_center = vec3([w//2, h//2, r])
    v = p - sphere_center
    s = r * (v/abs(v))
    return np.array(s[:2])


def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0] * p2[1] - p2[0] * p1[1])
    return A, B, -C


def intersection(L1, L2):
    D = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x, y
    else:
        return False

