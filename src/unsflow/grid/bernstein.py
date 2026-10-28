import numpy as np
import math

def bernstein_n_i(n, i, u):
    return (
        math.comb(n, i)
        * u**i
        * (1 - u)**(n - i)
    )

def bezier_curve(points, u=np.linspace(0, 1, 50)):
    npoints = len(points)
    ndim = len(points[0])
    n = npoints - 1

    curve = np.zeros((ndim, len(u)))

    for ip in range(npoints):
        p = points[ip]
        b = bernstein_n_i(n, ip, u)

        for dim in range(ndim):
            curve[dim, :] += b * p[dim]

    return curve
