import numpy as np
import matplotlib.pyplot as plt
import Grid
from Grid.src.functions import elliptic_grid_generation
from scipy.interpolate import CubicSpline

""" 
test case to debug the elliptic grid generation method. Define a figure delimited by 4 curved
borders, and see how it goes
"""

nx = 60
ny = 40

# parameteric picture
L = 10
R = 1.5

"""  left border  """
x = np.zeros(ny)
y = np.linspace(0, L, ny)
c_left = np.array((x, y))

"""  bottom border  """
theta = np.linspace(0, 2 * np.pi, nx)
x = np.linspace(0, L, nx)
y = 0.6 * R * np.sin(4*theta)
c_bottom = np.array((x, y))

"""  right border  """
x = np.zeros(ny) + L
y = np.linspace(0, L + R, ny)
c_right = np.array((x, y))

"""  top border  """
theta = np.linspace(0, np.pi / 2, nx)
x = np.linspace(0, L, nx)
y = L + R * np.sin(theta) + 0.9 * np.sin(15 * theta)
c_top = np.array((x, y))

# GENERATION

t_streamwise = np.linspace(0, 1, nx)
t_spanwise = np.linspace(0, 1, ny)

splinex_bottom = CubicSpline(t_streamwise, c_bottom[0, :])
spliney_bottom = CubicSpline(t_streamwise, c_bottom[1, :])

splinex_top = CubicSpline(t_streamwise, c_top[0, :])
spliney_top = CubicSpline(t_streamwise, c_top[1, :])

splinex_left = CubicSpline(t_spanwise, c_left[0, :])
spliney_left = CubicSpline(t_spanwise, c_left[1, :])

splinex_right = CubicSpline(t_spanwise, c_right[0, :])
spliney_right = CubicSpline(t_spanwise, c_right[1, :])

plt.plot(splinex_bottom(t_streamwise), spliney_bottom(t_streamwise), 'o')
plt.plot(splinex_top(t_streamwise), spliney_top(t_streamwise), 'o')
plt.plot(splinex_left(t_spanwise), spliney_left(t_spanwise), 'o')
plt.plot(splinex_right(t_spanwise), spliney_right(t_spanwise), 'o')


def stretching_function(x, alpha):
    return 1 / (1 + np.exp(-alpha * (xi - 0.5)))


# COMPUTATIONAL DOMAIN
xi = np.linspace(0, 1, nx)
# xi = stretching_function(xi, 4)
eta = np.linspace(0, 1, ny)
# eta = stretching_function(eta, 7)
XI, ETA = np.meshgrid(xi, eta, indexing='ij')
X, Y = np.zeros_like(XI), np.zeros_like(ETA)

# TRANSFINITE INTERPOLATION
for i in range(nx):
    for j in range(ny):
        X[i, j] = (1 - XI[i, j]) * splinex_left(ETA[i, j]) + XI[i, j] * splinex_right(ETA[i, j]) + (
                1 - ETA[i, j]) * splinex_bottom(XI[i, j]) + ETA[i, j] * splinex_top(XI[i, j]) - (1 - XI[i, j]) * (
                          1 - ETA[i, j]) * splinex_left(0) - (1 - XI[i, j]) * ETA[i, j] * splinex_left(1) - (1 - ETA[i, j]) * XI[
                      i, j] * splinex_right(0) - XI[i, j] * ETA[i, j] * splinex_right(1)

        Y[i, j] = (1 - XI[i, j]) * spliney_left(ETA[i, j]) + XI[i, j] * spliney_right(ETA[i, j]) + (
                1 - ETA[i, j]) * spliney_bottom(XI[i, j]) + ETA[i, j] * spliney_top(XI[i, j]) - (1 - XI[i, j]) * (
                          1 - ETA[i, j]) * spliney_left(0) - (1 - XI[i, j]) * ETA[i, j] * spliney_left(1) - (1 - ETA[i, j]) * XI[
                      i, j] * spliney_right(0) - XI[i, j] * ETA[i, j] * spliney_right(1)

plt.figure()
for i in range(nx):
    plt.plot(X[i, :], Y[i, :], 'k')
for j in range(ny):
    plt.plot(X[:, j], Y[:, j], 'k')

plt.show()
