import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

nx = 30
ny = 15

# parameteric picture
L = 10
R = 1.5

"""  left border  """
theta = np.linspace(0, np.pi, ny)
x = -np.sin(theta)
y = np.linspace(0, L, ny)
c_left = np.array((x, y))

"""  bottom border  """
theta = np.linspace(0, 4 * np.pi, nx)
x = np.linspace(0, L, nx)
y = np.zeros(nx)
c_bottom = np.array((x, y))

"""  right border  """
theta = np.linspace(0, 4 * np.pi, ny)
x = L + np.zeros(ny)
y = np.linspace(0, L, ny)
c_right = np.array((x, y))

"""  top border  """
theta = np.linspace(0, np.pi / 2, nx)
x = np.linspace(0, L, nx)
y = L + np.zeros(nx)
c_top = np.array((x, y))

plt.figure()
plt.plot(c_left[0, :], c_left[1, :])
plt.plot(c_bottom[0, :], c_bottom[1, :])
plt.plot(c_right[0, :], c_right[1, :])
plt.plot(c_top[0, :], c_top[1, :])
# plt.show()


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

# COMPUTATIONAL DOMAIN
xi = np.linspace(0, 1, nx)
xi = xi ** 1.3  # stretching
eta = np.linspace(0, 1, ny)
eta = eta ** 1  # stretching
XI, ETA = np.meshgrid(xi, eta, indexing='ij')
X, Y = np.zeros_like(XI), np.zeros_like(ETA)


# TRANSFINITE INTERPOLATION
for i in range(nx):
    for j in range(ny):
        X[i, j] = (1 - XI[i, j]) * splinex_left(ETA[i, j]) + XI[i, j] * splinex_right(ETA[i, j]) + (
                    1 - ETA[i, j]) * splinex_bottom(XI[i, j]) + ETA[i, j] * splinex_top(XI[i, j]) - (1 - XI[i, j]) * (
                              1 - ETA[i, j]) * splinex_left(0) - (1 - XI[i, j]) * ETA[i, j] * splinex_left(1) - (1 - ETA[i, j]) * \
                  XI[i, j] * splinex_right(0) - XI[i, j] * ETA[i, j] * splinex_right(1)

        Y[i, j] = (1 - XI[i, j]) * spliney_left(ETA[i, j]) + XI[i, j] * spliney_right(ETA[i, j]) + (
                1 - ETA[i, j]) * spliney_bottom(XI[i, j]) + ETA[i, j] * spliney_top(XI[i, j]) - (1 - XI[i, j]) * (
                              1 - ETA[i, j]) * spliney_left(0) - (1 - XI[i, j]) * ETA[i, j] * spliney_left(1) - (1 - ETA[i, j]) * \
                  XI[i, j] * spliney_right(0) - XI[i, j] * ETA[i, j] * spliney_right(1)

plt.figure()
for i in range(nx):
    plt.plot(X[i, :], Y[i, :], 'k')
for j in range(ny):
    plt.plot(X[:, j], Y[:, j], 'k')

plt.show()
