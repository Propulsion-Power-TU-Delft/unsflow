import numpy as np
import matplotlib.pyplot as plt
import Grid
from Grid.src.functions import elliptic_grid_generation
from scipy.interpolate import CubicSpline
import pickle

""" 
test case to debug the elliptic grid generation method. Define a figure delimited by 4 curved
borders, and see how it goes
"""



# parameteric picture
L = 10
R = 1.5

pickle_mesh = 'nasar37_blade/mesh_21_21_26.pickle'
with open(pickle_mesh, 'rb') as f:
    data = pickle.load(f)

Z_mesh = data['z'][:,:,0]
X_mesh = data['x'][:,:,0]
Y_mesh = data['y'][:,:,0]

R_mesh = np.sqrt(X_mesh**2 + Y_mesh**2)

nx = Z_mesh.shape[0]
ny = Z_mesh.shape[1]



"""  left border  """
x = Z_mesh[0,:]
y = R_mesh[0,:]
c_left = np.array((x, y))

"""  bottom border  """
x = Z_mesh[:,0]
y = R_mesh[:,0]
c_bottom = np.array((x, y))

"""  right border  """
x = Z_mesh[-1,:]
y = R_mesh[-1,:]
c_right = np.array((x, y))

"""  top border  """
x = Z_mesh[:,-1]
y = R_mesh[:,-1]
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


def stretching_function(x_param, alpha):
    return 1 / (1 + np.exp(-alpha * (x_param - 0.5)))


# COMPUTATIONAL DOMAIN
nx = 30
ny = 25

xi = np.linspace(0, 1, nx)
# xi = stretching_function(xi, 6)
xi = xi**1.4
eta = np.linspace(0, 1, ny)
eta = stretching_function(eta, 7)
# eta = eta**1
XI, ETA = np.meshgrid(xi, eta, indexing='ij')
X, Y = XI*0, ETA*0

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
    plt.plot(X[i, :], Y[i, :], 'k', lw=0.5)
for j in range(ny):
    plt.plot(X[:, j], Y[:, j], 'k', lw=0.5)

plt.show()
