"""
WEIGHTED LEAST SQUARES APPROXIMATION TO SCATTERED F(x,y). Reference
"An As-Short-As-Possible Introduction to the Least Squares, Weighted Least Squares and
Moving Least Squares Methods for Scattered Data Approximation and Interpolation" - Nealen Andrew
"""
import matplotlib.pyplot as plt
import numpy as np
import pickle

from Grid.src.weighted_least_squares import *
from Grid.src.styles import color_map
import time

begin = time.time()

def function(X, Y):
    # return X ** 4 - X ** 2 * Y ** 2 + 3 * X ** 3 * Y - 6 * X * Y ** 2 - 2 * X + Y ** 2 + 1
    # return X+Y
    return np.exp(-1/X)+np.sin(X*Y)


def function_dx(X, Y):
    # return 4 * X ** 3 - 2 * X * Y ** 2 + 9 * X ** 2 * Y - 6 * Y ** 2 - 2
    # return X/X
    return np.exp(-1/X)/X**2 + Y*np.cos(X*Y)


def function_dy(X, Y):
    # return -X ** 2 * 2 * Y + 3 * X ** 3 - 12 * X * Y + 2 * Y
    # return Y/Y
    return X*np.cos(X*Y)


# GENERATE THE DATA, AND THE ANALYTIC RESULTS
L = 5
H = 5
nx, ny = 50, 30
x = np.linspace(0, L, nx)
y = np.linspace(0, H, ny)
delta = np.sqrt((x[1] - x[0]) ** 2 + (y[1] - y[0]) ** 2) / 10
X, Y = np.meshgrid(x, y, indexing='ij')
Z = function(X, Y)
Zdx = function_dx(X, Y)
Zdy = function_dy(X, Y)



# EVALUEA THE WLS FIELD
x_points = X.flatten()
y_points = Y.flatten()
z_points = Z.flatten()

Z_wls = np.zeros_like(Z)
Z_wls_dx = np.zeros_like(Z)
Z_wls_dy = np.zeros_like(Z)
for ii in range(nx):
    for jj in range(ny):
        print("Regression %i of %i" % (jj + ii * ny, nx * ny))
        f, dfdx, dfdy = evaluate_weight_least_square_regression(X[ii, jj], Y[ii, jj], x_points, y_points, z_points,
                                                                order=2, delta=delta, wfunc_type = 'gauss')
        Z_wls[ii, jj] = f
        Z_wls_dx[ii, jj] = dfdx
        Z_wls_dy[ii, jj] = dfdy

fig, ax = plt.subplots(1, 3, figsize=(16, 5))
contour1 = ax[0].contourf(X, Y, Z, levels=50, cmap=color_map)
fig.colorbar(contour1, ax=ax[0])
ax[0].set_title(r'$z$')
contour2 = ax[1].contourf(X, Y, Z_wls, levels=50, cmap=color_map)
fig.colorbar(contour2, ax=ax[1])
ax[1].set_title(r'$z_{wls}$')
contour3 = ax[2].contourf(X, Y, (Z - Z_wls), levels=50, cmap=color_map)
fig.colorbar(contour3, ax=ax[2])
ax[2].set_title(r'$\varepsilon$')
plt.savefig('pictures/f_%i_%i.pdf' %(nx, ny), bbox_inches='tight')


fig, ax = plt.subplots(1, 3, figsize=(16, 5))
contour1 = ax[0].contourf(X, Y, Zdx, levels=50, cmap=color_map)
fig.colorbar(contour1, ax=ax[0])
ax[0].set_title(r'$\frac{\partial z}{\partial x}$')
contour2 = ax[1].contourf(X, Y, Z_wls_dx, levels=50, cmap=color_map)
fig.colorbar(contour2, ax=ax[1])
ax[1].set_title(r'$\frac{\partial z_{wls}}{\partial x}$')
contour3 = ax[2].contourf(X, Y, (Zdx - Z_wls_dx), levels=50, cmap=color_map)
fig.colorbar(contour3, ax=ax[2])
ax[2].set_title(r'$\varepsilon$')
plt.savefig('pictures/dfdx_%i_%i.pdf' %(nx, ny), bbox_inches='tight')


fig, ax = plt.subplots(1, 3, figsize=(16, 5))
contour1 = ax[0].contourf(X, Y, Zdy, levels=50, cmap=color_map)
fig.colorbar(contour1, ax=ax[0])
ax[0].set_title(r'$\frac{\partial z}{\partial y}$')
contour2 = ax[1].contourf(X, Y, Z_wls_dy, levels=50, cmap=color_map)
fig.colorbar(contour2, ax=ax[1])
ax[1].set_title(r'$\frac{\partial z_{wls}}{\partial y}$')
contour3 = ax[2].contourf(X, Y, (Zdy - Z_wls_dy), levels=50, cmap=color_map)
fig.colorbar(contour3, ax=ax[2])
ax[2].set_title(r'$\varepsilon$')
plt.savefig('pictures/dfdy_%i_%i.pdf' %(nx, ny), bbox_inches='tight')

end = time.time()
print("Running time: %.2f" %(end-begin))
plt.show()
