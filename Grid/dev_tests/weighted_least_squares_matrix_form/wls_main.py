"""
WEIGHTED LEAST SQUARES APPROXIMATION TO SCATTERED F(x,y). Reference
"An As-Short-As-Possible Introduction to the Least Squares, Weighted Least Squares and
Moving Least Squares Methods for Scattered Data Approximation and Interpolation" - Nealen Andrew
"""
import matplotlib.pyplot as plt
import numpy as np
import pickle

from grid.src.weighted_least_squares import *
from utils.styles import color_map
import time

begin = time.time()
function_to_test = 5

def function(X, Y):
    if function_to_test == 1:
        return X ** 4 - X ** 2 * Y ** 2 + 3 * X ** 3 * Y - 6 * X * Y ** 2 - 2 * X + Y ** 2 + 1
    elif function_to_test == 2:
        return X + Y
    elif function_to_test == 3:
        return np.exp(-1 / X) + np.sin(X * Y)
    elif function_to_test == 4:
        return X**2 - Y**2 +3*X - Y
    elif function_to_test ==5:
        return np.sin(X)*np.cos(Y)


def function_dx(X, Y):
    if function_to_test == 1:
        return 4 * X ** 3 - 2 * X * Y ** 2 + 9 * X ** 2 * Y - 6 * Y ** 2 - 2
    elif function_to_test == 2:
        return X / X
    elif function_to_test == 3:
        return np.exp(-1 / X) / X ** 2 + Y * np.cos(X * Y)
    elif function_to_test == 4:
        return 2*X +3
    elif function_to_test ==5:
        return np.cos(X)*np.cos(Y)


def function_dy(X, Y):
    if function_to_test == 1:
        return -X ** 2 * 2 * Y + 3 * X ** 3 - 12 * X * Y + 2 * Y
    elif function_to_test == 2:
        return Y / Y
    elif function_to_test == 3:
        return X * np.cos(X * Y)
    elif function_to_test == 4:
        return -2*Y -1
    elif function_to_test ==5:
        return -np.sin(X)*np.sin(Y)


# GENERATE THE DATA, AND THE ANALYTIC RESULTS
L = 5
H = 5
nx, ny = 30, 20
x = np.linspace(0, L, nx)
y = np.linspace(0, H, ny)
delta = np.sqrt((x[1] - x[0]) ** 2 + (y[1] - y[0]) ** 2) / 10
X, Y = np.meshgrid(x, y, indexing='ij')
Z = function(X, Y)
Zdx = function_dx(X, Y)
Zdy = function_dy(X, Y)

# EVALUATE THE WLS SOLUTION
x_points = X.flatten()
y_points = Y.flatten()
z_points = Z.flatten()





Z_wls = np.zeros_like(Z)
Z_wls_dx = np.zeros_like(Z)
Z_wls_dy = np.zeros_like(Z)
for ii in range(nx):
    for jj in range(ny):
        print("Regression %i of %i" % (jj + ii * ny, nx * ny))
        Z_wls[ii, jj], Z_wls_dx[ii, jj], Z_wls_dy[ii, jj] = compute_function_and_gradient_approximation(
            X[ii, jj], Y[ii, jj], x_points, y_points, z_points)


fig, ax = plt.subplots(1, 3, figsize=(16, 4))
contour1 = ax[0].contourf(X, Y, Z, levels=50, cmap=color_map)
fig.colorbar(contour1, ax=ax[0])
ax[0].set_title(r'$z$')
contour2 = ax[1].contourf(X, Y, Z_wls, levels=50, cmap=color_map)
fig.colorbar(contour2, ax=ax[1])
ax[1].set_title(r'$z_{wls}$')
contour3 = ax[2].contourf(X, Y, (Z - Z_wls), levels=50, cmap=color_map)
fig.colorbar(contour3, ax=ax[2])
ax[2].set_title(r'$\varepsilon$')
plt.savefig('pictures/func_%i_f_%i_%i.pdf' % (function_to_test, nx, ny), bbox_inches='tight')

fig, ax = plt.subplots(1, 3, figsize=(16, 4))
contour1 = ax[0].contourf(X, Y, Zdx, levels=50, cmap=color_map)
fig.colorbar(contour1, ax=ax[0])
ax[0].set_title(r'$\frac{\partial z}{\partial x}$')
contour2 = ax[1].contourf(X, Y, Z_wls_dx, levels=50, cmap=color_map)
fig.colorbar(contour2, ax=ax[1])
ax[1].set_title(r'$\frac{\partial z_{wls}}{\partial x}$')
contour3 = ax[2].contourf(X, Y, (Zdx - Z_wls_dx), levels=50, cmap=color_map)
fig.colorbar(contour3, ax=ax[2])
ax[2].set_title(r'$\varepsilon$')
plt.savefig('pictures/func_%i_dfdx_%i_%i.pdf' % (function_to_test, nx, ny), bbox_inches='tight')

fig, ax = plt.subplots(1, 3, figsize=(16, 4))
contour1 = ax[0].contourf(X, Y, Zdy, levels=50, cmap=color_map)
fig.colorbar(contour1, ax=ax[0])
ax[0].set_title(r'$\frac{\partial z}{\partial y}$')
contour2 = ax[1].contourf(X, Y, Z_wls_dy, levels=50, cmap=color_map)
fig.colorbar(contour2, ax=ax[1])
ax[1].set_title(r'$\frac{\partial z_{wls}}{\partial y}$')
contour3 = ax[2].contourf(X, Y, (Zdy - Z_wls_dy), levels=50, cmap=color_map)
fig.colorbar(contour3, ax=ax[2])
ax[2].set_title(r'$\varepsilon$')
plt.savefig('pictures/func_%i_dfdy_%i_%i.pdf' % (function_to_test, nx, ny), bbox_inches='tight')


end = time.time()
print("Running time: %.2f" % (end - begin))
plt.show()
