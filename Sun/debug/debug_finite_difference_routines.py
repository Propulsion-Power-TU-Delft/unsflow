#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 09:29:59 2023
@author: F. Neri, TU Delft

test the different finite differences routines
"""
import sys

sys.path.append('../')

import matplotlib.pyplot as plt
import numpy as np
import Sun
from matplotlib import cm
from Sun.src.styles import *

# %%
nx = 20
ny = 20

# computational grid
x = np.linspace(0, 10, nx)
y = np.linspace(0, 10, ny)
Y, X = np.meshgrid(y, x)

# physical grid
Z = np.zeros_like(X)
R = np.zeros_like(Y)
for i in range(0, nx):
    for j in range(0, ny):
        # analytical transformation
        Z[i, j] = X[i, j] ** 2 + Y[i, j] ** 2
        R[i, j] = np.sin(X[i, j]) + np.cos(Y[i, j])

fig, ax = plt.subplots(1, 2, figsize=fig_size)
c0 = ax[0].scatter(Z, R, s=scatter_point_size, c='blue')
c1 = ax[1].scatter(X, Y, s=scatter_point_size, c='green')
ax[0].set_xlabel(r'$z$')
ax[0].set_ylabel(r'$r$')
ax[0].set_title(r'physical grid')
ax[1].set_xlabel(r'$\xi$')
ax[1].set_ylabel(r'$\eta$')
ax[1].set_title(r'computational grid')
fig.savefig('pictures/debug_finite_diff_grids.pdf', bbox_inches='tight')

# analytical transformation gradients:
dzdx = np.zeros_like(X)
dzdy = np.zeros_like(X)
drdx = np.zeros_like(X)
drdy = np.zeros_like(X)
for i in range(0, nx):
    for j in range(0, ny):
        dzdx[i, j] = 2 * X[i, j]
        dzdy[i, j] = 2 * Y[i, j]
        drdx[i, j] = np.cos(X[i, j])
        drdy[i, j] = -np.sin(Y[i, j])
J = dzdx * drdy - dzdy * drdx








# compute the finite difference version (_fd)

# hard coded version
dzdx_fd, dzdy_fd, drdx_fd, drdy_fd = Sun.src.general_functions.JacobianTransform(Z, R, X, Y)
J_fd = dzdx_fd * drdy_fd - dzdy_fd * drdx_fd

# numpy.gradient() version
dzdx_fd2, dzdy_fd2, drdx_fd2, drdy_fd2 = Sun.src.general_functions.JacobianTransform2(Z, R, X, Y)
J_fd2 = dzdx_fd2 * drdy_fd2 - dzdy_fd2 * drdx_fd2

# findiff version of second order
dzdx_fd3, dzdy_fd3, drdx_fd3, drdy_fd3 = Sun.src.general_functions.JacobianTransform3(Z, R, X, Y, order=2)
J_fd3 = dzdx_fd3 * drdy_fd3 - dzdy_fd3 * drdx_fd3

# findiff version of fourth order
dzdx_fd4, dzdy_fd4, drdx_fd4, drdy_fd4 = Sun.src.general_functions.JacobianTransform3(Z, R, X, Y, order=4)
J_fd4 = dzdx_fd4 * drdy_fd4 - dzdy_fd4 * drdx_fd4

# now compare the jacobians with the analytical one
J_fd_err = (J_fd - J) / (np.max(J) - np.min(J))
J_fd2_err = (J_fd2 - J) / (np.max(J) - np.min(J))
J_fd3_err = (J_fd3 - J) / (np.max(J) - np.min(J))
J_fd4_err = (J_fd4 - J) / (np.max(J) - np.min(J))


fig, ax = plt.subplots(1, 5, figsize=(30, 6))
contour10 = ax[0].contourf(X, Y, J, cmap=cm.jet, levels=N_levels)
contour11 = ax[1].contourf(X, Y, J_fd_err, cmap=cm.jet, levels=N_levels)
contour12 = ax[2].contourf(X, Y, J_fd2_err, cmap=cm.jet, levels=N_levels)
contour13 = ax[3].contourf(X, Y, J_fd3_err, cmap=cm.jet, levels=N_levels)
contour14 = ax[4].contourf(X, Y, J_fd4_err, cmap=cm.jet, levels=N_levels)
cbar = plt.colorbar(contour10, ax=ax[0])
cbar = plt.colorbar(contour11, ax=ax[1])
cbar = plt.colorbar(contour12, ax=ax[2])
cbar = plt.colorbar(contour13, ax=ax[3])
cbar = plt.colorbar(contour14, ax=ax[4])
ax[0].set_xticks([])
ax[1].set_xticks([])
ax[2].set_xticks([])
ax[3].set_xticks([])
ax[4].set_xticks([])
ax[0].set_yticks([])
ax[1].set_yticks([])
ax[2].set_yticks([])
ax[3].set_yticks([])
ax[4].set_yticks([])
ax[0].set_xlabel(r'$\xi$')
ax[1].set_xlabel(r'$\xi$')
ax[2].set_xlabel(r'$\xi$')
ax[3].set_xlabel(r'$\xi$')
ax[4].set_xlabel(r'$\xi$')
ax[0].set_ylabel(r'$\eta$')
ax[0].set_title(r'$J$')
ax[1].set_title(r'$J_{1,err}$')
ax[2].set_title(r'$J_{2,err}$')
ax[3].set_title(r'$J_{3,err}$')
ax[4].set_title(r'$J_{4,err}$')
fig.suptitle(r'$(%d \times %d) \ \mathrm{grid}$' %(nx, ny))
fig.savefig('pictures/debug_jacobian_errors_%d_%d.pdf' %(nx, ny), bbox_inches='tight')
plt.show()
