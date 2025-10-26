#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 09:29:59 2023
@author: F. Neri, TU Delft

test the grid finite differences implementation
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

fig, ax = plt.subplots(2, 5, figsize=(20, 16))
contour00 = ax[0, 0].contourf(X, Y, dzdx, cmap=cm.jet, levels=N_levels)
contour01 = ax[0, 1].contourf(X, Y, dzdy, cmap=cm.jet, levels=N_levels)
contour02 = ax[0, 2].contourf(X, Y, drdx, cmap=cm.jet, levels=N_levels)
contour03 = ax[0, 3].contourf(X, Y, drdy, cmap=cm.jet, levels=N_levels)
contour04 = ax[0, 4].contourf(X, Y, J, cmap=cm.jet, levels=N_levels)
cbar = plt.colorbar(contour00, ax=ax[0, 0])
cbar = plt.colorbar(contour01, ax=ax[0, 1])
cbar = plt.colorbar(contour02, ax=ax[0, 2])
cbar = plt.colorbar(contour03, ax=ax[0, 3])
cbar = plt.colorbar(contour04, ax=ax[0, 4])

# compute the finite difference version (_fd)
dzdx_fd, dzdy_fd, drdx_fd, drdy_fd = Sun.src.general_functions.JacobianTransform3(Z, R, X, Y)
J_fd = dzdx_fd * drdy_fd - dzdy_fd * drdx_fd

contour10 = ax[1, 0].contourf(X, Y, dzdx_fd, cmap=cm.jet, levels=N_levels)
contour11 = ax[1, 1].contourf(X, Y, dzdy_fd, cmap=cm.jet, levels=N_levels)
contour12 = ax[1, 2].contourf(X, Y, drdx_fd, cmap=cm.jet, levels=N_levels)
contour13 = ax[1, 3].contourf(X, Y, drdy_fd, cmap=cm.jet, levels=N_levels)
contour14 = ax[1, 4].contourf(X, Y, J, cmap=cm.jet, levels=N_levels)
cbar = plt.colorbar(contour10, ax=ax[1, 0])
cbar = plt.colorbar(contour11, ax=ax[1, 1])
cbar = plt.colorbar(contour12, ax=ax[1, 2])
cbar = plt.colorbar(contour13, ax=ax[1, 3])
cbar = plt.colorbar(contour14, ax=ax[1, 4])
ax[0, 0].set_xticks([])
ax[0, 1].set_xticks([])
ax[0, 2].set_xticks([])
ax[0, 3].set_xticks([])
ax[0, 4].set_xticks([])
ax[0, 1].set_yticks([])
ax[0, 2].set_yticks([])
ax[0, 3].set_yticks([])
ax[0, 4].set_yticks([])
ax[1, 1].set_yticks([])
ax[1, 2].set_yticks([])
ax[1, 3].set_yticks([])
ax[1, 4].set_yticks([])
ax[1, 0].set_xlabel(r'$\xi$')
ax[1, 1].set_xlabel(r'$\xi$')
ax[1, 2].set_xlabel(r'$\xi$')
ax[1, 3].set_xlabel(r'$\xi$')
ax[1, 4].set_xlabel(r'$\xi$')
ax[0, 0].set_ylabel(r'$\eta$')
ax[1, 0].set_ylabel(r'$\eta$')
ax[0, 0].set_title(r'$\partial z / \partial \xi$')
ax[0, 1].set_title(r'$\partial z / \partial \eta$')
ax[0, 2].set_title(r'$\partial r / \partial \xi$')
ax[0, 3].set_title(r'$\partial r / \partial \eta$')
ax[0, 4].set_title(r'$J$')
ax[1, 0].set_title(r'$(\partial z / \partial \xi)_{FD}$')
ax[1, 1].set_title(r'$(\partial z / \partial \eta)_{FD}$')
ax[1, 2].set_title(r'$(\partial r / \partial \xi)_{FD}$')
ax[1, 3].set_title(r'$(\partial r / \partial \eta)_{FD}$')
ax[1, 4].set_title(r'$J_{FD}$')
fig.savefig('pictures/debug_finite_diff_findiff4_%d_%d.pdf' %(nx, ny), bbox_inches='tight')
plt.show()
