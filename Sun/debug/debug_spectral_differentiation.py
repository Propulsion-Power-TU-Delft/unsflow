#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 09:29:59 2023
@author: F. Neri, TU Delft

test the spectral differentiation algorithm
"""
import sys
sys.path.append('../')

import matplotlib.pyplot as plt
import numpy as np
import Sun
from matplotlib import cm



#%%
n_fine = 1000 #fine grid
n_coarsex = 30
n_coarsey = 20

#analytical domain
x_fine = np.linspace(-1,1, n_fine)
y_fine = np.linspace(-1,1, n_fine)

X_fine, Y_fine = np.meshgrid(x_fine, y_fine)

def function2D(X,Y):
    return (X**2+2*Y-3*np.sin(5*X*Y)+Y**3)

def function2DGradient(X,Y):
    dZdX = 2*X -3*np.cos(5*X*Y)*5*Y
    dZdY = 2-3*np.cos(5*X*Y)*5*X + 3*Y**2
    return dZdX, dZdY

Z_fine = function2D(X_fine, Y_fine)
dZdX_fine, dZdY_fine = function2DGradient(X_fine, Y_fine)


N_levels = 50

fig, ax = plt.subplots(2,3,figsize=(16,8))
contour00 = ax[0,0].contourf(X_fine, Y_fine, Z_fine, cmap=cm.jet, levels=N_levels)
contour01 = ax[0,1].contourf(X_fine, Y_fine, dZdX_fine, cmap=cm.jet, levels=N_levels)
contour02 = ax[0,2].contourf(X_fine, Y_fine, dZdY_fine, cmap=cm.jet, levels=N_levels)
ax[0,0].set_ylabel(r'$\eta$')
ax[0,0].set_title(r'$f(\xi, \eta)$')
ax[0,0].set_xticks([])
ax[0,1].set_title(r'$\partial f / \partial \xi$')
ax[0,1].set_xticks([])
ax[0,1].set_yticks([])
ax[0,2].set_title(r'$\partial f / \partial \eta$')
ax[0,2].set_xticks([])
ax[0,2].set_yticks([])
ax[1,0].set_xlabel(r'$\xi$')
ax[1,0].set_ylabel(r'$\eta$')
ax[1,1].set_xlabel(r'$\xi$')
ax[1,1].set_yticks([])
ax[1,2].set_xlabel(r'$\xi$')
ax[1,2].set_yticks([])
cbar = plt.colorbar(contour00, ax=ax[0,0])
cbar = plt.colorbar(contour01, ax=ax[0,1])
cbar = plt.colorbar(contour02, ax=ax[0,2])

#%% gauss lobatto derivatives
x_gl = Sun.src.general_functions.GaussLobattoPoints(n_coarsex) #points ordered in descending way, not ascending!!
y_gl = Sun.src.general_functions.GaussLobattoPoints(n_coarsey)
Y_gl, X_gl = np.meshgrid(y_gl, x_gl)

# plt.figure()
# plt.scatter(X_gl, Y_gl)

Dx = Sun.src.general_functions.ChebyshevDerivativeMatrixBayliss(x_gl) #derivative operator in xi, Bayliss formulation
Dy = Sun.src.general_functions.ChebyshevDerivativeMatrixBayliss(y_gl)


fig2, ax2 = plt.subplots(1,2,figsize=(12,5))
c1 = ax2[0].imshow(Dx, cmap=cm.coolwarm)
cbar = plt.colorbar(c1, ax=ax2[0])
ax2[0].set_title(r'$D^{\xi}$')
c2 = ax2[1].imshow(Dy, cmap=cm.coolwarm)
cbar = plt.colorbar(c2, ax=ax2[1])

ax2[1].set_title(r'$D^{\eta}$')


Z_gl = function2D(X_gl, Y_gl)
contour10 = ax[1,0].contourf(X_gl, Y_gl, Z_gl, cmap=cm.jet, levels=N_levels)
cbar = plt.colorbar(contour10, ax=ax[1,0])
ax[1,0].set_title(r'$f_{GL}(\xi, \eta)$')

dZdX_gl = np.zeros_like(Z_gl)
dZdY_gl = np.zeros_like(Z_gl)

#critical part here. you should copy paste what is inside the sun model
for i in range(0,n_coarsex):
    for j in range(0,n_coarsey):
        tmpx = 0
        tmpy = 0
        for nx in range(0,n_coarsex):
            tmpx += Dx[i,nx]*Z_gl[nx,j]
        for ny in range(0,n_coarsey):
            tmpy += Dy[j,ny]*Z_gl[i,ny]
        dZdX_gl[i,j] = tmpx
        dZdY_gl[i,j] = tmpy

contour11 = ax[1,1].contourf(X_gl, Y_gl, dZdX_gl, cmap=cm.jet, levels=N_levels)
contour12 = ax[1,2].contourf(X_gl, Y_gl, dZdY_gl, cmap=cm.jet, levels=N_levels)
cbar = plt.colorbar(contour11, ax=ax[1,1])
cbar = plt.colorbar(contour12, ax=ax[1,2])
ax[1,1].set_title(r'$\partial f_{GL} / \partial \xi$')
ax[1,2].set_title(r'$\partial f_{GL} / \partial \eta$')
fig.suptitle('grid (%.1d X %.1d)' %(n_coarsex, n_coarsey))
fig.savefig('pictures/debug_spectral_diff_%.1d_%.1d.pdf' %(n_coarsex, n_coarsey), bbox_inches='tight')



plt.show()