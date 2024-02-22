#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 09:29:59 2023
@author: F. Neri, TU Delft
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import jv, yv, jvp, yvp
import Sun
import scipy
from scipy.optimize import fsolve
from scipy.sparse.linalg import eigs
from Sun.src.sun_model_multiblock import SunModelMultiBlock
from Grid.src.config import Config
import os
from Utils.styles import *


# input data of the problem (SI units)
r1 = 0.1826  # inner radius [m]
r2 = 0.2487  # outer radius [m]
M = 0.015  # Mach number
p = 100e3  # pressure [Pa]
T = 288  # temperature [K]
L = 0.08  # length [m]
R = 287.058  # air gas constant [kJ/kgK]
gmma = 1.4  # cp/cv ratio of air
rho = p / (R * T)  # density [kg/m3]
a = np.sqrt(gmma * p / rho)  # ideal speed of sound [m/s]
HARMONIC_ORDER = 1

# non-dimensionalization terms:
x_ref = r1
u_ref = M * a
rho_ref = rho
t_ref = x_ref / u_ref
omega_ref = 1 / t_ref
p_ref = rho_ref * u_ref ** 2

COMPUTE_ANALYTICAL = True

if COMPUTE_ANALYTICAL:
    # %% ANALYTICAL PART OF THE PROBLEM
    from scipy.optimize import fsolve

    # radial cordinate array span
    r = np.linspace(r1, r2, 300)
    lambda_span = np.linspace(1, 300, 300)  # we will do a loop for every possible value of lambda
    m = 1  # second circumferential mode


    def Bessel_determinant(lmbda, r1=r1, r2=r2):
        r = np.linspace(r1, r2)
        dJ1dr = jvp(m, lmbda * r, n=1)
        dN1dr = yvp(m, lmbda * r, n=1)
        det = dJ1dr[0] * dN1dr[-1] - dN1dr[0] * dJ1dr[-1]
        return det


    def lambda_root(lmbda_span, r1=r1, r2=r2):
        det = np.array(())
        for s in range(0, len(lambda_span)):
            lmbda = lambda_span[s]
            det = np.append(det, Bessel_determinant(lmbda, r1, r2))
        return det


    def find_multiple_zeros(f, xmin, xmax, intervals=9):
        roots = []
        x = np.linspace(xmin, xmax, intervals)
        for i in range(0, intervals):
            root = fsolve(Bessel_determinant, x[i])
            roots.append(root)
        roots = np.array(roots)
        return np.unique(roots.round(decimals=2))


    def compute_omega(alphas, lmbdas, M, L, a):
        print('{:<8s} {:<8s} {:<8s} {:<8s}'.format('THETA', 'R', 'Z', 'OMEGA'))
        omega_list = []

        for alpha in alphas:
            i = 1
            for lmbda in lmbdas:
                omega = a * np.sqrt(((1 - M ** 2) * alpha * np.pi / L) ** 2 + (1 - M ** 2) * lmbda ** 2)
                omega_list.append(omega)
                print('{:<8.0f} {:<8.0f} {:<8.0f} {:<8.0f}'.format(m, i, alpha, omega))
                i += 1  # radial mode order
        omega_list = np.array(omega_list)
        return np.sort(omega_list)

    det = lambda_root(lambda_span)
    zeros = np.zeros(len(det))
    roots = find_multiple_zeros(lambda_root, 1, 300)
    roots_y = np.zeros_like(roots)
    alpha = [1, 2, 3, 4, 5, 6, 7, 8]  # possible axial wavenumbers
    omega_analytical = compute_omega(alpha, roots[0:8], M, L, a)
    omega_analytical_zero = np.zeros_like(omega_analytical)

    # plot the determinant value
    fig, ax = plt.subplots()
    ax.plot(lambda_span, det, linewidth=medium_line_width)
    ax.plot(roots, roots_y, 'ro', label='zeros', markersize=marker_size)
    ax.plot(lambda_span, zeros, '--k', lw=light_line_width)
    # ax.set_title(r'$\lambda$ roots', fontsize=font_title)
    ax.set_xlabel(r'$\lambda \ \mathrm{[m^{-1}]}$', fontsize=font_labels)
    plt.xticks(fontsize=font_axes)
    plt.yticks(fontsize=font_axes)
    ax.set_xlim([0, 300])
    ax.set_ylim([-0.25, 0.25])
    ax.set_ylabel(r'$\det{\mathbf{Q}(\lambda)}$', fontsize=font_labels)
    ax.legend(fontsize=font_legend)
    plt.grid(alpha=0.2)
    fig.savefig('pictures/lambda_roots.pdf', bbox_inches='tight')





import pickle
import pickle

# Specify the file path from which you want to load the pickled object
file_path = "data/meta/60_20_06.pickle"

# Open the file in binary read mode and load the pickled object using pickle.load()
with open(file_path, 'rb') as obj:
    sun_obj = pickle.load(obj)


print()







# PLOT RESULTS
marker_size = 130
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(omega_analytical.real, omega_analytical.imag, marker='x', facecolors='blue',
           s=marker_size, label=r'analytical')
ax.scatter(sun_obj.real[2:], sun_obj.imag[2:], marker='o', facecolors='none', edgecolors='red',
           s=marker_size, label=r'numerical')
ax.set_xlabel(r'$\omega_{R}$ [rad/s]', fontsize=font_labels)
ax.set_ylabel(r'$\omega_{I}$ [rad/s]', fontsize=font_labels)
ax.legend(fontsize=font_legend)
ax.set_xlim([7500, 38000])
plt.xticks(fontsize=font_axes)
plt.yticks(fontsize=font_axes)
# ax.set_xlim([np.min(eigenvalues.real), np.max(eigenvalues.real)])
ax.set_ylim([-8000, 8000])
ax.grid(alpha=0.3)
fig.savefig('pictures/chi_map_arnoldi.pdf', bbox_inches='tight')

# # EIGENFUNCTIONS
# z_grid = sun_obj.data.zGrid
# r_grid = sun_obj.data.rGrid
#
# for ivec in range(np.shape(eigenvectors)[1]):
#     eigenvec = eigenvectors[:, ivec]
#     rho_eig = []
#     ur_eig = []
#     ut_eig = []
#     uz_eig = []
#     p_eig = []
#
#     for i in range(len(eigenvec)):
#         if (i) % 5 == 0:
#             rho_eig.append(eigenvec[i])
#         elif (i - 1) % 5 == 0 and i != 0:
#             ur_eig.append(eigenvec[i])
#         elif (i - 2) % 5 == 0 and i != 0:
#             ut_eig.append(eigenvec[i])
#         elif (i - 3) % 5 == 0 and i != 0:
#             uz_eig.append(eigenvec[i])
#         elif (i - 4) % 5 == 0 and i != 0:
#             p_eig.append(eigenvec[i])
#         else:
#             raise ValueError("Not correct indexing for eigenvector retrieval!")
#
#
#     def scaled_eigenvector_real(eig_list):
#         array = np.array(eig_list, dtype=complex)
#         array = np.reshape(array, (Nz, Nr))
#         array_real_scaled = array.real / (np.max(array.real) - np.min(array.real))
#         return array_real_scaled
#
#
#     rho_eig_r = scaled_eigenvector_real(rho_eig)
#     ur_eig_r = scaled_eigenvector_real(ur_eig)
#     ut_eig_r = scaled_eigenvector_real(ut_eig)
#     uz_eig_r = scaled_eigenvector_real(uz_eig)
#     p_eig_r = scaled_eigenvector_real(p_eig)
#
#     plt.figure(figsize=(7, 5))
#     cnt = plt.contourf(z_grid, r_grid, rho_eig_r, levels=200, cmap='bwr')
#     for c in cnt.collections:
#         c.set_edgecolor("face")
#     plt.ylabel(r'$r$ [-]')
#     plt.xlabel(r'$z$ [-]')
#     plt.title(r'$\tilde{\rho}_{%i}$' % (ivec + 1))
#     plt.colorbar()
#     plt.savefig('pictures/%i_%i/eigenfunction_rho_%i.pdf' % (Nz, Nr, ivec + 1), bbox_inches='tight')
#
#     plt.figure(figsize=(7, 5))
#     cnt = plt.contourf(z_grid, r_grid, ur_eig_r, levels=200, cmap='bwr')
#     for c in cnt.collections:
#         c.set_edgecolor("face")
#     plt.ylabel(r'$r$ [-]')
#     plt.xlabel(r'$z$ [-]')
#     plt.title(r'$\tilde{u}_{r,%i}$' % (ivec + 1))
#     plt.colorbar()
#     plt.savefig('pictures/%i_%i/eigenfunction_ur_%i.pdf' % (Nz, Nr, ivec + 1), bbox_inches='tight')
#
#     plt.figure(figsize=(7, 5))
#     cnt = plt.contourf(z_grid, r_grid, ut_eig_r, levels=200, cmap='bwr')
#     for c in cnt.collections:
#         c.set_edgecolor("face")
#     plt.ylabel(r'$r$ [-]')
#     plt.xlabel(r'$z$ [-]')
#     plt.title(r'$\tilde{u}_{\theta,%i}$' % (ivec + 1))
#     plt.colorbar()
#     plt.savefig('pictures/%i_%i/eigenfunction_ut_%i.pdf' % (Nz, Nr, ivec + 1), bbox_inches='tight')
#
#     plt.figure(figsize=(7, 5))
#     cnt = plt.contourf(z_grid, r_grid, uz_eig_r, levels=200, cmap='bwr')
#     for c in cnt.collections:
#         c.set_edgecolor("face")
#     plt.ylabel(r'$r$ [-]')
#     plt.xlabel(r'$z$ [-]')
#     plt.title(r'$\tilde{u}_{z,%i}$' % (ivec + 1))
#     plt.colorbar()
#     plt.savefig('pictures/%i_%i/eigenfunction_uz_%i.pdf' % (Nz, Nr, ivec + 1), bbox_inches='tight')
#
#     plt.figure(figsize=(7, 5))
#     cnt = plt.contourf(z_grid, r_grid, p_eig_r, levels=200, cmap='bwr')
#     for c in cnt.collections:
#         c.set_edgecolor("face")
#     plt.ylabel(r'$r$ [-]')
#     plt.xlabel(r'$z$ [-]')
#     plt.title(r'$\tilde{p}_{%i}$' % (ivec + 1))
#     plt.colorbar()
#     plt.savefig('pictures/%i_%i/eigenfunction_p_%i.pdf' % (Nz, Nr, ivec + 1), bbox_inches='tight')
#
#
# import pickle
# file_path = 'data/meta/%i_%i_%i.pickle'%(Nz, Nr, gradient_order)
# with open(file_path, 'wb') as file:
#     pickle.dump(eigenvalues, file)

plt.show()