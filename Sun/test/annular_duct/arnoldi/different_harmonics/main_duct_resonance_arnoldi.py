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

# non-dimensionalization terms:
x_ref = r1
u_ref = M * a
rho_ref = rho
t_ref = x_ref / u_ref
omega_ref = 1 / t_ref
p_ref = rho_ref * u_ref ** 2

# %% ANALYTICAL PART OF THE PROBLEM
from scipy.optimize import fsolve

# radial cordinate array span
r = np.linspace(r1, r2, 300)
lambda_span = np.linspace(1, 300, 300)  # we will do a loop for every possible value of lambda
m = 12  # second circumferential mode


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

# plot the determinant value
fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(lambda_span, det, label=r'$m=%d$' % (m))
ax.plot(roots, roots_y, 'ro', label='zeros')
ax.plot(lambda_span, zeros, '--k', lw=0.5)
ax.set_title(r'$\lambda$ roots')
ax.set_xlabel(r'$\lambda$')
ax.set_xlim([0, 300])
ax.set_ylim([-0.25, 0.25])
ax.set_ylabel(r'$\det{\mathbf{Q}(\lambda)}$')
ax.legend()
# fig.savefig('pictures/lambda_roots_m%d.pdf' % (m), bbox_inches='tight')

alpha = [1, 2, 3, 4, 5, 6, 7, 8]  # possible axial wavenumbers
omega_analytical = compute_omega(alpha, roots[0:8], M, L, a)
omega_analytical_zero = np.zeros_like(omega_analytical)












# %%%%%%%%%%%%%%%%%%%%%%% COMPUTATIONAL PART %%%%%%%%%%%%%%%%%%%%%%%
# number of grid nodes in the computational domain
Nz = 45
Nr = 15

# implement a constant uniform flow in the annulus duct
density = np.zeros((Nz, Nr))
axialVel = np.zeros((Nz, Nr))
radialVel = np.zeros((Nz, Nr))
tangentialVel = np.zeros((Nz, Nr))
pressure = np.zeros((Nz, Nr))
for ii in range(0, Nz):
    for jj in range(0, Nr):
        density[ii, jj] = rho
        axialVel[ii, jj] = M * a
        pressure[ii, jj] = p

# create a meridional object, having the same information of the meridional post-process object of a compressor
duct_Obj = Sun.src.AnnulusMeridional(0, L, r1, r2, Nz, Nr, density, radialVel, tangentialVel, axialVel, pressure)
duct_grid = Sun.src.sun_grid.SunGrid(duct_Obj)
duct_grid.ShowGrid()

# general workflow of the sun model
sun_obj = Sun.src.SunModel(duct_grid)
sun_obj.ComputeBoundaryNormals()
sun_obj.add_shaft_rpm(omega_ref)
sun_obj.AddNormalizationQuantities(rho_ref, u_ref, x_ref)
sun_obj.NormalizeData()
sun_obj.ComputeSpectralGrid()
gradient_routine = 'findiff'
gradient_order = 6
sun_obj.ComputeJacobianPhysical(routine=gradient_routine, order=gradient_order)
sun_obj.AddAMatrixToNodesFrancesco2()
sun_obj.AddBMatrixToNodesFrancesco2()
sun_obj.AddCMatrixToNodesFrancesco2(m=12)
sun_obj.AddEMatrixToNodesFrancesco2()
sun_obj.AddRMatrixToNodesFrancesco2()
sun_obj.AddSMatrixToNodes()
sun_obj.AddHatMatricesToNodes()
sun_obj.ApplySpectralDifferentiation()
sun_obj.build_A_global_matrix()
sun_obj.build_C_global_matrix()
sun_obj.build_R_global_matrix()
sun_obj.build_Z_global_matrix()
sun_obj.impose_boundary_conditions('zero pressure', 'zero pressure')
sun_obj.apply_boundary_conditions_generalized()

omega_search = 45000
mode_name = r'$[R,Z] = [2, 3]$'
sigma = omega_search / omega_ref
A = sun_obj.Z_g
M = sun_obj.A_g
C = np.linalg.inv(A - sigma * M)
C = np.dot(C, M)
number_search = 20
print('Searching Eigenvalues with ARPACK...')
eigenvalues, eigenvectors = eigs(C, k=number_search)
eigenvalues = sigma + 1 / eigenvalues
eigenvalues *= omega_ref

marker_size = 100
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(omega_analytical.real, omega_analytical.imag, marker='x', facecolors='blue',
           s=marker_size, label=r'analytical')
ax.scatter(eigenvalues.real, eigenvalues.imag, marker='o', facecolors='none', edgecolors='red',
           s=marker_size, label=r'numerical')
ax.set_xlabel(r'$\omega_{R}$ [rad/s]')
ax.set_ylabel(r'$\omega_{I}$ [rad/s]')
ax.legend()
ax.set_xlim([20000, 70000])
ax.set_ylim([-1000, 1000])
ax.grid(alpha=0.3)
# fig.savefig('pictures/%i/chi_map_arnoldi_%i_%i_%i.pdf' % (eigenvalues[0].real, Nz, Nr, eigenvalues[0].real), bbox_inches='tight')

# # EIGENFUNCTIONS
# z_grid = sun_obj.data.zGrid
# r_grid = sun_obj.data.rGrid
# rho_eig = []
# ur_eig = []
# ut_eig = []
# uz_eig = []
# p_eig = []
#
# for i in range(len(eigenvectors)):
#     if (i) % 5 == 0:
#         rho_eig.append(eigenvectors[i])
#     elif (i - 1) % 5 == 0 and i != 0:
#         ur_eig.append(eigenvectors[i])
#     elif (i - 2) % 5 == 0 and i != 0:
#         ut_eig.append(eigenvectors[i])
#     elif (i - 3) % 5 == 0 and i != 0:
#         uz_eig.append(eigenvectors[i])
#     elif (i - 4) % 5 == 0 and i != 0:
#         p_eig.append(eigenvectors[i])
#     else:
#         raise ValueError("Not correct indexing for eigenvector retrieval!")
#
#
# def scaled_eigenvector_real(eig_list):
#     array = np.array(eig_list, dtype=complex)
#     array = np.reshape(array, (Nz, Nr))
#     array_real_scaled = array.real / (np.max(array.real) - np.min(array.real))
#     return array_real_scaled
#
#
# rho_eig_r = scaled_eigenvector_real(rho_eig)
# ur_eig_r = scaled_eigenvector_real(ur_eig)
# ut_eig_r = scaled_eigenvector_real(ut_eig)
# uz_eig_r = scaled_eigenvector_real(uz_eig)
# p_eig_r = scaled_eigenvector_real(p_eig)
#
#
#
# plt.figure(figsize=(7, 5))
# plt.contourf(z_grid, r_grid, rho_eig_r, levels=30, cmap='RdBu')
# plt.ylabel(r'$r$ [-]')
# plt.xlabel(r'$z$ [-]')
# plt.title(r'$\tilde{\rho} \quad$'+mode_name)
# plt.colorbar()
# # plt.savefig('pictures/%i/eigenfunction_rho_2D_%i_%i_%i.pdf' % (eigenvalues[0].real, Nz, Nr, eigenvalues[0].real), bbox_inches='tight')
# plt.figure(figsize=(7, 5))
# plt.contourf(z_grid, r_grid, ur_eig_r, levels=30, cmap='RdBu')
# plt.ylabel(r'$r$ [-]')
# plt.xlabel(r'$z$ [-]')
# plt.title(r'$\tilde{u}_r \quad$' + mode_name)
# plt.colorbar()
# # plt.savefig('pictures/%i/eigenfunction_ur_2D_%i_%i_%i.pdf' % (eigenvalues[0].real, Nz, Nr, eigenvalues[0].real), bbox_inches='tight')
# plt.figure(figsize=(7, 5))
# plt.contourf(z_grid, r_grid, ut_eig_r, levels=30, cmap='RdBu')
# plt.ylabel(r'$r$ [-]')
# plt.xlabel(r'$z$ [-]')
# plt.title(r'$\tilde{u}_{\theta} \quad$' + mode_name)
# plt.colorbar()
# # plt.savefig('pictures/%i/eigenfunction_ut_2D_%i_%i_%i.pdf' % (eigenvalues[0].real, Nz, Nr, eigenvalues[0].real), bbox_inches='tight')
# plt.figure(figsize=(7, 5))
# plt.contourf(z_grid, r_grid, uz_eig_r, levels=30, cmap='RdBu')
# plt.ylabel(r'$r$ [-]')
# plt.xlabel(r'$z$ [-]')
# plt.title(r'$\tilde{u}_z \quad$' + mode_name)
# plt.colorbar()
# # plt.savefig('pictures/%i/eigenfunction_uz_2D_%i_%i_%i.pdf' % (eigenvalues[0].real, Nz, Nr, eigenvalues[0].real), bbox_inches='tight')
# plt.figure(figsize=(7, 5))
# plt.contourf(z_grid, r_grid, p_eig_r, levels=30, cmap='RdBu')
# plt.ylabel(r'$r$ [-]')
# plt.xlabel(r'$z$ [-]')
# plt.title(r'$\tilde{p} \quad$' + mode_name)
# plt.colorbar()
# # plt.savefig('pictures/%i/eigenfunction_p_2D_%i_%i_%i.pdf' % (eigenvalues[0].real, Nz, Nr, eigenvalues[0].real), bbox_inches='tight')
#
#
#
# # first axial order
# # plt.figure(figsize=(7, 5))
# # plt.plot(z_grid[:, 0], np.abs(p_eig_r[:, 0].real), '--o', label='numerical')
# # plt.plot(z_grid[:, 0], np.max(np.abs(p_eig_r[:, 0])) * np.sin(np.pi * z_grid[:, 0] / L * r1), label='analytical')
# # plt.ylabel(r'$p$ [-]')
# # plt.xlabel(r'$z$ [-]')
# # plt.title(mode_name)
# # plt.legend()
# # plt.savefig('pictures/%i/eigenfunction_z_%i_%i_%i.pdf' % (eigenvalues[0].real, Nz, Nr, eigenvalues[0].real), bbox_inches='tight')
#
# # # second axial order
# plt.figure(figsize=(7, 5))
# plt.plot(z_grid[:, 0], p_eig_r[:, 0], '--o', label='numerical')
# plt.plot(z_grid[:, 0], np.max(p_eig_r[:, 0]) * np.sin(-2 * np.pi * z_grid[:, 0] / L * r1), label='analytical')
# plt.ylabel(r'$p$ [-]')
# plt.xlabel(r'$z$ [-]')
# plt.title(mode_name)
# plt.legend()
# # plt.savefig('pictures/%i/eigenfunction_z_%i_%i_%i.pdf' % (eigenvalues[0].real, Nz, Nr, eigenvalues[0].real), bbox_inches='tight')
#
#
#
#
# # RADIAL CUT
# LAMBDA = roots[1]
# ALPHA = jvp(m, LAMBDA*r1, n=1) / yvp(m, LAMBDA*r1, n=1)
# ALPHA2 = jvp(m, LAMBDA*r2, n=1) / yvp(m, LAMBDA*r2, n=1)
# r_var = np.linspace(r1, r2, 100)
# eigen_analyt_r = jv(m, LAMBDA*r_var) - ALPHA*yv(m, LAMBDA*r_var)
# eigen_analyt_r_scaled = eigen_analyt_r / (np.max(eigen_analyt_r) - np.min(eigen_analyt_r))
#
#
# plt.figure(figsize=(7, 5))
# # if opposite signs
# plt.plot(r_grid[Nz // 2, :], (p_eig_r[Nz // 2, :]) / (np.max(p_eig_r[Nz // 2, :]) - np.min(p_eig_r[Nz // 2, :])), '--o', label='numerical')
# plt.plot(r_var/r1, eigen_analyt_r_scaled, label='analytical')
# #if same signs
# # plt.plot(r_grid[Nz // 2, :], np.abs(p_eig_r[Nz // 2, :])/np.max(np.abs(p_eig_r[Nz // 2, :])), '--o', label='numerical')
# # plt.plot(r_var/r1, np.abs(eigen_analyt_r_scaled)/np.max(np.abs(eigen_analyt_r_scaled)), label='analytical')
# plt.ylabel(r'$p$ [-]')
# plt.xlabel(r'$r$ [-]')
# plt.title(mode_name)
# plt.legend()
# # plt.savefig('pictures/%i/eigenfunction_r_%i_%i_%i.pdf' % (eigenvalues[0].real, Nz, Nr, eigenvalues[0].real), bbox_inches='tight')
plt.show()
