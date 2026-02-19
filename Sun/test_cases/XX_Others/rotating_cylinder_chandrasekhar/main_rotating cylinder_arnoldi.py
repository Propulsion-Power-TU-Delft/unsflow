#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 09:29:59 2023
@author: F. Neri, TU Delft

Tries to replicate problem 3.1 of hydrodynamic instability (Drazin)
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import jv, yv, jvp, yvp
import Sun
import scipy
from scipy.optimize import fsolve
from scipy.sparse.linalg import eigs


# input data
R2 = 1  # dimensional outer diameter
R1 = R2 / 100  # to simulate a cylinder, to avoid singularities at r=0
H = R2 * 10  # dimensional height of the cylinder
n_ax = np.linspace(1, 10, 10)  # axial mode numbers
a_list = n_ax * np.pi * R2 / H
m = 0  # investigate zero circumferential modes
omega = 1  # constant angular velocity of the rotating flow


def eigenvalue_function_plus(alpha, order):
    func = jvp(order, alpha, n=0)
    return func


alpha = np.linspace(0, 25, 1000)
eig_fun_plus = eigenvalue_function_plus(alpha, 1)


def find_multiple_zeros(f, xmin, xmax, order, intervals=10):
    roots = []
    x = np.linspace(xmin, xmax, intervals)
    for i in range(0, intervals):
        root = fsolve(f, x[i], args=(order))
        roots.append(root)
    roots = np.array(roots)
    return np.unique(roots.round(decimals=2))


roots_plus = find_multiple_zeros(eigenvalue_function_plus, alpha[0], alpha[-1], 1, intervals=11)
roots_plus_y = np.zeros_like(roots_plus)
roots = np.sort(np.unique(roots_plus))

fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(alpha, eig_fun_plus, label=r'$f_(\alpha)$')
ax.plot(roots_plus, roots_plus_y, 'o', label=r'roots')
ax.set_title(r'$\alpha$ roots')
ax.set_xlabel(r'$\alpha$')
ax.legend()
fig.savefig('pictures/alpha_roots_%d.pdf' % (m), bbox_inches='tight')


def eigenvalues(alpha_j, a, omega):
    # for a list of alpha_J, and for a single value of a find the related eigenvalues

    s_plus = 2 * omega / (1 + alpha_j ** 2 / a ** 2) ** (0.5)
    s_minus = -2 * omega / (1 + alpha_j ** 2 / a ** 2) ** (0.5)
    return s_plus, s_minus


rows = len(roots)*2
cols = len(a_list)
eigs_array = np.empty((rows, cols), dtype=complex)

i = 0
for a in a_list:
    eig_plus, eig_minus = eigenvalues(roots, a, omega)
    eigs_list = np.append(eig_plus, eig_minus)
    eigs_array[:, i] = eigs_list
    i += 1


fig, ax = plt.subplots(figsize=(10, 7))
for i in range(0, len(a_list)):
    ax.plot(eigs_array[:, i].real, eigs_array[:, i].imag, '.', label=r'$a = %d$' %(i+1))
ax.set_title('eigenvalues')
ax.set_ylabel(r'$\omega_{R} \ \mathrm{[rad/s]}$')
ax.set_xlabel(r'$\omega_{I} \ \mathrm{[rad/s]}$')
ax.legend()
# fig.savefig('pictures/eigs_%d_%d_%d.pdf' % (m, k, r0), bbox_inches='tight')











# now solve with sun model
# %%COMPUTATIONAL PART
# number of grid nodes in the computational domain
Nz = 20
Nr = 20

rho = 1
ur = 0
uz = 0
p = 1

# implement a constant uniform flow in the annulus duct
density = np.zeros((Nz, Nr))
axialVel = np.zeros((Nz, Nr))
radialVel = np.zeros((Nz, Nr))
tangentialVel = np.zeros((Nz, Nr))
pressure = np.zeros((Nz, Nr))
for ii in range(0, Nz):
    for jj in range(0, Nr):
        density[ii, jj] = rho
        tangentialVel[ii, jj] = R1 + jj * R2 / (Nr - 1) * omega
        pressure[ii, jj] = p

# non-dimensionalization terms:
x_ref = R2
u_ref = tangentialVel[0, -1]
rho_ref = rho
t_ref = x_ref / u_ref
omega_ref = 1 / t_ref
p_ref = rho_ref * u_ref ** 2

# create a meridional object, having the same information of the meridional post-process object of a compressor
duct_Obj = sun.src.AnnulusMeridional(0, H, R1, R2, Nz, Nr, density, radialVel, tangentialVel, axialVel, pressure)
duct_grid = sun.src.sun_grid.SunGrid(duct_Obj)
duct_grid.ShowGrid()

# general workflow of the sun model
sun_obj = sun.src.SunModel(duct_grid)
sun_obj.ComputeBoundaryNormals()
sun_obj.add_shaft_rpm(omega_ref)
sun_obj.AddNormalizationQuantities(rho_ref, u_ref, x_ref)
sun_obj.NormalizeData()
sun_obj.ComputeSpectralGrid()
gradient_routine = 'findiff'
gradient_order = 2
sun_obj.ComputeJacobianPhysical(routine=gradient_routine, order=gradient_order)
sun_obj.AddAMatrixToNodesFrancesco2()
sun_obj.AddBMatrixToNodesFrancesco2()
sun_obj.AddCMatrixToNodesFrancesco2(m=1)
sun_obj.AddEMatrixToNodesFrancesco2()
sun_obj.AddRMatrixToNodesFrancesco2()
sun_obj.AddSMatrixToNodes(turbo=False)
sun_obj.AddHatMatricesToNodes()
sun_obj.ApplySpectralDifferentiation()
sun_obj.build_A_global_matrix()
sun_obj.build_C_global_matrix()
sun_obj.build_R_global_matrix()
sun_obj.build_Z_global_matrix()
sun_obj.set_boundary_conditions('zero pressure', 'zero pressure')
sun_obj.apply_boundary_conditions_generalized()

omega_search = 0
mode_name = r'$[R,Z] = [3, 3]$'
sigma = omega_search / omega_ref
A = sun_obj.Z_g
M = 1j*sun_obj.A_g
C = np.linalg.inv(A - sigma * M)
C = np.dot(C, M)
number_search = 1
print('Searching Eigenvalues with ARPACK...')
eigenvalues, eigenvectors = eigs(C, k=number_search)
eigenvalues = sigma + 1 / eigenvalues
eigenvalues *= omega_ref

marker_size = 100
fig, ax = plt.subplots(figsize=(7, 5))
# ax.scatter(omega_analytical.real, omega_analytical.imag, marker='x', facecolors='blue',
#            s=marker_size, label=r'analytical')
ax.scatter(eigenvalues.real, eigenvalues.imag, marker='o', facecolors='none', edgecolors='red',
           s=marker_size, label=r'numerical')
ax.set_xlabel(r'$\omega_{R}$ [rad/s]')
ax.set_ylabel(r'$\omega_{I}$ [rad/s]')
ax.legend()
# ax.set_xlim([7500, 35000])
# ax.set_ylim([-8000, 8000])
ax.grid(alpha=0.3)


z_grid = sun_obj.data.zGrid
r_grid = sun_obj.data.rGrid
rho_eig = []
ur_eig = []
ut_eig = []
uz_eig = []
p_eig = []

for i in range(len(eigenvectors)):
    if (i) % 5 == 0:
        rho_eig.append(eigenvectors[i])
    elif (i - 1) % 5 == 0 and i != 0:
        ur_eig.append(eigenvectors[i])
    elif (i - 2) % 5 == 0 and i != 0:
        ut_eig.append(eigenvectors[i])
    elif (i - 3) % 5 == 0 and i != 0:
        uz_eig.append(eigenvectors[i])
    elif (i - 4) % 5 == 0 and i != 0:
        p_eig.append(eigenvectors[i])
    else:
        raise ValueError("Not correct indexing for eigenvector retrieval!")


def scaled_eigenvector_real(eig_list):
    array = np.array(eig_list, dtype=complex)
    array = np.reshape(array, (Nz, Nr))
    array_real_scaled = array.real / (np.max(array.real) - np.min(array.real))
    return array_real_scaled


rho_eig_r = scaled_eigenvector_real(rho_eig)
ur_eig_r = scaled_eigenvector_real(ur_eig)
ut_eig_r = scaled_eigenvector_real(ut_eig)
uz_eig_r = scaled_eigenvector_real(uz_eig)
p_eig_r = scaled_eigenvector_real(p_eig)



plt.figure(figsize=(7, 5))
plt.contourf(z_grid, r_grid, rho_eig_r, levels=200, cmap='RdBu')
plt.ylabel(r'$r$ [-]')
plt.xlabel(r'$z$ [-]')
plt.title(r'$\tilde{\rho} \quad$'+mode_name)
plt.colorbar()
# plt.savefig('pictures/%i/eigenfunction_rho_2D_%i_%i_%i.pdf' % (eigenvalues[0].real, Nz, Nr, eigenvalues[0].real), bbox_inches='tight')
plt.figure(figsize=(7, 5))
plt.contourf(z_grid, r_grid, ur_eig_r, levels=200, cmap='RdBu')
plt.ylabel(r'$r$ [-]')
plt.xlabel(r'$z$ [-]')
plt.title(r'$\tilde{u}_r \quad$' + mode_name)
plt.colorbar()
# plt.savefig('pictures/%i/eigenfunction_ur_2D_%i_%i_%i.pdf' % (eigenvalues[0].real, Nz, Nr, eigenvalues[0].real), bbox_inches='tight')
plt.figure(figsize=(7, 5))
plt.contourf(z_grid, r_grid, ut_eig_r, levels=200, cmap='RdBu')
plt.ylabel(r'$r$ [-]')
plt.xlabel(r'$z$ [-]')
plt.title(r'$\tilde{u}_{\theta} \quad$' + mode_name)
plt.colorbar()
# plt.savefig('pictures/%i/eigenfunction_ut_2D_%i_%i_%i.pdf' % (eigenvalues[0].real, Nz, Nr, eigenvalues[0].real), bbox_inches='tight')
plt.figure(figsize=(7, 5))
plt.contourf(z_grid, r_grid, uz_eig_r, levels=200, cmap='RdBu')
plt.ylabel(r'$r$ [-]')
plt.xlabel(r'$z$ [-]')
plt.title(r'$\tilde{u}_z \quad$' + mode_name)
plt.colorbar()
# plt.savefig('pictures/%i/eigenfunction_uz_2D_%i_%i_%i.pdf' % (eigenvalues[0].real, Nz, Nr, eigenvalues[0].real), bbox_inches='tight')
plt.figure(figsize=(7, 5))
plt.contourf(z_grid, r_grid, p_eig_r, levels=200, cmap='RdBu')
plt.ylabel(r'$r$ [-]')
plt.xlabel(r'$z$ [-]')
plt.title(r'$\tilde{p} \quad$' + mode_name)
plt.colorbar()
# plt.savefig('pictures/%i/eigenfunction_p_2D_%i_%i_%i.pdf' % (eigenvalues[0].real, Nz, Nr, eigenvalues[0].real), bbox_inches='tight')



plt.show()