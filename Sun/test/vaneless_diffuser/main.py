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
import os

# input data of the problem (SI units)
R1 = 1  # diffuser inlet radius [m]
R2 = 2  # diffuser outlet radius [m]
P = 1  # constant diffuser pressure [Pa], or maybe Bernoulli?
RHO = 1  # constand density [kg/m3]
W = R1/5  # air gas constant [kJ/kgK]
V = 1  # magnitude of velocity at inlet
VR = 0.05  # diffuser inlet radial velocity [m/s]
VTHETA = np.sqrt(1-VR**2)  # diffuser inlet tangential velocity [m/s]
VZ = 0  # diffuser inlet axial velocity [m/s]
Nr = 10
Nz = 50
M_HARMONIC = 1

# non-dimensionalization terms:
x_ref = R1
u_ref = 1
rho_ref = RHO
t_ref = x_ref / u_ref
omega_ref = 1 / t_ref
p_ref = rho_ref * u_ref ** 2

number_search = 10
gradient_routine = 'numpy'
gradient_order = 2
folder_path = "pictures/" + str(Nz) + "_" + str(Nr)  # Replace with the desired folder path
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# implement a constant uniform flow in the annulus duct
radius = np.linspace(R1, R2, Nz)
density = np.zeros((Nz, Nr))
radialVel = np.zeros((Nz, Nr))
azimuthalVel = np.zeros((Nz, Nr))
axialVel = np.zeros((Nz, Nr))
pressure = np.zeros((Nz, Nr))
dur_dr = np.zeros((Nz, Nr))
dut_dr = np.zeros((Nz, Nr))
dp_dr = np.zeros((Nz, Nr))
for ii in range(0, Nz):
    for jj in range(0, Nr):
        density[ii, jj] = RHO
        radialVel[ii, jj] = VR*R1/radius[ii]
        azimuthalVel[ii, jj] = VTHETA * R1 / radius[ii]
        pressure[ii, jj] = P +0.5*(V**2 - radialVel[ii, jj]**2 - azimuthalVel[ii, jj]**2)
        dur_dr[ii, jj] = -VR*R1/(radius[ii]**2)
        dut_dr[ii, jj] = -VTHETA * R1 / (radius[ii] ** 2)
        dp_dr[ii, jj] = R1/(radius[ii]**2) * (radialVel[ii, jj]*VR + azimuthalVel[ii, jj]*VTHETA)

# create a meridional object, having the same information of the meridional post-process object of a compressor
duct_Obj = Sun.src.DiffuserMeridional(R1, R2, W, density, radialVel, azimuthalVel, axialVel, pressure,
                                      dur_dr, dut_dr, dp_dr, grid_refinement=1)
duct_Obj.contour_fields()
duct_Obj.normalize_data(rho_ref, u_ref, x_ref)
duct_grid = Sun.src.sun_grid.SunGrid(duct_Obj)
duct_grid.ShowGrid()

# general workflow of the sun model
sun_obj = Sun.src.SunModel(duct_grid)
sun_obj.set_overwriting_equation_euler_wall('uz')
# sun_obj.ComputeBoundaryNormals()
# sun_obj.ShowNormals()
sun_obj.add_shaft_rpm(omega_ref)
sun_obj.set_normalization_quantities(mode='duct object')
sun_obj.ComputeSpectralGrid()
sun_obj.ComputeJacobianPhysical(routine=gradient_routine, order=gradient_order, method='nearest')
sun_obj.ContourTransformation(save_filename='%i_%i/transformation' % (Nz, Nr))
sun_obj.AddAMatrixToNodesFrancesco2(normalize=False)
sun_obj.AddBMatrixToNodesFrancesco2(normalize=False)
sun_obj.AddCMatrixToNodesFrancesco2(m=M_HARMONIC, normalize=False)
sun_obj.AddEMatrixToNodesFrancesco2(normalize=False)
sun_obj.AddRMatrixToNodesFrancesco2(normalize=False)
sun_obj.AddSMatrixToNodes(turbo=False, normalize=False)
sun_obj.AddHatMatricesToNodes()
sun_obj.ApplySpectralDifferentiation()
sun_obj.build_A_global_matrix()
sun_obj.build_C_global_matrix()
sun_obj.build_R_global_matrix()
sun_obj.build_Z_global_matrix()
sun_obj.build_S_global_matrix()
sun_obj.set_boundary_conditions('zero pressure', 'zero pressure', 'zero axial', 'zero axial')
sun_obj.apply_boundary_conditions_generalized()

omega_search = 0
sigma = omega_search / omega_ref
A = sun_obj.Z_g
M = 1j * sun_obj.A_g
C = np.linalg.inv(A - sigma * M)
C = np.dot(C, M)
print('Searching Eigenvalues with ARPACK...')
eigenvalues, eigenvectors = eigs(C, k=number_search)
eigenvalues = sigma + 1 / eigenvalues
eigenvalues *= omega_ref

# make copies of the arrays to sort
eigenfreqs = np.copy(eigenvalues)
df = np.copy(eigenvalues.imag)
rs = np.copy(eigenvalues.real)
eigenvecs = np.copy(eigenvectors)
sorted_indices = sorted(range(len(rs)), key=lambda i: rs[i], reverse=False)

# order the original arrays following the sorting indices
for i in range(len(sorted_indices)):
    eigenvalues[i] = eigenfreqs[sorted_indices[i]]
    eigenvectors[:, i] = eigenvecs[:, sorted_indices[i]]





# PLOT RESULTS
marker_size = 100
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(eigenvalues.real, eigenvalues.imag, marker='o', facecolors='none', edgecolors='red',
           s=marker_size, label=r'numerical')
ax.set_xlabel(r'$\omega_{R}$ [rad/s]')
ax.set_ylabel(r'$\omega_{I}$ [rad/s]')
ax.legend()
# ax.set_xlim([7500, 38000])
# ax.set_ylim([-8000, 8000])
ax.grid(alpha=0.3)
fig.savefig('pictures/%i_%i/chi_map_arnoldi.pdf' % (Nz, Nr), bbox_inches='tight')

# EIGENFUNCTIONS
z_grid = sun_obj.data.zGrid
r_grid = sun_obj.data.rGrid

for ivec in range(np.shape(eigenvectors)[1]):
    eigenvec = eigenvectors[:, ivec]
    rho_eig = []
    ur_eig = []
    ut_eig = []
    uz_eig = []
    p_eig = []

    for i in range(len(eigenvec)):
        if (i) % 5 == 0:
            rho_eig.append(eigenvec[i])
        elif (i - 1) % 5 == 0 and i != 0:
            ur_eig.append(eigenvec[i])
        elif (i - 2) % 5 == 0 and i != 0:
            ut_eig.append(eigenvec[i])
        elif (i - 3) % 5 == 0 and i != 0:
            uz_eig.append(eigenvec[i])
        elif (i - 4) % 5 == 0 and i != 0:
            p_eig.append(eigenvec[i])
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
    cnt = plt.contourf(z_grid, r_grid, rho_eig_r, levels=200, cmap='bwr')
    for c in cnt.collections:
        c.set_edgecolor("face")
    plt.ylabel(r'$r$ [-]')
    plt.xlabel(r'$z$ [-]')
    plt.title(r'$\tilde{\rho}_%i$' % (ivec + 1))
    plt.colorbar()
    plt.savefig('pictures/%i_%i/eigenfunction_rho_%i.pdf' % (Nz, Nr, ivec + 1), bbox_inches='tight')

    plt.figure(figsize=(7, 5))
    cnt = plt.contourf(z_grid, r_grid, ur_eig_r, levels=200, cmap='bwr')
    for c in cnt.collections:
        c.set_edgecolor("face")
    plt.ylabel(r'$r$ [-]')
    plt.xlabel(r'$z$ [-]')
    plt.title(r'$\tilde{u}_{r,%i}$' % (ivec + 1))
    plt.colorbar()
    plt.savefig('pictures/%i_%i/eigenfunction_ur_%i.pdf' % (Nz, Nr, ivec + 1), bbox_inches='tight')

    plt.figure(figsize=(7, 5))
    cnt = plt.contourf(z_grid, r_grid, ut_eig_r, levels=200, cmap='bwr')
    for c in cnt.collections:
        c.set_edgecolor("face")
    plt.ylabel(r'$r$ [-]')
    plt.xlabel(r'$z$ [-]')
    plt.title(r'$\tilde{u}_{\theta,%i}$' % (ivec + 1))
    plt.colorbar()
    plt.savefig('pictures/%i_%i/eigenfunction_ut_%i.pdf' % (Nz, Nr, ivec + 1), bbox_inches='tight')

    plt.figure(figsize=(7, 5))
    cnt = plt.contourf(z_grid, r_grid, uz_eig_r, levels=200, cmap='bwr')
    for c in cnt.collections:
        c.set_edgecolor("face")
    plt.ylabel(r'$r$ [-]')
    plt.xlabel(r'$z$ [-]')
    plt.title(r'$\tilde{u}_{z,%i}$' % (ivec + 1))
    plt.colorbar()
    plt.savefig('pictures/%i_%i/eigenfunction_uz_%i.pdf' % (Nz, Nr, ivec + 1), bbox_inches='tight')

    plt.figure(figsize=(7, 5))
    cnt = plt.contourf(z_grid, r_grid, p_eig_r, levels=200, cmap='bwr')
    for c in cnt.collections:
        c.set_edgecolor("face")
    plt.ylabel(r'$r$ [-]')
    plt.xlabel(r'$z$ [-]')
    plt.title(r'$\tilde{p}_%i$' % (ivec + 1))
    plt.colorbar()
    plt.savefig('pictures/%i_%i/eigenfunction_p_%i.pdf' % (Nz, Nr, ivec + 1), bbox_inches='tight')

# plt.show()
