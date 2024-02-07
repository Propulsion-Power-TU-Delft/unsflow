#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 09:29:59 2023
@author: F. Neri, TU Delft
"""

import matplotlib.pyplot as plt
import numpy as np
import Sun
import os
from Sun.src.sun_model_multiblock import SunModelMultiBlock
from Grid.src.config import Config
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
rho = p/R/T  # density [kg/m3]
a = np.sqrt(gmma * p / rho)  # ideal speed of sound [m/s]
HARMONIC_ORDER = 1

# non-dimensionalization terms:
x_ref = r1
u_ref = M * a
rho_ref = rho
t_ref = x_ref / u_ref
omega_ref = 1 / t_ref
rpm_ref = omega_ref*60/2/np.pi
p_ref = rho_ref * u_ref ** 2


# %%%%%%%%%%%%%%%%%%%%%%% COMPUTATIONAL PART %%%%%%%%%%%%%%%%%%%%%%%
# number of grid nodes in the computational domain
Nz = 60
Nr = 20
# Nz = 10//2
# Nr = 5

folder_path = "pictures/%02i_%02i" %(Nz, Nr)  # Replace with the desired folder path
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

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

config = Config('duct.ini')
duct_Obj1 = Sun.src.AnnulusMeridional(0, L, r1, r2, Nz, Nr,
                                     density, radialVel, tangentialVel, axialVel, pressure, config)
duct_Obj2 = Sun.src.AnnulusMeridional(L/2, L, r1, r2, Nz, Nr,
                                     density, radialVel, tangentialVel, axialVel, pressure, config)
duct_Obj1.normalize_data()
# duct_Obj2.normalize_data(rho_ref, u_ref, x_ref)
duct_grid1 = Sun.src.sun_grid.SunGrid(duct_Obj1)
# duct_grid2 = Sun.src.sun_grid.SunGrid(duct_Obj2)
sun_obj = Sun.src.SunModel(duct_grid1, config=config)
# sun_obj2 = Sun.src.SunModel(duct_grid2, config=config)

sun_blocks = [sun_obj]
ii = 0
for sun_obj in sun_blocks:
    sun_obj.ComputeBoundaryNormals()
    sun_obj.set_overwriting_equation_euler_wall('utheta')
    sun_obj.ComputeSpectralGrid()
    sun_obj.ComputeJacobianPhysical()
    sun_obj.AddAMatrixToNodesFrancesco2()
    sun_obj.AddBMatrixToNodesFrancesco2()
    sun_obj.AddCMatrixToNodesFrancesco2()
    sun_obj.AddEMatrixToNodesFrancesco2()
    sun_obj.AddRMatrixToNodesFrancesco2()
    sun_obj.AddSMatrixToNodes()
    sun_obj.AddHatMatricesToNodes()
    sun_obj.ApplySpectralDifferentiation()
    sun_obj.build_A_global_matrix()
    sun_obj.build_C_global_matrix()
    sun_obj.build_R_global_matrix()
    sun_obj.build_S_global_matrix()
    sun_obj.build_Z_global_matrix()
    sun_obj.compute_L_matrices(ii)
    sun_obj.set_boundary_conditions()
    sun_obj.apply_boundary_conditions_generalized()
    ii +=1

sun_multiblock = SunModelMultiBlock(sun_blocks, config)
sun_multiblock.construct_L_global_matrices()
# sun_multiblock.apply_matching_conditions()
# sun_multiblock.compute_P_Y_matrices()
# sun_multiblock.solve_evp(sort_mode='real increasing', sigma=21000/config.get_reference_omega())
# sun_multiblock.extract_eigenfields()
# sun_multiblock.plot_eigenfrequencies(save_filename='eigenfrequencies', normalization=False)
# sun_multiblock.plot_eigenfields(n=5, save_filename='eigenmode')
# sun_multiblock.write_results()

omega_search = 21000
sigma = omega_search / omega_ref


# Y1 = np.concatenate((-sun_multiblock.L0, np.zeros_like(sun_multiblock.L0)), axis=1)
# Y2 = np.concatenate((np.zeros_like(sun_multiblock.L0), np.eye(sun_multiblock.L0.shape[0])), axis=1)
# Y = np.concatenate((Y1, Y2), axis=0)  # Y matrix of EVP problem
# plt.figure()
# plt.spy(Y)
#
# P1 = np.concatenate((sun_multiblock.L1, sun_multiblock.L2), axis=1)
# P2 = np.concatenate((np.eye(sun_multiblock.L0.shape[0]), np.zeros_like(sun_multiblock.L0)), axis=1)
# P = np.concatenate((P1, P2), axis=0)  # P matrix of EVP problem
# plt.figure()
# plt.spy(P)
#
# print('Transforming the problem...')
# Ytilde = np.linalg.inv(Y-sigma*P) @ P
# print('Searching Eigenvalues with ARPACK...')
# eigenvalues, eigenvectors = eigs(Ytilde, k=config.get_research_number_omega_eigenvalues())

fig, ax = plt.subplots(1, 2, figsize=(12,5))
ax[0].spy(sun_multiblock.L0)
ax[0].set_title(r'$L_0$')
ax[1].spy(sun_multiblock.L1)
ax[1].set_title(r'$L_1$')
fig.savefig('pictures/%i_%i/L0_L1.pdf' % (Nz, Nr), bbox_inches='tight')

A = sun_multiblock.L0
M = -sun_multiblock.L1
C = np.linalg.inv(A - sigma * M)
C = np.dot(C, M)
print('Searching Eigenvalues with ARPACK...')
eigenvalues, eigenvectors = eigs(C, k=config.get_research_number_omega_eigenvalues())
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


omegar_an = [13450, 21077, 26721, 31296, 35049]
omegai_an = [0, 0, 0, 0, 0]

# PLOT RESULTS
marker_size = 50
fig, ax = plt.subplots()
ax.scatter(omegar_an, omegai_an, marker='x', facecolors='blue',
           s=marker_size, label=r'analytical')
ax.scatter(eigenvalues.real, eigenvalues.imag, marker='o', facecolors='none', edgecolors='red',
           s=marker_size, label=r'numerical')
ax.set_xlabel(r'$\omega_{R}$ [rad/s]')
ax.set_ylabel(r'$\omega_{I}$ [rad/s]')
ax.set_xlim([7500, 38000])
ax.set_ylim([-8000, 8000])
ax.legend()
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
    cnt = plt.contourf(z_grid, r_grid, rho_eig_r, levels=20, cmap='bwr')
    plt.ylabel(r'$r$ [-]')
    plt.xlabel(r'$z$ [-]')
    plt.title(r'$\tilde{\rho}_{%i}$' % (ivec + 1))
    plt.colorbar()
    plt.savefig('pictures/%i_%i/eigenfunction_rho_%i.pdf' % (Nz, Nr, ivec + 1), bbox_inches='tight')

    plt.figure(figsize=(7, 5))
    cnt = plt.contourf(z_grid, r_grid, ur_eig_r, levels=20, cmap='bwr')
    plt.ylabel(r'$r$ [-]')
    plt.xlabel(r'$z$ [-]')
    plt.title(r'$\tilde{u}_{r,%i}$' % (ivec + 1))
    plt.colorbar()
    plt.savefig('pictures/%i_%i/eigenfunction_ur_%i.pdf' % (Nz, Nr, ivec + 1), bbox_inches='tight')

    plt.figure(figsize=(7, 5))
    cnt = plt.contourf(z_grid, r_grid, ut_eig_r, levels=20, cmap='bwr')
    plt.ylabel(r'$r$ [-]')
    plt.xlabel(r'$z$ [-]')
    plt.title(r'$\tilde{u}_{\theta,%i}$' % (ivec + 1))
    plt.colorbar()
    plt.savefig('pictures/%i_%i/eigenfunction_ut_%i.pdf' % (Nz, Nr, ivec + 1), bbox_inches='tight')

    plt.figure(figsize=(7, 5))
    cnt = plt.contourf(z_grid, r_grid, uz_eig_r, levels=20, cmap='bwr')
    plt.ylabel(r'$r$ [-]')
    plt.xlabel(r'$z$ [-]')
    plt.title(r'$\tilde{u}_{z,%i}$' % (ivec + 1))
    plt.colorbar()
    plt.savefig('pictures/%i_%i/eigenfunction_uz_%i.pdf' % (Nz, Nr, ivec + 1), bbox_inches='tight')

    plt.figure(figsize=(7, 5))
    cnt = plt.contourf(z_grid, r_grid, p_eig_r, levels=20, cmap='bwr')
    plt.ylabel(r'$r$ [-]')
    plt.xlabel(r'$z$ [-]')
    plt.title(r'$\tilde{p}_{%i}$' % (ivec + 1))
    plt.colorbar()
    plt.savefig('pictures/%i_%i/eigenfunction_p_%i.pdf' % (Nz, Nr, ivec + 1), bbox_inches='tight')

plt.show()
