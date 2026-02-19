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
from sun.src.sun_model_multiblock import SunModelMultiBlock
from grid.src.config import Config
from scipy.sparse.linalg import eigs
from utils.styles import *
import pickle

# input data of the problem (SI units)
r2 = 0.2487  # outer radius [m]
r1 = r2/10  # inner radius [m]
M = 0.015  # Mach number
p = 100e3  # pressure [Pa]
T = 288  # temperature [K]
L = 0.25  # length [m]
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
Nz = 45
Nr = 15

with open("../analytical/analytical_eigenvalues.pickle", 'rb') as file:
    omegar_an = pickle.load(file)

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
        axialVel[ii, jj] = M*a
        pressure[ii, jj] = p

config = Config('duct.ini')
duct_Obj1 = sun.src.AnnulusMeridional(0, L, r1, r2, Nz, Nr,
                                     density, radialVel, tangentialVel, axialVel, pressure, config,
                                      mode='gauss-lobatto')
duct_Obj1.normalize_data()
duct_grid1 = sun.src.sun_grid.SunGrid(duct_Obj1)
sun_obj = sun.src.SunModel(duct_grid1, config=config)

sun_blocks = [sun_obj]
ii = 0
for sun_obj in sun_blocks:
    sun_obj.ComputeBoundaryNormals()
    sun_obj.set_overwriting_equation_euler_wall('uz')
    sun_obj.ComputeSpectralGrid()
    sun_obj.ComputeJacobianPhysical()
    sun_obj.AddAMatrixToNodes_francesco()
    sun_obj.AddBMatrixToNodes_francesco()
    sun_obj.AddCMatrixToNodes_francesco()
    sun_obj.AddEMatrixToNodes_francesco()
    sun_obj.AddRMatrixToNodes_francesco()
    sun_obj.AddSMatrixToNodes()
    sun_obj.AddHatMatricesToNodes()
    sun_obj.ApplySpectralDifferentiationKronecker()
    sun_obj.build_A_global_matrix()
    sun_obj.build_C_global_matrix()
    sun_obj.build_R_global_matrix()
    sun_obj.build_S_global_matrix()
    sun_obj.build_Z_global_matrix()
    sun_obj.compute_L_matrices(ii)
    sun_obj.set_boundary_conditions()
    sun_obj.apply_boundary_conditions_generalized()
    ii += 1

sun_multiblock = SunModelMultiBlock(sun_blocks, config)
sun_multiblock.construct_L_global_matrices()

omega_search = 10000
sigma = omega_search / omega_ref

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



# PLOT RESULTS
marker_size = 125
fig, ax = plt.subplots()
ax.scatter(omegar_an, omegar_an*0, marker='x', facecolors='blue',
           s=marker_size, label=r'analytical')
ax.scatter(eigenvalues.real, eigenvalues.imag, marker='o', facecolors='none', edgecolors='red',
           s=marker_size, label=r'numerical')
ax.set_xlabel(r'$\omega_{R}$ [rad/s]', fontsize=font_labels)
ax.set_ylabel(r'$\omega_{I}$ [rad/s]', fontsize=font_labels)
ax.set_xlim([4500, 17500])
ax.set_ylim([-2000, 2000])
plt.xticks(fontsize=font_axes)
plt.yticks(fontsize=font_axes)
ax.legend(fontsize=font_legend)
ax.grid(alpha=grid_opacity)
fig.savefig('pictures/%02i_%02i/chi_map_arnoldi.pdf' % (Nz, Nr), bbox_inches='tight')

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



    xtick_locations = [0, L/config.get_reference_length()]
    xtick_labels = [r'$0$', r'$L$']
    ytick_locations = [r1/config.get_reference_length(), r2/config.get_reference_length()]
    ytick_labels = [r'$r_1$', r'$r_2$']

    plt.figure()
    cnt = plt.contourf(z_grid, r_grid, rho_eig_r, levels=N_levels, cmap='bwr')
    for c in cnt.collections:
        c.set_edgecolor("face")
        c.set_linewidth(0.000000000001)
    plt.ylabel(r'$r$ [-]', fontsize=font_labels)
    plt.xlabel(r'$z$ [-]', fontsize=font_labels)
    plt.xticks(ticks=xtick_locations, labels=xtick_labels, fontsize=font_axes)
    plt.yticks(ticks=ytick_locations, labels=ytick_labels, fontsize=font_axes)
    plt.title(r'$\tilde{\rho}_{%i}$' % (ivec + 1), fontsize=font_title)
    cnbar = plt.colorbar(cnt)
    cnbar.ax.tick_params(labelsize=font_axes)
    plt.savefig('pictures/%02i_%02i/eigenfunction_rho_%i.pdf' % (Nz, Nr, ivec + 1), bbox_inches='tight')

    plt.figure()
    cnt = plt.contourf(z_grid, r_grid, ur_eig_r, levels=N_levels, cmap='bwr')
    for c in cnt.collections:
        c.set_edgecolor("face")
        c.set_linewidth(0.000000000001)
    plt.ylabel(r'$r$ [-]', fontsize=font_labels)
    plt.xlabel(r'$z$ [-]', fontsize=font_labels)
    plt.xticks(ticks=xtick_locations, labels=xtick_labels, fontsize=font_axes)
    plt.yticks(ticks=ytick_locations, labels=ytick_labels, fontsize=font_axes)
    plt.title(r'$\tilde{u}_{r,%i}$' % (ivec + 1), fontsize=font_title)
    cnbar = plt.colorbar(cnt)
    cnbar.ax.tick_params(labelsize=font_axes)
    plt.savefig('pictures/%02i_%02i/eigenfunction_ur_%i.pdf' % (Nz, Nr, ivec + 1), bbox_inches='tight')

    plt.figure()
    cnt = plt.contourf(z_grid, r_grid, ut_eig_r, levels=N_levels, cmap='bwr')
    for c in cnt.collections:
        c.set_edgecolor("face")
        c.set_linewidth(0.000000000001)
    plt.ylabel(r'$r$ [-]', fontsize=font_labels)
    plt.xlabel(r'$z$ [-]', fontsize=font_labels)
    plt.xticks(ticks=xtick_locations, labels=xtick_labels, fontsize=font_axes)
    plt.yticks(ticks=ytick_locations, labels=ytick_labels, fontsize=font_axes)
    plt.title(r'$\tilde{u}_{\theta,%i}$' % (ivec + 1), fontsize=font_title)
    cnbar = plt.colorbar(cnt)
    cnbar.ax.tick_params(labelsize=font_axes)
    plt.savefig('pictures/%02i_%02i/eigenfunction_ut_%i.pdf' % (Nz, Nr, ivec + 1), bbox_inches='tight')

    plt.figure()
    cnt = plt.contourf(z_grid, r_grid, uz_eig_r, levels=N_levels, cmap='bwr')
    for c in cnt.collections:
        c.set_edgecolor("face")
        c.set_linewidth(0.000000000001)
    plt.ylabel(r'$r$ [-]', fontsize=font_labels)
    plt.xlabel(r'$z$ [-]', fontsize=font_labels)
    plt.xticks(ticks=xtick_locations, labels=xtick_labels, fontsize=font_axes)
    plt.yticks(ticks=ytick_locations, labels=ytick_labels, fontsize=font_axes)
    plt.title(r'$\tilde{u}_{z,%i}$' % (ivec + 1), fontsize=font_title)
    cnbar = plt.colorbar(cnt)
    cnbar.ax.tick_params(labelsize=font_axes)
    plt.savefig('pictures/%02i_%02i/eigenfunction_uz_%i.pdf' % (Nz, Nr, ivec + 1), bbox_inches='tight')

    plt.figure()
    cnt = plt.contourf(z_grid, r_grid, p_eig_r, levels=N_levels, cmap='bwr')
    for c in cnt.collections:
        c.set_edgecolor("face")
        c.set_linewidth(0.000000000001)
    plt.ylabel(r'$r$ [-]', fontsize=font_labels)
    plt.xlabel(r'$z$ [-]', fontsize=font_labels)
    plt.xticks(ticks=xtick_locations, labels=xtick_labels, fontsize=font_axes)
    plt.yticks(ticks=ytick_locations, labels=ytick_labels, fontsize=font_axes)
    plt.title(r'$\tilde{p}_{%i}$' % (ivec + 1), fontsize=font_title)
    cnbar = plt.colorbar(cnt)
    plt.quiver(z_grid, r_grid, uz_eig_r, ur_eig_r)
    cnbar.ax.tick_params(labelsize=font_axes)
    plt.savefig('pictures/%02i_%02i/eigenfunction_p_%i.pdf' % (Nz, Nr, ivec + 1), bbox_inches='tight')


file_path = 'data/meta/%02i_%02i_%02i.pickle'%(Nz, Nr, config.get_grid_transformation_gradient_order())
with open(file_path, 'wb') as file:
    pickle.dump(eigenvalues, file)

# plt.show()
