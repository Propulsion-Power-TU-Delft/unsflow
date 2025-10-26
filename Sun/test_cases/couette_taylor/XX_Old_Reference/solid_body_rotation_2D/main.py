#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 09:29:59 2023
@author: F. Neri, TU Delft
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, sin, cos
import Sun
import os
from Sun.src.sun_model_multiblock import SunModelMultiBlock
from Grid.src.config import Config
from scipy.sparse.linalg import eigs
from Utils.styles import *
import pickle

# input data of the problem (SI units)
R1 = 1  # inner radius [m]
D = 0.05  # radial gap [m]
R2 = R1 + D  # outer radius [m]
P1 = 100e3  # pressure at hub [Pa]
T = 288  # temperature everywhere [K]
L = 3 * D  # length [m]
GMMA = 1.4  # cp/cv ratio of air
RHO = 1.014  # density everywhere [kg/m3]
MU = 1  # omega ratio
OMEGA2 = 10  # omega outer cylinder [rad/s]
OMEGA1 = OMEGA2 / MU  # omega inner cylinder [rad/s]
HARMONIC_ORDER = 0  # axisymmetric perturbations

# non-dimensionalization terms:
x_ref = (R1 + R2) / 2
omega_ref = (OMEGA1 + OMEGA2) / 2
u_ref = omega_ref * x_ref
rho_ref = RHO
t_ref = x_ref / u_ref
omega_ref = 1 / t_ref
rpm_ref = omega_ref * 60 / 2 / pi
p_ref = rho_ref * u_ref ** 2








# %%%%%%%%%%%%%%%%%%%%%%% ANALYTICAL PART %%%%%%%%%%%%%%%%%%%%%%%
LIMIT = 3
radial_orders = np.linspace(1, LIMIT, LIMIT)
axial_orders = np.linspace(1, LIMIT, LIMIT)


class EigenFrequency:
    """class who stores the information of a mode. For now consider only the positive frequencies"""

    def __init__(self, radial_order, axial_order, eigenfrequency_pos, eigenfrequency_neg):
        self.r_ord = radial_order
        self.z_ord = axial_order
        self.omega_pos = eigenfrequency_pos
        self.omega_neg = eigenfrequency_neg


eigs_an_list = []
for m in radial_orders:
    for alpha in axial_orders:
        omega_plus = (2 * OMEGA1 * (pi * alpha * D / L) ** 2) / ((pi * alpha * D / L) ** 2 + m ** 2 * pi ** 2)
        omega_minus = -(2 * OMEGA1 * (pi * alpha * D / L) ** 2) / ((pi * alpha * D / L) ** 2 + m ** 2 * pi ** 2)
        eigs_an_list.append(EigenFrequency(m, alpha, omega_plus, omega_minus))

plt.figure()
for eig in eigs_an_list:
    plt.scatter(eig.omega_pos.real, eig.omega_pos.imag, c='r', s=50)
    plt.scatter(eig.omega_neg.real, eig.omega_neg.imag, c='b', s=50)
plt.xlabel(r'$\omega_R \ \rm{[rad/s]}$')
plt.ylabel(r'$\omega_I \ \rm{[rad/s]}$')
plt.grid(alpha=grid_opacity)
# plt.show()









# %%%%%%%%%%%%%%%%%%%%%%% COMPUTATIONAL PART %%%%%%%%%%%%%%%%%%%%%%%
# number of grid nodes in the computational domain
Nz = 30
Nr = 10

folder_path = "pictures/%02i_%02i" % (Nz, Nr)  # Replace with the desired folder path
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

density = np.ones((Nz, Nr))*RHO
radialVel = np.zeros((Nz, Nr))
tangentialVel = np.zeros((Nz, Nr))
axialVel = np.zeros((Nz, Nr))
pressure = np.ones((Nz, Nr))*P1


config = Config('config.ini')
duct_obj = Sun.src.AnnulusMeridional(0, L, R1, R2, Nz, Nr, density, radialVel, tangentialVel, axialVel, pressure,
                                      config, mode='gauss-lobatto')


# override the coutte-taylor baseflow solution
duct_obj.zeta = (duct_obj.r_grid-R1)/D

duct_obj.ut = (R1 + duct_obj.zeta*D)*OMEGA1*(1-(1-MU)*duct_obj.zeta)
duct_obj.p = P1 + duct_obj.zeta*R1 + duct_obj.zeta**2*(D/2-R1*(1-MU)) + duct_obj.zeta**3*(R1*(1-MU)**2/3 -
                                                            2*(1-MU)*D/3) + duct_obj.zeta**4*D*(1-MU)**2/4
duct_obj.dut_dr = OMEGA1*(-R1*(1-MU)+D-2*duct_obj.zeta*D*(1-MU))/D
duct_obj.dp_dr = duct_obj.rho*duct_obj.ut**2/duct_obj.r_grid

plt.figure()
plt.contourf(duct_obj.z_grid, duct_obj.r_grid, duct_obj.zeta, levels=N_levels)
plt.xlabel('z')
plt.ylabel('r')
plt.title(r'$\zeta \ \rm{[-]}$')
plt.colorbar()

plt.figure()
plt.contourf(duct_obj.z_grid, duct_obj.r_grid, duct_obj.ut, levels=N_levels)
plt.xlabel('z')
plt.ylabel('r')
plt.title(r'$u_{\theta} \ \rm{[m/s]}$')
plt.colorbar()

plt.figure()
plt.contourf(duct_obj.z_grid, duct_obj.r_grid, duct_obj.dut_dr, levels=N_levels)
plt.xlabel('z')
plt.ylabel('r')
plt.title(r'$du_{\theta}/dr \ \rm{[Pa/m]}$')
plt.colorbar()

plt.figure()
plt.contourf(duct_obj.z_grid, duct_obj.r_grid, duct_obj.p, levels=N_levels)
plt.xlabel('z')
plt.ylabel('r')
plt.title(r'$p \ \rm{[Pa]}$')
plt.colorbar()

plt.figure()
plt.contourf(duct_obj.z_grid, duct_obj.r_grid, duct_obj.dp_dr, levels=N_levels)
plt.xlabel('z')
plt.ylabel('r')
plt.title(r'$dp/dr \ \rm{[Pa/m]}$')
plt.colorbar()

# plt.show()

duct_obj.normalize_data()
duct_grid1 = Sun.src.sun_grid.SunGrid(duct_obj)
sun_obj = Sun.src.SunModel(duct_grid1, config=config)

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
    ii += 1

sun_multiblock = SunModelMultiBlock(sun_blocks, config)
sun_multiblock.construct_L_global_matrices()

omega_search = 0.25
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
ax.scatter(eigenvalues.real, eigenvalues.imag, marker='o', facecolors='none', edgecolors='red', s=marker_size,
           label=r'numerical')
ax.set_xlabel(r'$\omega_{R}$ [rad/s]', fontsize=font_labels)
ax.set_ylabel(r'$\omega_{I}$ [rad/s]', fontsize=font_labels)
# ax.set_xlim([12000, 36000])
# ax.set_ylim([-8000, 8000])
plt.xticks(fontsize=font_axes)
plt.yticks(fontsize=font_axes)
ax.legend(fontsize=font_legend)
ax.grid(alpha=grid_opacity)
fig.savefig('pictures/%02i_%02i/chi_map_arnoldi.pdf' % (Nz, Nr), bbox_inches='tight')
# plt.show()

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

    xtick_locations = [0, L / config.get_reference_length()]
    xtick_labels = [r'$0$', r'$L$']
    ytick_locations = [R1 / config.get_reference_length(), R2 / config.get_reference_length()]
    ytick_labels = [r'$r_1$', r'$r_2$']

    # plt.figure()
    # cnt = plt.contourf(z_grid, r_grid, rho_eig_r, levels=N_levels_medium, cmap='bwr')
    # for c in cnt.collections:
    #     c.set_edgecolor("face")
    #     c.set_linewidth(0.000000000001)
    # plt.ylabel(r'$r$ [-]', fontsize=font_labels)
    # plt.xlabel(r'$z$ [-]', fontsize=font_labels)
    # plt.xticks(ticks=xtick_locations, labels=xtick_labels, fontsize=font_axes)
    # plt.yticks(ticks=ytick_locations, labels=ytick_labels, fontsize=font_axes)
    # plt.title(r'$\tilde{\rho}_{%i}$' % (ivec + 1), fontsize=font_title)
    # cnbar = plt.colorbar(cnt)
    # cnbar.ax.tick_params(labelsize=font_axes)
    # plt.savefig('pictures/%02i_%02i/eigenfunction_rho_%i.pdf' % (Nz, Nr, ivec + 1), bbox_inches='tight')
    #
    plt.figure()
    cnt = plt.contourf(z_grid, r_grid, ur_eig_r, levels=N_levels_medium, cmap='bwr')
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
    #
    # plt.figure()
    # cnt = plt.contourf(z_grid, r_grid, ut_eig_r, levels=N_levels_medium, cmap='bwr')
    # for c in cnt.collections:
    #     c.set_edgecolor("face")
    #     c.set_linewidth(0.000000000001)
    # plt.ylabel(r'$r$ [-]', fontsize=font_labels)
    # plt.xlabel(r'$z$ [-]', fontsize=font_labels)
    # plt.xticks(ticks=xtick_locations, labels=xtick_labels, fontsize=font_axes)
    # plt.yticks(ticks=ytick_locations, labels=ytick_labels, fontsize=font_axes)
    # plt.title(r'$\tilde{u}_{\theta,%i}$' % (ivec + 1), fontsize=font_title)
    # cnbar = plt.colorbar(cnt)
    # cnbar.ax.tick_params(labelsize=font_axes)
    # plt.savefig('pictures/%02i_%02i/eigenfunction_ut_%i.pdf' % (Nz, Nr, ivec + 1), bbox_inches='tight')
    #
    # plt.figure()
    # cnt = plt.contourf(z_grid, r_grid, uz_eig_r, levels=N_levels_medium, cmap='bwr')
    # for c in cnt.collections:
    #     c.set_edgecolor("face")
    #     c.set_linewidth(0.000000000001)
    # plt.ylabel(r'$r$ [-]', fontsize=font_labels)
    # plt.xlabel(r'$z$ [-]', fontsize=font_labels)
    # plt.xticks(ticks=xtick_locations, labels=xtick_labels, fontsize=font_axes)
    # plt.yticks(ticks=ytick_locations, labels=ytick_labels, fontsize=font_axes)
    # plt.title(r'$\tilde{u}_{z,%i}$' % (ivec + 1), fontsize=font_title)
    # cnbar = plt.colorbar(cnt)
    # cnbar.ax.tick_params(labelsize=font_axes)
    # plt.savefig('pictures/%02i_%02i/eigenfunction_uz_%i.pdf' % (Nz, Nr, ivec + 1), bbox_inches='tight')

    plt.figure()
    cnt = plt.contourf(z_grid, r_grid, p_eig_r, levels=N_levels, cmap='bwr')
    plt.quiver(z_grid, r_grid, uz_eig_r, ur_eig_r)
    # plt.gca().set_aspect('equal', adjustable='box')
    for c in cnt.collections:
        c.set_edgecolor("face")
        c.set_linewidth(0.000000000001)
    plt.ylabel(r'$r$ [-]', fontsize=font_labels)
    plt.xlabel(r'$z$ [-]', fontsize=font_labels)
    plt.xticks(ticks=xtick_locations, labels=xtick_labels, fontsize=font_axes)
    plt.yticks(ticks=ytick_locations, labels=ytick_labels, fontsize=font_axes)
    plt.title(r'$\tilde{p}_{%i}$' % (ivec + 1), fontsize=font_title)
    cnbar = plt.colorbar(cnt)
    cnbar.ax.tick_params(labelsize=font_axes)
    plt.savefig('pictures/%02i_%02i/eigenfunction_p_%i.pdf' % (Nz, Nr, ivec + 1), bbox_inches='tight')


file_path = 'data/meta/%02i_%02i_%02i.pickle' % (Nz, Nr, config.get_grid_transformation_gradient_order())
with open(file_path, 'wb') as file:
    pickle.dump(eigenvalues, file)

plt.show()
