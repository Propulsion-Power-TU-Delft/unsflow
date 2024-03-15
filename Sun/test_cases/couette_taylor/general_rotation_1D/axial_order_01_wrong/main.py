#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 09:29:59 2023
@author: F. Neri, TU Delft
"""

import matplotlib.pyplot as plt
import numpy as np
import Sun
import scipy
from numpy import pi, sin, cos
import os
from Grid.src.config import Config
from Utils.styles import *
from Sun.src.general_functions import GaussLobattoPoints, ChebyshevDerivativeMatrixBayliss

# input data of the problem (SI units)
R1 = 1  # inner radius [m]
D = 0.001  # radial gap [m]
R2 = R1 + D  # outer radius [m]
P1 = 100e3  # pressure at hub [Pa]
T = 288  # temperature everywhere [K]
L = 3 * D  # length [m]
GMMA = 1.4  # cp/cv ratio of air
R = 287.05  # air gas constant [kj/kgK]
RHO1 = P1/R/T  # density everywhere [kg/m3]
MU = -1  # omega ratio = omega2/omega1
OMEGA2 = 10  # omega outer cylinder [rad/s]
OMEGA1 = OMEGA2 / MU  # omega inner cylinder [rad/s]
HARMONIC_ORDER = 0  # axisymmetric perturbations
plots = False

# non-dimensionalization terms:
x_ref = R1
omega_ref = OMEGA2
u_ref = omega_ref * x_ref
rho_ref = RHO1
t_ref = x_ref / u_ref
omega_ref = 1 / t_ref
rpm_ref = omega_ref * 60 / 2 / pi
p_ref = rho_ref * u_ref ** 2








# %%%%%%%%%%%%%%%%%%%%%%% ANALYTICAL PART %%%%%%%%%%%%%%%%%%%%%%%
LIMIT = 50
# radial_orders = np.linspace(1, LIMIT, LIMIT)
# axial_orders = np.linspace(1, 1, 1)
#
#
# class EigenFrequency:
#     """class who stores the information of a mode. For now consider only the positive frequencies"""
#
#     def __init__(self, radial_order, axial_order, eigenfrequency_pos, eigenfrequency_neg):
#         self.r_ord = radial_order
#         self.z_ord = axial_order
#         self.omega_pos = eigenfrequency_pos
#         self.omega_neg = eigenfrequency_neg
#
#
# eigs_an_list = []
# for m in radial_orders:
#     for alpha in axial_orders:
#         omega_plus = (2 * OMEGA1 * (pi * alpha * D / L) ** 2) / ((pi * alpha * D / L) ** 2 + m ** 2 * pi ** 2)
#         omega_minus = -(2 * OMEGA1 * (pi * alpha * D / L) ** 2) / ((pi * alpha * D / L) ** 2 + m ** 2 * pi ** 2)
#         eigs_an_list.append(EigenFrequency(m, alpha, omega_plus, omega_minus))
#
# plt.figure()
# for eig in eigs_an_list:
#     plt.scatter(eig.omega_pos.real, eig.omega_pos.imag, c='r', s=50)
#     plt.scatter(eig.omega_neg.real, eig.omega_neg.imag, c='b', s=50)
# plt.xlabel(r'$\omega_R \ \rm{[rad/s]}$')
# plt.ylabel(r'$\omega_I \ \rm{[rad/s]}$')
# plt.grid(alpha=grid_opacity)









# %%%%%%%%%%%%%%%%%%%%%%% COMPUTATIONAL PART %%%%%%%%%%%%%%%%%%%%%%%
# number of grid nodes in the computational domain
N = LIMIT*2
folder_path = "pictures/%02i" % (N)  # Replace with the desired folder path
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

config = Config('config.ini')
duct_obj = Sun.src.CouetteTaylor1D(R1, R2, N, config, mode='gauss-lobatto')
duct_obj.zeta = (duct_obj.r-R1)/D
duct_obj.ut = (R1 + duct_obj.zeta*D)*OMEGA1*(1-(1-MU)*duct_obj.zeta)
duct_obj.p = P1 + duct_obj.zeta*R1 + duct_obj.zeta**2*(D/2-R1*(1-MU)) + duct_obj.zeta**3*(R1*(1-MU)**2/3 -
                                                            2*(1-MU)*D/3) + duct_obj.zeta**4*D*(1-MU)**2/4
duct_obj.rho = duct_obj.p/R/T
duct_obj.dut_dr = OMEGA1*(-R1*(1-MU)+D-2*duct_obj.zeta*D*(1-MU))/D
duct_obj.dp_dr = duct_obj.rho*duct_obj.ut**2/duct_obj.r
duct_obj.drho_dr = duct_obj.dp_dr/R/T

if plots:
    plt.figure()
    plt.plot(duct_obj.r, duct_obj.zeta)
    plt.xlabel(r'$r \ \rm{[m]}$')
    plt.ylabel(r'$\zeta$')
    plt.grid(alpha=grid_opacity)
    plt.savefig('pictures/%02i/zeta.pdf' % N, bbox_inches='tight')

    plt.figure()
    plt.plot(duct_obj.r, duct_obj.ut)
    plt.xlabel(r'$r \ \rm{[m]}$')
    plt.ylabel(r'$u_{\theta} \ \rm{[m/s]}$')
    plt.grid(alpha=grid_opacity)
    plt.savefig('pictures/%02i/ut.pdf' % N, bbox_inches='tight')

    plt.figure()
    plt.plot(duct_obj.r, duct_obj.dut_dr)
    plt.xlabel(r'$r \ \rm{[m]}$')
    plt.ylabel(r'$du_{\theta}/dr \ \rm{[Pa/m]}$')
    plt.grid(alpha=grid_opacity)
    plt.savefig('pictures/%02i/dut_dr.pdf' % N, bbox_inches='tight')

    plt.figure()
    plt.plot(duct_obj.r, duct_obj.p)
    plt.xlabel(r'$r \ \rm{[m]}$')
    plt.ylabel(r'$p \ \rm{[Pa]}$')
    plt.grid(alpha=grid_opacity)
    plt.savefig('pictures/%02i/p.pdf' % N, bbox_inches='tight')

    plt.figure()
    plt.plot(duct_obj.r, duct_obj.dp_dr)
    plt.xlabel(r'$r \ \rm{[m]}$')
    plt.ylabel(r'$dp/dr \ \rm{[Pa/m]}$')
    plt.grid(alpha=grid_opacity)
    plt.savefig('pictures/%02i/dp_dr.pdf' % N, bbox_inches='tight')

    # plt.show()

duct_obj.normalize_data()
alpha_axial = 10
k_axial = pi*alpha_axial/(L/x_ref)
a_axial = k_axial*D/x_ref
sun_obj = Sun.src.SunModel1D(duct_obj, config=config)
sun_obj.L0 = np.zeros((N, N))
for i in range(N):
    sun_obj.L0[i] = a_axial**2 * (1-(1-MU)*duct_obj.zeta[i]/x_ref)
sun_obj.L1 = np.zeros_like(sun_obj.L0)
xi = GaussLobattoPoints(N)
D = ChebyshevDerivativeMatrixBayliss(xi)
D2 = D@D
sun_obj.L2 = -2*D2-a_axial**2
sun_obj.set_boundary_conditions()
sun_obj.apply_boundary_conditions_generalized()

Y1 = np.concatenate((-sun_obj.L0, np.zeros_like(sun_obj.L0)), axis=1)
Y2 = np.concatenate((np.zeros_like(sun_obj.L0), np.eye(sun_obj.L0.shape[0])), axis=1)
sun_obj.Y = np.concatenate((Y1, Y2), axis=0)  # Y matrix of EVP problem

P1 = np.concatenate((sun_obj.L1, sun_obj.L2), axis=1)
P2 = np.concatenate((np.eye(sun_obj.L0.shape[0]), np.zeros_like(sun_obj.L0)), axis=1)
sun_obj.P = np.concatenate((P1, P2), axis=0)  # P matrix of EVP problem

eigenvalues, eigenvectors = scipy.linalg.eig(a=sun_obj.Y, b=sun_obj.P)
eigenvalues *= omega_ref

# PLOT RESULTS
marker_size = 125
fig, ax = plt.subplots()
ax.scatter(eigenvalues.real, eigenvalues.imag, marker='o', facecolors='none', edgecolors='red', s=marker_size,
           label=r'numerical')
# for eig in eigs_an_list:
#     plt.scatter(eig.omega_pos.real, eig.omega_pos.imag, marker='x', c='blue', s=marker_size)
#     plt.scatter(eig.omega_neg.real, eig.omega_neg.imag, marker='x', c='blue', s=marker_size)
ax.set_xlabel(r'$\omega_{R}$ [rad/s]', fontsize=font_labels)
ax.set_ylabel(r'$\omega_{I}$ [rad/s]', fontsize=font_labels)
ax.set_xlim([-5, 5])
ax.set_ylim([-1, 1])
plt.xticks(fontsize=font_axes)
plt.yticks(fontsize=font_axes)
ax.legend(fontsize=font_legend)
ax.grid(alpha=grid_opacity)
fig.savefig('pictures/%02i/chi_map_arnoldi.pdf' % (N), bbox_inches='tight')
# plt.show()

# EIGENFUNCTIONS
r = sun_obj.data.r

for ivec in range(N//2):
    eigenvec = eigenvectors[:, ivec]
    ur_eig = eigenvec[0:len(eigenvec)//2]

    def scaled_eigenvector_real(eig_list):
        array = np.array(eig_list, dtype=complex)
        array_real_scaled = array.real / (np.max(array.real) - np.min(array.real))
        return array_real_scaled

    ur_eig_r = scaled_eigenvector_real(ur_eig)


    plt.figure()
    plt.plot(r, ur_eig_r, '-x', linewidth=0.8)
    plt.ylabel(r'$\tilde{u}_r$ [-]')
    plt.xlabel(r'$r$ [-]', fontsize=font_labels)
    plt.title(r'$\omega = (%.2f, %.2f j) \quad \rm{[rad/s]}$' % (eigenvalues[ivec].real,
                                                                    eigenvalues[ivec].imag), fontsize=font_title)
    plt.savefig('pictures/%02i/eigenfunction_ur_%i.pdf' % (N, ivec + 1), bbox_inches='tight')

#
#
# file_path = 'data/meta/%02i_%02i_%02i.pickle' % (Nz, Nr, config.get_grid_transformation_gradient_order())
# with open(file_path, 'wb') as file:
#     pickle.dump(eigenvalues, file)

# plt.show()
