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
from sun.src.sun_model_multiblock import SunModelMultiBlock
from grid.src.config import Config
import os
from utils.styles import *
import pickle


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






# Specify the file path from which you want to load the pickled object
file_path = "eigenfunction.pickle"

# Open the file in binary read mode and load the pickled object using pickle.load()
with open(file_path, 'rb') as obj:
    dict = pickle.load(obj)

r_grid = dict['r']
z_grid = dict['z']
p_eig_r = dict['p']
Nz = p_eig_r.shape[0]
Nr = p_eig_r.shape[1]

xtick_locations = [z_grid[0, 0], z_grid[-1, 0]]
xtick_labels = [r'$0$', r'$L$']
ytick_locations = [r_grid[0, 0], r_grid[0, -1]]
ytick_labels = [r'$r_1$', r'$r_2$']

plt.figure()
cnt = plt.contourf(z_grid, r_grid, p_eig_r, levels=N_levels_medium, cmap='RdBu')
for c in cnt.collections:
    c.set_edgecolor("face")
    c.set_linewidth(0.000000000001)
plt.xlabel(r'$z$ [-]', fontsize=font_labels)
plt.ylabel(r'$r$ [-]', fontsize=font_labels)
plt.xticks(ticks=xtick_locations, labels=xtick_labels, fontsize=font_axes)
plt.yticks(ticks=ytick_locations, labels=ytick_labels, fontsize=font_axes)
plt.title(r'$\tilde{p}$ [-]', fontsize=font_title)
cnbar = plt.colorbar(cnt)
cnbar.set_ticks(np.linspace(-0.5, 0.5, 9))
cnbar.ax.tick_params(labelsize=font_axes)
plt.savefig('eigenfunction_p_2D_%02i_%02i.pdf' %(z_grid.shape[0], z_grid.shape[1]), bbox_inches='tight')

z_grid_refined = np.linspace(z_grid[0,0], z_grid[-1, 0], 200)
# # second axial order
plt.figure()
plt.plot(z_grid_refined, np.max(p_eig_r[:, 0]) * np.sin(2 * np.pi * z_grid_refined / L * r1), label='analytical',
         lw=medium_line_width)
plt.plot(z_grid[:, 0], -p_eig_r[:, 0], 'ro', label='numerical', lw=medium_line_width, markerfacecolor='none',
         markeredgewidth=1.5, markersize=8)
plt.ylabel(r'$\tilde{p}$ [-]', fontsize=font_labels)
plt.xlabel(r'$z$ [-]', fontsize=font_labels)
plt.xticks(fontsize=font_axes)
plt.yticks(fontsize=font_axes)
plt.grid(alpha=grid_opacity)
plt.legend(fontsize=font_legend)
plt.savefig('eigenfunction_r_%02i_%02i.pdf' %(z_grid.shape[0], z_grid.shape[1]), bbox_inches='tight')




# RADIAL CUT
LAMBDA = roots[1]
ALPHA = jvp(m, LAMBDA*r1, n=1) / yvp(m, LAMBDA*r1, n=1)
ALPHA2 = jvp(m, LAMBDA*r2, n=1) / yvp(m, LAMBDA*r2, n=1)
r_var = np.linspace(r1, r2, 200)
eigen_analyt_r = jv(m, LAMBDA*r_var) - ALPHA*yv(m, LAMBDA*r_var)
eigen_analyt_r_scaled = eigen_analyt_r / (np.max(eigen_analyt_r) - np.min(eigen_analyt_r))


plt.figure()
# if opposite signs
plt.plot(r_var/r1, eigen_analyt_r_scaled, label='analytical', lw=medium_line_width)
plt.plot(r_grid[Nz // 2, :], (p_eig_r[Nz // 2, :]) / (np.max(p_eig_r[Nz // 2, :]) - np.min(p_eig_r[Nz // 2, :])),
         'ro', label='numerical', lw=medium_line_width, markerfacecolor='none', markeredgewidth=1.5,
         markersize=8)
#if same signs
# plt.plot(r_grid[Nz // 2, :], np.abs(p_eig_r[Nz // 2, :])/np.max(np.abs(p_eig_r[Nz // 2, :])), '--o', label='numerical')
# plt.plot(r_var/r1, np.abs(eigen_analyt_r_scaled)/np.max(np.abs(eigen_analyt_r_scaled)), label='analytical')
plt.ylabel(r'$\tilde{p}$ [-]', fontsize=font_labels)
plt.xlabel(r'$r$ [-]', fontsize=font_labels)
plt.xticks(fontsize=font_axes)
plt.yticks(fontsize=font_axes)
plt.grid(alpha=grid_opacity)
# plt.title()
plt.legend(fontsize=font_legend)
plt.savefig('eigenfunction_z_%02i_%02i.pdf' %(z_grid.shape[0], z_grid.shape[1]), bbox_inches='tight')
plt.show()




plt.show()