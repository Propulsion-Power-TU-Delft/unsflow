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

# input data
R2 = 1  # dimensional outer diameter
R1 = R2 / 100  # to simulate a cylinder, to avoid singularities at r=0
H = R2 * 10  # dimensional height of the cylinder
n_ax = np.linspace(1, 6, 6)  # axial mode numbers
a_list = n_ax * np.pi * R2 / H
m = 0  # investigate zero circumferential modes
omega = 0.5  # constant angular velocity of the rotating flow


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
    eigs = np.append(eig_plus, eig_minus)
    eigs_array[:, i] = eigs
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
duct_Obj = Sun.src.AnnulusMeridional(0, H, R1, R2, Nz, Nr, density, radialVel, tangentialVel, axialVel, pressure)
duct_grid = Sun.src.sun_grid.SunGrid(duct_Obj)
duct_grid.ShowGrid()

# general workflow of the sun model
sun_obj = Sun.src.SunModel(duct_grid)
sun_obj.ComputeBoundaryNormals()
sun_obj.AddNormalizationQuantities(rho_ref, u_ref, x_ref, 0)
sun_obj.NormalizeData()
sun_obj.ComputeSpectralGrid()
gradient_routine = 'findiff'
gradient_order = 2
sun_obj.ComputeJacobianPhysical(routine=gradient_routine, order=gradient_order)
sun_obj.AddAMatrixToNodesFrancesco2()
sun_obj.AddBMatrixToNodesFrancesco2()
sun_obj.AddCMatrixToNodesFrancesco2(m=0)
sun_obj.AddEMatrixToNodesFrancesco2()
sun_obj.AddRMatrixToNodesFrancesco2()
sun_obj.AddSMatrixToNodes()
sun_obj.AddHatMatricesToNodes()
sun_obj.ApplySpectralDifferentiation()
sun_obj.impose_boundary_conditions('zero pressure', 'zero pressure')

omega_domain = [-3, 3, -3, 3]
grid_omega = [100, 10]
sun_obj.ComputeSVD(omega_domain / omega_ref, grid_omega=grid_omega)
sun_obj.PlotInverseConditionNumber(save_filename='chi_map_%d_%d' %(Nz, Nr))

plt.show()
