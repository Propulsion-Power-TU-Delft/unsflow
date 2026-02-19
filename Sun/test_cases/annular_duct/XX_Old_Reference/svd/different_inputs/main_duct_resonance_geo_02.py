#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 09:29:59 2023
@author: F. Neri, TU Delft

annulus duct exercise, with different gemotrical quantities
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import jv, yv, jvp, yvp
import Sun
import scipy
from scipy.optimize import fsolve

# input data of the problem (SI units)
r1 = 0.6  # inner radius [m]
r2 = 0.65  # outer radius [m]
M = 0.015  # Mach number
p = 100e3  # pressure [Pa]
T = 288  # temperature [K]
L = 0.2  # length [m]
R = 287  # air gas constant [kJ/kgK]
gmma = 1.4  # cp/cv ratio of air
rho = p / (R * T)  # density [kg/m3]
a = np.sqrt(gmma * p / rho)  # ideal speed of sound [m/s]

# non-dimensionalization terms:
x_ref = r2
u_ref = M * a
rho_ref = rho
t_ref = x_ref / u_ref
omega_ref = 1 / t_ref
p_ref = rho_ref * u_ref ** 2

# %% ANALYTICAL PART OF THE PROBLEM
from scipy.optimize import fsolve

# radial cordinate array span
r = np.linspace(r1, r2, 300)
lambda_min = 1
lambda_max = 450
lambda_span = np.linspace(lambda_min, lambda_max, 500)  # we will do a loop for every possible value of lambda
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


def find_multiple_zeros(f, xmin, xmax, intervals=8):
    roots = []
    x = np.linspace(xmin, xmax, intervals)
    for i in range(0, intervals):
        root = fsolve(Bessel_determinant, x[i])
        roots.append(root)
    roots = np.array(roots)
    return np.unique(roots.round(decimals=2))


def compute_omega(alphas, lmbdas, M, L, a):
    print('{:<8s} {:<8s} {:<8s} {:<8s}'.format('m', 'lambda', 'alpha', 'omega'))
    omega_list = []

    for alpha in alphas:
        for lmbda in lmbdas:
            omega = a * np.sqrt(((1 - M ** 2) * alpha * np.pi / L) ** 2 + (1 - M ** 2) * lmbda ** 2)
            omega_list.append(omega)
            print('{:<8.0f} {:<8.1f} {:<8.0f} {:<8.0f}'.format(1, lmbda, alpha, omega))
    omega_list = np.array(omega_list)
    return np.sort(omega_list)


det = lambda_root(lambda_span)
zeros = np.zeros(len(det))
roots = find_multiple_zeros(lambda_root, lambda_min, lambda_max)
roots_y = np.zeros_like(roots)

# plot the determinant value
fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(lambda_span, det, label=r'$m=%d$' %(m))
ax.plot(roots, roots_y, 'ro', label='zeros')
ax.plot(lambda_span, zeros, '--k', lw=0.5)
ax.set_title(r'$\lambda$ roots')
ax.set_xlabel(r'$\lambda$')
ax.set_xlim([lambda_min, lambda_max])
ax.set_ylim([-0.2, 0.2])
ax.set_ylabel(r'$\det{\mathbf{Q}(\lambda)}$')
ax.legend()
fig.savefig('pictures/lambda_roots_geo2_m%d.pdf' %(m), bbox_inches='tight')

alpha = [1, 2, 3, 4, 5, 6, 7, 8]  # possible axial wavenumbers
omega_analytical = compute_omega(alpha, roots[0:8], M, L, a)
omega_analytical_zero = np.zeros_like(omega_analytical)








# %%COMPUTATIONAL PART
# number of grid nodes in the computational domain
Nz = 30
Nr = 10

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
duct_Obj = sun.src.AnnulusMeridional(0, L, r1, r2, Nz, Nr, density, radialVel, tangentialVel, axialVel, pressure)
duct_grid = sun.src.sun_grid.SunGrid(duct_Obj)
duct_grid.ShowGrid()

# general workflow of the sun model
sun_obj = sun.src.SunModel(duct_grid)
sun_obj.ComputeBoundaryNormals()
sun_obj.AddNormalizationQuantities(rho_ref, u_ref, x_ref, 0)
sun_obj.NormalizeData()
sun_obj.ComputeSpectralGrid()
gradient_routine = 'findiff'
gradient_order = 6
sun_obj.ComputeJacobianPhysical(routine=gradient_routine, order=gradient_order)
sun_obj.AddAMatrixToNodesFrancesco2()
sun_obj.AddBMatrixToNodesFrancesco2()
sun_obj.AddCMatrixToNodesFrancesco2(m=m)
sun_obj.AddEMatrixToNodesFrancesco2()
sun_obj.AddRMatrixToNodesFrancesco2()
sun_obj.AddSMatrixToNodes()
sun_obj.AddHatMatricesToNodes()
sun_obj.ApplySpectralDifferentiation()

omega_domain = [3e3, 23e3, -6e3, 6e3]
grid_omega = [150, 10]
sun_obj.ComputeSVD(omega_domain = omega_domain / omega_ref, grid_omega = grid_omega)
sun_obj.PlotInverseConditionNumber(ref_solution=omega_analytical, save_filename='chi_map_geo2_%s-%s_%d_%d_m%d'
                                                                                %(gradient_routine, gradient_order, Nz, Nr, m))

plt.show()
