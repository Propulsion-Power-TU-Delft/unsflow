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

# input data of the problem (SI units)
r1 = 0.1826  # inner radius [m]
r2 = 0.2487  # outer radius [m]
M = 0.1  # Mach number
p = 100e4  # pressure [Pa]
T = 420  # temperature [K]
L = 1  # length [m]
R = 287.058  # air gas constant [kJ/kgK]
gmma = 1.4  # cp/cv ratio of air
rho = p / (R * T)  # density [kg/m3]
a = np.sqrt(gmma * p / rho)  # ideal speed of sound [m/s]

# non-dimensionalization terms:
x_ref = r1
# x_ref =1
u_ref = M * a
# u_ref = 1
rho_ref = rho
# rho_ref = 1
t_ref = x_ref / u_ref
# t_ref = 1
omega_ref = 1 / t_ref
p_ref = rho_ref * u_ref ** 2

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
roots = find_multiple_zeros(lambda_root, 1, 300)
roots_y = np.zeros_like(roots)

# plot the determinant value
fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(lambda_span, det, label=r'$m=%d$' %(m))
ax.plot(roots, roots_y, 'ro', label='zeros')
ax.plot(lambda_span, zeros, '--k', lw=0.5)
ax.set_title(r'$\lambda$ roots')
ax.set_xlabel(r'$\lambda$')
ax.set_xlim([0, 300])
ax.set_ylim([-0.25, 0.25])
ax.set_ylabel(r'$\det{\mathbf{Q}(\lambda)}$')
ax.legend()
fig.savefig('pictures/lambda_roots_m%d.pdf' %(m), bbox_inches='tight')

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

omega_domain = [1e3, 8e3, -1e3, 1e3]
grid_omega = [150, 10]
sun_obj.ComputeSVD(omega_domain = omega_domain/omega_ref, grid_omega = grid_omega)
sun_obj.PlotInverseConditionNumber(ref_solution=omega_analytical, save_filename='chi_map_set05_ur_%s-%s_%d_%d_m%d'
                                                                                %(gradient_routine, gradient_order, Nz, Nr, m))

plt.show()
