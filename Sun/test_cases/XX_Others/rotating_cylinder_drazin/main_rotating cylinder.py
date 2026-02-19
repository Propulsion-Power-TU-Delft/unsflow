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


def eigenvalue_function_plus(alpha, m, k, r0):
    a = k * r0
    func = alpha * jvp(m, alpha, n=1) + m * (1 + alpha ** 2 / a ** 2) ** (1 / 2) * jvp(m, alpha, n=0)
    return func


def eigenvalue_function_minus(alpha, m, k, r0):
    a = k * r0
    func = alpha * jvp(m, alpha, n=1) - m * (1 + alpha ** 2 / a ** 2) ** (1 / 2) * jvp(m, alpha, n=0)
    return func


alpha = np.linspace(0, 25, 1000)
m = 1
k = 1
r0 = 1
a = k*r0
eig_fun_plus = eigenvalue_function_plus(alpha, m, k, r0)
eig_fun_minus = eigenvalue_function_minus(alpha, m, k, r0)


def find_multiple_zeros(f, xmin, xmax, m, k, r0, intervals=15):
    roots = []
    x = np.linspace(xmin, xmax, intervals)
    for i in range(0, intervals):
        root = fsolve(f, x[i], args=(m, k, r0))
        roots.append(root)
    roots = np.array(roots)
    return np.unique(roots.round(decimals=2))


roots_plus = find_multiple_zeros(eigenvalue_function_plus, alpha[0], alpha[-1], m, k, r0)
roots_plus_y = np.zeros_like(roots_plus)
roots_minus = find_multiple_zeros(eigenvalue_function_minus, alpha[0], alpha[-1], m, k, r0)
roots_minus_y = np.zeros_like(roots_minus)
roots = np.append(roots_minus, roots_plus)
roots = np.sort(np.unique(roots))

fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(alpha, eig_fun_plus, 'blue', label=r'$f_1(\alpha)$')
ax.plot(alpha, eig_fun_minus, 'green', label=r'$f_2(\alpha)$')
ax.plot(roots_plus, roots_plus_y, 'bo', label=r'roots $f_1$')
ax.plot(roots_minus, roots_minus_y, 'go', label=r'roots $f_2$')
ax.set_title(r'$\alpha$ roots')
ax.set_xlabel(r'$\alpha$')
ax.legend()
fig.savefig('pictures/alpha_roots_%d_%d_%d.pdf' % (m, k, r0), bbox_inches='tight')


def eigenvalues(alpha, m, a):
    # return s/omega_0
    s_plus = 2j/(1 + alpha**2 / a**2)**(0.5) - 1j*m
    s_minus = - 2j/(1 + alpha**2 / a**2)**(0.5) - 1j*m
    return s_plus, s_minus

eig_plus, eig_minus = eigenvalues(roots, m, a)
eigs_analytical = np.append(eig_plus, eig_minus)


fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(eig_plus.imag, eig_plus.real, 'bo', label=r'$plus$')
ax.plot(eig_minus.imag, eig_minus.real, 'go', label=r'$minus$')
ax.set_title('eigenvalues')
ax.set_ylabel(r'$\mathcal{Re}(\frac{s}{\Omega_0})$')
ax.set_xlabel(r'$\mathcal{Im}(\frac{s}{\Omega_0})$')
ax.legend()
fig.savefig('pictures/eigs_%d_%d_%d.pdf' % (m, k, r0), bbox_inches='tight')


# now solve with sun model
# %%COMPUTATIONAL PART
# number of grid nodes in the computational domain
Nz = 15
Nr = 5

rho = 1
ur = 0
uz = 0
p = 1e5

r1 = r0/100
r2 = r0
L = 10*r2

# implement a constant uniform flow in the annulus duct
density = np.zeros((Nz, Nr))
axialVel = np.zeros((Nz, Nr))
radialVel = np.zeros((Nz, Nr))
tangentialVel = np.zeros((Nz, Nr))
pressure = np.zeros((Nz, Nr))
for ii in range(0, Nz):
    for jj in range(0, Nr):
        density[ii, jj] = rho
        tangentialVel[ii, jj] = r1 + jj * r2 / (Nr-1)
        pressure[ii, jj] = p

# non-dimensionalization terms:
x_ref = r0
u_ref = tangentialVel[0, -1]
rho_ref = rho
t_ref = x_ref / u_ref
omega_ref = 1 / t_ref
p_ref = rho_ref * u_ref ** 2

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
gradient_order = 2
sun_obj.ComputeJacobianPhysical(routine=gradient_routine, order=gradient_order)
sun_obj.AddAMatrixToNodesFrancesco2()
sun_obj.AddBMatrixToNodesFrancesco2()
sun_obj.AddCMatrixToNodesFrancesco2(m=m)
sun_obj.AddEMatrixToNodesFrancesco2()
sun_obj.AddRMatrixToNodesFrancesco2()
sun_obj.AddSMatrixToNodes()
sun_obj.AddHatMatricesToNodes()
sun_obj.ApplySpectralDifferentiation()
sun_obj.impose_boundary_conditions('zero pressure', 'zero pressure')

omega_domain = [-0.2, 0.2, -4, 2]
grid_omega = [5, 100]
sun_obj.ComputeSVD(omega_domain / omega_ref, grid_omega = grid_omega)
sun_obj.PlotInverseConditionNumber(ref_solution=eigs_analytical, save_filename='chi_map_%s-%s_%d_%d_m%d'
                                                                                %(gradient_routine, gradient_order, Nz, Nr, m))

plt.show()

