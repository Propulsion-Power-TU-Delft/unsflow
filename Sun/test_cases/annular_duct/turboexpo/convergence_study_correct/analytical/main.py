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
from Utils.styles import *

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


import pickle
file_path = 'analytical_eigenvalues.pickle'
with open(file_path, 'wb') as file:
    pickle.dump(omega_analytical, file)


plt.figure()
plt.plot(lambda_span, det)
plt.plot(roots, roots*0, 'ro', label='zeros')
plt.xlim([0, lambda_span.max()])
plt.ylim([-0.2, 0.2])
plt.xlabel(r'$\lambda \ \mathrm{[m^{-1}]}$', fontsize=font_labels)
plt.ylabel(r'$\det \mathbf{Q}(\lambda)$', fontsize=font_labels)
plt.grid(alpha=grid_opacity)
plt.legend()
plt.savefig('lambda_roots.pdf', bbox_inches='tight')
# plt.show()