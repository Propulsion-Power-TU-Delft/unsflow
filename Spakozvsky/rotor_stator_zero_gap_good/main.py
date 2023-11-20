#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 12:21:40 2023

@author: fneri

@author: fneri
Exercise 5.4.3 of Spakovszky thesis
"""

import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.insert(1, '../src/')  # to add function folder
from Spakozvsky.src.functions import *
from Spakozvsky.src.axial_duct import AxialDuct
from Spakozvsky.src.axial_stator import AxialStator
import os

format_fig = (7, 7)

# create directory for pictures
path = "pics"
isExist = os.path.exists(path)
if not isExist:
    os.makedirs(path)

# %%INPUT DATA for the implementation of the generic actuator disk of Greitzer, zero gap
Vy1 = 0  # non dimensional background azimuthal flow velocity at outlet
DeltaX = 5

# from the inertia parameters we need to go back to axial spacing in order to recostruct the matrix

# rotor parameters (pag. 147)
beta1 = -71.1 * np.pi / 180  # relative inlet swirl
Vx1 = np.abs(1 / np.tan(beta1))
alfa1 = 0 * np.pi / 180  # absolute inlet swirl
beta2 = -35 * np.pi / 180  # relative outlet swirl
alfa2 = 65.7 * np.pi / 180  # absolute outlet swirl
dLr_dPhi = -0.6938  # steady state rotor loss derivative at background condition
dLr_dTanb = dLr_dPhi / ((np.tan(alfa1) - np.tan(beta1)) ** 2)  # steady state rotor loss derivative at background condition
c_r = 0.135  # blade chord
gamma_r = -50.2 * np.pi / 180  # stagger angle rotor blades
# lambda_r = 0.212 #inertia parameter rotor
lambda_r = c_r / np.cos(gamma_r) ** 2

# stator parameters (pag. 147)
beta3 = -35 * np.pi / 180  # relative inlet swirl
alfa3 = 65.7 * np.pi / 180  # absolute inlet swirl
beta4 = -71.1 * np.pi / 180  # relative outlet swirl
alfa4 = 0.0 * np.pi / 180  # absolute outlet swirl
dLs_dTana = 0.0411  # steady state stator loss at inlet condition of the stator
dLs_dphi = -dLs_dTana * ((np.tan(alfa3) - np.tan(beta3)) ** 2)
c_s = 0.121  # blade chord
gamma_s = 61.8 * np.pi / 180  # stagger angle rotor blades
lambda_s = c_s / np.cos(gamma_s)  # inertia parameter stator

# axial cordinates
x1 = 0
x2 = c_r * np.cos(gamma_r)
x3 = x2 + DeltaX
x4 = x3 + c_s * np.cos(gamma_s)

# velocities across the stages
Vx2 = Vx1
Vy2 = Vx2 * np.tan(alfa2)
Vx3 = Vx1
Vy3 = Vy2
Vx4 = Vx1
Vy4 = 0

# %% compute the results for each harmonic
LAMBDA = 1.08  # inertia parameter of the rotor row only
MU = 1.788  # inertia parameter of rotor+stator rows
tau_u = 1  # from thesis
tau_s = tau_u * c_s / (np.cos(gamma_s) * Vx1) * 0
tau_r = tau_u * c_r / (np.cos(gamma_r) * Vx1) * 0


inlet_duct = AxialDuct(Vy1, Vx1)
stator = AxialStator()


# system function from matrix stacking method
def rotor_stator(s, n, theta=0):
    m1 = np.linalg.inv(Tax_n(x4, s, n, Vx4, Vy4, theta=theta))
    m2 = Bsta_n(s, n, Vx3, Vy3, Vy4, alfa3, alfa4, lambda_s, dLs_dTana, theta=theta, tau_s=tau_s)
    m3 = Bgap_n(x2, x3, s, n, Vx3, Vy2, theta)
    m4 = Brot_n(s, n, Vx1, Vy1, Vy2, alfa1, beta1, beta2, lambda_r, dLr_dTanb, theta=theta, tau_r=tau_r)
    m5 = Tax_n(x1, s, n, Vx1, Vy1, theta=theta)
    m6 = m1 @ (m2 @ (m3 @ (m4 @ m5)))
    EC = np.array([[1, 0, 0]])
    IC = np.array([[0, 1, 0],
                   [0, 0, 1]])
    Y = np.concatenate((np.matmul(EC, m6), IC))
    return np.linalg.det(Y)


domain = [-0.15, 0.35, -4.5, 4.5, 5]
grid = [1, 1]
n = np.arange(1, 7)
plt.figure(figsize=format_fig)
poles_fneri = {}
poles_spak = {}
for nn in n:
    poles = Shot_Gun(rotor_stator, domain, grid, n=nn, attempts=30, tol=1e-6)
    poles_fneri[nn] = poles
    plt.plot(poles.real, -poles.imag, 'o', label='n:' + str(nn))
real_axis_x = np.linspace(domain[0], domain[1], 100)
real_axis_y = np.zeros(len(real_axis_x))
imag_axis_y = np.linspace(domain[2], domain[3], 100)
imag_axis_x = np.zeros(len(imag_axis_y))
plt.plot(real_axis_x, real_axis_y, '--k', linewidth=0.5)
plt.plot(imag_axis_x, imag_axis_y, '--k', linewidth=0.5)
plt.xlim([-0.15, 0.35])
plt.ylim([0, 4.5])
plt.legend()
plt.xlabel(r'$\sigma_{n}$')
plt.ylabel(r'$j \omega_{n}$')
plt.title('Root locus')
plt.grid(alpha=0.2)
plt.savefig(path + '/poles_rotor_stator_deltax_%.1f.pdf' %(DeltaX))

# domain = [-2.5, 0.5, -4.5, 4.5, 5]
# grid = [1, 1]
# n = np.arange(1, 7)
# plt.figure(figsize=format_fig)
# poles_fneri = {}
# poles_spak = {}
# for nn in n:
#     poles = Shot_Gun(rotor_stator, domain, grid, n=nn, attempts=30, tol=1e-6)
#     poles_fneri[nn] = poles
#     plt.plot(poles.real, -poles.imag, 'o', label='n:' + str(nn))
# real_axis_x = np.linspace(domain[0], domain[1], 100)
# real_axis_y = np.zeros(len(real_axis_x))
# imag_axis_y = np.linspace(domain[2], domain[3], 100)
# imag_axis_x = np.zeros(len(imag_axis_y))
# plt.plot(real_axis_x, real_axis_y, '--k', linewidth=0.5)
# plt.plot(imag_axis_x, imag_axis_y, '--k', linewidth=0.5)
# plt.xlim([-2.5, 0.5])
# plt.ylim([-0.5, 4.5])
# plt.legend()
# plt.xlabel(r'$\sigma_{n}$')
# plt.ylabel(r'$j \omega_{n}$')
# plt.title('Root locus')
# plt.grid(alpha=0.2)
# plt.savefig(path + '/poles_rotor_stator_lag.pdf')
plt.show()
