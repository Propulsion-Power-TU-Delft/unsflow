#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 14:11:33 2023

@author: fneri
Exercise 5.4.3 of Spakovszky thesis
"""

import matplotlib.pyplot as plt
import numpy as np
import Spakozvsky
from Spakozvsky.src.functions import *
import os
import time

# Preamble: customization of matplotlib
# Configuration for plots
plt.rc('text', usetex=False)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rcParams['font.size'] = 14
format_fig = (9, 5)

# create directory for pictures
path = "pics"
isExist = os.path.exists(path)
if not isExist:
    os.makedirs(path)
poles_fneri = {}

Vx1 = 0.34  # non-dimensional background axial flow velocity at inlet
Vy1 = 0  # non-dimensional background azimuthal flow velocity at outlet
DeltaX = 0.3  # gap between rotor and stator

# rotor parameters (pag. 147)
beta1 = -71.1 * np.pi / 180  # relative inlet swirl
alfa1 = 0 * np.pi / 180  # absolute inlet swirl
beta2 = -35 * np.pi / 180  # relative outlet swirl
alfa2 = 65.7 * np.pi / 180  # absolute outlet swirl
dLr_dPhi = -0.6938  # steady state rotor loss derivative at background condition
dLr_dTanb = dLr_dPhi / ((np.tan(alfa1) - np.tan(beta1)) ** 2)  # steady state rotor loss derivative at background condition
c_r = 0.135  # blade chord
gamma_r = -50.2 * np.pi / 180  # stagger angle rotor blades
lambda_r = 0.212  # inertia parameter rotor

# stator parameters (pag. 147)
beta3 = -35 * np.pi / 180  # relative inlet swirl
alfa3 = 65.7 * np.pi / 180  # absolute inlet swirl
beta4 = -71.1 * np.pi / 180  # relative outlet swirl
alfa4 = 0.0 * np.pi / 180  # absolute outlet swirl
dLs_dTana = 0.0411  # steady state stator loss at inlet condition of the stator
c_s = 0.121  # blade chord
gamma_s = 61.8 * np.pi / 180  # stagger angle rotor blades
lambda_s = 0.256  # inertia parameter rotor

# axial cordinates
x1 = 0
x2 = c_r * np.cos(gamma_r)
x3 = DeltaX
x4 = x3 + c_s * np.cos(gamma_s)

# velocities across the stages
Vx2 = Vx1
Vy2 = Vx2 * np.tan(alfa2)
Vx3 = Vx1
Vy3 = Vy2
Vx4 = Vx1
Vy4 = 0


# %% compute the results for each harmonic

# system function
def rotor_stator(s, n, theta=0):
    m1 = np.linalg.inv(Tax_n(x4, s, n, Vx4, Vy4, theta=theta))
    m2 = Bsta_n(s, n, Vx3, Vy3, Vy4, alfa3, alfa4, lambda_s, dLs_dTana, theta, 0)
    m3 = Bgap_n(x2, x3, s, n, Vx2, Vy2, theta)
    m4 = Brot_n(s, n, Vx1, Vy1, Vy2, alfa1, beta1, beta2, lambda_r, dLr_dTanb, theta, 0)
    m5 = Tax_n(x1, s, n, Vx1, Vy1, theta=theta)
    m6 = np.linalg.multi_dot([m1, m2, m3, m4, m5])
    EC = np.array([[1, 0, 0]])
    IC = np.array([[0, 1, 0],
                   [0, 0, 1]])
    Y = np.concatenate((np.matmul(EC, m6), IC))
    return np.linalg.det(Y)

def rotor_stator_matrix(s, n, theta=0):
    m1 = np.linalg.inv(Tax_n(x4, s, n, Vx4, Vy4, theta=theta))
    m2 = Bsta_n(s, n, Vx3, Vy3, Vy4, alfa3, alfa4, lambda_s, dLs_dTana, theta, 0)
    m3 = Bgap_n(x2, x3, s, n, Vx2, Vy2, theta)
    m4 = Brot_n(s, n, Vx1, Vy1, Vy2, alfa1, beta1, beta2, lambda_r, dLr_dTanb, theta, 0)
    m5 = Tax_n(x1, s, n, Vx1, Vy1, theta=theta)
    m6 = np.linalg.multi_dot([m1, m2, m3, m4, m5])
    EC = np.array([[1, 0, 0]])
    IC = np.array([[0, 1, 0],
                   [0, 0, 1]])
    Y = np.concatenate((np.matmul(EC, m6), IC))
    return Y


domain = [-2.5, 0.5, -0.5, 4.5]
grid = [5, 5]
# n = np.arange(1, 2)
# plt.figure(figsize=(16, 9))
# for nn in n:
#     poles = Shot_Gun(rotor_stator, domain, grid, n=nn, attempts=30)
#     poles_fneri[nn] = poles
#     plt.plot(poles.real, -poles.imag, 'o', label='n ' + str(nn))
# real_axis_x = np.linspace(domain[0], domain[1], 100)
# real_axis_y = np.zeros(len(real_axis_x))
# imag_axis_y = np.linspace(domain[2], domain[3], 100)
# imag_axis_x = np.zeros(len(imag_axis_y))
# plt.plot(real_axis_x, real_axis_y, '--k', linewidth=0.5)
# plt.plot(imag_axis_x, imag_axis_y, '--k', linewidth=0.5)
# plt.xlim([domain[0], domain[1]])
# plt.ylim([domain[2], domain[3]])
# plt.legend()
# plt.xlabel(r'$\sigma_{n}$')
# plt.ylabel(r'$j \omega_{n}$')
# plt.title('Root locus')
# plt.savefig(path+'/poles_rotor_stator_deltax_03.png')
# plt.show()

omR_min = domain[0]
omR_max = domain[1]
omI_min = domain[2]
omI_max = domain[3]
nR = 200
nI = 200
omR = np.linspace(omR_min, omR_max, nR)
omI = np.linspace(omI_min, omI_max, nI)
omegaI, omegaR = np.meshgrid(omI, omR)
chi = np.zeros((nR, nI))
sing_value_min = np.zeros((nR, nI))
sing_value_max = np.zeros((nR, nI))
start_time = time.time()

for ii in range(0, nR):
    for jj in range(0, nI):

        # this block is simply to print info, and time remaining for the full process
        current_time = time.time() - start_time
        if (ii == 0 and jj == 0):
            print('SVD %.1d of %1.d ..' % (ii * len(omI) + 1 + jj, len(omR) * len(omI)))
        if (jj == 1):  # update time whenever jj=1
            delta_time_svd = current_time
            total_time = (delta_time_svd / (
                    ii * nI + jj)) * nR * nI  # (time passed / number of SVD done) * number of total SVD to do
        if (ii != 0 or jj != 0):
            remaining_minutes = (total_time - current_time) / 60
            total_minutes = total_time / 60
            print('SVD %.1d of %1.d \t (%.1d min remaining)' % (
                ii * len(omI) + 1 + jj, len(omR) * len(omI), remaining_minutes + 1))  # keep track of the progress

        omega = omR[ii] - 1j * omI[jj]
        Y = rotor_stator_matrix(omega, 6, 0)
        u, s, v = np.linalg.svd(Y)
        sing_value_min[ii, jj] = np.min(s)
        sing_value_max[ii, jj] = np.max(s)
        chi[ii, jj] = np.min(s) / np.max(s)

end_time = time.time() - start_time
hrs = int(end_time / 3600)
mins = int((end_time - hrs * 3600) / 60)
sec = int(end_time - hrs * 3600 - mins * 60)
print('Total SVD time: \t %1.d hrs %1.d mins %1.d sec' % (hrs, mins, sec))

fig, ax = plt.subplots(figsize=format_fig)
cs = ax.contourf(omegaR, omegaI, chi, 100, cmap='jet')
ax.set_xlabel(r'$RS \ \mathrm{[-]}$')
ax.set_ylabel(r'$DF \ \mathrm{[-]}$')
ax.set_title(r'$\chi$')
cbar = fig.colorbar(cs)
plt.show()