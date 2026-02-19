#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 16:17:31 2022

@author: fn
3DOF linearized greitzer
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.optimize import fsolve
from utils.styles import *

# parameters set
N = 20e3  # [rpm] --> large N enlarges the cirle
r_ref = 100e-3  # radius [m]
omega = 2 * np.pi * N / 60  # [rad/s]
U_ref = omega * r_ref  # reference speed of compressor[m/s]
a = 340  # speed of sound [m/s]
Vp = 0.0002  # plenum volume stable [m3]
# Vp = 0.0005                                 # plenum volume unstable [m3]
dc = 10e-3  # inlet diameter [m]
Ac = (np.pi * dc ** 2) / 4  # inlet area [m2]
Lc = 0.1  # inlet length [m]
dt = 10e-3  # throttle diameter [m]
At = (np.pi * dt ** 2) / 4  # throttle area [m2]
Lt = 0.1  # throttle length [m]
B_real = 0.3 * (U_ref / (2 * a)) * np.sqrt(Vp / (Ac * Lc))  # B parameter of the described compressor
G_real = Lt * Ac / (Lc * At)  # G parameter of the described compressor
k_valve = 3.2  # 2.85 is unstable, 2.8 is stable. depends of course on parameters

# compressor working points
phi_p = np.array([0.2, 0.4, 0.5, 0.6, 0.8])
psi_p = np.array([0.5, 0.8, 0.95, 1.0, 0.8])

# polynomial interpolation for the compressor curve
z_coeff = np.polyfit(phi_p, psi_p, 3)
phi = np.linspace(0, 1, 100)
psi_c = np.polyval(z_coeff, phi)

# throttle valve curve
psi_v = k_valve * phi ** 2


# find the intersection flow coefficient between compressor and throttle
def func_work_coefficient(phi):
    return np.polyval(z_coeff, phi) - k_valve * phi ** 2


initial_phi_guess = 0.9
phi_eq = fsolve(func_work_coefficient, initial_phi_guess)
psi_eq = k_valve * phi_eq ** 2

# calculate derivatives of the compressor and throttle characteristic at equilibrium point
delta_phi = phi_eq * 0.001
phi_left = phi_eq - delta_phi
phi_right = phi_eq + delta_phi
psi_c_right = np.polyval(z_coeff, phi_right)
psi_c_left = np.polyval(z_coeff, phi_left)
psi_c_prime = (psi_c_right - psi_c_left) / (2 * delta_phi)
psi_v_prime = 2 * k_valve * phi_eq


def Greitzer(y, t, B, G, k_valve, closure_coeff):
    """
    Defines the differential equations for the Greitzer model
    (as found in the thesis of Sündstrom).

    Arguments:
        y :  vector of the state variables
        t :  time
        B :  B parameter
        G :  G parameter
        
    State variables:
        x1 : compressor flow coefficient disturbance
        x2 : throttle flow coefficient disturbance
        x3 : plenum work coefficient disturbance
    """
    x1, x2, x3 = y
    dydt = [B * (psi_c_prime * x1 - x3), (-psi_v_prime * x2 + x3) * B / G, (x1 - x2) / B]
    return dydt


t = np.linspace(0, 100, 5000)

# initial conditions
y0 = [-0.01 * phi_eq[0], 0.01 * phi_eq[0], -0.005 * psi_eq[0]]

# initial perturbations
IC_flow_c = [y0[0]]
IC_flow_t = [y0[1]]
IC_psi = [y0[2]]

# initial absolute values
IC_flow_c_abs = [phi_eq[0] + y0[0]]
IC_flow_t_abs = [phi_eq[0] + y0[1]]
IC_psi_abs = [psi_eq[0] + y0[2]]

from scipy.integrate import odeint

closure_coeff = 1.0
sol = odeint(Greitzer, y0, t, args=(B_real, G_real, k_valve, closure_coeff))
initialDerivative = Greitzer(y0, t[0], B_real, G_real, k_valve, closure_coeff)

path = "pics"
# Check whether the specified path exists or not
isExist = os.path.exists(path)
if not isExist:
    os.makedirs(path)

# temporal evolution
fig, axes = plt.subplots(3, 1)
axes[0].set_ylabel(r'$\widetilde{\Phi_{c}}$')
axes[0].plot(t, sol[:, 0])
axes[1].set_ylabel(r'$\widetilde{\Phi_{t}}$')
axes[1].plot(t, sol[:, 1])
axes[2].set_ylabel(r'$\widetilde{\Psi}$')
axes[2].plot(t, sol[:, 2])
axes[2].set_xlabel(r'$\xi $')
fig.suptitle('perturbation transient')
# fig.savefig(path + '/perturbation_transient.pdf', bbox_inches='tight')

# reconstruct absolute values from equilibrium plus perturbation:
sol_abs = np.zeros(sol.shape)
sol_abs[:, 0] = sol[:, 0] + phi_eq
sol_abs[:, 1] = sol[:, 1] + phi_eq
sol_abs[:, 2] = sol[:, 2] + psi_eq

# plots in the phase space
fig, axes = plt.subplots(1, 3)
axes[0].set_xlabel(r'$\widetilde{\Phi_{c}}$')
axes[0].set_ylabel(r'$\widetilde{\Phi_{t}}$')
axes[0].plot(sol[:, 0], sol[:, 1], linewidth=0.8)
axes[0].plot(IC_flow_c, IC_flow_t, marker='o', color='b', label='IC')
axes[1].set_ylabel(r'$\widetilde{\Psi}$')
axes[1].set_xlabel(r'$\widetilde{\Phi_{c}}$')
axes[1].plot(sol[:, 0], sol[:, 2], linewidth=0.8)
axes[1].plot(IC_flow_c, IC_psi, marker='o', color='b', label='IC')
axes[2].set_ylabel(r'$\widetilde{\Psi}$')
axes[2].set_xlabel(r'$\widetilde{\Phi_{t}}$')
axes[2].plot(sol[:, 1], sol[:, 2], linewidth=0.8)
axes[2].plot(IC_flow_t, IC_psi, marker='o', color='b', label='IC')
fig.suptitle('phase space trajectory')
# fig.savefig(path + '/phase_space_perturbations.pdf', bbox_inches='tight')

# plot the compressor curve
plt.figure()
plt.plot(phi, psi_c, linewidth=1.5, label='compressor curve')
plt.plot(phi, psi_v, '--', linewidth=1.5, label='throttle curve')
plt.scatter(phi_eq, psi_eq, color='red', s=50, label='equilibrium point')
plt.ylabel(r'$\Psi_c$')
plt.xlabel(r'$\Phi_c$')
plt.legend()
plt.ylim(0, 1.25)
# plt.title('characteristi')
# plt.savefig(path + '/characteristic_curves.pdf', bbox_inches='tight')

# plot the compressor curve
plt.figure()
plt.plot(phi, psi_c, linewidth=0.6, label='Compressor line')
plt.plot(phi, psi_v, linewidth=0.6, label='Throttle line')
plt.plot(sol_abs[:, 0], sol_abs[:, 2], '--k', linewidth=1, label='Transient')
plt.plot(IC_flow_c_abs, IC_psi_abs, 'ko', label='Initial condition')
plt.ylabel(r'$\Psi$')
plt.xlabel(r'$\Phi_c$')
plt.legend()
plt.ylim(0, 2)
plt.title('characteristic transient')
# plt.savefig(path + '/characteristic_transient.pdf', bbox_inches='tight')

# %% 3D trajectory in phase space

fig = plt.figure()
ax = plt.axes(projection='3d')

# Data for a three-dimensional line
xline = sol[:, 0]  # compressor flow coeff
yline = sol[:, 1]  # throttle flow coeff
zline = sol[:, 2]  # work coeff.
ax.plot3D(xline, yline, zline, linewidth=1)
ax.set_title('3D trajectory')
ax.set_xlabel(r'$\Phi_{c}$')
ax.set_ylabel(r'$\Phi_{t}$')
ax.set_zlabel(r'$\Psi$')
# fig.savefig(path + '/3D_phase_trajectory.pdf', bbox_inches='tight')

# %%
grid_num = 100
B_min = 0.001
B_max = 1.25
G_min = 0.001
G_max = 3
B = np.linspace(B_min, B_max, grid_num)
G = np.linspace(G_min, G_max, grid_num)
B_grid, G_grid = np.meshgrid(B, G)
stability = np.zeros((grid_num, grid_num))

for i in range(len(B)):
    for j in range(len(G)):
        B_r = B_grid[i, j]
        G_r = G_grid[i, j]
        coeffs = [-1, B_r * psi_c_prime - B_r * psi_v_prime / G_r, (psi_c_prime * psi_v_prime * B_r ** 2) / G_r - 1 / G_r - 1,
                  (B_r / G_r) * (psi_c_prime - psi_v_prime)]

        # Find the roots of the polynomial
        roots = np.roots(coeffs)
        roots_real = roots.real

        if (roots_real[0] >= 0 or roots_real[1] >= 0 or roots_real[2] >= 0):
            stability[i, j] = 1.0  # unstable
        else:
            stability[i, j] = -1.0  # stable

plt.figure()
plt.contourf(B_grid, G_grid, stability, cmap='bwr')
# plt.plot(B_real, G_real, 'ow')
plt.xlabel(r'$B$')
plt.ylabel(r'$G$')
# plt.colorbar()
plt.savefig(path + '/stability_map.pdf', bbox_inches='tight')
plt.show()
