#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 16:58:47 2023

@author: fneri
exercise taken from 2.5.2 of Spakovszky model
"""

import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.insert(1, '../src/')  # to add function folder
from functions import *

# Preamble: customization of matplotlib
# Configuration for plots
plt.rc('text', usetex=False)
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rcParams['font.size'] = 10
format_fig = (10, 8)

# %%DATA INPUT OF THE EXERCISE
R2 = 1  # initial radius
R3 = 3  # outlet radius
Nbl = 4  # number of blades
Q = 0.215  # source term
GAMMA = 0.7032  # circulation term
# GAMMA = 0 #circulation term
theta_max = 2 * np.pi / Nbl  # period of the perturbation
azimuthal_sampling_points = 100  # number of points along the period
theta = np.linspace(0, theta_max, azimuthal_sampling_points)  # theta domain
theta_deg = theta * 180 / np.pi  # theta domain in degrees
n = Nbl  # interested in the 20th harmonic, since we are looking at perturbation coming from the 20th harmonics
theta0 = theta[0]
omega = -Nbl

# INITIAL CONDITIONS
Wr_2 = np.exp(1j * n * theta0)
Wt_2 = np.exp(1j * n * theta0) * (GAMMA - 1) / Q

# radial position where we want the data
# radii = np.array([R2,1.1, 1.2, 1.3, 1.4, R3])
radii = np.array([R2, 1.05, 1.1, 1.15, 1.2, R3])
# radii = np.array([R2,1.002,1.004,1.006,1.008, 1.01])

r0 = R2  # this is the big problem. no reason for it. there is no way to understand what is its meaning
# boundary conditions are prescribed inlet velocities, and outlet pressure at R3 = 0
T2 = Trad_n(R2, R2, n, 1j * omega, Q, GAMMA, 0)
T3 = Trad_n(R3, R2, n, 1j * omega, Q, GAMMA, 0)
Y = np.zeros((3, 3), dtype=complex)
Y[0, :] = T2[0, :]
Y[1, :] = T2[1, :]
Y[2, :] = T3[2, :]
BC_vec = np.zeros((3, 1), dtype=complex)
BC_vec[0] = np.exp(1j * n * (theta0))
BC_vec[1] = ((GAMMA - 1) / Q) * np.exp(1j * n * (theta0))
BC_vec[2] = 0
# find the potential and vortical modes in the system that satisfy the BC
DEN_mode = np.matmul(np.linalg.inv(Y), BC_vec)
Wr0 = np.sqrt(1 + ((GAMMA - 1) / Q) ** 2)
fig, axes = plt.subplots(2, 2, figsize=format_fig)
axes[0, 0].set_ylabel(r'$\delta w_{r}$')
axes[1, 0].set_ylabel(r'$\delta w_{\theta}$')
axes[0, 1].set_ylabel(r'$\delta p_t $')
axes[1, 1].set_ylabel(r'$\delta p $')
axes[1, 0].set_xlabel(r'$\theta $')
axes[1, 1].set_xlabel(r'$\theta $')

for k in range(0, len(radii)):
    radius = radii[k]
    # compute now the flow solutions
    vec = np.zeros((3, len(theta)), dtype=complex)
    i = 0
    for t in theta:
        # shift of period/4 in order to match the plots presented in the thesis
        vec[:, i] = np.matmul(Trad_n(radius, r0, n, -1j * n, Q, GAMMA, theta=t + 1 / 4 * theta_max), DEN_mode).reshape(3)
        i = i + 1
    axes[0, 0].plot(theta_deg, vec[0, :], label='r=' + str(radius))
    axes[1, 0].plot(theta_deg, vec[1, :])
    axes[1, 1].plot(theta_deg, vec[2, :])
    axes[0, 1].plot(theta_deg, vec[2, :] + 0.5 * (vec[1, :] ** 2 + vec[0, :] ** 2))

fig.legend()

# %% VARIATIONS WITH RADIUS
radii = np.linspace(R2, R3, 3000)
fig, axes = plt.subplots(3, 1, figsize=format_fig)
axes[0].set_ylabel(r'$\delta w_{r}$')
axes[1].set_ylabel(r'$\delta w_{\theta}$')
axes[2].set_ylabel(r'$\delta p $')
axes[2].set_xlabel(r'$r $')
vec_rad = np.zeros((3, len(radii)), dtype=complex)
for k in range(0, len(radii)):
    vec_rad[:, k] = np.matmul(Trad_n(radii[k], r0, n, 1j * omega, Q, GAMMA, 0), DEN_mode).reshape(3)
axes[0].plot(radii, vec_rad[0, :].real)
axes[0].plot(radii, vec_rad[0, :].imag)
axes[1].plot(radii, vec_rad[1, :].real)
axes[1].plot(radii, vec_rad[1, :].imag)
axes[2].plot(radii, vec_rad[2, :].real)
axes[2].plot(radii, vec_rad[2, :].imag)

# plot of the phasors of the perturbation
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Perturbation Phasors')
axes[0].set_title(r'$\dot{\delta W_r}$')
axes[0].set_xlabel('Real')
axes[0].set_ylabel('Im')
axes[1].set_title(r'$\dot{\delta W_{\theta}}$')
axes[1].set_xlabel('Real')
axes[1].set_ylabel('Im')
axes[2].set_title(r'$\dot{\delta p}$')
axes[2].set_ylabel('Im')
axes[2].set_xlabel('Real')
axes[0].plot(vec_rad[0, :].real, vec_rad[0, :].imag)
axes[0].plot(vec_rad[0, 0].real, vec_rad[0, 0].imag, 'ko')
axes[0].plot(0, 0, 'kx')
axes[1].plot(vec_rad[1, :].real, vec_rad[1, :].imag)
axes[1].plot(vec_rad[1, 0].real, vec_rad[1, 0].imag, 'ko')
axes[1].plot(0, 0, 'kx')
axes[2].plot(vec_rad[2, :].real, vec_rad[2, :].imag)
axes[2].plot(vec_rad[2, 0].real, vec_rad[2, 0].imag, 'ko')
axes[2].plot(0, 0, 'kx')

Wr_mag = np.abs(vec_rad[0, :]) / np.abs(vec_rad[0, 0])
Wt_mag = np.abs(vec_rad[1, :]) / np.abs(vec_rad[1, 0])

phase_Wr = np.unwrap(np.angle(vec_rad[0, :])) * 180 / np.pi
# plt.figure()
# plt.plot(phase_Wr)
phase_Wtheta = np.unwrap(np.angle(vec_rad[1, :])) * 180 / np.pi
# plt.figure()
# plt.plot(phase_Wtheta)
delta_phase = -phase_Wr + phase_Wtheta + 360
fig, axes = plt.subplots(2, 1, figsize=(7, 9))
axes[0].set_ylabel(r'$\frac{|\delta w|}{|\delta w_{0}|}$')
# axes[0].set_ylim([0,1.1])
axes[1].set_ylabel(r'$\varphi(\delta w_r -\delta w_{\theta} )$')
axes[1].set_xlabel(r'$r $')
axes[0].plot(radii, Wr_mag, label=r'$|\delta w_r|$')
axes[0].plot(radii, Wt_mag, label=r'$|\delta w_{\theta}|$')
axes[1].plot(radii, delta_phase)
# axes[1].set_ylim([175,195])
axes[0].legend()
plt.show()
