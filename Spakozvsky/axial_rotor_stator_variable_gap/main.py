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
from Spakozvsky.src.axial_gap import AxialGap
from Spakozvsky.src.axial_rotor import AxialRotor
from Spakozvsky.src.driver import Driver



format_fig = (7, 7)

# %%INPUT DATA for the implementation of the generic actuator disk of Greitzer, zero gap
Vy1 = 0  #  non dimensional background azimuthal flow velocity at outlet
DeltaX = 0.3

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

inlet = AxialDuct(Vy1, Vx1, x1)
rotor = AxialRotor(Vx1, Vy1, Vy2, alfa1, beta1, beta2, lambda_r, dLr_dTanb, tau_r)
gap = AxialGap(x2, x3, Vx3, Vy3)
stator = AxialStator(Vx3, Vy3, Vy4, alfa3, alfa4, lambda_s, dLs_dTana, tau_s)
outlet = AxialDuct(Vy4, Vx4, x4)
driver = Driver('axial')
driver.add_component(inlet)
driver.add_component(rotor)
driver.add_component(gap)
driver.add_component(stator)
driver.add_component(outlet)
driver.set_inlet_boundary_conditions()
driver.set_outlet_boundary_conditions()

domain = [-2.5, 0.5, -4.5, 4.5]
grid = [1, 1]
attempts = 15
tol = 1e-3
driver.set_eigenvalues_research_settings(domain, grid, attempts, tol)

N = np.arange(1, 7)
driver.find_eigenvalues(N)
driver.plot_eigenvalues([-2.5, 0.5, -0.5, 4.5], save_filename='rotor_stator_gap_%.1f' %(DeltaX))
driver.store_results_pickle(save_filename='rotor_stator_gap_%.1f' %(DeltaX))

plt.show()
