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

from Sun.src.annulus_meridional import AnnulusMeridional


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

# non-dimensionalization terms:
x_ref = r1
u_ref = M * a
rho_ref = rho
t_ref = x_ref / u_ref
omega_ref = 1 / t_ref
p_ref = rho_ref * u_ref ** 2

# number of grid nodes in the computational domain
Nz = 15
Nr = 5

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
duct_Obj = AnnulusMeridional(0, L, r1, r2, Nz, Nr, density, radialVel, tangentialVel, axialVel, pressure)

duct_grid = Sun.src.sun_grid.SunGrid(duct_Obj)


# general workflow of the sun model
sun_obj = Sun.src.SunModel(duct_grid)
sun_obj.ComputeBoundaryNormals()
# sun_obj.ShowNormals()
sun_obj.AddNormalizationQuantities(rho_ref, u_ref, x_ref, 0)
sun_obj.NormalizeData()
# sun_obj.ShowPhysicalGrid()
sun_obj.ComputeSpectralGrid()
# sun_obj.ShowSpectralGrid()
sun_obj.ComputeJacobianPhysical()
sun_obj.ContourTransformation()
sun_obj.AddAMatrixToNodes()
sun_obj.AddBMatrixToNodes()
sun_obj.AddCMatrixToNodes(m=1)
sun_obj.AddEMatrixToNodes()
sun_obj.AddRMatrixToNodes()
sun_obj.AddSMatrixToNodes()
sun_obj.AddHatMatricesToNodes()
sun_obj.ApplySpectralDifferentiation()

omega_domain = np.array([7.5e3, 35e3, -8e3, 8e3]) / omega_ref
grid_omega = np.array([50, 15])
sun_obj.ComputeSVD(omega_domain = omega_domain, grid_omega = grid_omega)

sun_obj.PlotInverseConditionNumber()

plt.show()
