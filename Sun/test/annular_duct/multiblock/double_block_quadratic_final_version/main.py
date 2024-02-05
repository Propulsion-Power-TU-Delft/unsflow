#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 09:29:59 2023
@author: F. Neri, TU Delft
"""

import matplotlib.pyplot as plt
import numpy as np
import Sun
import os
from Sun.src.sun_model_multiblock import SunModelMultiBlock
from Grid.src.config import Config
from scipy.sparse.linalg import eigs

# input data of the problem (SI units)
r1 = 0.1826  # inner radius [m]
r2 = 0.2487  # outer radius [m]
M = 0.015  # Mach number
p = 100e3  # pressure [Pa]
T = 288  # temperature [K]
L = 0.08  # length [m]
R = 287.058  # air gas constant [kJ/kgK]
gmma = 1.4  # cp/cv ratio of air
rho = p/R/T  # density [kg/m3]
a = np.sqrt(gmma * p / rho)  # ideal speed of sound [m/s]
HARMONIC_ORDER = 1

# non-dimensionalization terms:
x_ref = r1
u_ref = M * a
rho_ref = rho
t_ref = x_ref / u_ref
omega_ref = 1 / t_ref
rpm_ref = omega_ref*60/2/np.pi
p_ref = rho_ref * u_ref ** 2


# %%%%%%%%%%%%%%%%%%%%%%% COMPUTATIONAL PART %%%%%%%%%%%%%%%%%%%%%%%
# number of grid nodes in the computational domain
Nz = 15
Nr = 10
# Nz = 10//2
# Nr = 5

config = Config('duct.ini')
duct_Obj1 = Sun.src.AnnulusMeridional(0, L/2, r1, r2, Nz, Nr, rho, 0, 0, M*a, p, config)
duct_Obj2 = Sun.src.AnnulusMeridional(L/2, L, r1, r2, Nz, Nr, rho, 0, 0, M*a, p, config)

duct_Obj1.normalize_data()
duct_Obj2.normalize_data()

duct_grid1 = Sun.src.sun_grid.SunGrid(duct_Obj1)
duct_grid2 = Sun.src.sun_grid.SunGrid(duct_Obj2)

sun_obj = Sun.src.SunModel(duct_grid1, config=config)
sun_obj2 = Sun.src.SunModel(duct_grid2, config=config)

sun_blocks = [sun_obj, sun_obj2]
ii = 0
for sun_obj in sun_blocks:
    sun_obj.ComputeBoundaryNormals()
    sun_obj.set_overwriting_equation_euler_wall('utheta')
    sun_obj.ComputeSpectralGrid()
    sun_obj.ComputeJacobianPhysical()
    sun_obj.AddAMatrixToNodesFrancesco2()
    sun_obj.AddBMatrixToNodesFrancesco2()
    sun_obj.AddCMatrixToNodesFrancesco2()
    sun_obj.AddEMatrixToNodesFrancesco2()
    sun_obj.AddRMatrixToNodesFrancesco2()
    sun_obj.AddSMatrixToNodes()
    sun_obj.AddHatMatricesToNodes()
    sun_obj.ApplySpectralDifferentiation()
    sun_obj.build_A_global_matrix()
    sun_obj.build_C_global_matrix()
    sun_obj.build_R_global_matrix()
    sun_obj.build_S_global_matrix()
    sun_obj.build_Z_global_matrix()
    sun_obj.compute_L_matrices(ii)
    sun_obj.set_boundary_conditions()
    sun_obj.apply_boundary_conditions_generalized()
    ii +=1

sun_multiblock = SunModelMultiBlock(sun_blocks, config)
sun_multiblock.construct_L_global_matrices()
sun_multiblock.apply_matching_conditions()
sun_multiblock.compute_P_Y_matrices()
sun_multiblock.solve_evp(sort_mode='real increasing')
sun_multiblock.extract_eigenfields()
sun_multiblock.plot_eigenfrequencies(save_filename='eigenfrequencies', normalization=False, delimit=(7500, 38000, -8000, 8000))
sun_multiblock.plot_eigenfields(save_filename='eigenmode')
sun_multiblock.write_results()

plt.show()
