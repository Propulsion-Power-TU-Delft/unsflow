#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 09:29:59 2023
@author: F. Neri, TU Delft

test the non-dimensionalisation procedure
"""
import sys
sys.path.append('../')
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import jv, yv, jvp, yvp
from src.compressor import Compressor
from src.grid import DataGrid
from src.sun_model import SunModel
import scipy
from scipy.optimize import fsolve
from src.styles import *

#input data of the problem (SI units)
r1 = 0.1826                             #inner radius [m]
r2 = 0.2487                             #outer radius [m]
M = 0.015                               #Mach number
p = 100e3                               #pressure [Pa]
T = 288                                 #temperature [K]
L = 0.08                                #length [m]
R = 287                                 #air gas constant [kJ/kgK]
gmma = 1.4                              #cp/cv ratio of air
rho = p/(R*T)                           #density [kg/m3]
a = np.sqrt(gmma*p/rho)                 #ideal speed of sound [m/s]

#non-dimensionalization terms:
x_ref = r1
u_ref = M*a
rho_ref = rho
t_ref = x_ref/u_ref
omega_ref = 1/t_ref
p_ref = rho_ref*u_ref**2

    

#%%COMPUTATIONAL PART, DIMENSIONAL VERSION

#number of grid nodes in the computational domain
Nz = 15
Nr = 5

#non-dimensionalization terms:
x_ref = r1/r1
u_ref = M*a/(M*a)
rho_ref = rho/rho
t_ref = x_ref/u_ref
omega_ref = 1/t_ref
p_ref = rho_ref*u_ref**2

#implement a constant uniform flow in the annulus duct
density = np.zeros((Nz,Nr))
axialVel = np.zeros((Nz,Nr))
radialVel = np.zeros((Nz,Nr))
tangentialVel = np.zeros((Nz,Nr))
pressure = np.zeros((Nz,Nr))
for ii in range(0,Nz):
    for jj in range(0,Nr):
        #there could be a need for normalizing the data? (or normalizing the NS equations directly)
        density[ii,jj] = rho
        axialVel[ii,jj] = M*a      
        pressure[ii,jj] = p

ductObj = DataGrid(0, L/x_ref, r1/x_ref, r2/x_ref, Nz, Nr, density/rho_ref, radialVel/u_ref, tangentialVel/u_ref, axialVel/u_ref, pressure/p_ref)


#general workflow of the sun model
sunObj = SunModel(ductObj)
sunObj.AddNormalizationQuantities(1,1,1)
sunObj.ComputeSpectralGrid()
sunObj.ComputeJacobianPhysical()
sunObj.AddAMatrixToNodes()
sunObj.AddBMatrixToNodes()
sunObj.AddCMatrixToNodes()
sunObj.AddEMatrixToNodes()
sunObj.AddRMatrixToNodes()
sunObj.AddSMatrixToNodes()
sunObj.AddHatMatricesToNodes()
sunObj.ApplySpectralDifferentiation()

omega_domain=[7.5e3, 35e3, -8e3, 8e3]
grid_omega=[150,20]
sunObj.ComputeSVD(omega_domain=omega_domain/omega_ref, grid_omega=grid_omega)
sunObj.PlotSingularValues(save_filename='debug_sing_val_map_dimensional_%1.d_%1.d_%1.d_%1.d' %(Nz,Nr,omega_domain[0],omega_domain[1]))
sunObj.PlotInverseConditionNumber(ref_solution = None, save_filename='debug_chi_map_dimensional_%1.d_%1.d_%1.d_%1.d' %(Nz,Nr,omega_domain[0],omega_domain[1]))


#%% NON-DIMENSIONAL PART
#general workflow of the sun model
#non-dimensionalization terms:
x_ref = r1
u_ref = M*a
rho_ref = rho
t_ref = x_ref/u_ref
omega_ref = 1/t_ref
p_ref = rho_ref*u_ref**2

ductObj = DataGrid(0, L/x_ref, r1/x_ref, r2/x_ref, Nz, Nr, density/rho_ref, radialVel/u_ref, tangentialVel/u_ref, axialVel/u_ref, pressure/p_ref)
sunObj = SunModel(ductObj)
sunObj.AddNormalizationQuantities(rho_ref, u_ref, x_ref)
sunObj.ComputeSpectralGrid()
sunObj.ComputeJacobianPhysical()
sunObj.AddAMatrixToNodes()
sunObj.AddBMatrixToNodes()
sunObj.AddCMatrixToNodes()
sunObj.AddEMatrixToNodes()
sunObj.AddRMatrixToNodes()
sunObj.AddSMatrixToNodes()
sunObj.AddHatMatricesToNodes()
sunObj.ApplySpectralDifferentiation()
sunObj.ComputeSVD(omega_domain=omega_domain/omega_ref, grid_omega=grid_omega)
sunObj.PlotSingularValues(save_filename='debug_sing_val_map_nondimensional_%1.d_%1.d_%1.d_%1.d' %(Nz,Nr,omega_domain[0],omega_domain[1]))
sunObj.PlotInverseConditionNumber(ref_solution = None, save_filename='debug_chi_map_nondimensional_%1.d_%1.d_%1.d_%1.d' %(Nz,Nr,omega_domain[0],omega_domain[1]))







 