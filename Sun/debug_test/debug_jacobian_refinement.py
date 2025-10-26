#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 09:29:59 2023
@author: F. Neri, TU Delft

test the jacobian and mapping gradients accuracy
"""
import sys
sys.path.append('../')

import matplotlib.pyplot as plt
import numpy as np
from src.compressor import Compressor
from src.grid import DataGrid
from src.sun_model import SunModel
from src.styles import *


#%%
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


#%% fine
#number of grid nodes in the computational domain
Nz = 10
Nr = 10

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
sunObj.AddNormalizationQuantities(rho_ref, u_ref, x_ref)
sunObj.ShowPhysicalGrid(save_filename='debug_phys_grid_%1.d_%1.d' %(Nz,Nr))
sunObj.ComputeSpectralGrid()
sunObj.ShowSpectralGrid(save_filename='debug_spect_grid_%1.d_%1.d' %(Nz,Nr))
sunObj.ComputeJacobianPhysical()
sunObj.ContourTransformation(save_filename='debug_transform_gradients_%1.d_%1.d' %(Nz,Nr))
sunObj.ShowJacobianSpectralAxis(save_filename='debug_transform_gradients_%1.d_%1.d' %(Nz,Nr))





#%%  coarse
#number of grid nodes in the computational domain
Nz = 100
Nr = 100

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
        tangentialVel[ii,jj] = axialVel[ii,jj]
        radialVel[ii,jj] = axialVel[ii,jj]
        pressure[ii,jj] = p

ductObj2 = DataGrid(0, L/x_ref, r1/x_ref, r2/x_ref, Nz, Nr, density/rho_ref, radialVel/u_ref, tangentialVel/u_ref, axialVel/u_ref, pressure/p_ref)

#general workflow of the sun model
sunObj2 = SunModel(ductObj2)
sunObj2.AddNormalizationQuantities(rho_ref, u_ref, x_ref)
# sunObj2.ShowPhysicalGrid(save_filename='dbg_jacobian_phys_grid_%1.d_%1.d' %(Nz,Nr))
sunObj2.ComputeSpectralGrid()
# sunObj2.ShowSpectralGrid(save_filename='dbg_jacobian_spec_grid_%1.d_%1.d' %(Nz,Nr))
sunObj2.ComputeJacobianPhysical()
# sunObj2.ContourTransformation(save_filename='dbg_jac_grad_%1.d_%1.d' %(Nz,Nr))
# sunObj2.ShowJacobianSpectralAxis(save_filename='dbg_jac_scatter_grad_%1.d_%1.d' %(Nz,Nr))








#%% blue print of how it should work
# import numpy as np
from scipy.interpolate import griddata

# # Define the original grid points (z, r) and function values f(z, r)
# z = np.array([0, 1, 2])   # Example grid points along the z-axis
# r = np.array([0, 1, 2])   # Example grid points along the r-axis
# f_values = np.array([[1, 2, 3],
#                      [4, 5, 6],
#                      [7, 8, 9]])

# # Create a meshgrid of the original grid points
# z_grid, r_grid = np.meshgrid(z, r)

# # Reshape the original grid points and function values for interpolation
# points = np.column_stack((z_grid.ravel(), r_grid.ravel()))
# values = f_values.ravel()

# # Define the new grid points (zgrid, rgrid)
# zgrid = np.array([0.5, 1.5])   # Example new grid points along the z-axis
# rgrid = np.array([0.5, 1.5])   # Example new grid points along the r-axis

# # Create a meshgrid of the new grid points
# zgrid_mesh, rgrid_mesh = np.meshgrid(zgrid, rgrid)

# # Reshape the new grid points for interpolation
# points_new = np.column_stack((zgrid_mesh.ravel(), rgrid_mesh.ravel()))

# # Perform linear interpolation using griddata
# f_interpolated = griddata(points, values, points_new, method='linear')

# # Reshape the interpolated values to match the shape of the new grid
# f_interpolated_reshaped = f_interpolated.reshape(zgrid_mesh.shape)

# print(f_interpolated_reshaped)

# plt.figure(figsize=(10,7))
# plt.scatter(z_grid, r_grid, c=f_values)
# # plt.colorbar()

# # plt.figure(figsize=(10,7))
# plt.scatter(zgrid_mesh, rgrid_mesh, marker = 's' , c=f_interpolated_reshaped)
# plt.colorbar()
#%%try to replicate

#interpolate derivative on the fine grid results, not the coarse one 
z_grid = sunObj2.dataSpectral.zGrid
r_grid =sunObj2.dataSpectral.rGrid

# Reshape the original grid points and function values for interpolation
points = np.column_stack((z_grid.ravel(), r_grid.ravel()))
values = sunObj2.J.ravel()

# # Create a meshgrid of the new grid points
zgrid_mesh = sunObj.dataSpectral.zGrid
rgrid_mesh =sunObj.dataSpectral.rGrid

# # Reshape the new grid points for interpolation
points_new = np.column_stack((zgrid_mesh.ravel(), rgrid_mesh.ravel()))

# Perform linear interpolation using griddata
f_interpolated = griddata(points, values, points_new, method='cubic')

# Reshape the interpolated values to match the shape of the new grid
f_interpolated_reshaped = f_interpolated.reshape(zgrid_mesh.shape)

#interpolated on a fine grid
plt.figure(figsize=(10,7))
plt.scatter(zgrid_mesh, rgrid_mesh, c=f_interpolated_reshaped, vmax=0.4)
plt.xlabel(r'$\xi$')
plt.ylabel(r'$\eta$')
plt.title(r'$J_{fine}$')
plt.colorbar()

#computed directly on a coarse grid
plt.figure(figsize=(10,7))
plt.scatter(zgrid_mesh, rgrid_mesh, c=sunObj.J, vmax=0.4)
plt.xlabel(r'$\xi$')
plt.ylabel(r'$\eta$')
plt.title(r'$J_{coarse}$')
plt.colorbar()

#difference
# error = np.abs((sunObj.J - f_interpolated_reshaped)/np.max(sunObj.J))
plt.plot()
plt.imshow(error, vmax=1)
plt.colorbar()

plt.figure()
plt.plot(rgrid_mesh[5,:], f_interpolated_reshaped[5,:], '--s', label='fine')
plt.plot(rgrid_mesh[5,:], sunObj.J[5,:], '--s', label='coarse')
plt.xlabel(r'$\eta$')
plt.ylabel(r'$J$')
plt.title(r'$\xi=%.3f$' %(zgrid_mesh[5,0]))
plt.legend()