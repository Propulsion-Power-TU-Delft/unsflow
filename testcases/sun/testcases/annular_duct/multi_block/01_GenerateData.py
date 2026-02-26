import matplotlib.pyplot as plt
import numpy as np
import os
from unsflow.sun.sun_model_multiblock import SunModelMultiBlock
from unsflow.sun.general_functions import gauss_lobatto_grid_generation 
from unsflow.sun.config import Config
from scipy.sparse.linalg import eigs
from unsflow.utils.plot_styles import *
import pickle


# INPUT VALUES
Nz = 10
Nr = 10
r1 = 0.1826  # inner radius [m]
r2 = 0.2487  # outer radius [m]
M = 0.015  # Mach number
p = 100e3  # pressure [Pa]
T = 288  # temperature [K]
nBlocks = 3
L = 0.08/nBlocks  # length [m]
R = 287.058  # air gas constant [kJ/kgK]
gmma = 1.4  # cp/cv ratio of air
rho = p/R/T  # density [kg/m3]
a = np.sqrt(gmma * p / rho)  # ideal speed of sound [m/s]

endPrevious = 0
for i in range(nBlocks):
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
            
    x = gauss_lobatto_grid_generation(Nz, 0+endPrevious, L+endPrevious)
    y = gauss_lobatto_grid_generation(Nr, r1, r2)
    X, Y = np.meshgrid(x, y, indexing='ij')

    data = {
        'AxialCoord': X,
        'RadialCoord': Y,
        'Density': density,
        'AxialVel': axialVel,
        'RadialVel': radialVel,
        'TangentialVel': tangentialVel,
        'Pressure': pressure
    }

    os.makedirs('Grids', exist_ok=True)
    with open('Grids/data_%02i_%02i_block_%02i.pkl' % (Nz, Nr, i), 'wb') as f:
        pickle.dump(data, f)

    print('Data saved in Grids/data_%02i_%02i_block_%02i.pkl' % (Nz, Nr, i))
    endPrevious += L
