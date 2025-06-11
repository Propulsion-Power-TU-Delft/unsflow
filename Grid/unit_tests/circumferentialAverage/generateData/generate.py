import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

"""
GENERATE AN EQUIVALENT CSV SET OF DATA TO REPRESENT A 3D SPLINE EXTRACTED SOLUTION FROM PARAVIEW.
THE FILE WILL CONTAIN DENSITY, VELOCITY AND PRESSURE QUANTITIES
"""

# generate the grid
ni, nj, nk = 25, 15, 100
z = np.linspace(0, 1, ni)
r = np.linspace(1, 1.5, nj)
theta = np.linspace(0, np.pi/10, nk)

Z, R, THETA = np.meshgrid(z, r, theta, indexing='ij')

X = R * np.cos(THETA)
Y = R * np.sin(THETA)

# generate the data
density = np.zeros_like(Z)+1.014
velX = np.zeros_like(Z)+0.5
velY = np.zeros_like(Z)+5.0
velZ = np.zeros_like(Z)-0.3
pressure = np.zeros_like(Z)+101325.0

# save the data in csv dataset:
os.makedirs('Dataset', exist_ok=True)

for i in range(ni):
    for j in range(nj):
        with open('Dataset/spline_data_%03i_%03i.csv' %(i, j), 'w') as file:
            file.write('Points_0,Points_1,Points_2,Velocity (m/s)_0,Velocity (m/s)_1,Velocity (m/s)_2,Grid Velocity (m/s)_0,Grid Velocity (m/s)_1,Grid Velocity (m/s)_2,Velocity_Gradient (1/s)_0,Velocity_Gradient (1/s)_1,Velocity_Gradient (1/s)_2,Velocity_Gradient (1/s)_3,Velocity_Gradient (1/s)_4,Velocity_Gradient (1/s)_5,Velocity_Gradient (1/s)_6,Velocity_Gradient (1/s)_7,Velocity_Gradient (1/s)_8,Eddy Viscosity (N·s/m²),Density (kg/m³),Pressure (Pa)\n')

            for k in range(nk):
                file.write(f"{X[i,j,k]:.6f},{Y[i,j,k]:.6f},{Z[i,j,k]:.6f},"
                            f"{velX[i,j,k]:.6f},{velY[i,j,k]:.6f},{velZ[i,j,k]:.6f},"
                            f"{0:.6f},{0:.6f},{0:.6f},"
                            f"{0:.6f},{0:.6f},{0:.6f},"
                            f"{0:.6f},{0:.6f},{0:.6f},"
                            f"{0:.6f},{0:.6f},{0:.6f},"
                            f"{1:.6f},{density[i,j,k]:.6f},{pressure[i,j,k]:.6f}\n")
        
        print(f"Written file ({i},{j})")
                

