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
import pickle
import os

data_folder = 'results/'
files_and_directories = os.listdir(data_folder)
filenames = [data_folder + file_name for file_name in files_and_directories if os.path.isfile(data_folder + file_name)]
filenames.sort()

poles = []
deltax = []
for file in filenames:
    try:
        with open(file, 'rb') as pik:
            driver = pickle.load(pik)
            poles.append(driver.poles_dict)
            deltax.append(file[-12:-7])
    except:
        pass

deltax = np.array(deltax, dtype=float)

pole_up = []
pole_dn = []
dx = []
for i in range(len(deltax)):
    first_key = list(poles[i].keys())[0]
    pole = poles[i][first_key]
    idx_up = np.where(-pole.imag>0.5)
    idx_dn = np.where(-pole.imag<=0.5)
    pole_up.append(pole[idx_up])
    pole_dn.append(pole[idx_dn])
    dx.append(deltax[i])

pole_up = np.array([item for sublist in pole_up for item in sublist])
pole_dn = np.array([item for sublist in pole_dn for item in sublist])
colormap = 'viridis'
plt.figure()
colors = np.linspace(0, 1, len(pole_up))
plt.scatter(pole_up.real, -pole_up.imag, c=colors, s=20, cmap=colormap)
idx = np.where(-pole_dn.imag>-0.5)
pole_dn = pole_dn[idx]
colors = np.linspace(0, 1, len(pole_dn))
scatter = plt.scatter(pole_dn.real, -pole_dn.imag, c=colors, s=20, cmap=colormap)
colorbar = plt.colorbar(scatter)
colorbar.set_label(r'$\Delta x$ [-]', rotation=90, labelpad=15)
plt.xlabel(r'$\sigma$')
plt.ylabel(r'$\omega$')
plt.grid(alpha=0.2)
plt.xlim([-4.5, 0.5])
plt.ylim([-0.5, 2])
# plt.title('Root Locus')
plt.savefig('pictures/root_locus_complex_plane.pdf', bbox_inches='tight')




plt.figure()
alpha=1
for i in range(len(deltax)):
    first_key = list(poles[i].keys())[0]
    real_part = poles[i][first_key].real
    imag_part = -poles[i][first_key].imag
    deltax_var = np.zeros(len(real_part)) + deltax[i]
    if i == 0:
        plt.scatter(deltax_var, real_part, facecolors='none', edgecolors='red', marker='s', label='Growth Rate', alpha=alpha)
        plt.scatter(deltax_var, imag_part, facecolors='none', edgecolors='blue', marker='^', label='Rotation Rate', alpha=alpha)
    else:
        if deltax[i]>0.4:  # avoid spurious eigenvalues
            idx = np.where(imag_part>-0.5)
            plt.scatter(deltax_var[idx], real_part[idx], facecolors='none', edgecolors='red', marker='s', alpha=alpha)
            plt.scatter(deltax_var[idx], imag_part[idx], facecolors='none', edgecolors='blue', marker='^', alpha=alpha)
        else:
            plt.scatter(deltax_var, real_part, facecolors='none', edgecolors='red', marker='s', alpha=alpha)
            plt.scatter(deltax_var, imag_part, facecolors='none', edgecolors='blue', marker='^', alpha=alpha)
plt.xlim([0, 1])
plt.ylim([-2, 2])
# plt.title('Root Locus')
plt.grid(alpha=0.2)
plt.legend()
plt.xlabel(r'$\Delta x$ [-]')
plt.savefig('pictures/root_locus_variable_deltx.pdf', bbox_inches='tight')
plt.show()
