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
from utils.styles import *

data_folder = 'results/'
files_and_directories = os.listdir(data_folder)
filenames = [data_folder + file_name for file_name in files_and_directories if file_name.split('.')[-1] == 'pkl']
filenames.sort()

poles = []
deltax = []
for file in filenames:
    try:
        with open(file, 'rb') as pik:
            driver = pickle.load(pik)
            poles.append(driver.poles_dict)
            deltax.append(file.split('_')[-1][0:-4])
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
plt.figure(figsize=(6, 4))
colors = np.linspace(0, 1, len(pole_up))
plt.scatter(pole_up.real, -pole_up.imag, c=colors, s=20, cmap=colormap)
idx = np.where(-pole_dn.imag>-0.5)
pole_dn = pole_dn[idx]
colors = np.linspace(0, 1, len(pole_dn))
scatter = plt.scatter(pole_dn.real, -pole_dn.imag, c=colors, s=20, cmap=colormap)
colorbar = plt.colorbar(scatter)
colorbar.set_label(r'$\Delta x$', rotation=90, labelpad=15)
plt.xlabel(r'$\sigma_3$')
plt.ylabel(r'$\omega_3$')
plt.grid(alpha=0.2)
plt.xlim([-4.5, 0.5])
plt.ylim([-0.4, 2])
plt.tight_layout()
# plt.title('Root Locus')
plt.savefig('pictures/root_locus_complex_plane.pdf', bbox_inches='tight')




plt.figure(figsize=(6, 5))
step = 1
alpha=1
ms=10
for i in range(len(deltax)):
    first_key = list(poles[i].keys())[0]w
    real_part = poles[i][first_key].real
    imag_part = -poles[i][first_key].imag
    deltax_var = np.zeros(len(real_part)) + deltax[i]
    if i == 0:
        plt.scatter(deltax_var[::step], real_part[::step], edgecolors='C0', facecolors='none', marker='o', label=r'$\sigma_3$', alpha=alpha, s=ms)
        # plt.scatter(deltax_var[::step], imag_part[::step], c='C0', marker='^', label=r'$\omega_3$', alpha=alpha, s=ms)
    else:
        if deltax[i]>0.4:  # avoid spurious eigenvalues
            idx = np.where(imag_part>-0.4)
            plt.scatter(deltax_var[idx][::step], real_part[idx][::step], edgecolors='C0', facecolors='none', marker='o', alpha=alpha, s=ms)
            # plt.scatter(deltax_var[idx][::step], imag_part[idx][::step], c='C0', marker='^', alpha=alpha, s=ms)
        else:
            plt.scatter(deltax_var[::step], real_part[::step], edgecolors='C0', facecolors='none', marker='o', alpha=alpha, s=ms)
            # plt.scatter(deltax_var[::step], imag_part[::step], c='C0', marker='^', alpha=alpha, s=ms)
plt.xlim([0, 1])
plt.ylim([-1.5, 0.25])
plt.axhline(0, color='red', linestyle='--')
# plt.title('Root Locus')
plt.grid(alpha=0.2)
# plt.legend()
plt.xlabel(r'$\Delta x$')
plt.ylabel(r'$\sigma_3$')
plt.tight_layout()

plt.savefig('pictures/root_locus_variable_deltx.pdf', bbox_inches='tight')
plt.show()
