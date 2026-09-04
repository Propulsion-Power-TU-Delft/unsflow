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
from unsflow.utils.thesis_plots import create_figure, set_thesis_style
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



# read ref data
ref_data = []
for n in ["1", "2", "3", "4"]:
    ref = np.loadtxt(f"ref_data/ref_line_{n}.csv", skiprows=1, dtype=float, delimiter=',')
    ref_data.append(ref)






pole_up = np.array([item for sublist in pole_up for item in sublist])
pole_dn = np.array([item for sublist in pole_dn for item in sublist])
colormap = 'viridis'

set_thesis_style()
fig, (ax1, ax2) = create_figure(fraction=1.00, aspect_ratio=1.3, subplots=(1, 2))
ms = 3

# --- First Subplot: Complex Plane Root Locus ---
colors_up = np.linspace(0, 1, len(pole_up))
ax1.scatter(pole_up.real, -pole_up.imag, c=colors_up, cmap=colormap, s=ms)

idx_dn = np.where(-pole_dn.imag > -0.38)
pole_dn_filtered = pole_dn[idx_dn]
colors_dn = np.linspace(0, 1, len(pole_dn_filtered))
scatter = ax1.scatter(pole_dn_filtered.real, -pole_dn_filtered.imag, c=colors_dn, cmap=colormap, s=ms)

colorbar = fig.colorbar(scatter, ax=ax1)
colorbar.set_label(r'$\Delta x$', rotation=90, labelpad=2.5)

ax1.set_xlabel(r'$\sigma_3$')
ax1.set_ylabel(r'$\omega_3$')
ax1.grid(alpha=0.2)
ax1.set_xlim([-4.5, 0.5])
ax1.set_ylim([-0.4, 2])


# --- Second Subplot: Variable Delta X ---
step = 1
alpha = 1

for i in range(len(deltax)):
    first_key = list(poles[i].keys())[0]
    real_part = poles[i][first_key].real
    imag_part = -poles[i][first_key].imag
    deltax_var = np.zeros(len(real_part)) + deltax[i]
    
    if i == 0:
        ax2.scatter(deltax_var[0], real_part[0], c='C0', marker='o', label=r'$\sigma_3$', alpha=alpha, s=ms)
        ax2.scatter(deltax_var[0], imag_part[0], c='C1', marker='^', label=r'$\omega_3$', alpha=alpha, s=ms)
    else:
        if deltax[i] > 0.4:  # avoid spurious eigenvalues
            idx_var = np.where(imag_part > -0.35)
            ax2.scatter(deltax_var[idx_var][::step], real_part[idx_var][::step], c='C0', marker='o', alpha=alpha, facecolor='none', s=ms)
            ax2.scatter(deltax_var[idx_var][::step], imag_part[idx_var][::step], c='C1', marker='^', alpha=alpha, facecolor='none', s=ms)
        else:
            ax2.scatter(deltax_var[::step], real_part[::step], c='C0', marker='o', facecolor='none', alpha=alpha, s=ms)
            ax2.scatter(deltax_var[::step], imag_part[::step], c='C1', marker='^', facecolor='none', alpha=alpha, s=ms)

for i in range(len(ref_data)):
    if i == 0:
        ax2.plot(ref_data[i][:, 0], ref_data[i][:, 1], '--kx', label='Reference')
    else:
        ax2.plot(ref_data[i][:, 0], ref_data[i][:, 1], '--kx')

ax2.set_xlim([0, 1])
ax2.set_ylim([-1.5, 2.0])
ax2.axhline(0, color='red', linestyle='--')
ax2.grid(alpha=0.2)
ax2.legend()
ax2.set_xlabel(r'$\Delta x$')
ax2.set_ylabel(r'$\sigma_3, \omega_3$')

plt.savefig('pictures/root_locus_combined.pdf')
plt.show()
