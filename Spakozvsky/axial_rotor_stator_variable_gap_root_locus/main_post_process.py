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

plt.figure()
for i in range(len(deltax)):
    first_key = list(poles[i].keys())[0]
    real_part = poles[i][first_key].real
    imag_part = -poles[i][first_key].imag
    plt.scatter(real_part, imag_part, c='blue')


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
plt.title('Root Locus')
plt.grid(alpha=0.2)
plt.legend()
plt.xlabel(r'$\Delta x$ [-]')
plt.savefig('root_locus_variable_deltx.pdf', bbox_inches='tight')
plt.show()
