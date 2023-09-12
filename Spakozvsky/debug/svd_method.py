#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 14:11:33 2023

@author: fneri
Exercise 5.4.3 of Spakovszky thesis
"""

import matplotlib.pyplot as plt
import numpy as np
import Spakozvsky
from Spakozvsky.src.functions import *
import os
import time

# Preamble: customization of matplotlib
# Configuration for plots
plt.rc('text', usetex=False)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rcParams['font.size'] = 14
format_fig = (10, 7)

# create directory for pictures
path = "pics"
isExist = os.path.exists(path)
if not isExist:
    os.makedirs(path)

poles = np.array([1 + 2j, -2 + 0.5j, -1j])
# system function
def matrix(s):

    # poles = np.array([0.1, 0, 1j])

    M = np.array([[poles[0] - s, 0, 0],
                  [0, poles[1] - s, 0],
                  [0, 0, poles[2] - s]], dtype=complex)
    return M


domain = [-3.5, 2.5, -3, 5]
omR_min = domain[0]
omR_max = domain[1]
omI_min = domain[2]
omI_max = domain[3]
nR = 200
nI = 200
omR = np.linspace(omR_min, omR_max, nR)
omI = np.linspace(omI_min, omI_max, nI)
omegaI, omegaR = np.meshgrid(omI, omR)
chi = np.zeros((nR, nI))
sing_value_min = np.zeros((nR, nI))
sing_value_max = np.zeros((nR, nI))
start_time = time.time()

for ii in range(0, nR):
    for jj in range(0, nI):

        # this block is simply to print info, and time remaining for the full process
        current_time = time.time() - start_time
        if (ii == 0 and jj == 0):
            print('SVD %.1d of %1.d ..' % (ii * len(omI) + 1 + jj, len(omR) * len(omI)))
        if (jj == 1):  # update time whenever jj=1
            delta_time_svd = current_time
            total_time = (delta_time_svd / (
                    ii * nI + jj)) * nR * nI  # (time passed / number of SVD done) * number of total SVD to do
        if (ii != 0 or jj != 0):
            remaining_minutes = (total_time - current_time) / 60
            total_minutes = total_time / 60
            print('SVD %.1d of %1.d \t (%.1d min remaining)' % (
                ii * len(omI) + 1 + jj, len(omR) * len(omI), remaining_minutes + 1))  # keep track of the progress

        omega = omR[ii] + 1j * omI[jj]
        Y = matrix(omega)
        u, s, v = np.linalg.svd(Y)
        sing_value_min[ii, jj] = np.min(s)
        sing_value_max[ii, jj] = np.max(s)
        chi[ii, jj] = np.min(s) / np.max(s)

end_time = time.time() - start_time
hrs = int(end_time / 3600)
mins = int((end_time - hrs * 3600) / 60)
sec = int(end_time - hrs * 3600 - mins * 60)
print('Total SVD time: \t %1.d hrs %1.d mins %1.d sec' % (hrs, mins, sec))

fig, ax = plt.subplots(figsize=format_fig)
cs = ax.contourf(omegaR, omegaI, np.log(chi), 100, cmap='jet')
# ax.plot(poles.real, poles.imag, 'w^')
ax.set_xlabel(r'$\omega_{R} \ \mathrm{[-]}$')
ax.set_ylabel(r'$\omega_{I} \ \mathrm{[-]}$')
ax.set_title(r'$\log \chi$')
cbar = fig.colorbar(cs)
plt.show()
