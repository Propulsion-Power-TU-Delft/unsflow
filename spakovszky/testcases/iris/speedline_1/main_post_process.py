import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from utils.styles import *

data_folder = "../data/IRIS_single_stage/design0_beta_3.450/operating_map/"
with open(data_folder + 'mass_flow.pkl', 'rb') as f:
    mass_flow = pickle.load(f)
with open(data_folder + 'beta_ts.pkl', 'rb') as f:
    beta_ts = pickle.load(f)  # total to static pressure ratio
with open(data_folder + 'rpm.pkl', 'rb') as f:
    rpm = pickle.load(f)

plt.figure()
for i in range(0, np.shape(mass_flow)[0] - 1):
    mdot = mass_flow[i, :]
    beta = beta_ts[i, :]
    idx = np.where(mdot > 0)
    plt.plot(mdot[idx], beta[idx], label='%.1f krpm' % (rpm[i] / 1000), linewidth=medium_line_width)
    plt.plot(mdot[0], beta[0], 'ks', markersize=4)
plt.xlabel(r'$\dot{m}$ [kg/s]', fontsize=font_labels)
plt.ylabel(r'$\beta_{ts}$ [-]', fontsize=font_labels)
plt.xticks(fontsize=font_axes)
plt.yticks(fontsize=font_axes)
plt.title('Compressor Curves', fontsize=font_title)
plt.grid(alpha=0.2)
plt.legend(fontsize=font_legend)
plt.savefig('pictures/iris_characteristic_curves.pdf', bbox_inches='tight')

speedline = 1  # choose the speedline to be used
mass_flow = mass_flow[speedline, :]
beta_ts = beta_ts[speedline, :]

data_folder = 'results/'
files_and_directories = os.listdir(data_folder)
filenames = [data_folder + file_name for file_name in files_and_directories if (os.path.isfile(data_folder +
                                                                    file_name) and file_name[-6:] == 'pickle')]
filenames.sort()

poles = []
mass_idx = []
for file in filenames:
    with open(file, 'rb') as pik:
        driver = pickle.load(pik)
        poles.append(driver.poles_dict)
        mass_idx.append(int(file[-9: -7]))

# the first index is the number of flow conditions, the second is the number of harmonics
n_harmonics = len(poles[0].keys())
GFactor = np.zeros((len(poles), n_harmonics))
RSpeed = np.zeros((len(poles), n_harmonics))
Mflow = np.zeros((len(poles), n_harmonics))

for i in range(len(poles)):
    pole = poles[i]
    idx = mass_idx[i]
    j = 0
    # the second index is the number of harmonics considered
    for key in pole.keys():
        real_part = pole[key].real
        imag_part = pole[key].imag
        idx_max = np.where(real_part == np.max(real_part))
        growth_factor = real_part[idx_max]
        rot_speed = imag_part[idx_max]
        GFactor[i, j] = growth_factor
        RSpeed[i, j] = rot_speed / key
        Mflow[i, j] = mass_flow[idx]
        j += 1

plt.figure()
for j in range(n_harmonics):
    plt.plot(Mflow[:, j], GFactor[:, j], '-s', label='n= %i' % (j + 1), linewidth=medium_line_width, markersize=3)
boundary_x = np.linspace(np.min(Mflow), np.max(Mflow))
boundary_y = np.zeros_like(boundary_x)
plt.plot(boundary_x, boundary_y, '--k', linewidth=medium_line_width)
plt.legend(fontsize=font_legend)
plt.xlabel(r'$\dot{m}$ [kg/s]', fontsize=font_labels)
plt.ylabel(r'GF', fontsize=font_labels)
plt.xticks(fontsize=font_axes)
plt.yticks(fontsize=font_axes)
plt.title(r'$%.1f$ [krpm]' % (rpm[speedline] / 1000), fontsize=font_title)
plt.grid(alpha=0.2)
plt.savefig('pictures/iris_growth_factors_speedline_%i.pdf' %(speedline), bbox_inches='tight')

plt.figure()
for j in range(n_harmonics):
    plt.plot(Mflow[:, j], RSpeed[:, j], '-s', label='n:%i' % (j + 1), linewidth=medium_line_width, markersize=3)
boundary_x = np.linspace(np.min(Mflow), np.max(Mflow))
boundary_y = np.zeros_like(boundary_x)
plt.plot(boundary_x, boundary_y, '--k')
plt.legend(fontsize=font_legend)
plt.xlabel(r'$\dot{m}$ [kg/s]', fontsize=font_labels)
plt.ylabel(r'RS', fontsize=font_labels)
plt.xticks(fontsize=font_axes)
plt.yticks(fontsize=font_axes)
plt.title(r'$%.1f$ [krpm]' % (rpm[speedline] / 1000), fontsize=font_title)
plt.grid(alpha=0.2)
plt.savefig('pictures/iris_rot_speed_speedline_%i.pdf' %(speedline), bbox_inches='tight')

idx_instability = 5
plt.figure()
idx = np.where(mass_flow > 0)
plt.plot(mass_flow[idx], beta_ts[idx], linewidth=medium_line_width)
plt.plot(mass_flow[0], beta_ts[0], 'ks', label='Senoo')
plt.plot(mass_flow[idx_instability], beta_ts[idx_instability], 'k^', label='Spakovszky')
plt.xlabel(r'$\dot{m}$ [kg/s]', fontsize=font_labels)
plt.ylabel(r'$\beta_{\rm ts}$', fontsize=font_labels)
plt.xticks(fontsize=font_axes)
plt.yticks(fontsize=font_axes)
plt.title('Compressor Curve', fontsize=font_title)
plt.grid(alpha=0.2)
plt.legend(fontsize=font_legend)
plt.savefig('pictures/iris_curve_speedline_%i.pdf' % (speedline), bbox_inches='tight')

plt.show()
