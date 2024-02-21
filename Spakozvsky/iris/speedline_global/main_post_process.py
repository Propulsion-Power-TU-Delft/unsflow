import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from Utils.styles import *


data_folder = "../data/IRIS_single_stage/design0_beta_3.450/operating_map/"
with open(data_folder + 'mass_flow.pkl', 'rb') as f:
    mass_flow = pickle.load(f)
with open(data_folder + 'beta_ts.pkl', 'rb') as f:
    beta_ts = pickle.load(f)  # total to static pressure ratio
with open(data_folder + 'rpm.pkl', 'rb') as f:
    rpm = pickle.load(f)

plt.figure(figsize=(7, 5))
stall_idxes = [6, 6, 4, 4, 4] # stall indices seen from the speedlines analysis
mdot_senoo = []
beta_senoo = []
mdot_spak = []
beta_spak = []
for i in range(0, np.shape(mass_flow)[0]-1):
    stall_idx = stall_idxes[i]
    mdot = mass_flow[i, :]
    beta = beta_ts[i, :]
    idx = np.where(mdot>0)
    plt.plot(mdot[idx], beta[idx], label='%.1f krpm' %(rpm[i]/1000), linewidth=medium_line_width)
    mdot_senoo.append(mdot[0])
    beta_senoo.append(beta[0])
    mdot_spak.append(mdot[stall_idx])
    beta_spak.append(beta[stall_idx])
plt.xlabel(r'$\dot{m}$ [kg/s]', fontsize=font_labels)
plt.ylabel(r'$\beta_{ts}$ [-]', fontsize=font_labels)
plt.xticks(fontsize=font_axes)
plt.yticks(fontsize=font_axes)
plt.grid(alpha=0.2)
plt.legend(fontsize=font_legend)
plt.savefig('pictures/iris_characteristic_curves.pdf', bbox_inches='tight')







plt.figure(figsize=(7, 5))
stall_idxes = [6, 6, 4, 4, 4] # stall indices seen from the speedlines analysis
mdot_senoo = []
beta_senoo = []
mdot_spak = []
beta_spak = []
for i in range(0, np.shape(mass_flow)[0]-1):
    stall_idx = stall_idxes[i]
    mdot = mass_flow[i, :]
    beta = beta_ts[i, :]
    idx = np.where(mdot>0)
    plt.plot(mdot[idx], beta[idx], label='%.1f krpm' %(rpm[i]/1000), linewidth = medium_line_width)
    plt.plot(mdot[0], beta[0], 'ks', markersize=marker_size)
    plt.plot(mdot[stall_idx], beta[stall_idx], 'k^', markersize=marker_size)
    mdot_senoo.append(mdot[0])
    beta_senoo.append(beta[0])
    mdot_spak.append(mdot[stall_idx])
    beta_spak.append(beta[stall_idx])
plt.plot(mdot_senoo, beta_senoo, '--sk', linewidth=0.5, label='Senoo')
plt.plot(mdot_spak, beta_spak, '--^k', linewidth=0.5, label='Spakovszky')
plt.xlabel(r'$\dot{m}$ [kg/s]', fontsize=font_labels)
plt.ylabel(r'$\beta_{ts}$ [-]', fontsize=font_labels)
plt.xticks(fontsize=font_axes)
plt.yticks(fontsize=font_axes)
plt.grid(alpha=0.2)
plt.legend(fontsize=font_legend, loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=3)
plt.savefig('pictures/iris_characteristic_curves_stall.pdf', bbox_inches='tight')


plt.show()
