import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from unsflow.utils.thesis_plots import create_figure, set_thesis_style


data_folder = "../data/IRIS_single_stage/design0_beta_3.450/operating_map/"
with open(data_folder + 'mass_flow.pkl', 'rb') as f:
    mass_flow = pickle.load(f)
with open(data_folder + 'beta_ts.pkl', 'rb') as f:
    beta_ts = pickle.load(f)  # total to static pressure ratio
with open(data_folder + 'rpm.pkl', 'rb') as f:
    rpm = pickle.load(f)

set_thesis_style()
fig, ax = create_figure(fraction=0.49, aspect_ratio=1.25, subplots=(1,1))

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
    ax.plot(mdot[idx], beta[idx], label='%.1f krpm' %(rpm[i]/1000))
    mdot_senoo.append(mdot[0])
    beta_senoo.append(beta[0])
    mdot_spak.append(mdot[stall_idx])
    beta_spak.append(beta[stall_idx])
ax.set_xlabel(r'$\dot{m}$ [kg/s]')
ax.set_ylabel(r'$\beta_{\rm ts}$')
ax.grid(alpha=.3)
ax.legend()
fig.savefig('pictures/iris_characteristic_curves.pdf')







set_thesis_style()
fig, ax = create_figure(fraction=0.6, aspect_ratio=1.3, subplots=(1,1))
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
    ax.plot(mdot[idx], beta[idx], label='%.1f krpm' %(rpm[i]/1000))
    ax.plot(mdot[0], beta[0], 'ks')
    ax.plot(mdot[stall_idx], beta[stall_idx], 'k^')
    mdot_senoo.append(mdot[0])
    beta_senoo.append(beta[0])
    mdot_spak.append(mdot[stall_idx])
    beta_spak.append(beta[stall_idx])
ax.plot(mdot_senoo, beta_senoo, '--sk', linewidth=0.5, label='Senoo')
ax.plot(mdot_spak, beta_spak, '--^k', linewidth=0.5, label='Spakovszky')
ax.set_xlabel(r'$\dot{m}$ [kg/s]')
ax.set_ylabel(r'$\beta_{\rm ts}$')
ax.grid(alpha=.4)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=3)
fig.savefig('pictures/iris_characteristic_curves_stall.pdf')


plt.show()
