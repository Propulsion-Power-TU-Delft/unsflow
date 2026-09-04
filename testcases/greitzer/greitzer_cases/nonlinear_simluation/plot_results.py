import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.optimize import fsolve
# from unsflow.utils.plot_styles import *
import pickle
from unsflow.greitzer.greitzer import Greitzer
from unsflow.greitzer.config import Config
from unsflow.utils.thesis_plots import create_figure, set_thesis_style

with open('results/unstable.pkl', 'rb') as f:
    greitzer = pickle.load(f)

set_thesis_style()
fig, ax = create_figure(fraction=1.00, aspect_ratio=1.3, subplots=(1, 2))

ax[0].plot(greitzer.phi, greitzer.psi_c, label=r'Compressor')
ax[0].plot(greitzer.phi, greitzer.psi_v, label=r'Throttle')
ax[0].plot(greitzer.solutionGreitzer[:,0], greitzer.solutionGreitzer[:,2], '-.k', label=r'Solution')
ax[0].plot(greitzer.solutionGreitzer[0,0], greitzer.solutionGreitzer[0,2], 'ok')
ax[0].grid(alpha=0.2)
ax[0].set_xlabel(r'$\Phi$')
ax[0].set_ylabel(r'$\Psi$')
ax[0].set_xlim([-0.2,0.65])
ax[0].set_ylim([0.15,0.8])
ax[0].legend()

ax[1].plot(greitzer.xi, greitzer.solutionGreitzer[:,0]/greitzer.solutionGreitzer[0,0], '-', label=r'$\Phi_{\rm c}/\Phi_{\rm c0}$')
ax[1].plot(greitzer.xi, greitzer.solutionGreitzer[:,1]/greitzer.solutionGreitzer[0,1], '--', label=r'$\Phi_{\rm t}/\Phi_{\rm t0}$')
ax[1].plot(greitzer.xi, greitzer.solutionGreitzer[:,2]/greitzer.solutionGreitzer[0,2], '-.', label=r'$\Psi_{\rm c}/\Psi_{\rm c0}$')
ax[1].set_ylabel(r'$\mathbf{x} / \mathbf{x}_{0}$')
ax[1].set_xlabel(r'$\xi$')
ax[1].set_xlim(right=80)
ax[1].legend()
ax[1].grid(alpha=0.2)


fig.savefig('pics/greitzer_limit_cycle.pdf')
plt.show()



