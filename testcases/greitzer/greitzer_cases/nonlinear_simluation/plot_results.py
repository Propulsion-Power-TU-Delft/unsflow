import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.optimize import fsolve
from unsflow.utils.plot_styles import *
import pickle
from greitzer.src.greitzer import Greitzer
from greitzer.src.config import Config


with open('results/unstable.pkl', 'rb') as f:
    greitzer = pickle.load(f)

plt.figure()
plt.plot(greitzer.xi, greitzer.solutionGreitzer[:,0], label=r'$\Phi_{\rm c}$')
plt.plot(greitzer.xi, greitzer.solutionGreitzer[:,1], label=r'$\Phi_{\rm t}$')
plt.plot(greitzer.xi, greitzer.solutionGreitzer[:,2], label=r'$\Psi_{\rm c}$')
plt.xlabel(r'$\xi$')
plt.xlim(right=75)
plt.legend()
plt.grid(alpha=0.2)
plt.savefig('pics/greitzer_signals_together.pdf', bbox_inches='tight')


plt.figure()
plt.plot(greitzer.phi, greitzer.psi_c, label=r'Compressor')
plt.plot(greitzer.phi, greitzer.psi_v, label=r'Throttle')
plt.plot(greitzer.solutionGreitzer[:,0], greitzer.solutionGreitzer[:,2], '-.k', label=r'Solution')
plt.plot(greitzer.solutionGreitzer[0,0], greitzer.solutionGreitzer[0,2], 'ok')
plt.grid(alpha=0.2)
plt.xlabel(r'$\Phi$')
plt.ylabel(r'$\Psi$')
plt.xlim([-0.2,0.65])
plt.ylim([0.15,0.8])
plt.legend()
plt.savefig('pics/greitzer_limit_cycle.pdf', bbox_inches='tight')
plt.show()



