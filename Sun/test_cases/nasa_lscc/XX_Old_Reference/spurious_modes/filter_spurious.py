import pickle
import matplotlib.pyplot as plt
import Sun
from sun.src.sun_model_multiblock import SunModelMultiBlock
from grid.src.config import Config
from numpy import genfromtxt

markers = ['v', '^', '<', '>', 'o', 's', 'D']
colors = ['red', 'green', 'blue', 'purple', 'black', 'orange', 'cyan', 'magenta', 'yellow']

sims = ['20_40_30_ur', '20_40_30_ut', '20_40_30_uz']
data = []
for sim in sims:
    files = ['../' + sim + '/pictures/eigenvalues.csv']
    for file in files:
        data.append(genfromtxt(file, delimiter=',', skip_header=1))

plt.figure()
for ii,datasim in enumerate(data):
    eigs = datasim
    plt.scatter(eigs[:, 0], eigs[:, 1], label=sims[ii], marker=markers[ii], s=50, edgecolors=colors[ii], facecolors='none')
    plt.grid(alpha=0.2)
    plt.legend(bbox_to_anchor=(0, 1.35), loc='upper left', ncol=2)
    plt.xlabel('RS')
    plt.ylabel('DF')
    plt.savefig('pictures/simulations.pdf', bbox_inches='tight')

plt.show()
