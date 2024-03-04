import pickle
import matplotlib.pyplot as plt
import Sun
from Sun.src.sun_model_multiblock import SunModelMultiBlock
from Grid.src.config import Config
from numpy import genfromtxt

sim = ['p128', 'p128.5', 'p128.75', 'p128.9']
for case in sim:
    resolutions = ['10_10_17_10', '10_10_17_20', '15_15_28_15', '20_20_35_20']
    files = ['../eigs_tracking_' + res + '_regressed/' + case + '/pictures/eigenvalues.csv' for res in resolutions]

    data = []
    for file in files:
        data.append(genfromtxt(file, delimiter=',', skip_header=1))

    markers = ['>', 'o', '<', 's']
    colors = ['red', 'green', 'blue', 'purple']
    plt.figure()
    for ii in range(len(data)):
        eigs = data[ii]
        plt.scatter(eigs[:, 0], eigs[:, 1], label=resolutions[ii], marker=markers[ii], s=50, facecolors='none',
                    edgecolors=colors[ii])
    plt.grid(alpha=0.2)
    plt.legend(bbox_to_anchor=(0, 1.25), loc='upper left', ncol=2)
    plt.xlabel('RS')
    plt.ylabel('DF')
    plt.savefig('regressed_simulations_%s.pdf' %case, bbox_inches='tight')

# plt.show()
