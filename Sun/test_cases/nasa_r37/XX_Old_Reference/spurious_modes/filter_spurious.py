import pickle
import matplotlib.pyplot as plt
import Sun
from sun.src.sun_model_multiblock import SunModelMultiBlock
from grid.src.config import Config
from numpy import genfromtxt

sim = ['p128', 'p128.5', 'p128.75', 'p128.9']
for case in sim:
    resolutions = ['15_15_21_15_collocation_radial_force', '21_21_25_21_collocation_radial_force', '25_25_31_25_collocation_radial_force']
    files = [ '../' + res + '/' + case + '/pictures/eigenvalues.csv' for res in resolutions]

    data = []
    for file in files:
        data.append(genfromtxt(file, delimiter=',', skip_header=1))

    markers = ['v', '^', '<', '>', 'o', 's', 'D']
    colors = ['red', 'green', 'blue', 'purple', 'black', 'orange', 'cyan', 'magenta', 'yellow']
    plt.figure()
    for ii in range(len(data)):
        eigs = data[ii]
        plt.scatter(eigs[:, 0], eigs[:, 1], label=resolutions[ii], marker=markers[ii], s=50, facecolors='none',
                    edgecolors=colors[ii])
    plt.grid(alpha=0.2)
    plt.legend(bbox_to_anchor=(0, 1.35), loc='upper left', ncol=2)
    plt.xlabel('RS')
    plt.ylabel('DF')
    plt.savefig('pictures/simulations_%s.pdf' %case, bbox_inches='tight')

plt.show()
