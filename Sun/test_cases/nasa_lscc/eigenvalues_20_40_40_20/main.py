import matplotlib.pyplot as plt
from numpy import genfromtxt
import os

current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)
files_and_folders = os.listdir(parent_directory)
simulations = [folder for folder in files_and_folders if os.path.isdir(os.path.join(parent_directory, folder))
               and folder[0:11] == '20_40_40_20']

data = []
labels = [name for name in simulations]
for file in simulations:
    datafile = '../' + file + '/pictures/eigenvalues.csv'
    data.append(genfromtxt(datafile, delimiter=',', skip_header=1))

plt.figure()
for ii in range(len(data)):
    eigs = data[ii]
    plt.plot(eigs[:, 0], eigs[:, 1], 's', label='%s' %labels[ii], mfc='none')
plt.grid(alpha=0.2)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)
plt.xlabel('RS')
plt.ylabel('DF')
plt.savefig('eig_map.pdf', bbox_inches='tight')

# plt.show()
