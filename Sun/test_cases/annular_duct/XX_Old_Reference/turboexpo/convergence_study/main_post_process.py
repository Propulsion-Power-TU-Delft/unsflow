import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import isdir
import pickle
from Utils.styles import *

# READ SIMULATION RESULTS
directory = './'
folders = [i for i in listdir(directory) if isdir(directory + i)]
folders.sort()
filenames = [directory + folder + '/data/meta/' + folder + '.pickle' for folder in folders]
data = []
names = []
for file in filenames:
    try:
        with open(file, 'rb') as file_pickle:
            data.append(pickle.load(file_pickle))
            names.append(file.split("/")[1])  # extract the name
    except:
        pass

# READ ANALYTICAL RESULTS
file = './analytical/analytical_eigenvalues.pickle'
with open(file, 'rb') as file_pickle:
    eigenvalues = pickle.load(file_pickle)
eigenvalues = eigenvalues[0:5]  # consider only the first 5 values

error = np.zeros((len(eigenvalues)+1, len(data)))
for i in range(len(eigenvalues)):
    eigenvalue = eigenvalues[i]
    for j in range(len(data)):
        numerical_results = data[j]
        error[i, j] = (np.min(np.abs(numerical_results - eigenvalue))) / eigenvalue
error[-1,:] = np.sum(error, axis=0)/error.shape[0]

plot_yticks = [r"$\varepsilon_{%i}$" %(s+1) for s in range(len(eigenvalues))]
plot_yticks.append(r'$\bar{\varepsilon}$')

plt.figure(figsize=(15,5))
plt.imshow(np.log10(error), cmap='coolwarm')
rows, cols = error.shape
for i in range(1, rows):
    plt.axhline(i - 0.5, color='white', linewidth=1)  # Add horizontal lines
for j in range(1, cols):
    plt.axvline(j - 0.5, color='white', linewidth=1)  # Add vertical lines
plt.xticks(range(len(names)), names, rotation=45, ha="right", rotation_mode="anchor", fontsize=font_labels-2)
plt.yticks(range(len(plot_yticks)), plot_yticks, fontsize=font_labels)
for i in range(len(names)):
    for j in range(len(plot_yticks)):
        plt.text(i, j, f'{error[j, i]:.0e}', ha='center', va='center', color='black', fontsize=18)
# plt.colorbar()
plt.savefig('doe_results.pdf', bbox_inches='tight')




# plt.figure(figsize=(7,7))
# for i in range(len(data)):
#     plt.scatter(data[i].real, data[i].imag, label=names[i])
# plt.legend()
# plt.xlabel(r'$\omega_R \ \mathrm{[rad/s]}$')
# plt.ylabel(r'$\omega_I \ \mathrm{[rad/s]}$')
plt.show()
