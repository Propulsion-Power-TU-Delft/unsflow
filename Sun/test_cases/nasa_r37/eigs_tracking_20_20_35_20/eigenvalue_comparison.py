import pickle
import matplotlib.pyplot as plt
import Sun
from Sun.src.sun_model_multiblock import SunModelMultiBlock
from Grid.src.config import Config
from numpy import genfromtxt

data = []
sims = ['p128', 'p128.5', 'p128.75', 'p128.9']
simsmass = [20.033, 19.771, 19.425, 19.338]
for file in sims:
    file += '/pictures/eigenvalues.csv'
    data.append(genfromtxt(file, delimiter=',', skip_header=1))

plt.figure()

for ii in range(len(data)):
    eigs = data[ii]
    plt.scatter(eigs[:, 0], eigs[:, 1], label='$\dot{m} = %.2f$ [kg/s]' %simsmass[ii])
# plt.scatter(data2[:,0], data2[:,1], label='$\dot{m}=20.0 \ \mathrm{[kg/s]}$')
# # plt.scatter(data3[:,0], data3[:,1], label='$\dot{m}=17.8 \ \mathrm{[kg/s]}$')
plt.grid(alpha=0.2)
plt.legend()
plt.xlabel('RS')
plt.ylabel('DF')
plt.savefig('tracking_eigs.pdf', bbox_inches='tight')

plt.show()
