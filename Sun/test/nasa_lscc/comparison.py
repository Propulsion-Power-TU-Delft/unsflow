2import pickle
import matplotlib.pyplot as plt
import Sun
from Sun.src.sun_model_multiblock import SunModelMultiBlock
from Grid.src.config import Config

data_file_1 = 'pictures/cfg09_20_20_30_20/eigenvalues.csv'
data_file_2 = 'pictures/cfg15_20_20_30_20/eigenvalues.csv'
data_file_3 = 'pictures/cfg12_20_20_30_20/eigenvalues.csv'

from numpy import genfromtxt
data1 = genfromtxt(data_file_1, delimiter=',', skip_header=1)
data2 = genfromtxt(data_file_2, delimiter=',', skip_header=1)
data3 = genfromtxt(data_file_3, delimiter=',', skip_header=1)

plt.figure()
plt.scatter(data1[:,0], data1[:,1], label='$\dot{m}=20.7 \ \mathrm{[kg/s]}$')
plt.scatter(data2[:,0], data2[:,1], label='$\dot{m}=20.0 \ \mathrm{[kg/s]}$')
# plt.scatter(data3[:,0], data3[:,1], label='$\dot{m}=17.8 \ \mathrm{[kg/s]}$')
plt.grid(alpha=0.2)
plt.legend()

plt.show()