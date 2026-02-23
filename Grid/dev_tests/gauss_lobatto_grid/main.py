import numpy as np
import matplotlib.pyplot as plt
from sun.src.general_functions import GaussLobattoPoints

Nz = 25
Nr = 15

xi = GaussLobattoPoints(Nz)
eta = GaussLobattoPoints(Nr)

XI, ETA = np.meshgrid(xi, eta, indexing='ij')

plt.figure()
for ii in range(Nz):
    plt.plot(XI[ii, :], ETA[ii, :], 'k', lw=1)
for jj in range(Nr):
    plt.plot(XI[:, jj], ETA[:, jj], 'k', lw=1)
plt.xticks([])
plt.yticks([])
plt.savefig('pictures/gauss_lobatto_grid_%02i_%02i.pdf' %(Nz, Nr), bbox_inches='tight')
plt.show()