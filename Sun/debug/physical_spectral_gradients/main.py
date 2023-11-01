import numpy
import numpy as np
import matplotlib.pyplot as plt
from numpy import cos, sin, pi, linspace
from findiff import FinDiff

ALPHA_GRID = 0.2
N = 50  # number of grid points
L = 1

i = linspace(0, N - 1, N)

x = np.zeros(N)
xi = np.zeros(N)
for i in range(N):
    x[i] = L*i/(N-1)
    xi[i] = cos(i*pi/(N-1))

# plt.figure()
# plt.plot(x, xi, label='analytical')
# plt.plot(x, cos(x*pi/L), '--o', label='numerical')
# plt.xlabel('x')
# plt.ylabel(r'$\xi$')
# plt.legend()
# plt.grid(alpha=0.25)

#
# plt.figure()
# plt.plot(xi, x, label='analytical')
# plt.plot(xi, L/pi*np.arccos(xi), '--o', label='numerical')
# plt.xlabel(r'$\xi$')
# plt.ylabel(r'$x$')
# plt.legend()
# plt.grid(alpha=0.25)



dx_dxi = -L/pi/np.sqrt(1-xi**2)
dxi_dx = -sin(pi*x/L)*pi/L

dx_dxi_num = numpy.gradient(x, xi)
# dxi_dx_num = numpy.gradient(xi, x)
dxi_dx_num = 1/dx_dxi_num

d_dxi = FinDiff(0, xi, acc=4)
dx_dxi_num_findiff4 = d_dxi(x)
d_dx = FinDiff(0, x, acc=4)
# dxi_dx_num_findiff4 = d_dx(xi)
dxi_dx_num_findiff4 = 1/dx_dxi_num_findiff4

d_dxi = FinDiff(0, xi, acc=6)
dx_dxi_num_findiff6 = d_dxi(x)
d_dx = FinDiff(0, x, acc=6)
# dxi_dx_num_findiff6 = d_dx(xi)
dxi_dx_num_findiff6 = 1/dx_dxi_num_findiff6

d_dxi = FinDiff(0, xi, acc=8)
dx_dxi_num_findiff8 = d_dxi(x)
d_dx = FinDiff(0, x, acc=8)
# dxi_dx_num_findiff8 = d_dx(xi)
dxi_dx_num_findiff8 = 1/dx_dxi_num_findiff8

d_dxi = FinDiff(0, xi, acc=10)
dx_dxi_num_findiff10 = d_dxi(x)
d_dx = FinDiff(0, x, acc=10)
# dxi_dx_num_findiff10 = d_dx(xi)
dxi_dx_num_findiff10 = 1/dx_dxi_num_findiff10


plt.figure()
plt.plot(x[1:-1], dx_dxi[1:-1], label='analytical')
plt.plot(x, dx_dxi_num, '--o', label=r'$2^{nd}$ order', linewidth=0.5)
plt.plot(x, dx_dxi_num_findiff4, '--s', label=r'$4^{th}$ order', linewidth=0.5)
plt.plot(x, dx_dxi_num_findiff6, '--^', label=r'$6^{th}$ order', linewidth=0.5)
plt.plot(x, dx_dxi_num_findiff8, '--x', label=r'$8^{th}$ order', linewidth=0.5)
plt.plot(x, dx_dxi_num_findiff10, '--*', label=r'$10^{th}$ order', linewidth=0.5)
plt.xlabel('x')
plt.ylabel(r'$\partial x / \partial \xi$')
plt.legend()
plt.title(r'N:%i' %(N))
plt.grid(alpha=ALPHA_GRID)
plt.savefig('pictures/dx_dxi_%i.pdf' %(N), bbox_inches='tight')

plt.figure()
plt.plot(x, dxi_dx, label='analytical')
plt.plot(x, dxi_dx_num, '--o', label=r'$2^{nd}$ order', linewidth=0.5)
plt.plot(x, dxi_dx_num_findiff4, '--s', label=r'$4^{th}$ order', linewidth=0.5)
plt.plot(x, dxi_dx_num_findiff6, '--^', label=r'$6^{th}$ order', linewidth=0.5)
plt.plot(x, dxi_dx_num_findiff8, '--x', label=r'$8^{th}$ order', linewidth=0.5)
plt.plot(x, dxi_dx_num_findiff10, '--*', label=r'$10^{th}$ order', linewidth=0.5)
plt.xlabel('x')
plt.ylabel(r'$\partial \xi / \partial x$')
plt.legend()
plt.grid(alpha=ALPHA_GRID)
plt.title(r'N:%i' %(N))
plt.savefig('pictures/dxi_dx_%i.pdf' %(N), bbox_inches='tight')
# plt.show()