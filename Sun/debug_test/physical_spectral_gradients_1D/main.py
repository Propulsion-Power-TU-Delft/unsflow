import numpy
import numpy as np
import matplotlib.pyplot as plt
from numpy import cos, sin, pi, linspace
from findiff import FinDiff
from utils.styles import *

ALPHA_GRID = 0.2
N = 20  # number of grid points
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

# refined version of the analytical case to show the singularities at the extremities
xi_refined = np.linspace(-0.999, 0.999, 1000)
dx_dxi_refined = -L/pi/np.sqrt(1-xi_refined**2)

plt.figure()
plt.plot(xi_refined, dx_dxi_refined, label='Analytical', linewidth=1.5)
plt.plot(xi, dx_dxi_num, ':o', label=r'$2^{nd}$ order', linewidth=1.5, mfc='none')
plt.plot(xi, dx_dxi_num_findiff4, ':s', label=r'$4^{th}$ order', linewidth=1.5, mfc='none')
plt.plot(xi, dx_dxi_num_findiff6, ':^', label=r'$6^{th}$ order', linewidth=1.5, mfc='none')
plt.plot(xi, dx_dxi_num_findiff8, ':P', label=r'$8^{th}$ order', linewidth=1.5, mfc='none')
plt.plot(xi, dx_dxi_num_findiff10, ':D', label=r'$10^{th}$ order', linewidth=1.5, mfc='none')
plt.xlabel(r'$\xi$', fontsize=font_labels)
plt.ylabel(r'$d x / d \xi$', fontsize=font_labels)
plt.xticks(fontsize=font_axes)
plt.yticks(fontsize=font_axes)
plt.ylim([-5.5, 0])
plt.legend(fontsize=font_legend)
# plt.title(r'N:%i' %(N), fontsize=font_title)
plt.grid(alpha=ALPHA_GRID)
plt.savefig('pictures/dx_dxi_%i.pdf' %(N), bbox_inches='tight')
#
# plt.figure()
# plt.plot(x, dxi_dx, label='analytical')
# plt.plot(x, dxi_dx_num, '--o', label=r'$2^{nd}$ order', linewidth=0.5)
# plt.plot(x, dxi_dx_num_findiff4, '--s', label=r'$4^{th}$ order', linewidth=0.5)
# plt.plot(x, dxi_dx_num_findiff6, '--^', label=r'$6^{th}$ order', linewidth=0.5)
# plt.plot(x, dxi_dx_num_findiff8, '--x', label=r'$8^{th}$ order', linewidth=0.5)
# plt.plot(x, dxi_dx_num_findiff10, '--*', label=r'$10^{th}$ order', linewidth=0.5)
# plt.xlabel('x')
# plt.ylabel(r'$\partial \xi / \partial x$')
# plt.legend()
# plt.grid(alpha=ALPHA_GRID)
# plt.title(r'N:%i' %(N))
# plt.savefig('pictures/dxi_dx_%i.pdf' %(N), bbox_inches='tight')


degree = 3
coefficients = np.polyfit(xi_refined, dx_dxi_refined, degree)
y_interpolate = np.interp(xi, xi_refined, dx_dxi_refined)



plt.figure(figsize=(7, 5.5))
# plt.plot(xi_refined, dx_dxi_refined, label='analytical', linewidth=line_width)
# plt.plot(xi, y_interpolate, label='interpolated', linewidth=line_width)
plt.plot(xi, (dx_dxi_num-y_interpolate)/y_interpolate, '--o', label=r'$2^{nd}$ order', linewidth=medium_line_width,
         markersize=marker_size)
plt.plot(xi, (dx_dxi_num_findiff4-y_interpolate)/y_interpolate, '--s', label=r'$4^{th}$ order',
         linewidth=medium_line_width, markersize=marker_size)
plt.plot(xi, (dx_dxi_num_findiff6-y_interpolate)/y_interpolate, '--^', label=r'$6^{th}$ order',
         linewidth=medium_line_width, markersize=marker_size)
plt.plot(xi, (dx_dxi_num_findiff8-y_interpolate)/y_interpolate, '--P', label=r'$8^{th}$ order',
         linewidth=medium_line_width, markersize=marker_size)
plt.plot(xi, (dx_dxi_num_findiff10-y_interpolate)/y_interpolate, '--D', label=r'$10^{th}$ order',
         linewidth=medium_line_width, markersize=marker_size)
plt.xlabel(r'$\xi$ [-]', fontsize=font_labels)
plt.ylabel(r'$\varepsilon$ [-]', fontsize=font_labels)
plt.xticks(fontsize=font_axes)
plt.yticks(fontsize=font_axes)
# plt.ylim([-5.5, 0])
# plt.xlim([0,1.05])
plt.legend(fontsize=font_legend)
# plt.title(r'N:%i' %(N), fontsize=font_title)
plt.grid(alpha=ALPHA_GRID)
plt.savefig('pictures/dx_dxi_err_relative%i.pdf' %(N), bbox_inches='tight')


plt.show()