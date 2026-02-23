import numpy as np
import matplotlib.pyplot as plt
from grid.src.functions import transfinite_grid_generation, eriksson_stretching_function_initial, \
    eriksson_stretching_function_final, eriksson_stretching_function_both
import pickle

x = np.linspace(0, 1, 50)
ALPHA = np.array([1, 2, 5, 10, 15])


plt.figure()
for alpha in ALPHA:
    y = eriksson_stretching_function_initial(x, alpha)
    plt.plot(x, y, '-o', markersize=3, label=r'$\alpha=%i$' % alpha)
plt.xlabel(r'$x$')
plt.ylabel(r'$f_1(x)$')
plt.grid(alpha=0.3)
plt.legend()
plt.savefig('erikkson_initial.pdf', bbox_inches='tight')

plt.figure()
for alpha in ALPHA:
    y = eriksson_stretching_function_final(x, alpha)
    plt.plot(x, y, '-o', markersize=3, label=r'$\alpha=%i$' % alpha)
plt.xlabel(r'$x$')
plt.ylabel(r'$f_2(x)$')
plt.grid(alpha=0.3)
plt.legend()
plt.savefig('erikkson_final.pdf', bbox_inches='tight')

plt.figure()
for alpha in ALPHA:
    y = eriksson_stretching_function_both(x, alpha)
    plt.plot(x, y, '-o', markersize=3, label=r'$\alpha=%i$' % alpha)
plt.xlabel(r'$x$')
plt.ylabel(r'$f_3(x)$')
plt.grid(alpha=0.3)
plt.legend()
plt.savefig('erikkson_both.pdf', bbox_inches='tight')


plt.show()
