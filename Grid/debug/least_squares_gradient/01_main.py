import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

from Grid.src.functions import compute_gradient_least_square

SELECTION = 2
color_map = 'jet'

def functions_test(X, Y):
    # return X ** 4 - X ** 2 * Y ** 2 + 3 * X ** 3 * Y - 6 * X * Y ** 2 - 2 * X + Y ** 2 + 1
    # return X+Y
    if SELECTION==1:
        f = np.exp(-1/X)+np.sin(X*Y)
        dfdx = np.exp(-1/X)/X**2 + Y*np.cos(X*Y)
        dfdy = X*np.cos(X*Y)
    elif SELECTION==2:
        f = X ** 4 - X ** 2 * Y ** 2 + 3 * X ** 3 * Y - 6 * X * Y ** 2 - 2 * X + Y ** 2 + 1
        dfdx = 4 * X ** 3 - 2 * X * Y ** 2 + 9 * X ** 2 * Y - 6 * Y ** 2 - 2
        dfdy = -X ** 2 * 2 * Y + 3 * X ** 3 - 12 * X * Y + 2 * Y
    elif SELECTION==3:
        f = X+Y
        dfdx = X/X
        dfdy = Y/Y
    else:
        raise ValueError("Test function not valid")
    return f, dfdx, dfdy



# GENERATE THE DATA, AND THE ANALYTIC RESULTS
with open('grid_128_192.pik', 'rb') as file:
    data = pickle.load(file)


X, Y = data['X'], data['Y']
nx, ny = X.shape
Z, DZDX, DZDY = functions_test(X, Y)
DZDX_LS, DZDY_LS = compute_gradient_least_square(X, Y, Z)


out_folder = 'Pics_%i_%i' %(nx,ny)
os.makedirs(out_folder, exist_ok=True)

fig, ax = plt.subplots(1, 2, figsize=(11, 5))
for i in range(nx):
    ax[0].plot(X[i,:], Y[i,:], 'k', lw=0.4)
for j in range(ny):
    ax[0].plot(X[:,j], Y[:,j], 'k', lw=0.4)
contour1 = ax[1].contourf(X, Y, Z, levels=50, cmap=color_map)
fig.colorbar(contour1, ax=ax[1])
ax[1].set_title(r'$f(x,y)$')
plt.savefig(out_folder+'/f_%i_%i.pdf' %(nx, ny), bbox_inches='tight')


fig, ax = plt.subplots(1, 3, figsize=(16, 5))
contour1 = ax[0].contourf(X, Y, DZDX, levels=50, cmap=color_map)
fig.colorbar(contour1, ax=ax[0])
ax[0].set_title(r'$\frac{\partial f}{\partial x}$')
contour2 = ax[1].contourf(X, Y, DZDX_LS, levels=50, cmap=color_map)
fig.colorbar(contour2, ax=ax[1])
ax[1].set_title(r'$\frac{\partial f_{wls}}{\partial x}$')
contour3 = ax[2].contourf(X, Y, (DZDX - DZDX_LS), levels=50, cmap=color_map)
fig.colorbar(contour3, ax=ax[2])
ax[2].set_title(r'$\varepsilon$')
for axx in ax:
    axx.set_xticks([])
    axx.set_yticks([])
plt.savefig(out_folder+'/dfdx_%i_%i.pdf' %(nx, ny), bbox_inches='tight')


fig, ax = plt.subplots(1, 3, figsize=(16, 5))
contour1 = ax[0].contourf(X, Y, DZDY, levels=50, cmap=color_map)
fig.colorbar(contour1, ax=ax[0])
ax[0].set_title(r'$\frac{\partial f}{\partial y}$')
contour2 = ax[1].contourf(X, Y, DZDY_LS, levels=50, cmap=color_map)
fig.colorbar(contour2, ax=ax[1])
ax[1].set_title(r'$\frac{\partial f_{wls}}{\partial y}$')
contour3 = ax[2].contourf(X, Y, (DZDY - DZDY_LS), levels=50, cmap=color_map)
fig.colorbar(contour3, ax=ax[2])
ax[2].set_title(r'$\varepsilon$')
for axx in ax:
    axx.set_xticks([])
    axx.set_yticks([])
plt.savefig(out_folder+'/dfdy_%i_%i.pdf' %(nx, ny), bbox_inches='tight')


plt.show()
