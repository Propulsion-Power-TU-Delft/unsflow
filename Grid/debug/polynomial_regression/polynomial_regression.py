"""
POLYNOMIAL 2D REGRESSION. TAKEN FROM SLIDE 5, LECTURE 10-11, DDFM COURSE VKI
"""
import matplotlib.pyplot as plt
import numpy as np
import pickle

from Grid.src.polynomial_ls_regression import *
from Grid.src.styles import color_map

# FUNCTION TO REGRESS AND TEST. 4th ORDER SHOULD BE CAPTURED PERFECTLY

def random_function(X, Y):
    return X ** 4 - X ** 2 * Y ** 2 + 3 * X ** 3 * Y - 6 * X * Y ** 2 - 2 * X + Y ** 2 + 1
    # return X**4
    # return X + Y


def random_function_dx(X, Y):
    return 4 * X ** 3 - 2 * X * Y ** 2 + 9 * X ** 2 * Y - 6 * Y ** 2 - 2
    # return 4 * X ** 3
    # return np.zeros_like(X)+1

def random_function_dy(X, Y):
    return -X ** 2 * 2 * Y + 3 * X ** 3 - 12 * X * Y + 2 * Y
    # return X*0
    # return np.zeros_like(X)+1

# GENERATE THE DATA, READING FROM THE BLADE FILE
L = 5
H = 5
nx, ny = 50, 50
x = np.linspace(0, L, nx)
y = np.linspace(0, H, ny)
X, Y = np.meshgrid(x, y, indexing='ij')
Z = random_function(X, Y)
Zdx = random_function_dx(X, Y)
Zdy = random_function_dy(X, Y)

# USE THE FUNCTIONS IMPLEMENTED IN SRC
W = basis_function_matrix(X, Y, order=4)
Wdx, Wdy = basis_function_matrix_derivatives(W, X, Y)

# FIND THE COEFF VECTOR SOLUTION
coeff_vector = least_square_regression(W, Z)

# EVALUTE THE REGRESSED FIELD AND GRADIENTS
Znew = regression_evaluation(W, coeff_vector, nx, ny)
Znew_dx = regression_evaluation(Wdx, coeff_vector, nx, ny)
Znew_dy = regression_evaluation(Wdy, coeff_vector, nx, ny)

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
contour1 = ax[0].contourf(X, Y, Z, levels=50, cmap=color_map)
fig.colorbar(contour1, ax=ax[0])
ax[0].set_title(r'$z$')
contour2 = ax[1].contourf(X, Y, Znew, levels=50, cmap=color_map)
fig.colorbar(contour2, ax=ax[1])
ax[1].set_title(r'$z_R$')
fig.savefig('pictures/function_and_regression.pdf', bbox_inches='tight')


fig, ax = plt.subplots(2, 2, figsize=(12, 8))
contour1 = ax[0, 0].contourf(X, Y, Zdx, levels=50, cmap=color_map)
fig.colorbar(contour1, ax=ax[0, 0])
ax[0, 0].set_title(r'$\partial z / \partial x$')
contour2 = ax[0, 1].contourf(X, Y, Zdy, levels=50, cmap=color_map)
fig.colorbar(contour2, ax=ax[0, 1])
ax[0, 1].set_title(r'$\partial z / \partial y$')
contour3 = ax[1, 0].contourf(X, Y, Znew_dx, levels=50, cmap=color_map)
fig.colorbar(contour3, ax=ax[1, 0])
ax[1, 0].set_title(r'$\partial z_R / \partial x$')
contour4 = ax[1, 1].contourf(X, Y, Znew_dy, levels=50, cmap=color_map)
fig.colorbar(contour4, ax=ax[1, 1])
ax[1, 1].set_title(r'$\partial z_R / \partial y$')
fig.savefig('pictures/derivatives_and_regression.pdf', bbox_inches='tight')


# PLOT THE ERRORS
err = np.abs(Z - Znew)
plt.figure(figsize=(8, 5))
plt.contourf(X, Y, err, levels=50, cmap=color_map)
plt.colorbar()
plt.title(r'$z(x,y)$ error')
plt.savefig('pictures/function_error.pdf', bbox_inches='tight')


err_x = np.abs(Zdx - Znew_dx)
plt.figure(figsize=(8, 5))
plt.contourf(X, Y, err_x, levels=50, cmap=color_map)
plt.colorbar()
plt.title(r'$\partial z / \partial x$ error')
plt.savefig('pictures/dfdx_error.pdf', bbox_inches='tight')


err_y = np.abs(Zdy - Znew_dy)
plt.figure(figsize=(8, 5))
plt.contourf(X, Y, err_y, levels=50, cmap=color_map)
plt.colorbar()
plt.title(r'$\partial z / \partial y$ error')
plt.savefig('pictures/dfdy_error.pdf', bbox_inches='tight')

plt.show()
