"""
POLYNOMIAL 2D REGRESSION. TAKEN FROM SLIDE 5, LECTURE 10-11, DDFM COURSE VKI
"""
import matplotlib.pyplot as plt
import numpy as np
import pickle


# FUNCTION TO REGRESS
def cubic_monomyal(X, Y):
    return X ** 3 + Y ** 2


def monomyal_monomyal(X, Y):
    return X + Y


def quadratic_monomyal(X, Y):
    return X ** 2 + Y


def quadratic_quadratic(X, Y):
    return 3 * X - X ** 2 + 3 - 6 * Y + Y ** 2


def random_function(X, Y):
    return X**3 - Y**2 + 3*X + 6*X*Y - 2*X**2*Y + 1


def random_function_dx(X, Y):
    return 3*X**2 + 3 + 6*Y - 4*X*Y


def random_function_dy(X, Y):
    return -2*Y + 6*X - 2*X**2



# GENERATE THE DATA, READING FROM THE BLADE FILE
L = 5
H = 5
nx, ny = 50, 100
x = np.linspace(0, L, nx)
y = np.linspace(0, H, ny)
X, Y = np.meshgrid(x, y, indexing='ij')
Z = random_function(X, Y)
Zdx = random_function_dx(X, Y)
Zdy = random_function_dy(X, Y)


# BUILD THE BASIS FUNCTION MATRIX W(x, y)
def basis_function_matrix(X, Y, order_x=1, order_y=1):
    x = X.flatten()
    y = Y.flatten()
    nrows = len(x)
    if order_x == 1 and order_y == 1:
        ncols = 3
        # build the columns
        W = np.zeros((nrows, ncols))
        W[:, 0] = np.zeros(nrows) + 1  # the constant column
        W[:, 1] = x
        W[:, 2] = y
    elif order_x == 2 and order_y == 2:
        ncols = 6
        W = np.zeros((nrows, ncols))
        W[:, 0] = np.zeros(nrows) + 1  # the constant column
        W[:, 1] = x
        W[:, 2] = y
        W[:, 3] = x ** 2
        W[:, 4] = y ** 2
        W[:, 5] = x * y
    elif order_x == 3 and order_y == 3:
        ncols = 10
        W = np.zeros((nrows, ncols))
        W[:, 0] = np.zeros(nrows) + 1  # the constant column
        W[:, 1] = x
        W[:, 2] = y
        W[:, 3] = x ** 2
        W[:, 4] = y ** 2
        W[:, 5] = x * y
        W[:, 6] = x ** 3
        W[:, 7] = y ** 3
        W[:, 8] = x ** 2 * y
        W[:, 9] = x * y ** 2
    return W


def basis_function_matrix_derivatives(W, X, Y):
    """
    starting from the basis function matrix, calculate the basis function matrices corresponding to the x
    and y derivatives. only third order implemented
    """
    if np.shape(W)[1] != 10:
        raise ValueError('Only third order regression derivatives are implemented')

    x = X.flatten()
    y = Y.flatten()

    Wdx = np.zeros_like(W)
    Wdy = np.zeros_like(W)

    # x-derivatives
    Wdx[:, 0] = np.zeros(len(x))
    Wdx[:, 1] = np.zeros(len(x)) + 1
    Wdx[:, 2] = np.zeros(len(x))
    Wdx[:, 3] = 2 * x
    Wdx[:, 4] = np.zeros(len(x))
    Wdx[:, 5] = y
    Wdx[:, 6] = 3 * x ** 2
    Wdx[:, 7] = np.zeros(len(x))
    Wdx[:, 8] = 2 * x * y
    Wdx[:, 9] = y ** 2

    # y-derivatives
    Wdy[:, 0] = np.zeros(len(x))
    Wdy[:, 1] = np.zeros(len(x))
    Wdy[:, 2] = y
    Wdy[:, 3] = np.zeros(len(x))
    Wdy[:, 4] = 2 * y
    Wdy[:, 5] = x
    Wdy[:, 6] = np.zeros(len(x))
    Wdy[:, 7] = 3 * y ** 2
    Wdy[:, 8] = x ** 2
    Wdy[:, 9] = 2 * x * y

    return Wdx, Wdy


order_x = 3
order_y = 3
W = basis_function_matrix(X, Y, order_x, order_y)


def least_square_regression(W, Z):
    function_values = Z.flatten()
    weight_vector = np.linalg.inv(np.dot(W.T, W))
    weight_vector = np.dot(weight_vector, W.T)
    weight_vector = np.dot(weight_vector, function_values)
    return weight_vector


coeff_vector = least_square_regression(W, Z)
print(coeff_vector)


def regression_evaluation(W, coeff_vector):
    z = np.dot(W, coeff_vector)
    Z = np.reshape(z, (nx, ny))
    return Z


Znew = regression_evaluation(W, coeff_vector)
Wdx, Wdy = basis_function_matrix_derivatives(W, X, Y)
Znew_dx = regression_evaluation(Wdx, coeff_vector)
Znew_dy = regression_evaluation(Wdy, coeff_vector)

fig, ax = plt.subplots(1, 2, figsize=(12, 8))
contour1 = ax[0].contourf(X, Y, Z, levels=50)
fig.colorbar(contour1, ax=ax[0])
ax[0].set_title(r'$Z$')
contour2 = ax[1].contourf(X, Y, Znew, levels=50)
fig.colorbar(contour2, ax=ax[1])
ax[1].set_title(r'$Z_r$')
# plt.show()

fig, ax = plt.subplots(2, 2, figsize=(12, 8))
contour1 = ax[0, 0].contourf(X, Y, Zdx, levels=50)
fig.colorbar(contour1, ax=ax[0, 0])
ax[0, 0].set_title(r'$\partial Z / \partial z$')
contour2 = ax[0, 1].contourf(X, Y, Zdy, levels=50)
fig.colorbar(contour2, ax=ax[0, 1])
ax[0, 1].set_title(r'$\partial Z / \partial r$')
contour3 = ax[1, 0].contourf(X, Y, Znew_dx, levels=50)
fig.colorbar(contour3, ax=ax[1, 0])
ax[1, 0].set_title(r'$\partial Z_r / \partial z$')
contour4 = ax[1, 1].contourf(X, Y, Znew_dy, levels=50)
fig.colorbar(contour4, ax=ax[1, 1])
ax[1, 1].set_title(r'$\partial Z_r / \partial r$')
plt.show()
