"""
POLYNOMIAL 2D REGRESSION FUNCTIONS. TAKEN FROM SLIDE 5, LECTURE 10-11, DDFM COURSE VKI
"""
import numpy as np


def basis_function_matrix(X, Y, order=4):
    """
    given the matrices of cordinates x and y, build the basis function matrix, stacking
    all the basis column by column. Not elegant implementation but it works. Up to order 4, for a
    2-dimensional problem
    """
    x = X.flatten()
    y = Y.flatten()
    nrows = len(x)
    if order == 1:
        ncols = 3
        # build the columns
        W = np.zeros((nrows, ncols))
        W[:, 0] = np.zeros(nrows) + 1  # the constant column
        W[:, 1] = x
        W[:, 2] = y
    elif order == 2:
        ncols = 6
        W = np.zeros((nrows, ncols))
        W[:, 0] = np.zeros(nrows) + 1  # the constant column
        W[:, 1] = x
        W[:, 2] = y
        W[:, 3] = x ** 2
        W[:, 4] = y ** 2
        W[:, 5] = x * y
    elif order == 3:
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
    elif order == 4:
        ncols = 15
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
        W[:, 10] = x ** 4
        W[:, 11] = y ** 4
        W[:, 12] = x ** 3 * y
        W[:, 13] = x * y ** 3
        W[:, 14] = x ** 2 * y ** 2
    else:
        raise ValueError('Order of regression not recognized!')
    return W


def basis_function_matrix_derivatives(W, X, Y):
    """
    starting from the basis function matrix, calculate the basis function matrices corresponding to the x
    and y derivatives. only fourth order implemented since is the one used in the instability model
    """
    if np.shape(W)[1] != 15:
        raise ValueError('Only fourth order regression derivatives are implemented')

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
    Wdx[:, 10] = 4 * x ** 3
    Wdx[:, 11] = np.zeros(len(x))
    Wdx[:, 12] = 3 * x ** 2 * y
    Wdx[:, 13] = y ** 3
    Wdx[:, 14] = 2 * x * y ** 2

    # y-derivatives
    Wdy[:, 0] = np.zeros(len(x))
    Wdy[:, 1] = np.zeros(len(x))
    Wdy[:, 2] = np.zeros(len(x)) + 1
    Wdy[:, 3] = np.zeros(len(x))
    Wdy[:, 4] = 2 * y
    Wdy[:, 5] = x
    Wdy[:, 6] = np.zeros(len(x))
    Wdy[:, 7] = 3 * y ** 2
    Wdy[:, 8] = x ** 2
    Wdy[:, 9] = 2 * x * y
    Wdy[:, 10] = np.zeros(len(x))
    Wdy[:, 11] = 4 * y ** 3
    Wdy[:, 12] = x ** 3
    Wdy[:, 13] = 3 * x * y ** 2
    Wdy[:, 14] = 2 * x ** 2 * y
    return Wdx, Wdy


def least_square_regression(W, Z):
    """
    find the coefficient vector by means of the least square error minimization (analytical formula)
    """
    function_values = Z.flatten()
    weight_vector = np.linalg.inv(np.dot(W.T, W))
    weight_vector = np.dot(weight_vector, W.T)
    weight_vector = np.dot(weight_vector, function_values)
    return weight_vector


def regression_evaluation(W, coeff_vector, nx, ny):
    """
    evaluate the 2D function values matrix and return it in a good shape
    """
    z = np.dot(W, coeff_vector)
    Z = np.reshape(z, (nx, ny))
    return Z
