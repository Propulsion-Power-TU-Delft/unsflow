"""
POLYNOMIAL 2D REGRESSION FUNCTIONS. TAKEN FROM SLIDE 5, LECTURE 10-11, DDFM COURSE VKI
"""
import numpy as np
from numpy.polynomial.chebyshev import chebvander2d


def basis_function_matrix(X, Y, order=4):
    """
    given the matrices of cordinates x and y, build the basis function matrix, stacking
    all the basis column by column. Not elegant implementation but it works. Up to order 4, for a
    2-dimensional problem.
    :param X: grid of X cordinates (along the axis of the machine)
    :param Y: grid of Y cordinates (along the radius of the machine)
    :param order: order of the regression. 4 is the values used in the literature.
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
    Starting from the basis function matrix, calculate the basis function matrices corresponding to the x
    and y derivatives. only fourth order implemented since is the one used in the instability model.
    :param W: basis function matrix corresponding to X and Y cordinate grids.
    :param X: grid of X cordinates (along the axis of the machine)
    :param Y: grid of Y cordinates (along the radius of the machine)
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
    :param W: basis function matrix corresponding to X and Y cordinate grids.
    :param Z: function values to regress, coherent with X and Y cordinate grids.
    """
    function_values = Z.flatten()
    weight_vector = np.linalg.inv(np.dot(W.T, W))
    weight_vector = np.dot(weight_vector, W.T)
    weight_vector = np.dot(weight_vector, function_values)
    return weight_vector


def regression_evaluation(W, coeff_vector, nx, ny):
    """
    Evaluate the 2D function values matrix and return it in the right 2D shape.
    :param W: basis function matrix corresponding to X and Y cordinate grids.
    :param coeff_vector: coefficient vector found with least-square-based regression.
    :param nx: number of streamwise points in the original grid.
    :param ny: numebr of spanwise points in the original grid.
    """
    z = np.dot(W, coeff_vector)
    Z = np.reshape(z, (nx, ny))
    return Z

def chebyshev_polynomial(k, x):
    """
    Definition by recursion.
    :param k: order of the polynomial
    :param x: evaluation points
    """
    if k == 0:
        return np.ones_like(x)
    elif k == 1:
        return x
    else:
        return 2 * x * chebyshev_polynomial(k - 1, x) - chebyshev_polynomial(k - 2, x)


def chebyshev_derivative_recursive(k, x):
    """
    Definition by recursion.
    :param k: order of the polynomial
    :param x: evaluation points
    """
    if k == 0:
        return np.zeros_like(x)
    elif k == 1:
        return np.ones_like(x)
    else:
        return 2 * chebyshev_polynomial(k - 1, x) + 2 * x * chebyshev_derivative_recursive(k - 1,
                                                                                           x) - chebyshev_derivative_recursive(
            k - 2, x)


def compute_derivative_matrices_chebyshev(degrees, x, y):
    """
    Build the Van Der Monde matrices of the x and y derivatives.
    :param degrees: (x,y) degrees
    :param x: evaluation points
    :param y: evaluation points
    """
    n_samples = len(x)
    DX = np.zeros((n_samples, (degrees[0] + 1) * (degrees[1] + 1)))
    DY = np.zeros((n_samples, (degrees[0] + 1) * (degrees[1] + 1)))
    for i in range(degrees[0] + 1):
        for j in range(degrees[1] + 1):
            k = j + i * (degrees[0] + 1)
            DX[:, k] = chebyshev_derivative_recursive(i, x) * chebyshev_polynomial(j, y)
            DY[:, k] = chebyshev_polynomial(i, x) * chebyshev_derivative_recursive(j, y)
    return DX, DY

