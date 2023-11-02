import numpy as np
from .styles import total_chars

def JacobianTransform(X, Y, Z, R):
    """
    It computes the jacobian of the transformation between two sets of cordinates. It uses central differences in the central
    points and first order at the borders.
    X,Y are the cordinates of which we want the gradients
    Z,R are the cordinates, used in the gradients. They need to be cartesian in order to be mathematically consistent
    
    It returns the gradients of the first as a function of the second cordinates
    dxdz, dxdr, dydz, dydr
    
    Double-checked, with script "debug_finite_diff.py"
    """
    Nz, Nr = X.shape[0], X.shape[1]

    # instantiate matrices
    dxdr = np.zeros((Nz, Nr))
    dxdz = np.zeros((Nz, Nr))
    dydr = np.zeros((Nz, Nr))
    dydz = np.zeros((Nz, Nr))

    # 2nd order central difference for the grid. First take care of the corners, then of the edges, and then of the central points
    for ii in range(0, Nz):
        for jj in range(0, Nr):
            # the points are considered watching at indexes where i increase towards the top, and j increase towards the right.

            if (ii == 0 and jj == 0):  # lower-left corner
                dxdz[ii, jj] = (X[ii + 1, jj] - X[ii, jj]) / (Z[ii + 1, jj] - Z[ii, jj])
                dxdr[ii, jj] = (X[ii, jj + 1] - X[ii, jj]) / (R[ii, jj + 1] - R[ii, jj])
                dydz[ii, jj] = (Y[ii + 1, jj] - Y[ii, jj]) / (Z[ii + 1, jj] - Z[ii, jj])
                dydr[ii, jj] = (Y[ii, jj + 1] - Y[ii, jj]) / (R[ii, jj + 1] - R[ii, jj])
            elif (ii == Nz - 1 and jj == 0):  # upper-left corner
                dxdz[ii, jj] = (X[ii, jj] - X[ii - 1, jj]) / (Z[ii, jj] - Z[ii - 1, jj])
                dxdr[ii, jj] = (X[ii, jj + 1] - X[ii, jj]) / (R[ii, jj + 1] - R[ii, jj])
                dydz[ii, jj] = (Y[ii, jj] - Y[ii - 1, jj]) / (Z[ii, jj] - Z[ii - 1, jj])
                dydr[ii, jj] = (Y[ii, jj + 1] - Y[ii, jj]) / (R[ii, jj + 1] - R[ii, jj])
            elif (ii == 0 and jj == Nr - 1):  # bottom-right corner
                dxdz[ii, jj] = (X[ii + 1, jj] - X[ii, jj]) / (Z[ii + 1, jj] - Z[ii, jj])
                dxdr[ii, jj] = (X[ii, jj] - X[ii, jj - 1]) / (R[ii, jj] - R[ii, jj - 1])
                dydz[ii, jj] = (Y[ii + 1, jj] - Y[ii, jj]) / (Z[ii + 1, jj] - Z[ii, jj])
                dydr[ii, jj] = (Y[ii, jj] - Y[ii, jj - 1]) / (R[ii, jj] - R[ii, jj - 1])
            elif (ii == Nz - 1 and jj == Nr - 1):  # upper-right corner
                dxdz[ii, jj] = (X[ii, jj] - X[ii - 1, jj]) / (Z[ii, jj] - Z[ii - 1, jj])
                dxdr[ii, jj] = (X[ii, jj] - X[ii, jj - 1]) / (R[ii, jj] - R[ii, jj - 1])
                dydz[ii, jj] = (Y[ii, jj] - Y[ii - 1, jj]) / (Z[ii, jj] - Z[ii - 1, jj])
                dydr[ii, jj] = (Y[ii, jj] - Y[ii, jj - 1]) / (R[ii, jj] - R[ii, jj - 1])
            elif (ii == 0):  # bottom side
                dxdz[ii, jj] = (X[ii + 1, jj] - X[ii, jj]) / (Z[ii + 1, jj] - Z[ii, jj])
                dxdr[ii, jj] = (X[ii, jj + 1] - X[ii, jj - 1]) / (R[ii, jj + 1] - R[ii, jj - 1])
                dydz[ii, jj] = (Y[ii + 1, jj] - Y[ii, jj]) / (Z[ii + 1, jj] - Z[ii, jj])
                dydr[ii, jj] = (Y[ii, jj + 1] - Y[ii, jj - 1]) / (R[ii, jj + 1] - R[ii, jj - 1])
            elif (ii == Nz - 1):  # top side
                dxdz[ii, jj] = (X[ii, jj] - X[ii - 1, jj]) / (Z[ii, jj] - Z[ii - 1, jj])
                dxdr[ii, jj] = (X[ii, jj + 1] - X[ii, jj - 1]) / (R[ii, jj + 1] - R[ii, jj - 1])
                dydz[ii, jj] = (Y[ii, jj] - Y[ii - 1, jj]) / (Z[ii, jj] - Z[ii - 1, jj])
                dydr[ii, jj] = (Y[ii, jj + 1] - Y[ii, jj - 1]) / (R[ii, jj + 1] - R[ii, jj - 1])
            elif (jj == 0):  # left side
                dxdz[ii, jj] = (X[ii + 1, jj] - X[ii - 1, jj]) / (Z[ii + 1, jj] - Z[ii - 1, jj])
                dxdr[ii, jj] = (X[ii, jj + 1] - X[ii, jj]) / (R[ii, jj + 1] - R[ii, jj])
                dydz[ii, jj] = (Y[ii + 1, jj] - Y[ii - 1, jj]) / (Z[ii + 1, jj] - Z[ii - 1, jj])
                dydr[ii, jj] = (Y[ii, jj + 1] - Y[ii, jj]) / (R[ii, jj + 1] - R[ii, jj])
            elif (jj == Nr - 1):  # right side
                dxdz[ii, jj] = (X[ii + 1, jj] - X[ii - 1, jj]) / (Z[ii + 1, jj] - Z[ii - 1, jj])
                dxdr[ii, jj] = (X[ii, jj] - X[ii, jj - 1]) / (R[ii, jj] - R[ii, jj - 1])
                dydz[ii, jj] = (Y[ii + 1, jj] - Y[ii - 1, jj]) / (Z[ii + 1, jj] - Z[ii - 1, jj])
                dydr[ii, jj] = (Y[ii, jj] - Y[ii, jj - 1]) / (R[ii, jj] - R[ii, jj - 1])
            else:  # all ohter internal points
                dxdz[ii, jj] = (X[ii + 1, jj] - X[ii - 1, jj]) / (Z[ii + 1, jj] - Z[ii - 1, jj])
                dxdr[ii, jj] = (X[ii, jj + 1] - X[ii, jj - 1]) / (R[ii, jj + 1] - R[ii, jj - 1])
                dydz[ii, jj] = (Y[ii + 1, jj] - Y[ii - 1, jj]) / (Z[ii + 1, jj] - Z[ii - 1, jj])
                dydr[ii, jj] = (Y[ii, jj + 1] - Y[ii, jj - 1]) / (R[ii, jj + 1] - R[ii, jj - 1])

    return dxdz, dxdr, dydz, dydr


def JacobianTransform2(X, Y, Z, R):
    """
    It computes the jacobian of the transformation between two sets of cordinates using the numpy gradient function.
    X,Y are the function values of which we want the gradients.
    Z,R are the cordinates. They need to be cartesian in order to be mathematically consistent

    It returns the gradients of the first as a function of the second cordinates
    dxdz, dxdr, dydz, dydr

    Double-checked, with script "debug_finite_diff.py"
    """
    # Compute gradients with variable spacing
    dxdz, dxdr = np.gradient(X, Z[:, 0], R[0, :], edge_order=2)
    dydz, dydr = np.gradient(Y, Z[:, 0], R[0, :], edge_order=2)
    return dxdz, dxdr, dydz, dydr


def JacobianTransform3(X, Y, Z, R, order=2):
    """
    It computes the jacobian of the transformation between two sets of cordinates using the findiff package.
    X,Y are the function values of which we want the gradients.
    Z,R are the cordinates. They need to be cartesian in order to be mathematically consistent

    It returns the gradients of the first as a function of the second cordinates
    dxdz, dxdr, dydz, dydr

    Double-checked, with script "debug_finite_diff.py"
    """
    # Compute gradients with variable spacing

    from findiff import FinDiff
    d_dz = FinDiff(0, Z[:, 0], acc=order)
    d_dr = FinDiff(1, R[0, :], acc=order)

    dxdz = d_dz(X)
    dxdr = d_dr(X)
    dydz = d_dz(Y)
    dydr = d_dr(Y)
    return dxdz, dxdr, dydz, dydr


def ChebyshevDerivativeMatrix(x):
    """
    Define the first order derivative Chebyshev matrix, where x is the array of Gauss-Lobatto points. Expression from 
    Peyret book, page 50
    
    double-checked with script "debug_spectral_diff.py"
    """
    N = len(x)  # dimension of the square matrix
    D = np.zeros((N, N))
    for i in range(0, N):
        for j in range(0, N):
            xi = np.cos(np.pi * i / (N - 1))
            xj = np.cos(np.pi * j / (N - 1))

            # select the right ci, cj
            if i == 0 or i == N - 1:
                ci = 2
            else:
                ci = 1
            if j == 0 or j == N - 1:
                cj = 2
            else:
                cj = 1

            # matrix coefficients
            if (i != j):
                D[i, j] = (ci / cj) * (-1) ** (i + j) / (xi - xj)
            elif (i == j and 0 < i < N - 1):
                D[i, j] = -xi / 2 / (1 - xi ** 2)
            elif (i == 0 and j == 0):
                D[i, j] = (2 * (N ** 2) + 1) / 6
            elif (i == N - 1 and j == N - 1):
                D[i, j] = -(2 * (N ** 2) + 1) / 6
            else:
                raise ValueError('Some mistake in the differentiation matrix')

    return D


def ChebyshevDerivativeMatrixBayliss(x):
    """
    Define the first order derivative Chebyshev matrix, where x is the array of Gauss-Lobatto points. Expression from 
    Peyret book, page 50. Bayliss formulation for the diagonal term, as suggested by Peyret to fix the extremes
    
    double-checked with script "debug_spectral_diff.py"
    """
    N = len(x)  # dimension of the square matrix
    D = ChebyshevDerivativeMatrix(x)  # basic
    for i in range(0, N):
        row_tot = np.sum(D[i, :]) - D[i, i]  # sum of the terms out of the diagonal
        D[i, i] = -row_tot  # to make the sum of the elements in a row = 0 for every row
    return D


def Refinement(x, add_points):
    """
    it returns a refined array, which has an additional number add_points of equispaced points in every interval
    of the original array.
    """
    refined_x = np.array(())
    n = len(x)
    for i in range(0, n - 1):
        refined_x = np.append(refined_x, x[i])  # insert the original point
        tmp_cord = np.linspace(x[i], x[i + 1], add_points + 2)
        tmp_cord = tmp_cord[1:-1]
        refined_x = np.append(refined_x, tmp_cord)
    refined_x = np.append(refined_x, x[-1])
    return refined_x


def GaussLobattoPoints(N):
    """
    it returns an array of N Gauss-Lobatto points, between 1 and -1.
    """
    x = np.array(())

    # the difference in notation with respect to mathematics books is due to the python index notation, starting from 0 and not 1
    for i in range(0, N):
        xnew = np.cos(i * np.pi / (N - 1))  # gauss lobatto points
        x = np.append(x, xnew)
    return x


def scaled_eigenvector_real(eig_list, Nz, Nr):
    array = np.array(eig_list, dtype=complex)
    array = np.reshape(array, (Nz, Nr))
    array_real_scaled = array.real / (np.max(array.real) - np.min(array.real))
    return array_real_scaled


def print_banner_begin(string):
    """
    print the banner begin, including string in the middle
    """
    n = total_chars - 2
    print("+", f"{string:-^{n}}", "+", sep='')


def print_banner_end(string=''):
    """
    print the banner end
    """
    n = total_chars - 2
    print("+", f"{string:-^{n}}", "+", sep='')
    # print("\n")
