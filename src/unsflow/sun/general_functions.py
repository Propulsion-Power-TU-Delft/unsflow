import numpy as np
from findiff import FinDiff
import pandas as pd
import os
import pickle


def JacobianTransform_hardcoded(X, Y, Z, R):
    """
    It computes the jacobian of the transformation between two sets of cordinates. It uses central differences in the central
    points and first order at the borders.
    :param X: x cordinates of which we want the gradients
    :param Y: y cordinates of which we want the gradients
    :param Z: z cordinates of which we want the gradients
    :param R: r cordinates of which we want the gradients
    Z,R are the cordinates, used in the gradients. The latter need to be cartesian in order to be mathematically consistent
    It returns the gradients of the first as a function of the second cordinates dxdz, dxdr, dydz, dydr.
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


def JacobianTransform_numpy(X, Y, Z, R):
    """
    It computes the jacobian of the transformation between two sets of cordinates, using numpy.gradient().
    :param X: x cordinates of which we want the gradients (physical grid)
    :param Y: y cordinates of which we want the gradients (physical grid)
    :param Z: z cordinates of which we want the gradients (computational grid)
    :param R: r cordinates of which we want the gradients (computational grid)
    Z,R are the cordinates, used in the gradients. They need to be cartesian grids in order to be mathematically consistent.
    It returns the gradients of the first as a function of the second cordinates dxdz, dxdr, dydz, dydr.
    """
    # Compute gradients with variable spacing
    dxdz, dxdr = np.gradient(X, Z[:, 0], R[0, :], edge_order=2)
    dydz, dydr = np.gradient(Y, Z[:, 0], R[0, :], edge_order=2)
    return dxdz, dxdr, dydz, dydr


def JacobianTransform_findiff(X, Y, Z, R, order=2):
    """
    It computes the jacobian of the transformation between two sets of cordinates, using findiff library.
    :param X: x cordinates of which we want the gradients
    :param Y: y cordinates of which we want the gradients
    :param Z: z cordinates of which we want the gradients
    :param R: r cordinates of which we want the gradients
    :param order: finite difference order
    Z,R are the cordinates, used in the gradients. The latter need to be cartesian in order to be mathematically consistent
    It returns the gradients of the first as a function of the second cordinates dxdz, dxdr, dydz, dydr.
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
    Define the first order derivative Chebyshev operator, where x is the array of Gauss-Lobatto points along x.
    Expression from Peyret book, page 50.
    :param x: array of cordinates
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
    Define the first order derivative Chebyshev operator, where x is the array of Gauss-Lobatto points. Expression from
    Peyret book, page 50. Bayliss formulation for the diagonal term, as suggested by Peyret to fix the extremes.
    :param x: array of cordinates
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
    It returns a refined array, which has an additional number add_points of equispaced points in every interval
    of the original array.
    :param x: original array
    :param add_points: number of points to add in every interval
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
    It returns an array of ordered Gauss-Lobatto points, between 1 and -1.
    :param N: number of points (=maximum chebyshev polynomial order)
    """
    # x = np.array(())
    # # the difference in notation with respect to mathematics books is due to the python index notation, starting from 0 and not 1
    # for i in range(0, N):
    #     xnew = np.cos(i * np.pi / (N - 1))  # gauss lobatto points
    #     x = np.append(x, xnew)

    x = np.array([np.cos(i * np.pi / (N - 1)) for i in range(0, N)])
    return x


def scaled_eigenvector_real(eig_list, Nz, Nr):
    """
    Converts and scales an eigenvalue with max-min algorithm, to return a 2D eigenfunction.
    :param eig_list: list contaning eigenfunction
    :param Nz: number of z points.
    :param Nr: number of r points.
    """
    array = np.array(eig_list, dtype=complex)
    array = np.reshape(array, (Nz, Nr))
    array_real_scaled = array.real / (np.max(array.real) - np.min(array.real))
    return array_real_scaled

def annular_duct_analytical_transformation(z, L1, L2):
    """
    Analytical transformation for the annular problem (a rectangular one, evenly spaces in both z and r).
    L1,L2 represent the extremes. z is the physical array cordinate.
    It can be used for both axial and radial transformation, using proper inputs.
    """
    dcomputational_dphysical = -np.sin(np.pi * (z - L1) / (L2 - L1)) * np.pi / (L2 - L1)
    return dcomputational_dphysical


def enlarge_square_matrices(A_list):
    """
    Given a list of square arrays, construct a single one matrix including those arrays blocks on the diagonal
    """
    tot_rows = 0
    tot_cols = 0
    for A in A_list:
        tot_rows += A.shape[0]
        tot_cols += A.shape[0]
    A_g = np.zeros((tot_rows, tot_cols), dtype=complex)
    counter = 0
    for A in A_list:
        A_g[counter:counter + A.shape[0], counter:counter + A.shape[1]] = A
        counter += A.shape[0]
    return A_g


def gauss_lobatto_grid_generation(N, x_start, x_end):
    """
    Return the array of points distributed following gauss-lobatto structure
    """
    xi = np.zeros(N)
    for ii in range(len(xi)):
        xi[ii] = x_start + (x_end-x_start)*(1-np.cos(np.pi*ii/(N-1)))/2
    return xi


def write_gridfile_from_cturbobfm(filepath, outputName=None):
    """Take the path to results obtained with cturbobfm, supplemented by gradients, and produce the pkl file
    needed from the Sun module

    Args:
        filepath (_type_): path
        outputName (str, optional): _description_. Defaults to 'grid.pkl'.
    """
    
    # Read the first three lines to extract grid sizes
    with open(filepath, 'r') as f:
        ni = int(f.readline().strip().split('=')[1])
        nj = int(f.readline().strip().split('=')[1])
        nk = int(f.readline().strip().split('=')[1])

    # Read the rest of the CSV data into a DataFrame
    df = pd.read_csv(filepath, skiprows=3)

    print("Found the following grid sizes:")
    print(f"NI = {ni}, NJ = {nj}, NK = {nk}\n")

    if nk != 1:
        raise ValueError("Grid file is not 2D")

    print("With the following field values")

    # Convert DataFrame to dict of 2D float arrays
    field_dict = {
        key: np.array(values, dtype=float).reshape((ni, nj))
        for key, values in df.to_dict(orient='list').items()
    }

    print(field_dict.keys())
    
    # ok output the pkl file for Sun model (be careful for the convetion about directions)
    # CTurboBFM (x,y,z) corresponds to Sun (z, r, t)
    data = {
    'AxialCoord': field_dict['x'],
    'RadialCoord': field_dict['y'],
    'Density': field_dict['Density'],
    'AxialVel': field_dict['Velocity X'],
    'RadialVel': field_dict['Velocity Y'],
    'TangentialVel': field_dict['Velocity Z'],
    'Pressure': field_dict['Pressure'],
    'drho_dr': field_dict['Density Gradient Y'],
    'drho_dz': field_dict['Density Gradient X'],
    'duz_dr': field_dict['Velocity X Gradient Y'],
    'duz_dz': field_dict['Velocity X Gradient X'],
    'dur_dr': field_dict['Velocity Y Gradient Y'],
    'dur_dz': field_dict['Velocity Y Gradient X'],
    'dut_dr': field_dict['Velocity Z Gradient Y'],
    'dut_dz': field_dict['Velocity Z Gradient X'],
    'dp_dr': field_dict['Pressure Gradient Y'],
    'dp_dz': field_dict['Pressure Gradient X']
    }
    
    os.makedirs('Grids', exist_ok=True)
    with open('Grids/Sun_Data_%02i_%02i.pkl' % (ni, nj), 'wb') as f:
        pickle.dump(data, f)

    print('Data saved in Grids/data_%02i_%02i.pkl' % (ni, nj))
    
    
    