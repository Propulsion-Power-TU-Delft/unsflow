"""
WEIGHTED LEAST SQUARES APPROXIMATION TO SCATTERED F(x,y). Reference
"An As-Short-As-Possible Introduction to the Least Squares, Weighted Least Squares and
Moving Least Squares Methods for Scattered Data Approximation and Interpolation" - Nealen Andrew
"""
import numpy as np


def weight_function(x, y, xc, yc, delta, wfunc_type):
    """
    Compute the weight function to associate to the generic point.
    :param x: x-cordinate of the point we want to calculate the weight function
    :param y: y-cordinate of the point we want to calculate the weight function
    :param xc: x-cordinate of the center of the wight function, where the least square is fit
    :param yc: y-cordinate of the center of the wight function, where the least square is fit
    :param delta: smoothing parameter, to reduce small scale features
    :param wfunc_type: type of weight function to use
    """
    wfunction_list = ['gauss', 'wendland', 'risd']
    d = np.sqrt((x - xc) ** 2 + (y - yc) **2)
    h = delta

    if wfunc_type not in wfunction_list:
        wfunc_type = wfunction_list[0] # default choice

    if wfunc_type=='gauss':
        weight = np.exp(-d**2/h**2)
    elif wfunc_type=='wendland':
        print("WARNING: Wendland weight function not correctly working at the moment")
        weight = (1-d/h)**4 * (4*d/h + 1)
    elif wfunc_type=='risd':
        weight = 1/(d**2 + h**2)
    else:
        raise ValueError("Invalid weight function type")
    return weight


def basis_function(x, y, order=4):
    """
    For a certain point return the array defining the basis function values.
    :param x: x-cordinate of the point we want to calculate the basis function vector
    :param y: y-cordinate of the point we want to calculate the basis function vector
    :param order: order of the polynomial of the basis function
    """
    if order == 2:
        b = np.array([1, x, y, x ** 2, y ** 2, x * y])
        bdx = np.array([0, 1, 0, 2 * x, 0, y])
        bdy = np.array([0, 0, 1, 0, 2 * y, x])
    elif order == 3:
        b = np.array([1, x, y, x ** 2, y ** 2, x * y, x ** 3, y ** 3, x ** 2 * y, x * y ** 2])
        bdx = np.array([0, 1, 0, 2 * x, 0, y, 3 * x ** 2, 0, 2 * x * y, y ** 2])
        bdy = np.array([0, 0, 1, 0, 2 * y, x, 0, 3 * y ** 2, x ** 2, 2 * x * y])
    elif order == 4:
        b = np.array([1, x, y, x ** 2, y ** 2, x * y, x ** 3, y ** 3, x ** 2 * y, x * y ** 2,
                      x ** 4, y ** 4, x ** 3 * y, x * y ** 3, x ** 2 * y ** 2])
        bdx = np.array([0, 1, 0, 2 * x, 0, y, 3 * x ** 2, 0, 2 * x * y, y ** 2,
                        4 * x ** 3, 0, 3 * x ** 2 * y, y ** 3, 2 * x * y ** 2])
        bdy = np.array([0, 0, 1, 0, 2 * y, x, 0, 3 * y ** 2, x ** 2, 2 * x * y,
                        0, 4 * y ** 3, x ** 3, 3 * x * y ** 2, 2 * x ** 2 * y])
    else:
        raise ValueError("Invalid basis function order.")
    return b, bdx, bdy


def local_coeff_vector(xc, yc, x_points, y_points, z_points, order, delta, wfunc_type):
    """
    For the point (xc,yc) compute the coefficient vector of the regression.
    :param xc: x-cordinate of the point where we compute the WLS fit
    :param yc: y-cordinate of the point where we compute the WLS fit
    :param x_points: x cordinates of the dataset
    :param y_points: y cordinates of the dataset
    :param z_points: function values of the dataset
    :param order: order used for the basis functions
    :param delta: delta hyperparameter of the weight function
    :param wfunc_type: type of weighting function to use
    """
    sum1 = 0
    sum2 = 0
    for i in range(len(x_points)):
        sum1 += weight_function(xc, yc, x_points[i], y_points[i], delta, wfunc_type) * \
                np.dot(basis_function(x_points[i], y_points[i], order)[0], basis_function(x_points[i], y_points[i], order)[0])
        sum2 += weight_function(xc, yc, x_points[i], y_points[i], delta, wfunc_type) * \
                basis_function(x_points[i], y_points[i], order)[0] * z_points[i]
    coeff = sum2 / sum1
    return coeff


def evaluate_weight_least_square_regression(xc, yc, x_points, y_points, z_points, order, delta, wfunc_type):
    """
    For the point (xc,yc) compute the coefficient vector of the regression.
    :param xc: x-cordinate of the point where we compute the WLS fit
    :param yc: y-cordinate of the point where we compute the WLS fit
    :param x_points: x cordinates of the dataset
    :param y_points: y cordinates of the dataset
    :param z_points: function values of the dataset
    :param order: order used for the basis functions
    :param delta: delta hyperparameter of the weight function
    :param wfunc_type: type of weighting function to use
    """
    c = local_coeff_vector(xc, yc, x_points, y_points, z_points, order, delta, wfunc_type)
    w, wdx, wdy = basis_function(xc, yc, order)
    value = np.dot(w, c)
    value_dx = np.dot(wdx, c)
    value_dy = np.dot(wdy, c)
    return value, value_dx, value_dy
