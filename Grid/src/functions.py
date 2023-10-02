#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 18:29:29 2023
@author: F. Neri, TU Delft
"""
import matplotlib.pyplot as plt
import numpy as np
from numpy import sqrt, sin, cos, tan, arccos, arcsin, log
from .styles import *
import math
from scipy.optimize import fsolve
from scipy import interpolate


def cluster_sample_u(n, shrink_effect=3.5, border='default'):
    """
    routine to provide an array of numbers from 0 to 1, in which many points are clustered close to the borders.
    it makes use of sigmoid function. Change the parameters if needed
    """
    length = n
    # Generate the array with more points near the extremes
    if border == 'default':
        array = np.linspace(-shrink_effect, shrink_effect, length)
    elif border == 'left':
        array = np.linspace(-shrink_effect, 0, length)
    elif border == 'right':
        array = np.linspace(0, shrink_effect, length)

    sigmoid = 1 / (1 + np.exp(-array))  # Apply sigmoid function
    scaled_array = (sigmoid - sigmoid.min()) / (sigmoid.max() - sigmoid.min())
    return scaled_array


def eliminate_duplicates(x, y, z):
    """
    providing three arrays of cordinates, it returns the original arrays after deletion of the duplicates. 
    """
    matrix = np.array((x, y, z))
    unique_columns, indices = np.unique(matrix, axis=1, return_index=True)
    sorted_indices = np.sort(indices)
    # Select the unique columns from the original matrix
    unique_matrix = matrix[:, sorted_indices]
    return unique_matrix[0, :], unique_matrix[1, :], unique_matrix[2, :]


def project_vector_to_cylindrical(ux, uy, theta):
    """
    project a vector written in x-y components in the equivalent vector written in r-theta components
    """
    ur = ux * cos(theta) + uy * sin(theta)  # radial component
    ut = -ux * sin(theta) + uy * cos(theta)  # tangential component
    return ur, ut


def project_scalar_gradient_to_cylindrical(da_dx, da_dy, r, theta):
    """
    project the x-y gradients of a scalar field "a" in r-theta gradients. 
    """
    da_dr = da_dx * cos(theta) + da_dy * sin(theta)
    da_dtheta = r * (-da_dx * sin(theta) + da_dy * cos(theta))
    return da_dr, da_dtheta


def project_velocity_gradient_to_cylindrical(dux_dx, dux_dy, duy_dx, duy_dy, r, theta):
    """
    project the gradient of the velocity field in cylindrical cordinates
    (tensor transformation law T_{r,theta} = Q^T * T_{x,y} * Q) where Q is the rotation matrix, sines and cosines of theta.
    """

    dur_dr = dux_dx * cos(theta) ** 2 + (duy_dx + dux_dy) * sin(theta) * cos(theta) + duy_dy * sin(theta) ** 2
    dur_dtheta = r * (dux_dy * cos(theta) ** 2 + (duy_dy - dux_dx) * sin(theta) * cos(theta) - duy_dx * sin(theta) ** 2)
    dut_dr = duy_dx * cos(theta) ** 2 + (duy_dy - dux_dx) * sin(theta) * cos(theta) - dux_dy * sin(theta) ** 2
    dut_dtheta = r * (dux_dx * sin(theta) ** 2 - sin(theta) * cos(theta) * (duy_dx + dux_dy) + duy_dy * cos(theta) ** 2)

    return dur_dr, dur_dtheta, dut_dr, dut_dtheta


def second_order_finite_differences(x, y, z):
    """
    Args:
        x: grid values of x
        y: grid values of y
        z: grid values of z(x, y)

    Returns:
        dzdx: grid value of partial derivative dzdx
        dzdy: grid value of partial derivative dzdy
    """
    dim = z.shape
    dzdx = np.zeros_like(z)
    dzdy = np.zeros_like(z)

    # Calculate derivatives in the center with second order
    for i in range(1, dim[0] - 1):
        for j in range(1, dim[1] - 1):
            dzdx[i, j] = (z[i + 1, j] - z[i - 1, j]) / (x[i + 1, j] - x[i - 1, j])
            dzdy[i, j] = (z[i, j + 1] - z[i, j - 1]) / (y[i, j + 1] - y[i, j - 1])

    return dzdx, dzdy


def cartesian_to_cylindrical(x, y, z, v):
    """
    pass from the components in cartesian to cylindrical cordinate

    Args:
        x: x cordinate
        y: y cordinate
        z: z cordinate
        v: vector (3D)

    Returns:
        v_cylindrical
    """
    M = cartesian_to_cylindrical_matrix(x, y)
    v_cylindrical = np.dot(M, v)
    return v_cylindrical


def cartesian_to_cylindrical_matrix(x, y):
    """
    returns the matrix of the transformation
    Args:
        x: x cordinate
        y: y cordinate

    Returns:
        M: transformation matrix (3D)
    """
    theta = np.arctan2(y, x)
    M = np.array([[np.cos(theta), np.sin(theta), 0],
                  [-np.sin(theta), np.cos(theta), 0],
                  [0, 0, 1]])

    return M


def elliptic_grid_generation(c_left, c_bottom, c_right, c_top, orthogonality, x_stretching,
                             y_stretching, X0=None, Y0=None, tol=1e-3, save_filename=None, show=True,
                             pol_order=3, sigmoid_coeff_x=5, sigmoid_coeff_y=5, it_orth=-1):
    """
    create a structured grid, using elliptic method (Winslow equations). Inputs are the 4 borders
    delimiting the figure, and the structured X,Y initial conditions. Tol is used to choose when stopping
    the iterations.
    The features to be implemented are orthogonality, and stretching of the grid.
    """
    nx = np.shape(c_bottom)[1]
    ny = np.shape(c_left)[1]
    maxit = 500

    # computational domain between 0 and 1
    xi = np.linspace(0, 1, nx)
    dxi = xi[1] - xi[0]
    eta = np.linspace(0, 1, ny)
    deta = eta[1] - eta[0]

    X = np.zeros((nx, ny))
    Y = np.zeros((nx, ny))
    if X0 is not None and Y0 is not None:
        # inital grid attempt given in the args
        X = X0
        Y = Y0
    else:
        # if initial grid is not given, find it via interpolation of the borders
        X[0, :] = c_left[0, :]
        Y[0, :] = c_left[1, :]
        X[:, 0] = c_bottom[0, :]
        Y[:, 0] = c_bottom[1, :]
        X[-1, :] = c_right[0, :]
        Y[-1, :] = c_right[1, :]
        X[:, -1] = c_top[0, :]
        Y[:, -1] = c_top[1, :]
        for istream in range(1, nx - 1):
            for ispan in range(1, ny - 1):
                X[istream, ispan] = X[istream, 0] + (X[istream, -1] - X[istream, 0]) * ispan / (ny - 1)
                Y[istream, ispan] = Y[istream, 0] + (Y[istream, -1] - Y[istream, 0]) * ispan / (ny - 1)

    # plt.figure()
    # plt.plot(xi, chi, label=r'$f(\xi)$')
    # plt.legend()
    # plt.figure()
    # plt.plot(xi, chi_prime, label=r'$f_{1}(\xi)$')
    # plt.legend()
    # plt.figure()
    # plt.plot(xi, chi_second, label=r'$f_{2}(\xi)$')
    # plt.legend()
    # plt.show()

    # stretching functions block!
    f1 = np.zeros((nx, ny))
    f1_prime = np.zeros((nx, ny))
    f1_second = np.zeros((nx, ny))
    f2 = np.zeros((nx, ny))
    f2_prime = np.zeros((nx, ny))
    f2_second = np.zeros((nx, ny))
    for ispan in range(ny):
        if not x_stretching:
            f1[:, ispan], f1_prime[:, ispan], f1_second[:, ispan] = no_stretching_function(xi)
        elif x_stretching == 'sigmoid':
            f1[:, ispan], f1_prime[:, ispan], f1_second[:, ispan] = scaled_sigmoid(xi, sigmoid_coeff_x)
        elif x_stretching == 'polynomial':
            f1[:, ispan], f1_prime[:, ispan], f1_second[:, ispan] = polynomial_function(xi, pol_order)
        else:
            raise ValueError('Check the value of x-strecthing parameter!')
    for istream in range(0, nx):
        if not y_stretching:
            f2[istream, :], f2_prime[istream, :], f2_second[istream, :] = no_stretching_function(eta)
        elif y_stretching == 'sigmoid':
            f2[istream, :], f2_prime[istream, :], f2_second[istream, :] = scaled_sigmoid(eta, sigmoid_coeff_y)
        elif y_stretching == 'polynomial':
            f2[istream, :], f2_prime[istream, :], f2_second[istream, :] = polynomial_function(eta, pol_order)
        else:
            raise ValueError('Check the value of y-strecthing parameter!')

    g11 = np.zeros((nx, ny))
    g12 = np.zeros((nx, ny))
    g22 = np.zeros((nx, ny))
    g = np.zeros((nx, ny))
    a = np.zeros((nx, ny))
    b = np.zeros((nx, ny))
    c = np.zeros((nx, ny))
    d = np.zeros((nx, ny))
    e = np.zeros((nx, ny))

    WH_ratio = (np.max(X) - np.min(X)) / (np.max(Y) - np.min(Y))
    scale = 0.5 * ((np.max(X) - np.min(X)) + (np.max(Y) - np.min(Y)))  # scale of the dimensions
    tol *= scale  # scale the tolerance threshold
    if WH_ratio >= 1:
        pic_size = (8, 8 / WH_ratio)
    else:
        pic_size = (6 * WH_ratio, 6)

    if show:
        plt.figure(figsize=pic_size)

    for it in range(maxit):
        """
        main iteration loop. Follow the instructions given in the book <Basic structured grid generation with an introduction 
        to unstructured grid generation> from Farrashkhalvat, pag. 132. Thomas algorithm
        """

        # plot the grid at every iteration, as well as the original border in red, to check the mesh doesn't behave weird
        if it > it_orth:
            pass  # breakpoint to check
        if show:
            plt.clf()
            for ii in range(nx):
                plt.plot(X[ii, :], Y[ii, :], 'black', lw=0.5)
            for jj in range(ny):
                plt.plot(X[:, jj], Y[:, jj], 'black', lw=0.5)
            plt.plot(c_left[0, :], c_left[1, :], 'red', lw=0.5)
            plt.plot(c_bottom[0, :], c_bottom[1, :], 'red', lw=0.5)
            plt.plot(c_right[0, :], c_right[1, :], 'red', lw=0.5)
            plt.plot(c_top[0, :], c_top[1, :], 'red', lw=0.5)
            plt.xlabel(r'$X$')
            plt.ylabel(r'$Y$')
            plt.title('iteration %d' % (it))
            plt.pause(0.001)

        # store the previous iteration data to compute the convergence residual
        X_old = X.copy()
        Y_old = Y.copy()

        # internal slices of the matrices, p m stand for plus and minus
        i = slice(1, nx - 1)
        ip = slice(2, nx)
        im = slice(0, nx - 2)
        j = slice(1, ny - 1)
        jp = slice(2, ny)
        jm = slice(0, ny - 2)

        # terms of integration
        g11[i, j] = ((X[ip, j] - X[im, j]) / 2 / dxi) ** 2 + ((Y[ip, j] - Y[im, j]) / 2 / dxi) ** 2
        g22[i, j] = ((X[i, jp] - X[i, jm]) / 2 / deta) ** 2 + ((Y[i, jp] - Y[i, jm]) / 2 / deta) ** 2
        g12[i, j] = ((X[ip, j] - X[im, j]) / 2 / dxi) * ((X[i, jp] - X[i, jm]) / 2 / deta) + \
                    ((Y[ip, j] - Y[im, j]) / 2 / dxi) * ((Y[i, jp] - Y[i, jm]) / 2 / deta)

        # if orthogonality:
        #     """
        #     orthogonality condition given in the book, paragraph 5.3.1
        #     """
        #     g12 *= 0

        g[i, j] = g11[i, j] * g22[i, j] - g12[i, j] ** 2

        a[i, j] = g22[i, j] / dxi ** 2

        b[i, j] = 2 * g22[i, j] / (dxi ** 2) + 2 * g11[i, j] / deta ** 2

        c[i, j] = g22[i, j] / dxi ** 2

        d[i, j] = g11[i, j] / (deta ** 2) * (X[i, jp] + X[i, jm]) - 2 * g12[i, j] * (
                X[ip, jp] + X[im, jm] - X[im, jp] - X[ip, jm]) / 4 / dxi / deta

        e[i, j] = g11[i, j] / (deta ** 2) * (Y[i, jp] + Y[i, jm]) - 2 * g12[i, j] * (
                Y[ip, jp] + Y[im, jm] - Y[im, jp] - Y[ip, jm]) / 4 / dxi / deta

        # adjustments due to stretching functions (be careful to the signs)
        a[i, j] += +f1_second[i, j] / f1_prime[i, j] * g22[i, j] / 2 / dxi
        c[i, j] += -f1_second[i, j] / f1_prime[i, j] * g22[i, j] / 2 / dxi
        d[i, j] += -f2_second[i, j] / f2_prime[i, j] * g11[i, j] * (X[i, jp] - X[i, jm]) / 2 / deta
        e[i, j] += -f2_second[i, j] / f2_prime[i, j] * g11[i, j] * (Y[i, jp] - Y[i, jm]) / 2 / deta


        # solve the thomas algorithm
        P = np.zeros(nx)
        Q = np.zeros(nx)
        for jj in range(1, ny - 1):
            # fix the known terms for the first and last point of X equation
            d[1, jj] += a[1, jj] * X[0, jj]
            d[-1, jj] += c[-1, jj] * X[-1, jj]

            P[1] = c[1, jj] / (b[1, jj])
            Q[1] = d[1, jj] / (b[1, jj])

            for ii in range(2, nx - 1):
                P[ii] = c[ii, jj] / (b[ii, jj] - a[ii, jj] * P[ii - 1])
                Q[ii] = (d[ii, jj] + a[ii, jj] * Q[ii - 1]) / (b[ii, jj] - a[ii, jj] * P[ii - 1])

            for ii in range(nx - 2, 0, -1):
                X[ii, jj] = P[ii] * X[ii + 1, jj] + Q[ii]

            # fix the known terms for the first and last point of Y equation
            e[1, jj] += a[1, jj] * Y[0, jj]
            e[-1, jj] += c[-1, jj] * Y[-1, jj]

            P[1] = c[1, jj] / (b[1, jj])
            Q[1] = e[1, jj] / (b[1, jj])
            for ii in range(2, nx - 1):
                P[ii] = c[ii, jj] / (b[ii, jj] - a[ii, jj] * P[ii - 1])
                Q[ii] = (e[ii, jj] + a[ii, jj] * Q[ii - 1]) / (b[ii, jj] - a[ii, jj] * P[ii - 1])

            for ii in range(nx - 2, 0, -1):
                Y[ii, jj] = P[ii] * Y[ii + 1, jj] + Q[ii]

        if orthogonality and it > it_orth:
            # print('bottom edge fixing..')
            x = c_bottom[0, :]
            y = c_bottom[1, :]
            y_prime = np.gradient(y, x)
            for istream in range(1, nx - 1):
                xb_old = X_old[istream, 0]
                xp_new = X[istream, 1]
                yb_old = Y_old[istream, 0]
                yp_new = Y[istream, 1]
                yb_prime = y_prime[istream]
                sol = solve_linear_system(yb_prime, yp_new, xp_new, yb_old, xb_old)
                xb_new, yb_new = find_corresponding_point(sol[0], sol[1], x, y, xb_old, yb_old)
                X[istream, 0] = xb_new
                Y[istream, 0] = yb_new

            # print('top edge fixing..')
            x = c_top[0, :]
            y = c_top[1, :]
            y_prime = np.gradient(y, x)
            for istream in range(1, nx - 1):
                xb_old = X_old[istream, -1]
                xp_new = X[istream, -2]
                yb_old = Y_old[istream, -1]
                yp_new = Y[istream, -2]
                yb_prime = y_prime[istream]
                sol = solve_linear_system(yb_prime, yp_new, xp_new, yb_old, xb_old)
                xb_new, yb_new = find_corresponding_point(sol[0], sol[1], x, y, xb_old, yb_old)
                X[istream, -1] = xb_new
                Y[istream, -1] = yb_new

            # print('left edge fixing..')
            x = c_left[0, :]
            y = c_left[1, :]
            y_prime = np.gradient(y, x)
            for ispan in range(1, ny - 1):
                xb_old = X_old[0, ispan]
                xp_new = X[1, ispan]
                yb_old = Y_old[0, ispan]
                yp_new = Y[1, ispan]
                yb_prime = y_prime[ispan]
                sol = solve_linear_system(yb_prime, yp_new, xp_new, yb_old, xb_old)
                xb_new, yb_new = find_corresponding_point(sol[0], sol[1], x, y, xb_old, yb_old)
                X[0, ispan] = xb_new
                Y[0, ispan] = yb_new

            # print('right edge fixing..')
            x = c_right[0, :]
            y = c_right[1, :]
            y_prime = np.gradient(y, x)
            for ispan in range(1, ny - 1):
                xb_old = X_old[-1, ispan]
                xp_new = X[-2, ispan]
                yb_old = Y_old[-1, ispan]
                yp_new = Y[-2, ispan]
                yb_prime = y_prime[ispan]
                sol = solve_linear_system(yb_prime, yp_new, xp_new, yb_old, xb_old)
                xb_new, yb_new = find_corresponding_point(sol[0], sol[1], x, y, xb_old, yb_old)
                X[-1, ispan] = xb_new
                Y[-1, ispan] = yb_new

        err_x = np.linalg.norm(X_old - X)
        err_y = np.linalg.norm(Y_old - Y)
        if err_x < tol and err_y < tol and it > it_orth:
            print('convergence reached in %d sweeps' % (it))
            break

        if it == maxit - 1:
            print('convergence not reached')

    if save_filename is not None:
        plt.figure(figsize=pic_size)
        for ii in range(nx):
            plt.plot(X[ii, :], Y[ii, :], 'black', lw=0.5)
        for jj in range(ny):
            plt.plot(X[:, jj], Y[:, jj], 'black', lw=0.5)
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')
        plt.title('iteration %d' % (it))
        plt.savefig('pictures/' + save_filename + '_%d_%d.pdf' % (nx, ny), bbox_inches='tight')

    return X, Y


def scaled_sigmoid(x, alpha):
    """
    return sigmoid scaled function, first derivative, and second derivative over an array x. alpha decides the slope
    """
    f = 1 / (1 + np.exp(-alpha * (x - 0.5)))
    f_prime = (alpha * np.exp(-alpha * (x - 0.5))) / (1 + np.exp(-alpha * (x - 0.5))) ** 2
    f_second = (-alpha ** 2 * np.exp(-alpha * (x - 0.5)) * (1 + np.exp(-alpha * (x - 0.5))) + 2 * alpha ** 2 *
                np.exp(-2 * alpha * (x - 0.5))) / (1 + np.exp(-alpha * (x - 0.5))) ** 3
    return f, f_prime, f_second


def polynomial_function(x, n):
    """
    return polynomial x^n function, first derivative, and second derivative over an array x. alpha decides the slope
    """
    f = x ** n
    f_prime = n * x ** (n - 1)
    f_second = n * (n - 1) * x ** (n - 2)
    return f, f_prime, f_second


def no_stretching_function(x):
    """
    return the x function, first derivative, and second derivative over an array x. which defines the zero-stretching function
    """
    f = x
    f_prime = np.zeros(len(x)) + 1
    f_second = np.zeros(len(x))
    return f, f_prime, f_second


def solve_linear_system(yb_prime, yp_new, xp_new, yb_old, xb_old):
    """
    solve the linear system to fix the borders, handling zero,inf or nan slopes of the curves
    """
    if yb_prime == 0:
        # print('Horizontal point\n')
        sol = [xp_new, None]  # same x cordinate of the interior point
    elif math.isinf(yb_prime) or math.isnan(yb_prime):
        # print('Vertical point\n')
        sol = [None, yp_new]  # same y cordinate of the interior point
    else:
        # print('Inclined point\n')
        A_sys = np.array([[1 / yb_prime, 1],
                          [-yb_prime, 1]])
        B_sys = np.array([yp_new + xp_new / yb_prime,
                          yb_old - yb_prime * xb_old])
        sol = np.linalg.solve(A_sys, B_sys)
        sol = [sol[0], None]  # solve this case as the horizontal point case
    return sol



def find_corresponding_point(xb_new, yb_new, x, y, xb_old, yb_old):
    """
    depending on which (xb_new, yb_new) of is different from none, find the other one constraining it on the original border curve.
    xb_old, yb_old are the cordinates of the previous iteration
    """
    u = np.linspace(0, 1, len(x))  # curve parameterization
    degree = 8

    # Perform polynomial interpolation
    coefficients = np.polyfit(u, x, degree)
    interp_x = np.poly1d(coefficients)

    coefficients = np.polyfit(u, y, degree)
    interp_y = np.poly1d(coefficients)

    def zero_y_fx(u_param):
        return interp_y(u_param) - yb_new

    def zero_x_fy(u_param):
        return interp_x(u_param) - xb_new

    if xb_new is not None:
        # err = 1
        # tol = np.sqrt((np.max(y) - np.min(y)) ** 2 + (np.max(x) - np.min(x)) ** 2)
        # u_root = 0.5
        # while (err > tol):
        #     print('point (%.2f, %.2f)' % (xb_old, yb_old))
        u_root = fsolve(zero_x_fy, x0=yb_old)
        yb_new = interp_y(u_root)
            # err = np.abs(yb_new - yb_old)
    else:
        # err = 1
        # tol = np.sqrt((np.max(y) - np.min(y)) ** 2 + (np.max(x) - np.min(x)) ** 2)
        # u_root = 0.5
        # while (err > tol):
        #     print('point (%.2f, %.2f)' % (xb_old, yb_old))
        u_root = fsolve(zero_y_fx, x0=xb_old)
        xb_new = interp_x(u_root)
            # err = np.abs(xb_new - xb_old)

    return xb_new, yb_new
