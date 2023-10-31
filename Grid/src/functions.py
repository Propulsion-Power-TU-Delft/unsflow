#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 18:29:29 2023
@author: F. Neri, TU Delft
"""
import numpy as np
from numpy import sin, cos
import pickle
from .styles import *
import math
from scipy.optimize import fsolve
from scipy.optimize import minimize


def cluster_sample_u(n, shrink_effect=3.5, border='default'):
    """
    routine to provide an array of numbers from 0 to 1, in which many points are clustered close to the borders.
    it makes use of sigmoid functions.
    :param n: total number of sampling points
    :param shrink_effect: value of the shrink used in the sigmoid case. A higher value cluster the points more. 3.5 is good.
    :param border: set the borders at which the points will be clustered. left, right, or default (both).
    """
    length = n
    # Generate the array with more points near the extremes
    if border == 'default':
        array = np.linspace(-shrink_effect, shrink_effect, length)
    elif border == 'left':
        array = np.linspace(-shrink_effect, 0, length)
    elif border == 'right':
        array = np.linspace(0, shrink_effect, length)
    else:
        raise ValueError("Unknown type of border")

    sigmoid = 1 / (1 + np.exp(-array))  # Apply sigmoid function
    scaled_array = (sigmoid - sigmoid.min()) / (sigmoid.max() - sigmoid.min())
    return scaled_array


def eliminate_duplicates(x, y, z):
    """
    providing three arrays of cordinates, it returns the original arrays after deletion of the duplicates.
    :param x: array of x cordinates
    :param y: array of y cordinates
    :param z: array of z cordinates
    """
    matrix = np.array((x, y, z))
    unique_columns, indices = np.unique(matrix, axis=1, return_index=True)
    sorted_indices = np.sort(indices)
    # Select the unique columns from the original matrix
    unique_matrix = matrix[:, sorted_indices]
    return unique_matrix[0, :], unique_matrix[1, :], unique_matrix[2, :]


def project_vector_to_cylindrical(ux, uy, theta):
    """
    Project a vector written in x-y components in the equivalent vector written in r-theta components.
    :param ux: x-component of the vector
    :param uy: y-component of the vector
    :param theta: theta location of the considered point [rad]
    """
    ur = ux * cos(theta) + uy * sin(theta)  # radial component
    ut = -ux * sin(theta) + uy * cos(theta)  # tangential component
    return ur, ut


def project_scalar_gradient_to_cylindrical(da_dx, da_dy, r, theta):
    """
    Project the x-y gradients of a scalar field "a" in r-theta components.
    :param da_dx: x-derivative of the scalar field
    :param da_dy: y-derivative of the scalar field
    :param r: radial cordinate of the point
    :param theta: theta location of the considered point [rad]
    """
    da_dr = da_dx * cos(theta) + da_dy * sin(theta)
    da_dtheta = r * (-da_dx * sin(theta) + da_dy * cos(theta))
    return da_dr, da_dtheta


def project_velocity_gradient_to_cylindrical(dux_dx, dux_dy, duy_dx, duy_dy, r, theta):
    """
    Project the x-y gradients of a vector field "a" in r-theta components.
    :param dux_dx: x-derivative of the x-component field
    :param dux_dy: y-derivative of the x-component field
    :param duy_dx: x-derivative of the y-component field
    :param duy_dy: y-derivative of the y-component field
    :param r: radial cordinate of the point
    :param theta: theta location of the considered point [rad]
    (tensor transformation law T_{r,theta} = Q^T * T_{x,y} * Q) where Q is the rotation matrix (sines and cosines of theta).
    """

    dur_dr = dux_dx * cos(theta) ** 2 + (duy_dx + dux_dy) * sin(theta) * cos(theta) + duy_dy * sin(theta) ** 2
    dur_dtheta = r * (dux_dy * cos(theta) ** 2 + (duy_dy - dux_dx) * sin(theta) * cos(theta) - duy_dx * sin(theta) ** 2)
    dut_dr = duy_dx * cos(theta) ** 2 + (duy_dy - dux_dx) * sin(theta) * cos(theta) - dux_dy * sin(theta) ** 2
    dut_dtheta = r * (dux_dx * sin(theta) ** 2 - sin(theta) * cos(theta) * (duy_dx + dux_dy) + duy_dy * cos(theta) ** 2)

    return dur_dr, dur_dtheta, dut_dr, dut_dtheta


# def second_order_finite_differences(x, y, z):
#     """
#     Args:
#         x: grid values of x
#         y: grid values of y
#         z: grid values of z(x, y)
#
#     Returns:
#         dzdx: grid value of partial derivative dzdx
#         dzdy: grid value of partial derivative dzdy
#     """
#     dim = z.shape
#     dzdx = np.zeros_like(z)
#     dzdy = np.zeros_like(z)
#
#     # Calculate derivatives in the center with second order
#     for i in range(1, dim[0] - 1):
#         for j in range(1, dim[1] - 1):
#             dzdx[i, j] = (z[i + 1, j] - z[i - 1, j]) / (x[i + 1, j] - x[i - 1, j])
#             dzdy[i, j] = (z[i, j + 1] - z[i, j - 1]) / (y[i, j + 1] - y[i, j - 1])
#
#     return dzdx, dzdy


def cartesian_to_cylindrical(x, y, z, v):
    """
    Pass from the cordinates in cartesian to cylindrical cordinate.
    :param x: x cordinate
    :param y: y cordinate
    :param z: z cordinate
    :param v: vector in cartesian components
    """
    M = cartesian_to_cylindrical_matrix(x, y)
    v_cylindrical = np.dot(M, v)
    return v_cylindrical


def cartesian_to_cylindrical_matrix(x, y):
    """
    Returns the 3D matrix of the transformation from cartesian to cylindrical for the point located in (x,y)
    :param x: x cordinate
    :param y: y cordinate
    """
    theta = np.arctan2(y, x)
    M = np.array([[np.cos(theta), np.sin(theta), 0],
                  [-np.sin(theta), np.cos(theta), 0],
                  [0, 0, 1]])

    return M


def elliptic_grid_generation(c_left, c_bottom, c_right, c_top, orthogonality, x_stretching,
                             y_stretching, X0=None, Y0=None, tol=1e-3, save_filename=None, show=True,
                             pol_order=3, sigmoid_coeff_x=5, sigmoid_coeff_y=5, it_orth=-1, guardian=False,
                             method='minimize', fix_inlet=False, fix_outlet=False, save_animation=False):
    """
    Create a structured grid, using elliptic method (Winslow equations). Inputs are the 4 borders
    delimiting the figure, ordered in a certain way.
    :param c_left: left border of the domain, ordered from hub to shroud.
    :param c_bottom: bottom border of the domain, ordered from inlet to outlet.
    :param c_right: right border of the domain, ordered from hub to shroud.
    :param c_top: upper border of the domain, ordered from inlet to outlet.
    :param orthogonality: if True, enables orthogonality corrections, as well as border nodes update.
    :param x_stretching: stretching type of the grid in the streamwise direction.
    :param y_stretching: stretching type of the grid in the spanwise direction.
    :param X0: if set, is the initial condition  of the x PDE.
    :param Y0: if set, is the initial condition  of the y PDE.
    :param tol: threshold to stop iteration.
    :param save_filename: if set, saves the figure.
    :param show: if True shows the animation
    :param pol_order: polynomial order of the strecthing function (not validated yet.)
    :param sigmoid_coeff_x: x-coefficient of the sigmoid along the streamwise direction.
    :param sigmoid_coeff_y: y-coefficient of the sigmoid along the spanwise direction.
    :param it_orth: iteration number from which orthogonality is enabled.
    :param guardian: under-relaxation method
    :param method: choose method used to update points on the border. Suggested minimize, as fzero frequently diverges
    :param fix_inlet: if True, fixes the points on the inlet border
    :param fix_outlet: if True, fixes the points on the outlet border.
    :param save_animation: if specified,  saves figures with the defined filename path.
    """
    nx = np.shape(c_bottom)[1]
    ny = np.shape(c_left)[1]
    maxit = 500

    # computational domain between 0 and 1 in both directions
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

    # compute stretching functions, even in the case of no-stretching to avoid messing up things
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
        elif x_stretching == 'sigmoid_right':
            f1[:, ispan], f1_prime[:, ispan], f1_second[:, ispan] = scaled_sigmoid_right(xi, sigmoid_coeff_x)
        elif x_stretching == 'sigmoid_left':
            f1[:, ispan], f1_prime[:, ispan], f1_second[:, ispan] = scaled_sigmoid_left(xi, sigmoid_coeff_x)
        elif x_stretching == 'polynomial':
            f1[:, ispan], f1_prime[:, ispan], f1_second[:, ispan] = polynomial_function(xi, pol_order)
        else:
            raise ValueError('Check the value of x-strecthing parameter!')
    for istream in range(0, nx):
        if not y_stretching:
            f2[istream, :], f2_prime[istream, :], f2_second[istream, :] = no_stretching_function(eta)
        elif y_stretching == 'sigmoid':
            f2[istream, :], f2_prime[istream, :], f2_second[istream, :] = scaled_sigmoid(eta, sigmoid_coeff_y)
        elif y_stretching == 'sigmoid_down':
            f2[istream, :], f2_prime[istream, :], f2_second[istream, :] = scaled_sigmoid_left(eta, sigmoid_coeff_y)
        elif y_stretching == 'sigmoid_up':
            f2[istream, :], f2_prime[istream, :], f2_second[istream, :] = scaled_sigmoid_right(eta, sigmoid_coeff_y)
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

    # WH_ratio = (np.max(X) - np.min(X)) / (np.max(Y) - np.min(Y))
    # scale = 0.5 * ((np.max(X) - np.min(X)) + (np.max(Y) - np.min(Y)))  # reference lenght of the problem
    # tol *= scale  # scale the tolerance threshold
    pic_size_blank, pic_size_contour = compute_picture_size(X, Y)

    if show:
        plt.figure(figsize=pic_size_blank)

    if save_animation:
        X_animation = np.zeros((nx, ny, maxit))
        Y_animation = np.zeros((nx, ny, maxit))

    it = 0
    for it in range(maxit):
        """
        main iteration loop. Follow the instructions given in the book <Basic structured grid generation with an introduction 
        to unstructured grid generation> from Farrashkhalvat, pag. 132. Thomas algorithm
        """

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

        if orthogonality:
            """
            orthogonality condition given in the book, paragraph 5.3.1
            """
            g12 *= 0

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

        # solve the thomas algorithm, running sweeps along i for every j
        P = np.zeros(nx)
        Q = np.zeros(nx)
        for jj in range(1, ny - 1):
            # fix the known terms for the first and last point of X equation
            d[1, jj] += a[1, jj] * X[0, jj]
            d[-1, jj] += c[-1, jj] * X[-1, jj]

            # compute P and Q for every point, using the recursion from 1, for the x problem
            P[1] = c[1, jj] / (b[1, jj])
            Q[1] = d[1, jj] / (b[1, jj])
            for ii in range(2, nx - 1):
                P[ii] = c[ii, jj] / (b[ii, jj] - a[ii, jj] * P[ii - 1])
                Q[ii] = (d[ii, jj] + a[ii, jj] * Q[ii - 1]) / (b[ii, jj] - a[ii, jj] * P[ii - 1])

            # use P and Q to compute X
            for ii in range(nx - 2, 0, -1):
                X[ii, jj] = P[ii] * X[ii + 1, jj] + Q[ii]

            # fix the known terms for the first and last point of Y equation
            e[1, jj] += a[1, jj] * Y[0, jj]
            e[-1, jj] += c[-1, jj] * Y[-1, jj]

            # compute P and Q for every point, using the recursion from 1, for the y problem
            P[1] = c[1, jj] / (b[1, jj])
            Q[1] = e[1, jj] / (b[1, jj])
            for ii in range(2, nx - 1):
                P[ii] = c[ii, jj] / (b[ii, jj] - a[ii, jj] * P[ii - 1])
                Q[ii] = (e[ii, jj] + a[ii, jj] * Q[ii - 1]) / (b[ii, jj] - a[ii, jj] * P[ii - 1])

            # use P and Q to compute Y
            for ii in range(nx - 2, 0, -1):
                Y[ii, jj] = P[ii] * Y[ii + 1, jj] + Q[ii]

        # if orthogonality required, update all the borders points cordinates, except the vertices
        if orthogonality and it > it_orth:

            # BOTTOM EDGE
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
                if method == 'fzero':
                    xb_new, yb_new = find_corresponding_point(sol[0], sol[1], x, y, xb_old, yb_old, guardian=guardian)
                elif method == 'minimize':
                    xb_new, yb_new = find_optimized_point(sol[0], sol[1], x, y, xp_new, yp_new)
                else:
                    raise ValueError('Select a valid method for borders adjustment!')
                X[istream, 0] = xb_new
                Y[istream, 0] = yb_new

            # TOP EDGE
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
                if method == 'fzero':
                    xb_new, yb_new = find_corresponding_point(sol[0], sol[1], x, y, xb_old, yb_old, guardian=guardian)
                elif method == 'minimize':
                    xb_new, yb_new = find_optimized_point(sol[0], sol[1], x, y, xp_new, yp_new)
                else:
                    raise ValueError('Select a valid method for borders adjustment!')
                X[istream, -1] = xb_new
                Y[istream, -1] = yb_new

            # LEFT EDGE
            if not fix_inlet:
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
                    if method == 'fzero':
                        xb_new, yb_new = find_corresponding_point(sol[0], sol[1], x, y, xb_old, yb_old, guardian=guardian)
                    elif method == 'minimize':
                        xb_new, yb_new = find_optimized_point(sol[0], sol[1], x, y, xp_new, yp_new)
                    else:
                        raise ValueError('Select a valid method for borders adjustment!')
                    X[0, ispan] = xb_new
                    Y[0, ispan] = yb_new

            # RIGHT EDGE
            if not fix_outlet:
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
                    if method == 'fzero':
                        xb_new, yb_new = find_corresponding_point(sol[0], sol[1], x, y, xb_old, yb_old, guardian=guardian)
                    elif method == 'minimize':
                        xb_new, yb_new = find_optimized_point(sol[0], sol[1], x, y, xp_new, yp_new)
                    else:
                        raise ValueError('Select a valid method for borders adjustment!')
                    X[-1, ispan] = xb_new
                    Y[-1, ispan] = yb_new

        if save_animation:
            X_animation[:, :, it] = X
            Y_animation[:, :, it] = Y

        # compute differences from last iteration
        err_x = np.linalg.norm(X_old - X)
        err_y = np.linalg.norm(Y_old - Y)
        if err_x < tol and err_y < tol and it > it_orth:
            print('convergence reached in %d sweeps' % (it))
            if save_animation:
                X_animation = X_animation[:, :, 0:it]
                Y_animation = Y_animation[:, :, 0:it]
            break

        # give an indication that convergence was not reached even if exiting the loop
        if it == maxit - 1:
            print('convergence not reached')

    if save_filename is not None:
        pic_size, pic_size_contour = compute_picture_size(X, Y)
        plt.figure(figsize=pic_size)
        for ii in range(nx):
            plt.plot(X[ii, :], Y[ii, :], 'black', lw=0.5)
        for jj in range(ny):
            plt.plot(X[:, jj], Y[:, jj], 'black', lw=0.5)
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')
        plt.title('iteration %d' % (it))
        plt.savefig('pictures/' + save_filename + '_%d_%d.pdf' % (nx, ny), bbox_inches='tight')

    if save_animation:
        # Open the file in binary write mode and save the arrays
        with open('X_grid_animation.pickle', 'wb') as file:
            pickle.dump(X_animation, file)
        with open('Y_grid_animation.pickle', 'wb') as file:
            pickle.dump(Y_animation, file)

    return X, Y


def scaled_sigmoid(x, alpha):
    """
    Return sigmoid scaled function, first derivative, and second derivative over an array x.
    :param x: array of sigmoid argument
    :param alpha: coefficient of the sigmoid function f(x) = 1 / (1 + np.exp(-alpha * (x - 0.5)))
    """
    f = 1 / (1 + np.exp(-alpha * (x - 0.5)))
    f_prime = (alpha * np.exp(-alpha * (x - 0.5))) / (1 + np.exp(-alpha * (x - 0.5))) ** 2
    f_second = (-alpha ** 2 * np.exp(-alpha * (x - 0.5)) * (1 + np.exp(-alpha * (x - 0.5))) + 2 * alpha ** 2 *
                np.exp(-2 * alpha * (x - 0.5))) / (1 + np.exp(-alpha * (x - 0.5))) ** 3
    return f, f_prime, f_second


def scaled_sigmoid_right(x, alpha):
    """
    Return a straight line until half of the domain, and attach a sigmoid scaled function, first derivative,
    and second derivative over an array x.
    :param x: array of sigmoid argument
    :param alpha: coefficient of the sigmoid function f(x) = 1 / (1 + np.exp(-alpha * (x - 0.5)))
    """
    f, f_prime, f_second = scaled_sigmoid(x, alpha)

    # now overwrite the first half with a straight line function
    N = len(x)
    f[0:N // 2] = x[0:N // 2]
    f_prime[0:N // 2] = np.zeros(N // 2) + 1
    f_second[0:N // 2] = np.zeros(N // 2)
    return f, f_prime, f_second


def scaled_sigmoid_left(x, alpha):
    """
    Return a straight line from half of the domain, and attach a sigmoid scaled function, first derivative,
    and second derivative over an array x.
    :param x: array of sigmoid argument
    :param alpha: coefficient of the sigmoid function f(x) = 1 / (1 + np.exp(-alpha * (x - 0.5)))
    """
    f, f_prime, f_second = scaled_sigmoid(x, alpha)

    # now overwrite the first half with a straight line function
    N = len(x)
    f[N // 2:] = x[N // 2:]
    f_prime[N // 2:] = np.zeros_like(f_prime[N // 2:]) + 1
    f_second[N // 2:] = np.zeros_like(f_second[N // 2:])
    return f, f_prime, f_second


def polynomial_function(x, n):
    """
    return polynomial x^n function, first derivative, and second derivative over an array x.
    :param x: array of polynomial argument
    :param n: order of the polynomial
    """
    f = x ** n
    f_prime = n * x ** (n - 1)
    f_second = n * (n - 1) * x ** (n - 2)
    return f, f_prime, f_second


def no_stretching_function(x):
    """
    return the x function, first derivative, and second derivative over an array x that defines the zero-stretching function.
    :param x: array of function argument
    """
    f = x
    f_prime = np.zeros(len(x)) + 1
    f_second = np.zeros(len(x))
    return f, f_prime, f_second


def solve_linear_system(yb_prime, yp_new, xp_new, yb_old, xb_old):
    """
    Solve the linear system found to fix the borders during ellipti grid generation.
    It handles zeros, inf or nan slopes of the curves.
    :param yb_prime: derivative at the border point
    :param yp_new: new y cordinate of inner point close to the border.
    :param xp_new: new x cordinate of inner point close to the border.
    :param yb_old: old y cordinate of the border point.
    :param xb_old: new x cordinate of the border point.
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


def find_corresponding_point(xb_new, yb_new, x, y, xb_old, yb_old, guardian=True):
    """
    Depending on which (xb_new, yb_new) is different from none, find the other one constraining it on the original border
    from fzero function.
    :param xb_new: new x cordinate of the border point.
    :param yb_new: new y cordinate of the border point.
    :param x: set of x points of the border curve.
    :param y: set of y points of the border curve.
    :param yb_old: old y cordinate of the border point.
    :param xb_old: old x cordinate of the border point.
    :param guardian: if True it enables under-relaxation to improve stability
    """
    print("WARNING: method not stable. If it diverges, consider passing to find_optimized_point()")
    Deltax = np.max(x) - np.min(x)
    Deltay = np.max(y) - np.min(y)
    tol_x = Deltax / 5
    tol_y = Deltay / 5
    u = np.linspace(0, 1, len(x))  # curve parameterization
    degree = 10

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
        # this is the problem
        u_root = fsolve(zero_x_fy, x0=xb_new)
        yb_new = interp_y(u_root)
        if guardian:
            if (np.abs(yb_new - yb_old) > tol_y or u_root > 1 or u_root < 0):
                print("im not finding the right point")
                xb_new = xb_old
                yb_new = yb_old
    else:
        u_root = fsolve(zero_y_fx, x0=yb_new)
        xb_new = interp_x(u_root)
        if guardian:
            if (np.abs(xb_new - xb_old) > tol_x or u_root > 1 or u_root < 0):
                print("im not finding the right point")
                xb_new = xb_old
                yb_new = yb_old

    return xb_new, yb_new


def find_optimized_point(xb_new, yb_new, x, y, xp_new, yp_new):
    """
    Depending on which (xb_new, yb_new) is different from none, find the updated border point through an optimization problem,
    minimizing the distance of the new point distance from the old point.
    xb_old, yb_old are the cordinates of the previous iteration
    :param xb_new: new x cordinate of the border point.
    :param yb_new: new y cordinate of the border point.
    :param x: set of x points of the border curve.
    :param y: set of y points of the border curve.
    :param xp_new: new x cordinate of the inner point close to the border.
    :param yp_new: new y cordinate of the inner point close to the border.
    """
    u = np.linspace(0, 1, len(x))  # curve parameterization
    degree = 7

    # Perform polynomial interpolation
    coefficients = np.polyfit(u, x, degree)
    interp_x = np.poly1d(coefficients)

    coefficients = np.polyfit(u, y, degree)
    interp_y = np.poly1d(coefficients)

    def objective(u_param, xpoint, ypoint):
        y_u = interp_y(u_param)
        x_u = interp_x(u_param)
        obj = (y_u - ypoint) ** 2 + (x_u - xpoint) ** 2
        return obj

    initial_guess = np.mean(u)  # initial guess for u
    u_bounds = (np.min(u), np.max(u))  # u cannot go outside the initial parameterization space
    u_result = minimize(objective, initial_guess, args=(xp_new, yp_new), bounds=[u_bounds])
    xb_new = interp_x(u_result.x)
    yb_new = interp_y(u_result.x)
    return xb_new, yb_new


def compute_picture_size(x, y):
    """
    given the x and y dimension of a domain, compute the picture size in order to be scaled. Blank stands for the size
    without using colorbars, contour for cases where a part of the width (e.g. 10%) is used for the colorbar.
    :param x: grid of x cordinates
    :param y: grid of y cordinates
    """
    W = np.max(x) - np.min(x)
    H = np.max(y) - np.min(y)
    WH_ratio = W / H
    color_bar_span = 0.15
    if WH_ratio >= 1:
        pic_size_blank = (8, 8 / WH_ratio)
        pic_size_contour = (8*(1+color_bar_span), 8 / WH_ratio)
    else:
        pic_size_blank = (6 * WH_ratio, 6)
        pic_size_contour = (6 * WH_ratio*(1+color_bar_span), 6)
    return pic_size_blank, pic_size_contour
