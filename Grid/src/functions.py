#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 18:29:29 2023
@author: F. Neri, TU Delft
"""
import os.path
import matplotlib.pyplot as plt
import sys
import numpy as np
from numpy import sin, cos
import pickle
from Utils.styles import *
import math
from scipy.optimize import fsolve
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline, griddata
from scipy import interpolate
from shapely.geometry import LineString


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


def project_scalar_gradient_to_cylindrical(da_dx, da_dy, da_dz, r, theta):
    """
    Project the x-y gradients of a scalar field "a" in r-theta components.
    :param da_dx: x-derivative of the scalar field
    :param da_dy: y-derivative of the scalar field
    :param da_dz: z-derivative of the scalar field
    :param r: radial cordinate of the point
    :param theta: theta location of the considered point [rad]
    """
    da_dr = da_dx * cos(theta) + da_dy * sin(theta)
    da_dtheta = r * (-da_dx * sin(theta) + da_dy * cos(theta))
    return da_dr, da_dtheta, da_dz


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
    # dur_dtheta = r * (dux_dy * cos(theta) ** 2 + (duy_dy - dux_dx) * sin(theta) * cos(theta) - duy_dx * sin(theta) ** 2)
    dut_dr = duy_dx * cos(theta) ** 2 + (duy_dy - dux_dx) * sin(theta) * cos(theta) - dux_dy * sin(theta) ** 2
    # dut_dtheta = r * (dux_dx * sin(theta) ** 2 - sin(theta) * cos(theta) * (duy_dx + dux_dy) + duy_dy * cos(theta) ** 2)

    return dur_dr, dut_dr


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
    M = np.array([[np.cos(theta), np.sin(theta), 0], [-np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

    return M


def elliptic_grid_generation(c_left, c_bottom, c_right, c_top, orthogonality, x_stretching, y_stretching, tol=1e-3,
                             save_filename=None, show=True, pol_order=3, sigmoid_coeff_x=5, sigmoid_coeff_y=5, it_orth=-1,
                             guardian=False, method='intersection', fix_inlet=False, fix_outlet=False, save_animation=False,
                             border_adjustment=False, inlet_block=False, outlet_block=False):
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
    :param tol: threshold to stop iteration.
    :param save_filename: if set, saves the figure.
    :param show: if True shows the animation
    :param pol_order: polynomial order of the strecthing function (not validated yet.)
    :param sigmoid_coeff_x: x-coefficient of the sigmoid along the streamwise direction.
    :param sigmoid_coeff_y: y-coefficient of the sigmoid along the spanwise direction.
    :param it_orth: iteration number from which orthogonality is enabled.
    :param guardian: under-relaxation method
    :param method: choose method used to update points on the border. Suggested minimize, as fzero frequently diverges, and
    intersection doesn't work 100% good at the moment.
    :param fix_inlet: if True, fixes the points on the inlet border
    :param fix_outlet: if True, fixes the points on the outlet border.
    :param save_animation: if specified,  saves figures with the defined filename path.
    """
    nx = np.shape(c_bottom)[1]
    ny = np.shape(c_left)[1]
    maxit = 1000

    # computational domain between 0 and 1 in both directions
    xi = np.linspace(0, 1, nx)
    dxi = xi[1] - xi[0]
    eta = np.linspace(0, 1, ny)
    deta = eta[1] - eta[0]

    X = np.zeros((nx, ny))
    Y = np.zeros((nx, ny))

    # if initial grid is not given, find it via interpolation of the borders
    X[0, :] = sample_spline(c_left[0, :], sample_method=y_stretching, sample_coeff=sigmoid_coeff_y, sampling_points=ny)
    Y[0, :] = sample_spline(c_left[1, :], sample_method=y_stretching, sample_coeff=sigmoid_coeff_y, sampling_points=ny)
    X[:, 0] = sample_spline(c_bottom[0, :], sample_method=x_stretching, sample_coeff=sigmoid_coeff_x, sampling_points=nx,
                            inlet_block=inlet_block, outlet_block=outlet_block)
    Y[:, 0] = sample_spline(c_bottom[1, :], sample_method=x_stretching, sample_coeff=sigmoid_coeff_x, sampling_points=nx,
                            inlet_block=inlet_block, outlet_block=outlet_block)
    X[-1, :] = sample_spline(c_right[0, :], sample_method=y_stretching, sample_coeff=sigmoid_coeff_y, sampling_points=ny)
    Y[-1, :] = sample_spline(c_right[1, :], sample_method=y_stretching, sample_coeff=sigmoid_coeff_y, sampling_points=ny)
    X[:, -1] = sample_spline(c_top[0, :], sample_method=x_stretching, sample_coeff=sigmoid_coeff_x, sampling_points=nx,
                             inlet_block=inlet_block, outlet_block=outlet_block)
    Y[:, -1] = sample_spline(c_top[1, :], sample_method=x_stretching, sample_coeff=sigmoid_coeff_x, sampling_points=nx,
                             inlet_block=inlet_block, outlet_block=outlet_block)

    for istream in range(1, nx - 1):
        for ispan in range(1, ny - 1):
            X[istream, ispan] = X[istream, 0] + (X[istream, -1] - X[istream, 0]) * ispan / (ny - 1)
            Y[istream, ispan] = Y[istream, 0] + (Y[istream, -1] - Y[istream, 0]) * ispan / (ny - 1)
    X0 = X
    Y0 = Y

    # compute stretching functions, even in the case of no-stretching to avoid messing up things
    f1 = np.zeros((nx, ny))
    f1_prime = np.zeros((nx, ny))
    f1_second = np.zeros((nx, ny))
    f2 = np.zeros((nx, ny))
    f2_prime = np.zeros((nx, ny))
    f2_second = np.zeros((nx, ny))
    for ispan in range(ny):
        if x_stretching == 'default':
            f1[:, ispan], f1_prime[:, ispan], f1_second[:, ispan] = no_stretching_function(xi)
        elif x_stretching == 'sigmoid':
            f1[:, ispan], f1_prime[:, ispan], f1_second[:, ispan] = scaled_sigmoid(xi, sigmoid_coeff_x)
        elif x_stretching == 'gauss-lobatto':
            f1[:, ispan], f1_prime[:, ispan], f1_second[:, ispan] = scaled_gauss_lobatto(xi, inlet_block=inlet_block,
                                                                                         outlet_block=outlet_block)
        elif x_stretching == 'sigmoid_right':
            f1[:, ispan], f1_prime[:, ispan], f1_second[:, ispan] = scaled_sigmoid_right(xi, sigmoid_coeff_x)
        elif x_stretching == 'sigmoid_left':
            f1[:, ispan], f1_prime[:, ispan], f1_second[:, ispan] = scaled_sigmoid_left(xi, sigmoid_coeff_x)
        elif x_stretching == 'polynomial':
            f1[:, ispan], f1_prime[:, ispan], f1_second[:, ispan] = polynomial_function(xi, pol_order)
        else:
            raise ValueError('Check the value of x-strecthing parameter!')
    for istream in range(0, nx):
        if y_stretching == 'default':
            f2[istream, :], f2_prime[istream, :], f2_second[istream, :] = no_stretching_function(eta)
        elif y_stretching == 'sigmoid':
            f2[istream, :], f2_prime[istream, :], f2_second[istream, :] = scaled_sigmoid(eta, sigmoid_coeff_y)
        elif y_stretching == 'gauss-lobatto':
            f2[istream, :], f2_prime[istream, :], f2_second[istream, :] = scaled_gauss_lobatto(eta)
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
            # plt.plot(c_left[0, :], c_left[1, :], 'red', lw=0.5)
            # plt.plot(c_bottom[0, :], c_bottom[1, :], 'red', lw=0.5)
            # plt.plot(c_right[0, :], c_right[1, :], 'red', lw=0.5)
            # plt.plot(c_top[0, :], c_top[1, :], 'red', lw=0.5)
            plt.plot(X0[0, :], Y0[0, :], 'red', lw=0.5)
            plt.plot(X0[-1, :], Y0[-1, :], 'red', lw=0.5)
            plt.plot(X0[:, -1], Y0[:, -1], 'red', lw=0.5)
            plt.plot(X0[:, 0], Y0[:, 0], 'red', lw=0.5)
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
        g12[i, j] = ((X[ip, j] - X[im, j]) / 2 / dxi) * ((X[i, jp] - X[i, jm]) / 2 / deta) + ((Y[ip, j] - Y[im, j]) / 2 / dxi) * (
                (Y[i, jp] - Y[i, jm]) / 2 / deta)

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
        if border_adjustment and it > it_orth:

            # BOTTOM EDGE
            x = c_bottom[0, :]
            y = c_bottom[1, :]
            # x = X[:, 0]
            # y = Y[:, 0]
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
                elif method == 'intersection':
                    xb_new, yb_new = find_updated_point(X0[:, 0], Y0[:, 0], xp_new, yp_new, xb_old, yb_old, yb_prime)
                else:
                    raise ValueError('Select a valid method for borders adjustment!')
                X[istream, 0] = xb_new
                Y[istream, 0] = yb_new

            # TOP EDGE
            x = c_top[0, :]
            y = c_top[1, :]
            # x = X[:, -1]
            # y = Y[:, -1]
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
                elif method == 'intersection':
                    xb_new, yb_new = find_updated_point(X0[:, -1], Y0[:, -1], xp_new, yp_new, xb_old, yb_old, yb_prime)
                else:
                    raise ValueError('Select a valid method for borders adjustment!')
                X[istream, -1] = xb_new
                Y[istream, -1] = yb_new

            # LEFT EDGE
            if not fix_inlet:
                x = c_left[0, :]
                y = c_left[1, :]
                # x = X[0, :]
                # y = Y[0, :]
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
                    elif method == 'intersection':
                        xb_new, yb_new = find_updated_point(X0[0, :], Y0[0, :], xp_new, yp_new, xb_old, yb_old, yb_prime)
                    else:
                        raise ValueError('Select a valid method for borders adjustment!')
                    X[0, ispan] = xb_new
                    Y[0, ispan] = yb_new

            # RIGHT EDGE
            if not fix_outlet:
                x = c_right[0, :]
                y = c_right[1, :]
                # x = X[-1, :]
                # y = Y[-1, :]
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
                    elif method == 'intersection':
                        xb_new, yb_new = find_updated_point(X0[-1, :], Y0[-1, :], xp_new, yp_new, xb_old, yb_old, yb_prime)
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


def transfinite_grid_generation(c_left, c_bottom, c_right, c_top, block_topology, streamwise_coeff, spanwise_coeff, nx=None,
                                ny=None):
    """
    Method used to generate the grid with transfinite grid interpolation method.
    :param c_left: left border points (x, y)
    :param c_bottom: bottom border points (x, y)
    :param c_right: right border points (x, y)
    :param c_top: top border points (x, y)
    :param block_topology: inlet, internal, or outlet to impose the right stretching functions
    :param streamwise_coeff: coefficient of stretching function along streamwise direction
    :param spanwise_coeff: coefficient of stretching function along spanwise direction
    :param nx: number of points in streamwise direction (if None, default is used)
    :param ny: number of points in spanwise direction (if None, default is used)
    """
    if nx is None:
        nx = c_bottom.shape[1]
    if ny is None:
        ny = c_left.shape[1]

    t_streamwise = np.linspace(0, 1, nx)
    t_spanwise = np.linspace(0, 1, ny)

    splinex_bottom = CubicSpline(t_streamwise, c_bottom[0, :])
    spliney_bottom = CubicSpline(t_streamwise, c_bottom[1, :])

    splinex_top = CubicSpline(t_streamwise, c_top[0, :])
    spliney_top = CubicSpline(t_streamwise, c_top[1, :])

    splinex_left = CubicSpline(t_spanwise, c_left[0, :])
    spliney_left = CubicSpline(t_spanwise, c_left[1, :])

    splinex_right = CubicSpline(t_spanwise, c_right[0, :])
    spliney_right = CubicSpline(t_spanwise, c_right[1, :])

    plt.figure()
    plt.plot(splinex_bottom(t_streamwise), spliney_bottom(t_streamwise), '-o', label='spline bottom border')
    plt.plot(splinex_top(t_streamwise), spliney_top(t_streamwise), '-s', label='spline top border')
    plt.plot(splinex_left(t_spanwise), spliney_left(t_spanwise), '-^', label='spline left border')
    plt.plot(splinex_right(t_spanwise), spliney_right(t_spanwise), '-x', label='spline right border')
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.legend()
    

    xi = np.linspace(0, 1, nx)
    eta = np.linspace(0, 1, ny)

    # stretching functions applied to the computational cordinates. if the coefficients are equal to 1 and 1, no stretching
    # (this is needed because eriksson with a value of 1 is different from no stretching)
    if streamwise_coeff != 1:
        if block_topology.lower() == 'inlet':
            xi = eriksson_stretching_function_final(xi, streamwise_coeff)
        elif block_topology.lower() == 'internal':
            xi = eriksson_stretching_function_both(xi, streamwise_coeff)
        elif block_topology.lower() == 'outlet':
            xi = eriksson_stretching_function_initial(xi, streamwise_coeff)
        else:
            raise ValueError('Unrecognized block topology')

    if spanwise_coeff != 1:
        eta = eriksson_stretching_function_both(eta, spanwise_coeff)

    XI, ETA = np.meshgrid(xi, eta, indexing='ij')
    X, Y = XI * 0, ETA * 0

    # TRANSFINITE INTERPOLATION
    for i in range(nx):
        for j in range(ny):
            X[i, j] = (1 - XI[i, j]) * splinex_left(ETA[i, j]) + XI[i, j] * splinex_right(ETA[i, j]) + (
                    1 - ETA[i, j]) * splinex_bottom(XI[i, j]) + ETA[i, j] * splinex_top(XI[i, j]) - (1 - XI[i, j]) * (
                              1 - ETA[i, j]) * splinex_left(0) - (1 - XI[i, j]) * ETA[i, j] * splinex_left(1) - (1 - ETA[i, j]) * \
                      XI[i, j] * splinex_right(0) - XI[i, j] * ETA[i, j] * splinex_right(1)

            Y[i, j] = (1 - XI[i, j]) * spliney_left(ETA[i, j]) + XI[i, j] * spliney_right(ETA[i, j]) + (
                    1 - ETA[i, j]) * spliney_bottom(XI[i, j]) + ETA[i, j] * spliney_top(XI[i, j]) - (1 - XI[i, j]) * (
                              1 - ETA[i, j]) * spliney_left(0) - (1 - XI[i, j]) * ETA[i, j] * spliney_left(1) - (1 - ETA[i, j]) * \
                      XI[i, j] * spliney_right(0) - XI[i, j] * ETA[i, j] * spliney_right(1)

    plt.figure()
    for i in range(nx):
        plt.plot(X[i, :], Y[i, :], 'k', lw=0.5)
    for j in range(ny):
        plt.plot(X[:, j], Y[:, j], 'k', lw=0.5)
    
    ax = plt.gca()
    ax.set_aspect('equal')

    return X, Y


def eriksson_stretching_function_initial(x, alpha):
    """
    equation 4.93 Farrashkhalvat Book. Gives clustering at the initial part of the computational domain
    """
    f = (np.exp(alpha * x) - 1) / (np.exp(alpha) - 1)
    # plt.figure()
    # plt.plot(x, f, '-o', label=r'$f/f_{max}$')
    # plt.xlabel(r'$x$')
    # plt.xlabel(r'$f$')
    # plt.grid(alpha=grid_opacity)
    return f


def eriksson_stretching_function_final(x, alpha):
    """
    equation 4.95 Farrashkhalvat Book. Gives clustering at the final part of the computational domain
    """
    f = (np.exp(alpha) - np.exp(alpha * (1 - x))) / (np.exp(alpha) - 1)
    # plt.figure()
    # plt.plot(x, f, '-o', label=r'$f/f_{max}$')
    # plt.xlabel(r'$x$')
    # plt.xlabel(r'$f$')
    # plt.grid(alpha=grid_opacity)
    return f


def eriksson_stretching_function_both(x, alpha):
    """
    equation 4.97 Farrashkhalvat Book. Gives clustering at the initial and final part of the computational domain
    """
    x_midpoint = x[len(x)//2]
    f = np.zeros_like(x)

    for i in range(len(x)):
        if x[i] <= x_midpoint:
            f[i] = x_midpoint * (np.exp(alpha * x[i] / x_midpoint) - 1) / (np.exp(alpha) - 1)
        else:
            f[i] = 1 - (1 - x_midpoint) * (np.exp(alpha * (1 - x[i]) / (1 - x_midpoint)) - 1) / (np.exp(alpha) - 1)

    # plt.figure()
    # plt.plot(x, f, '-o', label=r'$f/f_{max}$')
    # plt.xlabel(r'$x$')
    # plt.ylabel(r'$f$')
    # plt.grid(alpha=grid_opacity)
    return f


def scaled_sigmoid(x, alpha):
    """
    Return sigmoid scaled function, first derivative, and second derivative over an array x.
    :param x: array of sigmoid argument
    :param alpha: coefficient of the sigmoid function f(x) = 1 / (1 + np.exp(-alpha * (x - 0.5)))
    """
    f = 1 / (1 + np.exp(-alpha * (x - 0.5)))
    f_prime = (alpha * np.exp(-alpha * (x - 0.5))) / (1 + np.exp(-alpha * (x - 0.5))) ** 2
    f_second = (-alpha ** 2 * np.exp(-alpha * (x - 0.5)) * (1 + np.exp(-alpha * (x - 0.5))) + 2 * alpha ** 2 * np.exp(
        -2 * alpha * (x - 0.5))) / (1 + np.exp(-alpha * (x - 0.5))) ** 3

    plt.figure()
    plt.plot(x, f, '-o', label=r'$f/f_{max}$')
    # plt.plot(x, f_prime / np.max(f_prime), '-o', label=r"$f ' / f' _{max}$")
    # plt.plot(x, f_second / np.max(f_second), '-o', label=r"$f'' / f''_{max}$")
    plt.xlabel(r'$\xi$')
    plt.xlabel(r'$\eta$')
    plt.grid(alpha=grid_opacity)
    plt.legend()

    return f, f_prime, f_second


def scaled_gauss_lobatto(x, inlet_block=False, outlet_block=False):
    """
    Return a gauss-lobatto spacing from [0,1] to [0,1].
    :param x: array of sigmoid argument
    """
    f = 0.5 * (1 - np.cos(np.pi * x))
    f_prime = np.pi / 2 * np.sin(np.pi * x)
    f_second = np.pi ** 2 / 2 * np.cos(np.pi * x)

    if inlet_block:
        f[0:len(f) // 2] = np.linspace(0, 0.5, len(f) // 2)
        f_prime[0:len(f) // 2] = np.ones(len(f) // 2)
        f_second[0:len(f) // 2] = np.zeros(len(f) // 2)
    if outlet_block:
        f[len(f) // 2:] = np.linspace(0.5, 1, len(f) // 2)
        f_prime[len(f) // 2:] = np.ones(len(f) // 2)
        f_second[len(f) // 2:] = np.zeros(len(f) // 2)

    # plt.figure()
    # plt.plot(x, f / (np.max(f)-np.min(f)), label=r'$f_{scaled}$')
    # plt.plot(x, f_prime / (np.max(f_prime)-np.min(f_prime)), label=r"$f '_{scaled}$")
    # plt.plot(x, f_second / (np.max(f_second)-np.min(f_second)), label=r"$f''_{scaled}$")
    # plt.xlabel(r'$\zeta$')
    # plt.ylabel(r'$f$')
    # plt.grid(alpha=grid_opacity)
    # plt.legend()
    # plt.savefig('stretching_functions.pdf', bbox_inches='tight')

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
        A_sys = np.array([[1 / yb_prime, 1], [-yb_prime, 1]])
        B_sys = np.array([yp_new + xp_new / yb_prime, yb_old - yb_prime * xb_old])
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


def find_updated_point(x, y, xp_new, yp_new, xb_old, yb_old, yb_prime):
    """
    Depending on which (xb_new, yb_new) is different from none, find the updated border point through an optimization problem,
    minimizing the distance of the new point distance from the old point.
    xb_old, yb_old are the cordinates of the previous iteration
    :param x: set of x points of the border curve.
    :param y: set of y points of the border curve.
    :param xp_new: new x cordinate of the inner point close to the border.
    :param yp_new: new y cordinate of the inner point close to the border.
    :param xb_old: old x cordinate of the border point.
    :param yb_old: old y cordinate of the border point.
    :param yb_prime: derivative of the function at the border
    """
    from intersect import intersection
    u = np.linspace(-0.1, 0.1, 5000)
    from scipy.interpolate import make_interp_spline
    u_border = np.linspace(0, 1, len(x))
    xspline_border = make_interp_spline(u_border, x, k=1)
    yspline_border = make_interp_spline(u_border, y, k=1)
    x_grid_line = xp_new + u * (-yb_prime)
    y_grid_line = yp_new + u * 1
    xb_new, yb_new = intersection(x_grid_line, y_grid_line, xspline_border(u_border), yspline_border(u_border))
    try:
        xb_new, yb_new = xb_new[0], yb_new[0]
    except:
        xb_new, yb_new = xb_old, yb_old

    # plt.figure()
    # plt.plot(xspline_border(u_border), yspline_border(u_border), label='border line')
    # plt.plot(xp_new, yp_new, 'ko', label='internal point')
    # plt.plot(x_grid_line, y_grid_line, 'k', label='internal grid line')
    # plt.plot(xb_new, yb_new, 'ro', label='intersection')
    # # plt.xlim([-0.1, 0.4])
    # # plt.ylim([0.6, 1.1])
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.legend()

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
    degree = 10

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


def sample_spline(x, sample_method, sample_coeff, sampling_points, inlet_block=False, outlet_block=False):
    """
    Sample the curve denoted by a generic x-cordinate, parametrized as cubic spline, in a certain method.
    :param x:cordinates
    :param sample_method: method used to sample (linear, sigmoid, sigmoid_left, sigmoid_right)
    :param sample_coeff: coefficient of the sigmoid slope (alpha parameters)
    :param sampling_points: number of sampling points
    """
    t = np.linspace(0, 1, sampling_points)
    spline = CubicSpline(t, x)

    if sample_method == 'default':
        t_scaled = t.copy()
    elif sample_method == 'sigmoid':
        t_scaled = scaled_sigmoid(t, sample_coeff)[0]
    elif sample_method == 'sigmoid_left' or sample_method == 'sigmoid_down':
        t_scaled = scaled_sigmoid_left(t, sample_coeff)[0]
    elif sample_method == 'sigmoid_right' or sample_method == 'sigmoid_up':
        t_scaled = scaled_sigmoid_right(t, sample_coeff)[0]
    elif sample_method == 'gauss-lobatto':
        t_scaled = scaled_gauss_lobatto(t, inlet_block, outlet_block)[0]
    else:
        raise ValueError("Unrecognized sample method")
    t_scaled[0] = 0
    t_scaled[-1] = 1
    x_spline = spline(t_scaled)

    # plt.figure()
    # plt.plot(t, x, 'k')
    # plt.plot(t_scaled, x_spline, 'ro')
    return x_spline


def project_2d_gradient_to_cylindrical(du_dx, du_dy, r, theta):
    """
    Project a gradient in cartesian cordinates in cylindrical cordinates
    """
    r_vers = np.array([cos(theta), sin(theta)])
    theta_vers = np.array([-sin(theta), cos(theta)])
    grad_xy = np.array([du_dx, du_dy])
    du_dr = np.dot(grad_xy, r_vers)
    du_dtheta = np.dot(grad_xy, theta_vers)
    return du_dr, r * du_dtheta


def print_object_memory_info(Object):
    """
    For the object argument, print the information related to the memory usage
    """
    tot_size = 0
    for attribute_name in dir(Object):
        attribute = getattr(Object, attribute_name)
        size_in_bytes = sys.getsizeof(attribute)
        tot_size += size_in_bytes
        if size_in_bytes < 1000:
            print(f"Size of {attribute_name}: {size_in_bytes} bytes")
        elif 1e3 <= size_in_bytes <= 1e6:
            print(f"Size of {attribute_name}: {size_in_bytes / 1e3} kbytes")
        elif 1e6 <= size_in_bytes <= 1e9:
            print(f"Size of {attribute_name}: {size_in_bytes / 1e6} Mbytes")
        else:
            print(f"Size of {attribute_name}: {size_in_bytes / 1e9} Gbytes")
    print(f"Total size: {tot_size / 1e6} Mbytes")


def create_folder(foldername):
    """
    If a folder doesn't exist, create it.
    """
    if not os.path.exists(foldername):
        os.makedirs(foldername)


def compute_2d_curvilinear_gradient(z, r, f):
    """
    Compute a gradient of the field f, defined on z,r (2d arrays), where the coordinate lines may be curvilinear.
    Use of central finite differences. Deprecated, use the least square version
    """
    nstream = z.shape[0]
    nspan = z.shape[1]
    dfdz = np.zeros_like(f)
    dfdr = np.zeros_like(f)
    for ii in range(0, nstream):
        for jj in range(0, nspan):

            # selection of proper stencil
            if ii == 0 and jj == 0:
                ip = 1
                im = 0
                jp = 1
                jm = 0
            elif ii == 0 and jj == nspan - 1:
                ip = 1
                im = 0
                jp = 0
                jm = -1
            elif ii == nstream - 1 and jj == 0:
                ip = 0
                im = -1
                jp = 1
                jm = 0
            elif ii == nstream - 1 and jj == nspan - 1:
                ip = 0
                im = -1
                jp = 0
                jm = -1
            elif ii == 0:
                ip = 1
                im = 0
                jp = 1
                jm = -1
            elif ii == nstream - 1:
                ip = 0
                im = -1
                jp = 1
                jm = -1
            elif jj == 0:
                ip = 1
                im = -1
                jp = 1
                jm = 0
            elif jj == nspan - 1:
                ip = 1
                im = -1
                jp = 0
                jm = -1
            else:
                ip = 1
                im = -1
                jp = 1
                jm = -1

            dstream = np.array([z[ii + ip, jj] - z[ii + im, jj], r[ii + ip, jj] - r[ii + im, jj]])
            dspan = np.array([z[ii, jj + jp] - z[ii, jj + jm], r[ii, jj + jp] - r[ii, jj + jm]])

            dstream_mag = np.linalg.norm(dstream)
            dspan_mag = np.linalg.norm(dspan)

            dfdstream = (f[ii + ip, jj] - f[ii + im, jj]) / dstream_mag
            dfdspan = (f[ii, jj + jp] - f[ii, jj + jm]) / dspan_mag

            dfdz[ii, jj] = dfdstream * dstream[0] / dstream_mag + dfdspan * dspan[0] / dspan_mag
            dfdr[ii, jj] = dfdstream * dstream[1] / dstream_mag + dfdspan * dspan[1] / dspan_mag

    return dfdz, dfdr


def clip_negative_values(f):
    """
    Remove all the negative values from a numpy array `f`
    """
    return np.sqrt(f ** 2)


def compute_curvilinear_abscissa(x, y):
    """
    having 2 arrays of data points on a curve in a x-y plane, compute the curvilinear abscissa coordinate for every point
    """
    s = np.zeros_like(x)
    for i in range(1, len(x)):
        s[i] = s[i - 1] + np.sqrt((x[i] - x[i - 1]) ** 2 + (y[i] - y[i - 1]) ** 2)
    if x[0] < x[-1]:
        return s
    else:
        return np.flip(s)


def compute_3dSpline_curve(x, y, z, num_points=250, u_param=None, spacing=None):
    """
    Given points in the space x,y,z, return the points lying on the spline passing throug them
    """
    numberPoints = len(x)
    uniquePointsX = np.unique(x)
    uniquePointsY = np.unique(y)
    uniquePointsZ = np.unique(z)
    
    if numberPoints != uniquePointsX.size or numberPoints != uniquePointsY.size or numberPoints != uniquePointsZ.size:
        print("Warning: the points are not unique")
        uniqueIndicesX = np.unique(x, return_index=True)[1]
        uniqueIndicesY = np.unique(y, return_index=True)[1]
        uniqueIndicesZ = np.unique(z, return_index=True)[1]
        
        
        if len(uniqueIndicesX)==1:
            if len(uniquePointsY)<len(uniquePointsZ):
                x, y, z = x[np.sort(uniqueIndicesY)], y[np.sort(uniqueIndicesY)], z[np.sort(uniqueIndicesY)]
            else:
                x, y, z = x[np.sort(uniqueIndicesZ)], y[np.sort(uniqueIndicesZ)], z[np.sort(uniqueIndicesZ)]
        
        elif len(uniqueIndicesY)==1:
            if len(uniquePointsX)<len(uniquePointsZ):
                x, y, z = x[np.sort(uniqueIndicesX)], y[np.sort(uniqueIndicesX)], z[np.sort(uniqueIndicesX)]
            else:   
                x, y, z = x[np.sort(uniqueIndicesZ)], y[np.sort(uniqueIndicesZ)], z[np.sort(uniqueIndicesZ)]
        
        elif len(uniqueIndicesZ)==1:
            if len(uniquePointsX)<len(uniquePointsY):
                x, y, z = x[np.sort(uniqueIndicesX)], y[np.sort(uniqueIndicesX)], z[np.sort(uniqueIndicesX)]
            else:
                x, y, z = x[np.sort(uniqueIndicesY)], y[np.sort(uniqueIndicesY)], z[np.sort(uniqueIndicesY)]
        
        elif len(uniqueIndicesX)<len(uniqueIndicesY) or len(uniqueIndicesX)<len(uniqueIndicesZ):
            x, y, z = x[np.sort(uniqueIndicesX)], y[np.sort(uniqueIndicesX)], z[np.sort(uniqueIndicesX)]
        
        elif len(uniqueIndicesY)<len(uniqueIndicesX) or len(uniqueIndicesY)<len(uniqueIndicesZ):
            x, y, z = x[np.sort(uniqueIndicesY)], y[np.sort(uniqueIndicesY)], z[np.sort(uniqueIndicesY)]
        
        else:
            x, y, z = x[np.sort(uniqueIndicesZ)], y[np.sort(uniqueIndicesZ)], z[np.sort(uniqueIndicesZ)]

    tck, u = interpolate.splprep([x, y, z], s=0, k=1)
    u_fine = np.linspace(0, 1, num_points)
    if u_param is not None:
        u_fine = u_param
    if spacing is not None:
        u_fine = eriksson_stretching_function_both(u_fine, spacing)
    xnew, ynew, znew = interpolate.splev(u_fine, tck)
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot(x, y, z, '-o', label='sample points filtered')
    # ax.plot(xnew, ynew, znew, '-o', label='spline')
    # ax.legend()    
    
    return xnew, ynew, znew


def compute_2dSpline_curve(x, y, num_points, spacing=None):
    """
    Given points in the space x,y, return the points lying on the spline passing throug them
    """
    tck, u = interpolate.splprep([x, y], s=0, k=3)
    u_fine = np.linspace(0, 1, num_points)
    if spacing is not None:
        u_fine = eriksson_stretching_function_both(u_fine, spacing)
    x, y = interpolate.splev(u_fine, tck)
    return x, y


def find_intersection(x1, y1, x2, y2):
    """
    Given two curves on a x-y plane find and return the intersection points (if any).
    """
    line_1 = LineString(np.column_stack((x1, y1)))
    line_2 = LineString(np.column_stack((x2, y2)))
    intersection = line_1.intersection(line_2)
    try:
        return intersection.xy[0], intersection.xy[1]
    except:
        return 0, 0


def compute_gradient_least_square(x, y, z):
    """
    Compute the gradient of z with respect to x and y using the least squares method.

    Parameters:
        x (numpy.ndarray): 2D array of x-coordinates.
        y (numpy.ndarray): 2D array of y-coordinates.
        z (numpy.ndarray): 2D array of z-values, a function of x and y.

    Returns:
        Gradients dz/dx and dz/dy.
    """
    ni,nj = x.shape
    dzdx = np.zeros((ni,nj))
    dzdy = np.zeros((ni,nj))

    for i in range(ni):
        for j in range(nj):
            
            # take the i,j of the neighbours to use depending on the location on the grid
            if i==0 and j==0:
                neighbours = [(1,0), (1,1), (0,1)]
            elif i==ni-1 and j==0:
                neighbours = [(-1,0), (-1,1), (0,1)]
            elif i==0 and j==nj-1:
                neighbours = [(0,-1), (1,-1), (1,0)]
            elif i==ni-1 and j==nj-1:
                neighbours = [(-1,0), (-1,-1), (0,-1)]
            elif i==0:
                neighbours = [(0,-1), (1,-1), (1,0), (1,1), (0,1)]
            elif i==ni-1:
                neighbours = [(0,-1), (-1,-1), (-1,0), (-1,1), (0,1)]
            elif j==0:
                neighbours = [(-1,0), (-1,1), (0,1), (1,1), (1,0)]
            elif j==nj-1:
                neighbours = [(-1,0), (-1,-1), (0,-1), (1,-1), (1,0)]
            else:
                neighbours = [(-1,-1), (0,-1), (0,1), (-1,0), (1,0), (-1,1), (0,1), (1,1)] 

            A = np.zeros((len(neighbours), 2))
            b = np.zeros((len(neighbours), 1))
            grad = np.zeros((2,1))
            for k in range(len(neighbours)):
                istep, jstep = neighbours[k]
                A[k,0] = x[i+istep, j+jstep]-x[i,j]
                A[k,1] = y[i+istep, j+jstep]-y[i,j]
                b[k,0] = z[i+istep, j+jstep]-z[i,j]
            grad = np.linalg.inv(A.T@A) @A.T@b
            dzdx[i,j] = grad[0,0]
            dzdy[i,j] = grad[1,0]
    return dzdx, dzdy


def contour_template(z, r, f, name, vmin=None, vmax=None):
        """
        Template function to create contours.

        Parameters
        -----------------------------------

        `z`: 2d array of x coordinates
        
        `r`: 2d array of y coordinates

        `f`: 2d array of function values

        `name`: string name of the plot title

        `vmin`: minimum value to truncate the color range

        `vmax`: max value to truncate the color range

        """
        if vmin == None:
            minval = np.min(f)
        else:
            minval = vmin
        if vmax == None:
            maxval = np.max(f)
        else:
            maxval = vmax
        levels = np.linspace(minval, maxval, N_levels)
        fig, ax = plt.subplots()
        contour = ax.contourf(z, r, f, levels=levels, cmap=color_map, vmin = minval, vmax = maxval)
        cbar = fig.colorbar(contour)
        contour = ax.contour(z, r, f, levels=levels, colors='black', vmin = minval, vmax = maxval, linewidths=0.1)
        plt.title(name)
        ax.set_aspect('equal', adjustable='box')


def rotate_cartesian_to_cylindric_tensor(theta, M_cart):
    """
    Express the cartesian tensor (x,y,z) in cylindrical ref frame
    """
    Q = np.array([[cos(theta),  sin(theta), 0],
                  [-sin(theta), cos(theta), 0],
                  [0,           0,          1]])
    M_cyl = (Q.T)@M_cart@Q
    return M_cyl


def griddata_interpolation_with_nearest_filler(xpoints, ypoints, zpoints, x_eval, y_eval, method='linear', filler = 1e10):
    """
    Interpolation using griddata, but the points lying out of the convex hull are treated with nearest neighbor.

    Parameters
    -------------------------------

    `xpoints`: 1 or 2D array of x points where data is known

    `ypoints`: 1 or 2D array of y points where data is known

    `zpoints`: 1D array of function values where data is known, related to `xpoints` and `ypoints`

    `x_eval`: 1 or 2D array where evaluating the function

    `y_eval`: 1 or 2D array where evaluating the function

    `method`: linear or cubic usually

    `filler`: value used to fill and recognize points outside the convex hull
    """
    z_eval = griddata((xpoints.flatten(), ypoints.flatten()), zpoints.flatten(), (x_eval, y_eval), method=method, fill_value=filler)
    
    ni,nj = z_eval.shape

    for i in range(ni):
        for j in range(nj):
            if z_eval[i,j] == filler:
                z_eval[i,j] = griddata((xpoints.flatten(), ypoints.flatten()), zpoints.flatten(), (x_eval[i,j], y_eval[i,j]), method='nearest')

    return z_eval


