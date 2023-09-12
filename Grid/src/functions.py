#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 18:29:29 2023
@author: F. Neri, TU Delft
"""
import numpy as np
from numpy import sqrt, sin, cos, tan, arccos, arcsin
from .styles import *


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
    # Scale the sigmoid values to the range 0 to 1
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
