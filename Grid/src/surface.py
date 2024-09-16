#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 11:41:53 2023
@author: F. Neri, TU Delft
"""
import warnings

import matplotlib.pyplot as plt
from numpy import array
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from .functions import cartesian_to_cylindrical, compute_2d_curvilinear_gradient
from Sun.src.general_functions import print_banner_begin, print_banner_end
from Utils.styles import total_chars, total_chars_mid
from Grid.src.functions import compute_picture_size, clip_negative_values, compute_curvilinear_abscissa, compute_3dSpline_curve, compute_2dSpline_curve, find_intersection
from Grid.src.profile import Profile
from Utils.styles import *
from scipy import interpolate
import math
import os
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import bisplrep, bisplev
from shapely.geometry import LineString
from scipy.spatial import KDTree

class Surface:
    """
    class used for the surface generation by means of curves lofting.
    """

    def __init__(self, name):
        """
        General constructor.
        :param name: string with the name of the surface
        """
        self.name = name
        self.coords = {}

    def add_curve(self, x, y, z):
        """
        add a curve to the dataset.
        :param x: array of x coordinates
        :param y: array of y coordinates
        :param z: array of z coordinates
        """
        iDum = len(self.coords)
        self.coords['Profile %i' % iDum] = {'x': x, 'y': y, 'z': z}

    def get_number_profiles(self):
        """
        Get the number of profiles
        """
        return len(self.coords)

    def get_number_lofts(self):
        """
        Get the number of surface lofts
        """
        n = len(self.surface)
        assert (n == self.get_number_profiles()-1)
        return n

    def plot_surface(self, surfaces=False):
        """
        Plot the profiles points, and the surface lofts if available.
        :param surfaces: bool value, True to plot the surface lofts
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        if self.get_number_profiles()>0:
            for key, values in self.coords.items():
                ax.plot(values['x'], values['y'], values['z'], lw=3, label=key)
            ax.legend()
            if surfaces:
                for key, values in self.surface.items():
                    ax.plot_surface(values['X'], values['Y'], values['Z'], alpha=0.5)
        else:
            raise ValueError('Not curves stored in the object yet')

    def loft_through_profiles(self, points_along_profile=100, points_between_profiles=40, extension=0.025):
        """
        Use interpolation between 2 profiles to assemble the overall surface of the camber.
        The camber is extended at the borders for 10% in order to cope for following griddata interpolation.
        :param points_along_profile: number of points used to interpolate splines along the generator curves
        :param points_between_profiles: number of points used in between the different lofts
        :param extension: percentage of extension at the borders
        """
        if self.get_number_profiles() < 2:
            raise ValueError('At least two profiles are needed for the loft function')
        self.surface = {}
        keys_list = list(self.coords.keys())
        for iSurf in range(self.get_number_profiles()-1):
            t = np.linspace(0-extension, 1+extension, points_along_profile)  # parameter flowing on one curve tangentially
            if iSurf==0:
                s = np.linspace(0-extension, 1, points_between_profiles)  # parameter connecting one curve to the other
            elif iSurf==self.get_number_profiles()-2:
                s = np.linspace(0, 1+extension, points_between_profiles)
            else:
                s = np.linspace(0, 1, points_between_profiles)

            T, S = np.meshgrid(t, s, indexing='ij')

            xint0, yint0, zint0 = compute_3dSpline_curve(self.coords[keys_list[iSurf]]['x'],
                                                         self.coords[keys_list[iSurf]]['y'],
                                                         self.coords[keys_list[iSurf]]['z'], u_param=t, spacing=3)
            xint1, yint1, zint1 = compute_3dSpline_curve(self.coords[keys_list[iSurf+1]]['x'],
                                                         self.coords[keys_list[iSurf+1]]['y'],
                                                         self.coords[keys_list[iSurf+1]]['z'], u_param=t, spacing=3)

            X, Y, Z = np.zeros_like(T), np.zeros_like(T), np.zeros_like(T)
            for ii in range(T.shape[0]):
                for jj in range(T.shape[1]):
                    X[ii, jj] = xint0[ii] + (xint1[ii] - xint0[ii]) * S[ii, jj]
                    Y[ii, jj] = yint0[ii] + (yint1[ii] - yint0[ii]) * S[ii, jj]
                    Z[ii, jj] = zint0[ii] + (zint1[ii] - zint0[ii]) * S[ii, jj]

            self.surface['Loft %i' % iSurf] = {'X': X, 'Y': Y, 'Z': Z}

    def get_global_surface(self, method):
        """
        Get the coordinates arrays for whole lofted surface.
        :param method: decide between cylindrical (r,theta,z) and cartesian (x, y, z) return
        """
        n_lofts = self.get_number_lofts()
        loft_dims = self.surface['Loft 0']['X'].shape
        Xg = np.zeros((loft_dims[0], loft_dims[1]*n_lofts))
        Yg = np.zeros_like(Xg)
        Zg = np.zeros_like(Xg)
        loft_keys = list(self.surface.keys())
        for iSurf in range(n_lofts):
            Xg[:, iSurf * loft_dims[1]:(iSurf+1) * loft_dims[1]] = self.surface[loft_keys[iSurf]]['X']
            Yg[:, iSurf * loft_dims[1]:(iSurf + 1) * loft_dims[1]] = self.surface[loft_keys[iSurf]]['Y']
            Zg[:, iSurf * loft_dims[1]:(iSurf + 1) * loft_dims[1]] = self.surface[loft_keys[iSurf]]['Z']

        if method=='cartesian':
            return Xg, Yg, Zg
        elif method=='cylindrical':
            return np.sqrt(Xg**2+Yg**2), np.arctan2(Yg,Xg), Zg
        else:
            raise ValueError('Unknown type of return method. Choose between cartesian and cylindrical.')

    def get_global_points(self, method):
        """
        Get the arrays with the coordinates of the points
        """
        xcoord, ycoord, zcoord = [], [], []
        for key, values in self.coords.items():
            xcoord.append(values['x'])
            ycoord.append(values['y'])
            zcoord.append(values['z'])
        x = np.concatenate(xcoord)
        y = np.concatenate(ycoord)
        z = np.concatenate(zcoord)
        if method == 'cartesian':
            return x, y, z
        else:
            return np.sqrt(x**2+y**2), np.arctan2(y,x), z















