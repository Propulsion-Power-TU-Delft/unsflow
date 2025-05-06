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
from Grid.src.functions import clip_negative_values, compute_curvilinear_abscissa, compute_3dSpline_curve, compute_2dSpline_curve, find_intersection, eriksson_stretching_function_both
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

    def __init__(self, name ,config):
        """
        General constructor.
        :param name: string with the name of the surface
        """
        self.name = name
        self.coords = {}
        self.config = config

    def add_curve(self, x, y, z):
        """
        add a curve to the dataset.
        :param x: array of x coordinates
        :param y: array of y coordinates
        :param z: array of z coordinates
        """
        iDum = len(self.coords)
        self.coords['Profile %i' % iDum] = {'x': x, 'y': y, 'z': z}

    def get_number_points_per_profile(self):
        """
        Get the number of points that define a single profile
        """
        nProf = self.get_number_profiles()
        nTot = len(self.get_global_points()[0])
        return nTot//nProf

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

    def plot_lofted_surface(self, surfaces=False):
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

    def plot_bspline_surface(self):
        """
        Plot the profiles points, and the surface lofts if available.
        :param surfaces: bool value, True to plot the surface lofts
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(*self.get_global_points('cartesian'), s=5, alpha=0.3, c='black')
        ax.plot_surface(self.Xg, self.Yg, self.Zg, alpha=0.6)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        
        


    def loft_through_profiles(self, points_along_profile=100, points_between_profiles=40, extension=0, order='linear'):
        """
        Use interpolation between 2 profiles to assemble the overall surface of the camber.
        The camber is extended at the borders for 10% in order to cope for following griddata interpolation.
        :param points_along_profile: number of points used to interpolate splines along the generator curves
        :param points_between_profiles: number of points used in between the different lofts
        :param extension: percentage of extension at the borders
        :param order: linear or quadratic bi-interpolation
        """
        if order == 'linear' and self.get_number_profiles() < 2:
            raise ValueError('At least two profiles are needed for a linear loft generation')
        elif order == 'quadratic' and self.get_number_profiles() < 3:
            raise ValueError('At least three profiles are needed for a quadratic loft surface generation. Not implemented yet!')

        self.surface = {}
        keys_list = list(self.coords.keys())
        for iSurf in range(self.get_number_profiles()-1):
            t = np.linspace(0-extension, 1+extension, points_along_profile)  # parameter flowing on one curve tangentially
            if iSurf == 0:
                s = np.linspace(0-extension, 1, points_between_profiles)  # parameter connecting one curve to the other
            elif iSurf == self.get_number_profiles()-2:
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

    def bspline_surface_generation(self, extension=0, stream_resolution=250, span_resolution=30):
        """
        Generation of surface by bi-variate spline
        """
        t = np.linspace(0-extension, 1+extension, stream_resolution)
        s = np.linspace(0-extension, 1+extension, span_resolution)

        t = eriksson_stretching_function_both(t, 1)

        # generate the spline along the profile (streamwise)
        prf_splx, prf_sply, prf_splz = [], [], []
        for i, (key, values) in enumerate(self.coords.items()):
            xint, yint, zint = compute_3dSpline_curve(values['x'], values['y'], values['z'], u_param=t)
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # ax.plot(values['x'], values['y'], values['z'], '-o', label='1d-Bspline-Points')
            # ax.plot(xint, yint, zint, ms=1, label='B-spline')
            # ax.legend()    
            # ax.set_title('Profile %i' % i)
            prf_splx.append(xint)
            prf_sply.append(yint)
            prf_splz.append(zint)
        
        # generate the dataset across the profiles (spanwise)
        self.cross_coords = {}
        for i in range(len(t)):
            self.cross_coords[i] = {}
            xtmp, ytmp, ztmp = [], [], []
            for iProf in range(len(prf_splx)):
                xtmp.append(prf_splx[iProf][i])
                ytmp.append(prf_sply[iProf][i])
                ztmp.append(prf_splz[iProf][i])
            x, y, z = np.array(xtmp), np.array(ytmp), np.array(ztmp)
            self.cross_coords[i]['x'] = x
            self.cross_coords[i]['y'] = y
            self.cross_coords[i]['z'] = z

        # generate the spline in the spanwise direction
        crs_splx, crs_sply, crs_splz = [], [], []
        for key, values in self.cross_coords.items():
            try:
                xint, yint, zint = compute_3dSpline_curve(values['x'], values['y'], values['z'], u_param=s)
                crs_splx.append(xint)
                crs_sply.append(yint)
                crs_splz.append(zint)
            except:
                pass

        
        # if self.config.get_visual_debug():
        #     fig = plt.figure()
        #     ax = fig.add_subplot(111, projection='3d')
        #     for i in range(len(prf_splx)):
        #         ax.plot(prf_splx[i], prf_sply[i], prf_splz[i], 'C0o', ms=1, label='streamwise', lw=0.5)
        #     for i in range(len(crs_splx)):
        #         ax.plot(crs_splx[i], crs_sply[i], crs_splz[i], 'C1', label='spanwise', lw=0.5)

            # ax.scatter(*self.get_global_points('cartesian'), s=20, alpha=0.3)

        streamPoints = len(crs_splx)
        spanPoints = len(crs_splx[0])
        
        self.Xg, self.Yg, self.Zg = np.zeros((streamPoints, spanPoints)), np.zeros((streamPoints, spanPoints)), np.zeros((streamPoints, spanPoints))
        for ii in range(streamPoints):
            for jj in range(spanPoints):
                self.Xg[ii, jj] = crs_splx[ii][jj]
                self.Yg[ii, jj] = crs_sply[ii][jj]
                self.Zg[ii, jj] = crs_splz[ii][jj]




    def get_global_lofted_surface(self, method):
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

    def get_global_bspline_surface(self, method):
        """
        Get the coordinates arrays for whole bspline surface.
        :param method: decide between cylindrical (r,theta,z) and cartesian (x, y, z) return
        """
        if method == 'cartesian':
            return self.Xg, self.Yg, self.Zg
        elif method == 'cylindrical':
            # return np.sqrt(self.Xg**2+self.Yg**2), np.arctan2(self.Yg, self.Xg), self.Zg
            return np.sqrt(self.Xg**2+self.Yg**2), np.arctan2(self.Yg,self.Xg), self.Zg
        else:
            raise ValueError('Unknown type of return method. Choose between cartesian and cylindrical.')

    def get_global_points(self, method='cartesian'):
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















