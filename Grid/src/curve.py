#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 18:29:29 2023
@author: F. Neri, TU Delft
"""
import matplotlib.pyplot as plt
from numpy import sqrt
from scipy.interpolate import splprep, splev
from .styles import *
from .functions import *


class Curve:
    """
    class that replicates a B-spline curve passing through points
    """

    def __init__(self, rescale_factor, x_ref, z=None, r=None, nstream=10, curve_filepath=None, units='mm',
                 mode='filedata', degree_spline=1):
        """
        overloaded constructor. You can give both the z and r cordinates (mode=cordinates), or you can provide 
        the filepath of the .curve files obtained from BladeGen (mode=filedata). 
        Units keep track of the units employed, rot_axis is usually the z axis.
        """
        if mode == 'filedata':
            self.read_from_curve_file(curve_filepath)

        elif mode == 'cordinates':
            self.r = r
            self.z = z

        self.r *= rescale_factor/x_ref
        self.z *= rescale_factor/x_ref

        self.nstream = nstream
        self.u_spline = np.linspace(0, 1, 10000)  # parametrization of the spline. increase points if needed
        self.r_spline, self.z_spline = self.compute_spline(self.u_spline, degree_spline=degree_spline)
        self.units = units



    def read_from_curve_file(self, filepath):
        """
        read the cordinates from the .curve file
        """
        self.data = np.loadtxt(filepath)
        self.x = self.data[:, 0]
        self.y = self.data[:, 1]
        self.z = self.data[:, 2]
        self.x, self.y, self.z = eliminate_duplicates(self.x, self.y,
                                                      self.z)  # because bladeGen exports some duplicate cordinates
        self.r = sqrt(self.x ** 2 + self.y ** 2)



    def compute_spline(self, u_ext, degree_spline=1):
        """
        compute spline passing through the points. it returns the cordinates along the spline, parametrized normally
        """
        self.tck, u = splprep([self.r, self.z], s=0, k=degree_spline)
        r_spline, z_spline = splev(u_ext, self.tck)
        return r_spline, z_spline



    def extend(self, u_min=-0.5, u_max=1.5, degree_spline=3):
        """
        extend the spline out of the normal domain of definition
        """
        u_spline_ext = np.linspace(u_min, u_max, 10000)  # parametrization of the spline. increase points if needed
        self.r_spline_ext, self.z_spline_ext = self.compute_spline(u_spline_ext, degree_spline=degree_spline)



    def trim_inlet(self, z_trim='span', r_trim='span'):
        """
        it deletes the points from the original spline before a certain inlet position point. 
        The trim plane can be horizontal or vertical depending on which trim cordinate has been given. 
        It OVERWRITES the original spline, losing the external points
        """
        if r_trim == 'span':
            idx = np.where(self.z_spline >= z_trim)
        elif z_trim == 'span':
            idx = np.where(self.r_spline >= r_trim)
        else:
            raise ValueError("Unknown trim type!")
        self.z_spline = self.z_spline[idx]
        self.r_spline = self.r_spline[idx]



    def trim_outlet(self, z_trim='span', r_trim='span'):
        """
        similar considerations of the trim_inlet, but deletes everything that comes after, not before.
        """
        if z_trim == 'span':
            idx = np.where(self.r_spline <= r_trim)
        elif r_trim == 'span':
            idx = np.where(self.z_spline <= z_trim)
        else:
            raise ValueError("Unknown trim type!")
        self.z_spline = self.z_spline[idx]
        self.r_spline = self.r_spline[idx]



    def sample(self, sampling_mode='default'):
        """
        having the spline data it computes a set of points on it, choosing an array of u values, from 0 (begin of 
        the spline) to 1 (end of spline). In default mode u is obtained with a linspace, in clustering mode 
        the points are clustered to a side, to highlight discontinuities        
        """
        if sampling_mode == 'default':
            self.u_sample = np.linspace(0, 1, self.nstream)
        elif sampling_mode == 'clustering':
            self.u_sample = cluster_sample_u(self.nstream)
        elif sampling_mode == 'clustering_left':
            self.u_sample = cluster_sample_u(self.nstream, border='left')
        elif sampling_mode == 'clustering_right':
            self.u_sample = cluster_sample_u(self.nstream, border='right')
        self.r_sample, self.z_sample = splev(self.u_sample, self.tck)



    def plot_spline(self):
        """
        it plots the spline, and the cordinates that have been given to generate it.
        If the spline has been trimmed, the trimmed parts are removed
        """
        plt.figure(figsize=fig_size)
        plt.plot(self.z, self.r, 'o', label='cordinates')
        plt.plot(self.z_spline, self.r_spline, label='B-spline')
        plt.legend()
        plt.xlabel(r'$z \ \mathrm{[%s]}$' % (self.units))
        plt.ylabel(r'$r \ \mathrm{[%s]}$' % (self.units))
