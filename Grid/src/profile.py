#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 11:41:53 2023
@author: F. Neri, TU Delft
"""
import matplotlib.pyplot as plt
from numpy import array
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from .functions import cartesian_to_cylindrical
from Sun.src.general_functions import print_banner_begin, print_banner_end
from Grid.src.functions import compute_picture_size
from Utils.styles import *


class Profile:
    """
    class that stores the information regarding the profile of a blade
    """

    def __init__(self, xss, yss, zss, xps, yps, zps):
        """
        :param xss: x coordinate of the suction side points

        :param xps: x coordinate of the pressure side points
        """

        self.xss = xss
        self.yss = yss
        self.zss = zss
        self.xps = xps
        self.yps = yps
        self.zps = zps

        self.rss = np.sqrt(self.xss**2 + self.yss**2)
        self.rps = np.sqrt(self.xps**2 + self.yps**2)
        self.thetass = np.arctan2(self.yss, self.xss)
        self.thetaps =np.arctan2(self.yps, self.xps)
        # self.thetac = 0.5*(self.thetass+self.thetaps)

    def plot_profile(self):
        plt.figure()
        plt.plot(self.zss, self.rss*self.thetass, '--ro', label='ss')
        plt.plot(self.zps, self.rps*self.thetaps, '--bo', label='ps')
        plt.legend()
        plt.xlabel(r'$z$')
        plt.ylabel(r'$r \theta$')





