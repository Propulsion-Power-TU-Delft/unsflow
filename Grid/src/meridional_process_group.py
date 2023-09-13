#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 17:07:05 2023
@author: F. Neri, TU Delft
"""

import numpy as np
from numpy import sqrt
from .styles import *
import matplotlib.path as mplpath
from scipy.ndimage import gaussian_filter
from scipy.interpolate import Rbf
import pickle


class MeridionalProcessGroup:
    """
    Group of meridional Process object, used only to plot stuff together
    """

    def __init__(self):
        self.group = []




    def add_to_group(self, meridional_obj):
        self.group.append(meridional_obj)

    def contour(self):
        plt.figure(figsize=fig_size)
        for obj in self.group:
            plt.contourf(obj.z_grid, obj.r_grid, obj.M, cmap='jet', levels=N_levels)
        plt.colorbar()

