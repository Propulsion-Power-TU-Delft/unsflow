#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 15:32:43 2023
@author: F. Neri, TU Delft
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
import time
from scipy.ndimage import minimum_filter
from scipy.sparse.linalg import eigs
from .sun_grid import SunGrid
from .annulus_meridional import AnnulusMeridional
from .general_functions import *
from .styles import *


class Eigenmode:
    """
    class used to store information related to an eigenmode. It will contain the fields of:
    -eigenpressure
    -eigenvelocities
    -eigendensities
    -eigenfrequency
    """
    def __init__(self, frequency, rho, ur, utheta, uz, p):
        self.eigenfrequency = frequency
        self.eigen_rho = rho
        self.eigen_ur = ur
        self.eigen_utheta = utheta
        self.eigen_uz = uz
        self.eigen_p = p

