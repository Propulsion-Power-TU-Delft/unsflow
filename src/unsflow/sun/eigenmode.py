import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
import time
from scipy.ndimage import minimum_filter
from scipy.sparse.linalg import eigs
from .sun_grid import SunGrid
from .general_functions import *
from unsflow.utils.plot_styles import *


class Eigenmode:
    """
    Class used to store information related to an eigenmode.
    """
    def __init__(self, frequency, rho, ur, utheta, uz, p):
        """
        Builds an eigenmode object.
        :param frequency: eigenfrequency
        :param rho: density eigenfunction
        :param ur: ur eigenfunction
        :param utheta: utheta eigenfunction
        :param uz: uz eigenfunction
        :param p: p eigenfunction
        """
        self.eigenfrequency = frequency
        self.eigen_rho = rho
        self.eigen_ur = ur
        self.eigen_utheta = utheta
        self.eigen_uz = uz
        self.eigen_p = p
        self.Nz = np.shape(rho)[0]
        self.Nr = np.shape(rho)[1]
        self.distinguish_physical_mode()

    def distinguish_physical_mode(self, threshold=0.3):
        """
        Distinguish between physical and sprious eigenmodes. The criterion is taken from 'Predicting the onset of flow
        unsteadiness based on global instability' by Crouch et al.
        """
        f = self.eigen_p  # classify based on the eigenpressure shape
        max_value = np.sqrt(np.max(f)**2)
        border_summation = np.sum(f[:, 0]**2) + np.sum(f[-1, 1:]**2) + np.sum(f[0, 1:]**2) + np.sum(f[1:-1, -1]**2)
        border_points = self.Nz*2 + (self.Nr-1)*2
        avg_borders = np.sqrt(border_summation/border_points)
        if np.abs(avg_borders)<threshold*max_value:
            self.is_physical = True
        else:
            self.is_physical = False
