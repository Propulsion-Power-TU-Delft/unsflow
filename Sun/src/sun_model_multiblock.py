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
import csv
import pickle
from scipy.ndimage import minimum_filter
from scipy.sparse.linalg import eigs
from .sun_grid import SunGrid
from .annulus_meridional import AnnulusMeridional
from .general_functions import *
from .styles import *
from .eigenmode import Eigenmode
from scipy.interpolate import griddata
from scipy.interpolate import Rbf
from Grid.src.functions import compute_picture_size


class SunModelMultiBlock:
    """
    Class used for Sun Model instability prediction, multiblock approach.
    """

    def __init__(self, sun_objects, config):
        """
        Instantiate the sun model Object, contaning all the attributes and methods necessary for the instability analysis.
        :param sun_objects: list of sun_model objects of the multiblock approach
        :param config: configuration file of the sun model
        """
        self.blocks = sun_objects
        self.config = config

        self.number_blocks = len(self.blocks)
        self.streamwise_points = [block.data.nAxialNodes for block in self.blocks]
        self.spanwise_points = self.blocks[0].data.nRadialNodes

    def construct_L_global_matrices(self):
        L0 = [block.L0 for block in self.blocks]
        self.L0 = enlarge_square_matrices(L0)

        plt.figure()
        plt.imshow(np.abs(L0[0]))
        plt.title('L0_0')

        plt.figure()
        plt.imshow(np.abs(L0[1]))
        plt.title('L0_1')

        plt.figure()
        plt.imshow(np.abs(L0[2]))
        plt.title('L0_2')

        plt.figure()
        plt.imshow(np.abs(self.L0))
        plt.title('L0')

        L1 = [block.L1 for block in self.blocks]
        self.L1 = enlarge_square_matrices(L1)

        plt.figure()
        plt.imshow(np.abs(L1[0]))
        plt.title('L1_0')

        plt.figure()
        plt.imshow(np.abs(L1[1]))
        plt.title('L1_1')

        plt.figure()
        plt.imshow(np.abs(L1[2]))
        plt.title('L1_2')

        plt.figure()
        plt.imshow(np.abs(self.L1))
        plt.title('L1')

        L2 = [block.L2 for block in self.blocks]
        self.L2 = enlarge_square_matrices(L2)

        plt.figure()
        plt.imshow(np.abs(L2[0]))
        plt.title('L2_0')

        plt.figure()
        plt.imshow(np.abs(L2[1]))
        plt.title('L2_1')

        plt.figure()
        plt.imshow(np.abs(L2[2]))
        plt.title('L2_2')

        plt.figure()
        plt.imshow(np.abs(self.L2))
        plt.title('L2')

    def apply_matching_conditions(self):
        """
        Apply the matching conditions for the blocks composing the multiblock. At this moment the system is composed by:
        (L0 + L1*omega + L2*omega^2)*x = 0. Since the matching conditions must be guaranteed for any possible omega, they
        must be applied on the matrix L0, while the corresponding rows of L1 and L2 must be set to 0
        """
        print('This is a bit of a problem since now the nodes are not the same, for the inlet and outlet. In other words, '
              'or we add the coincident nodes, or it becomes a problem. z_grid, r_grid rather than z_cg,r_cg')

    def compute_P_Y_matrices(self):
        """

        """
        Y1 = np.concatenate((-self.L0, np.zeros_like(self.L0)), axis=1)
        Y2 = np.concatenate((np.zeros_like(self.L0), np.eye(self.L0.shape[0])), axis=1)
        self.Y = np.concatenate((Y1, Y2), axis=0)  # Y matrix of EVP problem

        P1 = np.concatenate((self.L1, self.L2), axis=1)
        P2 = np.concatenate((np.eye(self.L0.shape[0]), np.zeros_like(self.L0)), axis=1)
        self.P = np.concatenate((P1, P2), axis=0)  # P matrix of EVP problem

    def solve_evp(self, sigma=0):
        print("Transforming generalized EVP in standard one...")
        Y_tilde = np.linalg.inv(self.Y - sigma * self.P)
        Y_tilde = np.dot(Y_tilde, self.P)

        print("Solving standard EVP...")
        self.eigenfreqs, self.eigenmodes = eigs(Y_tilde, k=self.config.get_research_number_omega_eigenvalues())
        self.eigenfreqs = sigma + 1 / self.eigenfreqs  # return of the initial shift
        self.eigenfreqs *= self.config.get_reference_omega()  # convert to dimensional frequencies
        self.eigenfreqs_df = self.eigenfreqs.imag / self.config.get_reference_omega()
        self.eigenfreqs_rs = self.eigenfreqs.real / self.config.get_reference_omega()
        self.sort_eigensolution()

    def sort_eigensolution(self):
        """
        Sort the eigenvalues and eigenvectors from the most unstable to the least one.
        """
        # make copies of the arrays to sort
        eigenfreqs = np.copy(self.eigenfreqs)
        df = np.copy(self.eigenfreqs_df)
        rs = np.copy(self.eigenfreqs_rs)
        eigenvectors = np.copy(self.eigenmodes)

        # get the sorting indices following descending order of the damping factor
        sorted_indices = sorted(range(len(df)), key=lambda i: df[i], reverse=True)

        # order the original arrays following the sorting indices
        for i in range(len(sorted_indices)):
            self.eigenfreqs[i] = eigenfreqs[sorted_indices[i]]
            self.eigenfreqs_df[i] = df[sorted_indices[i]]
            self.eigenfreqs_rs[i] = rs[sorted_indices[i]]
            self.eigenmodes[:, i] = eigenvectors[:, sorted_indices[i]]


