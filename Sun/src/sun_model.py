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


class SunModel:
    """
    Class used for Sun Model instability prediction based on the data contained in a Grid object containing the CFD results. 
    Matrix elements are taken from Aerodynamic Instabilities of Swept Airfoil Design in Transonic Axial-Flow Compressors,
    He et Al.
    The general stability equation is:
        (-j*omega*A + B*ddr + j*m*C/r + E*ddz + R + S)*Phi' = 0
    
    MATRICES:
        A : coefficient matrix of temporal derivatives
        B : coefficient matrix of radial derivatives
        C : coefficient matrix of azimuthal derivative s
        E : coefficient matrix of axial derivatives
        R : coefficient matrix of the known mean flow terms
        S : coefficient matrix of the body force model
    """

    def __init__(self, gridObject):
        """
        it builds the object and the related data
        """
        self.data = gridObject  # grid object containing also the meridional object with the data
        self.nPoints = (gridObject.nAxialNodes) * (gridObject.nRadialNodes)
        self.gmma = 1.4  # cp/cv for standard air for the moment
        print('Gamma set to default value: 1.4')
        self.substituted_equation = 'ur'  # decides which equation overwrite with the euler wall condition
        print('Default equation to overwrite with Euler Wall condition: Radial Momentum')
        print("\n")


    def set_overwriting_equation_euler_wall(self, equation):
        """
        select an equation to overwrite with the euler wall. Avilable options: ur, utheta, uz
        """
        if equation=='ur':
            self.substituted_equation = 'ur'
            print("Equation to overwrite with Euler Wall condition set to: Radial Momentum!")
        elif equation=='utheta':
            self.substituted_equation = 'utheta'
            print("Equation to overwrite with Euler Wall condition set to: Tangential Momentum!")
        elif equation=='uz':
            self.substituted_equation = 'ur'
            print("Equation to overwrite with Euler Wall condition set to: Axial Momentum!")
        else:
            raise ValueError("Not recognized option")

        print("\n")


    def AddNormalizationQuantities(self, rho_ref, u_ref, x_ref):
        """
        quantities needed to non-dimensionalize the conservatione equations, in order to make the system better posed numerically
        The non-dimensionalizations terms come from the advecation terms, and are [x]/[rho][u] for the continuity equation,
        [x]/[u]^2 for the momentum equations, and [x]/[rho][u]^3 for the pressure equation.
        The fundamental entities selected for non-dimensionalization are a reference density, a reference velocity, and a
        reference length. All the rest is obtained from these.
        """
        self.rho_ref = rho_ref
        self.u_ref = u_ref
        self.p_ref = rho_ref * u_ref ** 2
        self.x_ref = x_ref
        self.t_ref = 1 / self.omega_ref
        self.print_normalization_information()

        # normalization terms = inverse of advections, to be used for governing equations' normalization. They are correct
        # only for the form of the equations used. Otherwise, they must be changed
        self.continuity_norm = self.x_ref / self.rho_ref / self.u_ref
        self.momentum_norm = self.x_ref / self.u_ref ** 2
        self.pressure_norm = self.x_ref / self.rho_ref / self.u_ref ** 3



    def print_normalization_information(self):
        """
        print information on non-dimensionalization in the sun module. It should provide only ones if the data
        were already normalized in the meridional process
        """
        print("+----------------- NORMALIZATION -----------------+")
        print("Reference Length: %.2f [m]" % (self.x_ref))
        print("Reference Velocity: %.2f [m/s]" %(self.u_ref))
        print("Reference Density: %.2f [kg/m3]" % (self.rho_ref))
        print("Reference Pressure: %.2f [Pa]" % (self.p_ref))
        print("Reference Time: %.6f [s]" % (self.t_ref))
        print("Reference Angular Rate: %.2f [rad/s]" % (self.omega_ref))
        print("+--------------------------------------------------+")
        print("\n")

    def add_shaft_rpm(self, rpm):
        """
        add the rpm of the shaft
        Args:
            rpm: revolutions per minute
        """
        self.omega_shaft = 2 * np.pi * rpm / 60
        self.omega_ref = np.abs(np.copy(self.omega_shaft))

    def NormalizeData(self):
        """
        non-dimensionalise the node quantities
        """
        self.data.Normalize(self.rho_ref, self.u_ref, self.x_ref)

    def ComputeSpectralGrid(self):
        """
        it instanties a new grid object which has the same flow data of the original grid, but stored 
        on a computational grid suitable for spectral differentiation (grid poinst located on Gauss-Lobatto points)
        """
        self.dataSpectral = self.data.PhysicalToSpectralData()

    def ShowPhysicalGrid(self, save_filename=None, mode=None):
        """
        it shows the physical grid points, with different colors for the different parts of the domain
        """
        self.data.ShowGrid(mode=mode)
        plt.title('physical grid')
        plt.xlabel(r'$\hat{z} \quad  [-]$')
        plt.ylabel(r'$\hat{r} \quad  [-]$')
        if save_filename is not None:
            plt.savefig(folder_name + save_filename + '.pdf', bbox_inches='tight')

    def ShowSpectralGrid(self, save_filename=None, mode=None):
        """
        it shows the physical grid points, with different colors for the different parts of the domain
        """
        self.dataSpectral.ShowGrid(mode=mode)
        plt.title('spectral grid')
        plt.xlabel(r'$\xi \quad  [-]$')
        plt.ylabel(r'$\eta \quad  [-]$')
        if save_filename is not None:
            plt.savefig(folder_name + save_filename + '.pdf', bbox_inches='tight')

    def ComputeJacobianPhysical(self, routine='numpy', order=2, refinement=False):
        """
        The Jacobian for the physical grid as a function of the spectral grid cordinates is implemented here. 
        It computes the transformation derivatives for every grid point, and stores the value at the node level.
        NOTE: this approach is the only one correct if the nodes are set on curvilinear grids (as in compressors)
        """
        print("+-------------- TRANSFORMATION GRADIENTS --------------+")
        print("Routine used: %s" % routine)
        print("Order used: %i" %(order))
        print("Artificial Refinement: %s" %(refinement))
        print("+------------------------------------------------------+")
        print("\n")

        if not refinement:
            # grids (original)
            Z = self.data.zGrid
            R = self.data.rGrid
            X = self.dataSpectral.zGrid
            Y = self.dataSpectral.rGrid

            if routine == 'numpy':
                self.dzdx, self.dzdy, self.drdx, self.drdy = JacobianTransform2(Z, R, X, Y)
            elif routine == 'hard-coded':
                self.dzdx, self.dzdy, self.drdx, self.drdy = JacobianTransform(Z, R, X, Y)
            elif routine == 'findiff':
                self.dzdx, self.dzdy, self.drdx, self.drdy = JacobianTransform3(Z, R, X, Y, order=order)
            else:
                raise ValueError('select an available routine for the transformation gradient!')

            self.J = self.dzdx * self.drdy - self.dzdy * self.drdx
            for ii in range(0, self.data.nAxialNodes):
                for jj in range(0, self.data.nRadialNodes):
                    # add the inverse gradients information to every node
                    self.data.dataSet[ii, jj].AddTransformationGradients(self.dzdx[ii, jj], self.dzdy[ii, jj],
                                                                         self.drdx[ii, jj], self.drdy[ii, jj])
                    self.data.dataSet[ii, jj].AddJacobian(self.J[ii, jj])

        elif (isinstance(refinement, int) and refinement > 0):
            print('WARNING: refinement method is not validated and produces wrong results!')

            # refined physical grid
            ref_points = refinement  # refinement coefficient. additional points for every interval
            r = Refinement(self.data.rGrid[0, :], ref_points)  # it adds additional ref_points to every interval
            z = Refinement(self.data.zGrid[:, 0], ref_points)
            self.R_fine, self.Z_fine = np.meshgrid(r, z)

            # refined spectral grid
            x = GaussLobattoPoints(len(z))  # refined set of gauss lobatto points
            y = GaussLobattoPoints(len(r))
            self.Y_fine, self.X_fine = np.meshgrid(y, x)

            # compute jacobian
            self.dzdx_fine, self.dzdy_fine, self.drdx_fine, self.drdy_fine = JacobianTransform(self.Z_fine, self.R_fine,
                                                                                               self.X_fine,
                                                                                               self.Y_fine)
            self.J_fine = self.dzdx_fine * self.drdy_fine - self.dzdy_fine * self.drdx_fine

            # pick-up the values on the coarse grid points
            self.dzdx = self.dzdx_fine[::ref_points + 1, ::ref_points + 1]
            self.dzdy = self.dzdy_fine[::ref_points + 1, ::ref_points + 1]
            self.drdx = self.drdx_fine[::ref_points + 1, ::ref_points + 1]
            self.drdy = self.drdy_fine[::ref_points + 1, ::ref_points + 1]
            self.J = self.J_fine[::ref_points + 1, ::ref_points + 1]

            for ii in range(0, self.data.nAxialNodes):
                for jj in range(0, self.data.nRadialNodes):
                    # add the inverse gradients information to every node
                    self.data.dataSet[ii, jj].AddTransformationGradients(self.dzdx[ii, jj], self.dzdy[ii, jj],
                                                                         self.drdx[ii, jj], self.drdy[ii, jj])
                    self.data.dataSet[ii, jj].AddJacobian(self.dzdx[ii, jj] * self.drdy[ii, jj]
                                                          - self.dzdy[ii, jj] * self.drdx[ii, jj])
        else:
            raise Exception('Wrong refinement. Select a positive integer!')

    def ContourTransformation(self, save_filename=None):
        """
        Show the spectral gradients info as a function of the spectral grid cordinates.
        """
        plt.figure(figsize=fig_size)
        plt.contourf(self.dataSpectral.zGrid, self.dataSpectral.rGrid, self.J, levels=N_levels, cmap=color_map)
        plt.xlabel(r'$\xi \ \mathrm{[-]}$')
        plt.ylabel(r'$\eta \ \mathrm{[-]}$')
        plt.title(r'$J$')
        cb = plt.colorbar()
        cb.set_label(r'$\frac{\partial \hat{z}}{\partial \xi} \ \mathrm{[-]}$')
        if save_filename is not None:
            plt.savefig(folder_name + save_filename + '_J.pdf', bbox_inches='tight')

        plt.figure(figsize=fig_size)
        plt.contourf(self.dataSpectral.zGrid, self.dataSpectral.rGrid, self.dzdx, levels=N_levels, cmap=color_map)
        plt.xlabel(r'$\xi \ \mathrm{[-]}$')
        plt.ylabel(r'$\eta \ \mathrm{[-]}$')
        plt.title(r'$\frac{\partial \hat{z}}{\partial \xi}$')
        cb = plt.colorbar()
        cb.set_label(r'$\frac{\partial \hat{z}}{\partial \xi} \ \mathrm{[-]}$')
        if save_filename is not None:
            plt.savefig(folder_name + save_filename + '_1.pdf', bbox_inches='tight')

        plt.figure(figsize=fig_size)
        plt.contourf(self.dataSpectral.zGrid, self.dataSpectral.rGrid, self.dzdy, levels=N_levels, cmap=color_map)
        plt.xlabel(r'$\xi \ \mathrm{[-]}$')
        plt.ylabel(r'$\eta \ \mathrm{[-]}$')
        cb = plt.colorbar()
        plt.title(r'$\frac{\partial \hat{z}}{\partial \eta}$')
        cb.set_label(r'$\frac{\partial \hat{z}}{\partial \eta} \ \mathrm{[-]}$')
        if save_filename is not None:
            plt.savefig(folder_name + save_filename + '_2.pdf', bbox_inches='tight')

        plt.figure(figsize=fig_size)
        plt.contourf(self.dataSpectral.zGrid, self.dataSpectral.rGrid, self.drdx, levels=N_levels, cmap=color_map)
        plt.xlabel(r'$\xi \ \mathrm{[-]}$')
        plt.ylabel(r'$\eta \ \mathrm{[-]}$')
        cb = plt.colorbar()
        plt.title(r'$\frac{\partial \hat{r}}{\partial \xi}$')
        cb.set_label(r'$\frac{\partial \hat{r}}{\partial \xi} \ \mathrm{[-]}$')
        if save_filename is not None:
            plt.savefig(folder_name + save_filename + '_3.pdf', bbox_inches='tight')

        plt.figure(figsize=fig_size)
        plt.contourf(self.dataSpectral.zGrid, self.dataSpectral.rGrid, self.drdy, levels=N_levels, cmap=color_map)
        plt.xlabel(r'$\xi \ \mathrm{[-]}$')
        plt.ylabel(r'$\eta \ \mathrm{[-]}$')
        cb = plt.colorbar()
        plt.title(r'$\frac{\partial \hat{r}}{\partial \eta}$')
        cb.set_label(r'$\frac{\partial \hat{r}}{\partial \eta} \ \mathrm{[-]}$')
        if save_filename is not None:
            plt.savefig(folder_name + save_filename + '_4.pdf', bbox_inches='tight')

    def AddAMatrixToNodes(self):
        """
        compute and store at the node level the A matrix, not multiplied yet by j*omega. 
        """
        for ii in range(0, self.data.nAxialNodes):
            for jj in range(0, self.data.nRadialNodes):
                A = np.eye(5, dtype=complex)
                A = self.NormalizeMatrix(A)  # normalization
                self.data.dataSet[ii, jj].AddAMatrix(A)

    def AddAMatrixToNodesFrancesco(self):
        """
        compute and store at the node level the A matrix, not multiplied yet by j*omega.
        My version, checked
        """
        for ii in range(0, self.data.nAxialNodes):
            for jj in range(0, self.data.nRadialNodes):
                A = np.eye(5, dtype=complex)
                A = self.NormalizeMatrix(A)  # normalization
                self.data.dataSet[ii, jj].AddAMatrix(A)

    def AddAMatrixToNodesFrancesco2(self):
        """
        compute and store at the node level the A matrix, not multiplied yet by j*omega.
        My version, checked with the article of the annular duct and sympy
        """
        for ii in range(0, self.data.nAxialNodes):
            for jj in range(0, self.data.nRadialNodes):
                A = np.eye(5, dtype=complex)
                A[4, 0] = -self.data.dataSet[ii, jj].p * self.gmma / self.data.dataSet[ii, jj].rho
                A = self.NormalizeMatrix(A)  # normalization
                self.data.dataSet[ii, jj].AddAMatrix(A)

    def AddBMatrixToNodes(self):
        """
        compute and store at the node level the B matrix, needed to compute hat{B} later
        """
        for ii in range(0, self.data.nAxialNodes):
            for jj in range(0, self.data.nRadialNodes):
                B = np.zeros((5, 5), dtype=complex)

                B[0, 0] = self.data.dataSet[ii, jj].ur
                B[1, 1] = self.data.dataSet[ii, jj].ur
                B[2, 2] = self.data.dataSet[ii, jj].ur
                B[3, 3] = self.data.dataSet[ii, jj].ur
                B[4, 4] = self.data.dataSet[ii, jj].ur

                B[0, 1] = self.data.dataSet[ii, jj].rho
                B[1, 4] = 1 / self.data.dataSet[ii, jj].rho
                B[4, 1] = self.data.dataSet[ii, jj].p * self.gmma

                B = self.NormalizeMatrix(B)  # normalization
                self.data.dataSet[ii, jj].AddBMatrix(B)

    def AddBMatrixToNodesFrancesco(self):
        """
        compute and store at the node level the B matrix, needed to compute hat{B} later.
        my version checked

        """
        for ii in range(0, self.data.nAxialNodes):
            for jj in range(0, self.data.nRadialNodes):
                B = np.zeros((5, 5), dtype=complex)

                B[0, 0] = self.data.dataSet[ii, jj].ur
                B[1, 1] = self.data.dataSet[ii, jj].ur
                B[2, 2] = self.data.dataSet[ii, jj].ur
                B[3, 3] = self.data.dataSet[ii, jj].ur
                B[4, 4] = self.data.dataSet[ii, jj].ur

                B[0, 1] = self.data.dataSet[ii, jj].rho
                B[1, 4] = 1 / self.data.dataSet[ii, jj].rho
                B[4, 0] = -self.data.dataSet[ii, jj].p * self.data.dataSet[ii, jj].ur / self.data.dataSet[ii, jj].rho
                B[4, 1] = self.data.dataSet[ii, jj].p * (self.gmma - 1)

                B = self.NormalizeMatrix(B)  # normalization
                self.data.dataSet[ii, jj].AddBMatrix(B)

    def AddBMatrixToNodesFrancesco2(self):
        """
        compute and store at the node level the B matrix, needed to compute hat{B} later.
        my version checked

        """
        for ii in range(0, self.data.nAxialNodes):
            for jj in range(0, self.data.nRadialNodes):
                B = np.zeros((5, 5), dtype=complex)

                B[0, 0] = self.data.dataSet[ii, jj].ur
                B[1, 1] = self.data.dataSet[ii, jj].ur
                B[2, 2] = self.data.dataSet[ii, jj].ur
                B[3, 3] = self.data.dataSet[ii, jj].ur
                B[4, 4] = self.data.dataSet[ii, jj].ur

                B[0, 1] = self.data.dataSet[ii, jj].rho
                B[1, 4] = 1 / self.data.dataSet[ii, jj].rho
                B[4, 0] = - self.data.dataSet[ii, jj].p * self.data.dataSet[ii, jj].ur * self.gmma / self.data.dataSet[ii, jj].rho

                B = self.NormalizeMatrix(B)  # normalization
                self.data.dataSet[ii, jj].AddBMatrix(B)

    def AddCMatrixToNodes(self, m=1):
        """
        compute and store at node level the C matrix, already multiplied by j*m/r. Ready to be used in the final system of eqs.
        """
        for ii in range(0, self.data.nAxialNodes):
            for jj in range(0, self.data.nRadialNodes):
                C = np.zeros((5, 5), dtype=complex)

                C[0, 0] = self.data.dataSet[ii, jj].ut
                C[1, 1] = self.data.dataSet[ii, jj].ut
                C[2, 2] = self.data.dataSet[ii, jj].ut
                C[3, 3] = self.data.dataSet[ii, jj].ut
                C[4, 4] = self.data.dataSet[ii, jj].ut

                C[0, 2] = self.data.dataSet[ii, jj].rho
                C[2, 4] = 1 / self.data.dataSet[ii, jj].rho
                C[4, 2] = self.data.dataSet[ii, jj].p * self.gmma

                C = self.NormalizeMatrix(C)  # normalization
                C = C * 1j * m / self.data.dataSet[ii, jj].r

                self.data.dataSet[ii, jj].AddCMatrix(C)

    def AddCMatrixToNodesFrancesco(self, m=1):
        """
        compute and store at node level the C matrix, already multiplied by j*m/r. Ready to be used in the final system of eqs.
        my version, checked
        """
        for ii in range(0, self.data.nAxialNodes):
            for jj in range(0, self.data.nRadialNodes):
                C = np.zeros((5, 5), dtype=complex)

                C[0, 0] = self.data.dataSet[ii, jj].ut
                C[1, 1] = self.data.dataSet[ii, jj].ut
                C[2, 2] = self.data.dataSet[ii, jj].ut
                C[3, 3] = self.data.dataSet[ii, jj].ut
                C[4, 4] = self.data.dataSet[ii, jj].ut

                C[0, 2] = self.data.dataSet[ii, jj].rho
                C[2, 4] = 1 / self.data.dataSet[ii, jj].rho
                C[4, 0] = -self.data.dataSet[ii, jj].p * self.data.dataSet[ii, jj].ut / self.data.dataSet[ii, jj].rho
                C[4, 2] = self.data.dataSet[ii, jj].p * (self.gmma - 1)

                C = self.NormalizeMatrix(C)  # normalization
                C = C * 1j * m / self.data.dataSet[ii, jj].r

                self.data.dataSet[ii, jj].AddCMatrix(C)

    def AddCMatrixToNodesFrancesco2(self, m=1):
        """
        compute and store at node level the C matrix, already multiplied by j*m/r. Ready to be used in the final system of eqs.
        my version, checked
        """
        self.harmonic_order = m
        for ii in range(0, self.data.nAxialNodes):
            for jj in range(0, self.data.nRadialNodes):
                C = np.zeros((5, 5), dtype=complex)

                C[0, 0] = self.data.dataSet[ii, jj].ut
                C[1, 1] = self.data.dataSet[ii, jj].ut
                C[2, 2] = self.data.dataSet[ii, jj].ut
                C[3, 3] = self.data.dataSet[ii, jj].ut
                C[4, 4] = self.data.dataSet[ii, jj].ut

                C[0, 2] = self.data.dataSet[ii, jj].rho
                C[2, 4] = 1 / self.data.dataSet[ii, jj].rho
                C[4, 0] = -self.data.dataSet[ii, jj].p * self.data.dataSet[ii, jj].ut * self.gmma / self.data.dataSet[ii, jj].rho

                C = self.NormalizeMatrix(C)  # normalization
                C = C * 1j * m / self.data.dataSet[ii, jj].r

                self.data.dataSet[ii, jj].AddCMatrix(C)

    def AddEMatrixToNodes(self):
        """
        compute and store at the node level the E matrix, needed to compute hat{E}
        """
        for ii in range(0, self.data.nAxialNodes):
            for jj in range(0, self.data.nRadialNodes):
                E = np.zeros((5, 5), dtype=complex)

                E[0, 0] = self.data.dataSet[ii, jj].uz
                E[1, 1] = self.data.dataSet[ii, jj].uz
                E[2, 2] = self.data.dataSet[ii, jj].uz
                E[3, 3] = self.data.dataSet[ii, jj].uz
                E[4, 4] = self.data.dataSet[ii, jj].uz

                E[0, 3] = self.data.dataSet[ii, jj].rho
                E[3, 4] = 1 / self.data.dataSet[ii, jj].rho
                E[4, 3] = self.data.dataSet[ii, jj].p * self.gmma

                E = self.NormalizeMatrix(E)  # normalization
                self.data.dataSet[ii, jj].AddEMatrix(E)

    def AddEMatrixToNodesFrancesco(self):
        """
        compute and store at the node level the E matrix, needed to compute hat{E}. my version, checked
        """
        for ii in range(0, self.data.nAxialNodes):
            for jj in range(0, self.data.nRadialNodes):
                E = np.zeros((5, 5), dtype=complex)

                E[0, 0] = self.data.dataSet[ii, jj].uz
                E[1, 1] = self.data.dataSet[ii, jj].uz
                E[2, 2] = self.data.dataSet[ii, jj].uz
                E[3, 3] = self.data.dataSet[ii, jj].uz
                E[4, 4] = self.data.dataSet[ii, jj].uz

                E[0, 3] = self.data.dataSet[ii, jj].rho
                E[3, 4] = 1 / self.data.dataSet[ii, jj].rho
                E[4, 0] = -self.data.dataSet[ii, jj].p * self.data.dataSet[ii, jj].uz / self.data.dataSet[ii, jj].rho
                E[4, 3] = self.data.dataSet[ii, jj].p * (self.gmma - 1)

                E = self.NormalizeMatrix(E)  # normalization
                self.data.dataSet[ii, jj].AddEMatrix(E)

    def AddEMatrixToNodesFrancesco2(self):
        """
        compute and store at the node level the E matrix, needed to compute hat{E}. my version, checked
        """
        for ii in range(0, self.data.nAxialNodes):
            for jj in range(0, self.data.nRadialNodes):
                E = np.zeros((5, 5), dtype=complex)

                E[0, 0] = self.data.dataSet[ii, jj].uz
                E[1, 1] = self.data.dataSet[ii, jj].uz
                E[2, 2] = self.data.dataSet[ii, jj].uz
                E[3, 3] = self.data.dataSet[ii, jj].uz
                E[4, 4] = self.data.dataSet[ii, jj].uz

                E[0, 3] = self.data.dataSet[ii, jj].rho
                E[3, 4] = 1 / self.data.dataSet[ii, jj].rho
                E[4, 0] = -self.data.dataSet[ii, jj].p * self.data.dataSet[ii, jj].uz * self.gmma / self.data.dataSet[ii, jj].rho

                E = self.NormalizeMatrix(E)  # normalization
                self.data.dataSet[ii, jj].AddEMatrix(E)

    def AddRMatrixToNodes(self):
        """
        compute and store at the node level the R matrix, ready to be used in the final system of eqs.
        """
        for ii in range(0, self.data.nAxialNodes):
            for jj in range(0, self.data.nRadialNodes):
                R = np.zeros((5, 5), dtype=complex)

                R[0, 0] = self.data.dataSet[ii, jj].dur_dr + self.data.dataSet[ii, jj].duz_dz + (self.data.dataSet[ii, jj].ur
                                                                                                 / self.data.dataSet[ii, jj].r)
                R[0, 1] = self.data.dataSet[ii, jj].rho / self.data.dataSet[ii, jj].r + self.data.dataSet[ii, jj].drho_dr
                R[0, 2] = 0
                R[0, 3] = self.data.dataSet[ii, jj].drho_dz
                R[0, 4] = 0
                R[1, 0] = -self.data.dataSet[ii, jj].dp_dr / self.data.dataSet[ii, jj].rho ** 2
                R[1, 1] = self.data.dataSet[ii, jj].dur_dr
                R[1, 2] = -2 * self.data.dataSet[ii, jj].ut / self.data.dataSet[ii, jj].r
                R[1, 3] = self.data.dataSet[ii, jj].dur_dz
                R[1, 4] = 0
                R[2, 0] = 0
                R[2, 1] = self.data.dataSet[ii, jj].dut_dr + self.data.dataSet[ii, jj].ut / self.data.dataSet[ii, jj].r
                R[2, 2] = self.data.dataSet[ii, jj].ur / self.data.dataSet[ii, jj].r
                R[2, 3] = self.data.dataSet[ii, jj].dut_dz
                R[2, 4] = 0
                R[3, 0] = -self.data.dataSet[ii, jj].dp_dz / self.data.dataSet[ii, jj].rho ** 2
                R[3, 1] = self.data.dataSet[ii, jj].duz_dr
                R[3, 2] = 0
                R[3, 3] = self.data.dataSet[ii, jj].duz_dz
                R[3, 4] = 0
                R[4, 0] = 0
                R[4, 1] = self.data.dataSet[ii, jj].p * self.gmma / self.data.dataSet[ii, jj].r + self.data.dataSet[ii, jj].dp_dr
                R[4, 2] = 0
                R[4, 3] = self.data.dataSet[ii, jj].dp_dz
                R[4, 4] = self.gmma * (self.data.dataSet[ii, jj].duz_dz + self.data.dataSet[ii, jj].dur_dr +
                                       self.data.dataSet[ii, jj].ur / self.data.dataSet[ii, jj].r)
                R = self.NormalizeMatrix(R)  # normalization
                self.data.dataSet[ii, jj].AddRMatrix(R)

    def AddRMatrixToNodesFrancesco(self):
        """
        compute and store at the node level the R matrix, ready to be used in the final system of eqs.
        my version, checked
        """
        for ii in range(0, self.data.nAxialNodes):
            for jj in range(0, self.data.nRadialNodes):
                R = np.zeros((5, 5), dtype=complex)

                R[0, 0] = self.data.dataSet[ii, jj].dur_dr + self.data.dataSet[ii, jj].duz_dz + (self.data.dataSet[ii, jj].ur
                                                                                                 / self.data.dataSet[ii, jj].r)
                R[0, 1] = self.data.dataSet[ii, jj].rho / self.data.dataSet[ii, jj].r + self.data.dataSet[ii, jj].drho_dr
                R[0, 2] = 0
                R[0, 3] = self.data.dataSet[ii, jj].drho_dz
                R[0, 4] = 0
                R[1, 0] = self.data.dataSet[ii, jj].dp_dr / self.data.dataSet[ii, jj].rho ** 2
                R[1, 1] = self.data.dataSet[ii, jj].dur_dr
                R[1, 2] = -2 * self.data.dataSet[ii, jj].ut / self.data.dataSet[ii, jj].r
                R[1, 3] = self.data.dataSet[ii, jj].dur_dz
                R[1, 4] = 0
                R[2, 0] = 0
                R[2, 1] = self.data.dataSet[ii, jj].dut_dr + self.data.dataSet[ii, jj].ut / self.data.dataSet[ii, jj].r
                R[2, 2] = self.data.dataSet[ii, jj].ur / self.data.dataSet[ii, jj].r
                R[2, 3] = self.data.dataSet[ii, jj].dut_dz
                R[2, 4] = 0
                R[3, 0] = self.data.dataSet[ii, jj].dp_dz / self.data.dataSet[ii, jj].rho ** 2
                R[3, 1] = self.data.dataSet[ii, jj].duz_dr
                R[3, 2] = 0
                R[3, 3] = self.data.dataSet[ii, jj].duz_dz
                R[3, 4] = 0

                R[4, 0] = -self.data.dataSet[ii, jj].p / self.data.dataSet[ii, jj].rho ** 2 * (
                        self.data.dataSet[ii, jj].ur * self.data.dataSet[ii, jj].drho_dr +
                        self.data.dataSet[ii, jj].uz * self.data.dataSet[ii, jj].drho_dz
                )
                R[4, 1] = self.data.dataSet[ii, jj].dp_dr - self.data.dataSet[ii, jj].p / self.data.dataSet[ii, jj].rho * \
                          self.data.dataSet[ii, jj].drho_dr + (self.gmma - 1) * self.data.dataSet[ii, jj].p / self.data.dataSet[
                              ii, jj].r
                R[4, 2] = 0
                R[4, 3] = self.data.dataSet[ii, jj].dp_dz - self.data.dataSet[ii, jj].p / self.data.dataSet[ii, jj].rho * \
                          self.data.dataSet[ii, jj].drho_dz
                R[4, 4] = -1 / self.data.dataSet[ii, jj].rho * (
                        self.data.dataSet[ii, jj].ur * self.data.dataSet[ii, jj].drho_dr + self.data.dataSet[ii, jj].uz *
                        self.data.dataSet[ii, jj].drho_dz
                ) + (self.gmma - 1) * (self.data.dataSet[ii, jj].ur / self.data.dataSet[ii, jj].r + self.data.dataSet[
                    ii, jj].dur_dr +
                                       self.data.dataSet[ii, jj].duz_dz)

                R = self.NormalizeMatrix(R)  # normalization
                self.data.dataSet[ii, jj].AddRMatrix(R)

    def AddRMatrixToNodesFrancesco2(self):
        """
        compute and store at the node level the R matrix, ready to be used in the final system of eqs.
        """
        for ii in range(0, self.data.nAxialNodes):
            for jj in range(0, self.data.nRadialNodes):
                R = np.zeros((5, 5), dtype=complex)

                R[0, 0] = self.data.dataSet[ii, jj].dur_dr + self.data.dataSet[ii, jj].duz_dz + (self.data.dataSet[ii, jj].ur
                                                                                                 / self.data.dataSet[ii, jj].r)

                R[0, 1] = self.data.dataSet[ii, jj].rho / self.data.dataSet[ii, jj].r + self.data.dataSet[ii, jj].drho_dr
                R[0, 2] = 0
                R[0, 3] = self.data.dataSet[ii, jj].drho_dz
                R[0, 4] = 0
                R[1, 0] = self.data.dataSet[ii, jj].dp_dr / (self.data.dataSet[ii, jj].rho ** 2)
                R[1, 1] = self.data.dataSet[ii, jj].dur_dr
                R[1, 2] = -2 * self.data.dataSet[ii, jj].ut / self.data.dataSet[ii, jj].r
                R[1, 3] = self.data.dataSet[ii, jj].dur_dz
                R[1, 4] = 0
                R[2, 0] = 0
                R[2, 1] = self.data.dataSet[ii, jj].dut_dr + self.data.dataSet[ii, jj].ut / self.data.dataSet[ii, jj].r
                R[2, 2] = self.data.dataSet[ii, jj].ur / self.data.dataSet[ii, jj].r
                R[2, 3] = self.data.dataSet[ii, jj].dut_dz
                R[2, 4] = 0
                R[3, 0] = self.data.dataSet[ii, jj].dp_dz / (self.data.dataSet[ii, jj].rho ** 2)
                R[3, 1] = self.data.dataSet[ii, jj].duz_dr
                R[3, 2] = 0
                R[3, 3] = self.data.dataSet[ii, jj].duz_dz
                R[3, 4] = 0
                R[4, 0] = (1 / self.data.dataSet[ii, jj].rho) * (self.data.dataSet[ii, jj].ur * self.data.dataSet[ii, jj].dp_dr +
                                                                 self.data.dataSet[ii, jj].uz * self.data.dataSet[ii, jj].dp_dz)

                R[4, 1] = -self.data.dataSet[ii, jj].p * self.data.dataSet[ii, jj].drho_dr * self.gmma / \
                          self.data.dataSet[ii, jj].rho + self.data.dataSet[ii, jj].dp_dr

                R[4, 2] = 0
                R[4, 3] = self.data.dataSet[ii, jj].dp_dz - self.gmma / self.data.dataSet[ii, jj].rho * \
                          self.data.dataSet[ii, jj].p * self.data.dataSet[ii, jj].drho_dz

                R[4, 4] = (-self.data.dataSet[ii, jj].ur * self.data.dataSet[ii, jj].drho_dr -
                           self.data.dataSet[ii, jj].uz * self.data.dataSet[ii, jj].drho_dz) * \
                          self.gmma / self.data.dataSet[ii, jj].rho

                R = self.NormalizeMatrix(R)  # normalization
                self.data.dataSet[ii, jj].AddRMatrix(R)

    def AddSMatrixToNodes(self, turbo=True):
        """
        compute and store at the node level the S matrix, ready to be used in the final system of eqs. The matrix formulation
        depends on the selected body-force model
        """
        print("S Body Force Matrix: %s" %(turbo))
        for ii in range(0, self.data.nAxialNodes):
            for jj in range(0, self.data.nRadialNodes):
                S = np.zeros((5, 5), dtype=complex)

                if turbo:
                    S[0, 0] = self.data.meridional_obj.S00[ii, jj]
                    S[0, 1] = self.data.meridional_obj.S01[ii, jj]
                    S[0, 2] = self.data.meridional_obj.S02[ii, jj]
                    S[0, 3] = self.data.meridional_obj.S03[ii, jj]
                    S[0, 4] = self.data.meridional_obj.S04[ii, jj]

                    S[1, 0] = self.data.meridional_obj.S10[ii, jj]
                    S[1, 1] = self.data.meridional_obj.S11[ii, jj]
                    S[1, 2] = self.data.meridional_obj.S12[ii, jj]
                    S[1, 3] = self.data.meridional_obj.S13[ii, jj]
                    S[1, 4] = self.data.meridional_obj.S14[ii, jj]

                    S[2, 0] = self.data.meridional_obj.S20[ii, jj]
                    S[2, 1] = self.data.meridional_obj.S21[ii, jj]
                    S[2, 2] = self.data.meridional_obj.S22[ii, jj]
                    S[2, 3] = self.data.meridional_obj.S23[ii, jj]
                    S[2, 4] = self.data.meridional_obj.S24[ii, jj]

                    S[3, 0] = self.data.meridional_obj.S30[ii, jj]
                    S[3, 1] = self.data.meridional_obj.S31[ii, jj]
                    S[3, 2] = self.data.meridional_obj.S32[ii, jj]
                    S[3, 3] = self.data.meridional_obj.S33[ii, jj]
                    S[3, 4] = self.data.meridional_obj.S34[ii, jj]

                    S[4, 0] = self.data.meridional_obj.S40[ii, jj]
                    S[4, 1] = self.data.meridional_obj.S41[ii, jj]
                    S[4, 2] = self.data.meridional_obj.S42[ii, jj]
                    S[4, 3] = self.data.meridional_obj.S43[ii, jj]
                    S[4, 4] = self.data.meridional_obj.S44[ii, jj]

                S = self.NormalizeMatrix(S)  # normalization
                self.data.dataSet[ii, jj].AddSMatrix(S)

    def AddHatMatricesToNodes(self):
        """
        compute and store at the node level the hat{B},hat{E} matrix, needed for following multiplication with the spectral
        differential operators
        """
        for ii in range(0, self.data.nAxialNodes):
            for jj in range(0, self.data.nRadialNodes):
                Bhat = -(1 / self.data.dataSet[ii, jj].J) * (self.data.dataSet[ii, jj].B * self.data.dataSet[ii, jj].dzdy -
                                                             self.data.dataSet[ii, jj].E * self.data.dataSet[ii, jj].drdy)

                Ehat = (1 / self.data.dataSet[ii, jj].J) * (self.data.dataSet[ii, jj].B * self.data.dataSet[ii, jj].dzdx -
                                                            self.data.dataSet[ii, jj].E * self.data.dataSet[ii, jj].drdx)

                self.data.dataSet[ii, jj].AddHatMatrices(Bhat, Ehat)

    def ApplySpectralDifferentiation(self, verbose=False):
        """
       This method applies Chebyshev-Gauss-Lobatto differentiation method to express the perturbation derivatives as a function
       of the perturbation at the other nodes. It saves a new global (related to all the nodes) matrix Q_const, which is a part 
       of the global stability matrix (nPoints*5 X nPoints*5).
       The spectral differentiation formula has been double-checked in debug files
       """

        # the cordinates on the spectral grid directions the spectral matrix Dx and Dy
        x = self.dataSpectral.z
        y = self.dataSpectral.r

        # compute the spectral Matrices for x and y direction with the Bayliss formulation
        Dx = ChebyshevDerivativeMatrixBayliss(x)  # derivative operator in xi, Bayliss formulation
        Dy = ChebyshevDerivativeMatrixBayliss(y)  # derivative operator in eta, Bayliss formulation

        # Q_const is the global matrix storing B and E effects after spectral differentiation
        self.Q_const = np.zeros((self.nPoints * 5, self.nPoints * 5),
                                dtype=complex)  # instantiate the full stability matrix, that will be filled block by block

        # differentiation of a general perturbation vector (for the node (i,j)) along xi  and eta. (formula double-checked)
        for ii in range(0, self.dataSpectral.nAxialNodes):
            for jj in range(0, self.dataSpectral.nRadialNodes):
                B_ij = self.data.dataSet[ii, jj].Bhat  # Bhat matrix of the ij node
                E_ij = self.data.dataSet[ii, jj].Ehat  # Ehat matrix of the ij node
                node_counter = self.data.dataSet[
                    ii, jj].nodeCounter  # needed to keep track of the row in the stability equations.
                # every new node, increase the row number by 5

                # xi differentiation. m is in the range of axial nodes, first axis of the matrix
                for m in range(0, self.dataSpectral.nAxialNodes):
                    tmp = Dx[ii, m] * B_ij  # 5x5 matrix to be added to a certain block of Q
                    row = node_counter * 5  # this selects the correct block along i of Q
                    column = (m * self.dataSpectral.nRadialNodes + jj) * 5  # it selects the correct block along
                    # the second axis the matrix

                    if verbose:
                        print('Node [i,j] = (%.1d,%.1d)' % (ii, jj))
                        print('Element along i [m,j] = (%.1d,%.1d)' % (m, jj))
                        print('Derivative element [ii,m] = (%.1d,%.1d)' % (ii, m))
                        print('[row,col] = (%.1d,%.1d)' % (row, column))
                    self.AddToQ_const(tmp, row, column)

                # xi differentiation. n is in the range of radial nodes, second axis of the matrix
                for n in range(0, self.dataSpectral.nRadialNodes):
                    tmp = Dy[jj, n] * E_ij  # 5x5 matrix to be added to a certain block of Q
                    row = node_counter * 5  # this selects the correct block along i of Q
                    column = (ii * self.dataSpectral.nRadialNodes + n) * 5  # this is the important point
                    if verbose:
                        print('Node [i,j] = (%.1d,%.1d)' % (ii, jj))
                        print('Element along j [i,n] = (%.1d,%.1d)' % (jj, n))
                        print('Derivative element [jj,n] = (%.1d,%.1d)' % (jj, n))
                        print('[row,col] = (%.1d,%.1d)' % (row, column))
                    self.AddToQ_const(tmp, row, column)

    def NormalizeMatrix(self, M):
        """
        normalize every 5x5 matrix, where the first row is continuity coeffs, then 3 equations for momentum, and one for pressure.
        The multiplication factors guarantee that the final equations are non-dimensional (only if the matrix coefficients
        are taken from the same source)
        """
        M[0, :] = M[0, :] * self.continuity_norm
        M[1:4, :] = M[1:4, :] * self.momentum_norm
        M[4, :] = M[4, :] * self.pressure_norm
        return M

    def AddToQ_const(self, block, row, column):
        """
        add elements to the Bddxi+Eddeta matrix, specifying the first element location
        """
        self.Q_const[row:row + 5, column:column + 5] += block

    def AddToQ_var(self, block, row, column):
        """
        add elements to the variable part of the stability matrix specifying the first telement location
        """
        self.Q_var[row:row + 5, column:column + 5] += block

    def add_to_Y(self, block, row, column):
        """
        add elements to the constant part of the stability matrix specifying the first element location
        """
        self.Y[row:row + 5, column:column + 5] += block

    def add_to_A_g(self, block, row, column):
        """
        add elements to the constant part of the stability matrix specifying the first element location
        """
        self.A_g[row:row + 5, column:column + 5] += block

    def add_to_C_g(self, block, row, column):
        """
        add elements to the constant part of the stability matrix specifying the first element location
        """
        self.C_g[row:row + 5, column:column + 5] += block

    def add_to_R_g(self, block, row, column):
        """
        add elements to the constant part of the stability matrix specifying the first element location
        """
        self.R_g[row:row + 5, column:column + 5] += block

    def add_to_S_g(self, block, row, column):
        """
        add elements to the constant part of the stability matrix specifying the first element location
        """
        self.S_g[row:row + 5, column:column + 5] += block

    def ApplyBoundaryConditions(self):
        """
        it applies the correct set of boundary conditions to all the points marked with a boundary marker. 
        Every BC will modify the 5 equations for the respective node. The BCs are taken from what previously given.
        Boundary matching conditions could be given here.
        """
        for ii in range(0, self.data.nAxialNodes):
            for jj in range(0, self.data.nRadialNodes):
                marker = self.data.dataSet[ii, jj].marker
                counter = self.data.dataSet[ii, jj].nodeCounter
                row = counter * 5  # 5 equations per node
                if marker == 'inlet':
                    self.ApplyBCCondition(row, self.inlet_bc)

                elif marker == 'outlet':
                    self.ApplyBCCondition(row, self.outlet_bc)

                elif (marker == 'hub'):
                    self.ApplyBCCondition(row, self.hub_bc)

                elif marker == 'shroud':
                    self.ApplyBCCondition(row, self.shroud_bc)

                elif (marker != 'internal'):
                    raise Exception('Boundary condition unknown. Check the grid markers!')

    def AddRemainingMatrices(self, omega):
        """
        it adds the remaning diagonal block matrices to the full Qtot = Q_const + Q_var. Q_const is constant for every model,
        while Q_var depends on omega, which should be already provided as non-dimensional
        """
        # Q_var is composed by A, C, R, S. Q_const is composed by B,E spectrally differentiated. Q_tot is the global sum
        self.Q_var = np.zeros((self.Q_const.shape),
                              dtype=complex)  # variable part of the stability matrix. Instantiated for every new value of omega
        for ii in range(0, self.dataSpectral.nAxialNodes):
            for jj in range(0, self.dataSpectral.nRadialNodes):
                # add all the remaining terms on the diagonal
                diag_block_ij = -1j * omega * self.data.dataSet[ii, jj].A + self.data.dataSet[ii, jj].C + self.data.dataSet[
                    ii, jj].R + self.data.dataSet[ii, jj].S
                node_counter = self.data.dataSet[ii, jj].nodeCounter
                row = node_counter * 5
                column = node_counter * 5  # diagonal block
                self.AddToQ_var(diag_block_ij, row, column)
        self.Q_tot = self.Q_const + self.Q_var  # compute the global stability matrix

    def ApplyBCCondition(self, row, condition, wall_normal=np.array([1, 0, 0])):
        """
        for the considered grid node, it modifes the 5 governing equations starting from row index,
        which is related to its continuity eq.
        Args:
            row: row index, specifying which grid node equations will be modified
            condition: type of boundary condition
            wall_normal: normal to the wall at that point

        Returns:

        """
        if condition == 'zero perturbation':
            # BC for zero perturbation vector
            self.Q_tot[row:row + 5, :] = np.zeros(self.Q_tot[row:row + 5, :].shape, dtype=complex)  # make it zero first
            self.Q_tot[row:row + 5, row:row + 5] = np.eye(5, dtype=complex)  # zero perturbation condition for every flow variable

        elif condition == 'zero pressure':
            # BC for zero pressure perturbation
            self.Q_tot[row + 4, :] = np.zeros(self.Q_tot[row + 1, :].shape, dtype=complex)
            self.Q_tot[row + 4, row + 4] = 1  # zero pressure coefficient at that node

        elif condition == 'euler wall':
            # BC for non-penetration condition at the walls. The variable substituted_equation decides which equation has
            # be overwritten from the boundary condition. In principle anyone of the momentum equations should be ok.

            if self.substituted_equation == 'rho':
                # density equation modified
                print('Attention, Euler wall condition overwriting continuity equation. It should be done only for testing!')
                self.Q_tot[row, :] = np.zeros(self.Q_tot[row + 3, :].shape,
                                              dtype=complex)  # first make zero the axial velocity equation
                self.Q_tot[row, row + 1:row + 4] = wall_normal  # impose non-penetration condition (u*nr + v*nt + w*nz)

            elif self.substituted_equation == 'ur':
                # radial velocity equation modified
                self.Q_tot[row + 1, :] = np.zeros(self.Q_tot[row + 1, :].shape,
                                                  dtype=complex)  # first make zero the radial velocity equation
                self.Q_tot[row + 1, row + 1:row + 4] = wall_normal  # impose non-penetration condition (u*nr + v*nt + w*nz)

            elif self.substituted_equation == 'ut':
                # tangential velocity equation overwritten
                self.Q_tot[row + 2, :] = np.zeros(self.Q_tot[row + 3, :].shape,
                                                  dtype=complex)  # first make zero the tangential velocity equation
                self.Q_tot[row + 2, row + 1:row + 4] = wall_normal  # impose non-penetration condition (u*nr + v*nt + w*nz)

            elif self.substituted_equation == 'uz':
                # axial velocity equation modified
                self.Q_tot[row + 3, :] = np.zeros(self.Q_tot[row + 3, :].shape,
                                                  dtype=complex)  # first make zero the axial velocity equation
                self.Q_tot[row + 3, row + 1:row + 4] = wall_normal  # impose non-penetration condition (u*nr + v*nt + w*nz)

            elif self.substituted_equation == 'p':
                # pressure equation modified
                print('Attention, Euler wall condition overwriting pressure equation. It should be done only for testing!')
                self.Q_tot[row + 4, :] = np.zeros(self.Q_tot[row + 4, :].shape,
                                                  dtype=complex)  # first make zero the axial velocity equation
                self.Q_tot[row + 4, row + 1:row + 4] = wall_normal  # impose non-penetration condition (u*nr + v*nt + w*nz)
            else:
                raise ValueError('Equation to overwritten not specified!')

        else:
            raise ValueError('Boundary condition unknown')

    def ComputeSVD(self, omega_domain=None, grid_omega=None):
        """
        compute the SVD for every omega in omega_domain, discretized as in grid_omega. It computes
        every time the part of Q that depends on omega (-j*omega*A), and computes the boundary conditions.
        Then it computes the singular values, and store the inverse of the condition number in the chi attribute. Singular values 
        max and min are also stored for code test.
        """

        if omega_domain is None:
            omega_domain = np.array([-1, 1, -1, 0.5])

        if grid_omega is None:
            grid_omega = [10, 10]

        omR_min = omega_domain[0]
        omR_max = omega_domain[1]
        omI_min = omega_domain[2]
        omI_max = omega_domain[3]
        nR = grid_omega[0]
        nI = grid_omega[1]
        omR = np.linspace(omR_min, omR_max, nR)
        omI = np.linspace(omI_min, omI_max, nI)
        self.omegaI, self.omegaR = np.meshgrid(omI, omR)
        self.chi = np.zeros((nR, nI))
        self.sing_value_min = np.zeros((nR, nI))
        self.sing_value_max = np.zeros((nR, nI))
        start_time = time.time()
        for ii in range(0, nR):
            for jj in range(0, nI):

                # this block is simply to print info, and time remaining for the full process
                current_time = time.time() - start_time
                if (ii == 0 and jj == 0):
                    print('SVD %.1d of %1.d ..' % (ii * len(omI) + 1 + jj, len(omR) * len(omI)))
                if (jj == 1):  # update time whenever jj=1
                    delta_time_svd = current_time
                    total_time = (delta_time_svd / (
                            ii * nI + jj)) * nR * nI  # (time passed / number of SVD done) * number of total SVD to do
                if (ii != 0 or jj != 0):
                    remaining_minutes = (total_time - current_time) / 60
                    total_minutes = total_time / 60
                    print('SVD %.1d of %1.d \t (%.1d min remaining)' % (
                        ii * len(omI) + 1 + jj, len(omR) * len(omI), remaining_minutes + 1))  # keep track of the progress

                omega = omR[ii] + 1j * omI[jj]
                self.AddRemainingMatrices(omega)  # add the non-constant parts of the matrices
                self.ApplyBoundaryConditions()  # apply boundary condtions
                u, s, v = np.linalg.svd(self.Q_tot)
                self.sing_value_min[ii, jj] = np.min(s)
                self.sing_value_max[ii, jj] = np.max(s)
                self.chi[ii, jj] = np.min(s) / np.max(s)

        end_time = time.time() - start_time
        hrs = int(end_time / 3600)
        mins = int((end_time - hrs * 3600) / 60)
        sec = int(end_time - hrs * 3600 - mins * 60)
        print('Total SVD time: \t %1.d hrs %1.d mins %1.d sec' % (hrs, mins, sec))
        # self.eigs = np.array(self.FindLocalMinima(self.chi))

    def ComputeSVD2(self, omega_domain=None, grid_omega=None):
        """
        compute the SVD for every omega in omega_domain, discretized as in grid_omega. It computes
        every time the part of Q that depends on omega (-j*omega*A), and computes the boundary conditions.
        Then it computes the singular values, and store the inverse of the condition number in the chi attribute. Singular values
        max and min are also stored for code test.
        """
        if omega_domain is None:
            omega_domain = [-1, 1, -1, 1]
        if grid_omega is None:
            grid_omega = [10, 10]

        omR_min = omega_domain[0]
        omR_max = omega_domain[1]
        omI_min = omega_domain[2]
        omI_max = omega_domain[3]
        nR = grid_omega[0]
        nI = grid_omega[1]
        omR = np.linspace(omR_min, omR_max, nR)
        omI = np.linspace(omI_min, omI_max, nI)
        self.omegaI, self.omegaR = np.meshgrid(omI, omR)
        self.chi = np.zeros((nR, nI))
        self.sing_value_min = np.zeros((nR, nI))
        self.sing_value_max = np.zeros((nR, nI))
        start_time = time.time()
        for ii in range(0, nR):
            for jj in range(0, nI):

                # this block is simply to print info, and time remaining for the full process
                current_time = time.time() - start_time
                if (ii == 0 and jj == 0):
                    print('SVD %.1d of %1.d ..' % (ii * len(omI) + 1 + jj, len(omR) * len(omI)))
                if (jj == 1):  # update time whenever jj=1
                    delta_time_svd = current_time
                    total_time = (delta_time_svd / (
                            ii * nI + jj)) * nR * nI  # (time passed / number of SVD done) * number of total SVD to do
                if (ii != 0 or jj != 0):
                    remaining_minutes = (total_time - current_time) / 60
                    total_minutes = total_time / 60
                    print('SVD %.1d of %1.d \t (%.1d min remaining)' % (
                        ii * len(omI) + 1 + jj, len(omR) * len(omI), remaining_minutes + 1))  # keep track of the progress

                omega = omR[ii] + 1j * omI[jj]
                u, s, v = np.linalg.svd(self.Z_g - 1j * omega * self.A_g)
                self.sing_value_min[ii, jj] = np.min(s)
                self.sing_value_max[ii, jj] = np.max(s)
                self.chi[ii, jj] = np.min(s) / np.max(s)

        end_time = time.time() - start_time
        hrs = int(end_time / 3600)
        mins = int((end_time - hrs * 3600) / 60)
        sec = int(end_time - hrs * 3600 - mins * 60)
        print('Total SVD time: \t %1.d hrs %1.d mins %1.d sec' % (hrs, mins, sec))
        self.eigs = self.FindLocalMinima(self.chi)

    def ComputeSVDcompressor(self, RS_domain=np.array([-1, 1]), DF_domain=np.array([-1, 1]),
                             grid=np.array([10, 10]), verbose=True):
        """
        compute the SVD for every omega in omega_domain, discretized as in grid_omega. It computes
        every time the part of Q that depends on omega (-j*omega*A), and computes the boundary conditions.
        Then it computes the singular values, and store the inverse of the condition number in the chi attribute. Singular values 
        max and min are also stored for code test.
        """
        # find the limits of the non-dimensional omega of reserach = omega/omega_ref, where omega real and imaginary are
        # defined starting from the Rotational Speed RS and Damping Factor DF of the instability lobe (m=1)
        # omega_{dimensional}=omega_shaft*RS+1j*U*DF/r where U and r taken where we want.
        # omega_{non-dimensinoal}=omega_{dimensional}/omega_ref

        omR_min = RS_domain[0]
        omR_max = RS_domain[1]
        omI_min = DF_domain[0]
        omI_max = DF_domain[1]
        self.RS_domain = RS_domain
        self.DF_domain = DF_domain

        # discretization of the domain of research
        nR = grid[0]
        nI = grid[1]
        omR = np.linspace(omR_min, omR_max, nR)
        omI = np.linspace(omI_min, omI_max, nI)
        self.omegaI, self.omegaR = np.meshgrid(omI, omR)
        self.DF = self.omegaI
        self.RS = self.omegaR

        # instantiate results matrices and compute SVD for every point
        self.chi = np.zeros((nR, nI))  # inverse of the condition number
        self.sing_value_min = np.zeros((nR, nI))  # minimum singular value
        self.sing_value_max = np.zeros((nR, nI))  # maximum singular value

        start_time = time.time()
        for ii in range(0, nR):
            for jj in range(0, nI):

                # this block is simply to print info, and time remaining for the full process
                current_time = time.time() - start_time
                if (ii == 0 and jj == 0):
                    if verbose:
                        print('SVD %.1d of %1.d ..' % (ii * len(omI) + 1 + jj, len(omR) * len(omI)))
                if (jj == 1):  # update time whenever jj=1
                    delta_time_svd = current_time
                    total_time = (delta_time_svd / (
                            ii * nI + jj)) * nR * nI  # (time passed / number of SVD done) * number of total SVD to do
                if (ii != 0 or jj != 0):
                    remaining_minutes = (total_time - current_time) / 60
                    total_minutes = total_time / 60
                    if verbose:
                        print('SVD %.1d of %1.d \t (%.1d min remaining)' % (
                            ii * len(omI) + 1 + jj, len(omR) * len(omI), remaining_minutes + 1))  # keep track of the progress

                omega = omR[ii] + 1j * omI[jj]

                # the problem could be that every processor modify the matrices
                lmbda_factor = 1 / (1 + (1j * omega + 1 * self.omega_shaft / self.omega_ref))
                self.Q_tot = -1j * omega * self.A_g + self.Q_const + self.C_g + self.R_g + lmbda_factor * self.S_g
                self.ApplyBoundaryConditions()  # apply boundary condtions to boundary nodes
                u, s, v = np.linalg.svd(self.Q_tot)  # s contains the singular values in descending order
                self.sing_value_min[ii, jj] = np.min(s)
                self.sing_value_max[ii, jj] = np.max(s)
                self.chi[ii, jj] = np.min(s) / np.max(s)  # definition of chi

        # #print info on time progression
        end_time = time.time() - start_time
        hrs = int(end_time / 3600)
        mins = int((end_time - hrs * 3600) / 60)
        sec = int(end_time - hrs * 3600 - mins * 60)
        if (verbose):
            print('Total SVD time: \t %1.d hrs %1.d mins %1.d sec' % (hrs, mins, sec))

    def PlotInverseConditionNumberCompressor(self, sing_val=False, scale=None, save_filename=None, formatFig=(10, 6),
                                             ref_solution=None):
        """
        plots the chi map for the problem modeled. Log scale can be applied to see better the valleys
        """
        x = np.linspace(self.RS_domain[0], self.RS_domain[1])
        y = np.linspace(self.DF_domain[0], self.DF_domain[1])
        critical_line = np.zeros(len(x))  # superpose a critical line

        fig, ax = plt.subplots(figsize=formatFig)
        if scale == 'log':
            cs = ax.contourf(self.RS, self.DF, self.chi, N_levels, locator=ticker.LogLocator(), cmap=color_map)
        else:
            cs = ax.contourf(self.RS, self.DF, self.chi, N_levels, cmap=color_map)
        ax.plot(x, critical_line, '--w')
        ax.set_xlabel(r'$RS \ \mathrm{[-]}$')
        ax.set_ylabel(r'$DF \ \mathrm{[-]}$')
        ax.set_title(r'$\chi$')
        cbar = fig.colorbar(cs)
        if save_filename is not None:
            fig.savefig(folder_name + save_filename + '.pdf', bbox_inches='tight')

        if sing_val:
            self.PlotSingularValues()

    def FindLocalMinima(self, field):
        """
        Locate the local minima of the chi map
        """

        def find_local_minima(array):
            # Apply minimum filter to find the local minima
            neighborhood = np.ones((5, 5))  # 3x3 neighborhood
            local_minima = minimum_filter(array, footprint=neighborhood) == array
            # Get the indices of local minima
            indices = np.transpose(np.nonzero(local_minima))
            return indices

        minima_indices = find_local_minima(field)
        eigenvalues = []
        n_eig = minima_indices.shape[0]
        for s in range(0, n_eig):
            eig = self.omegaR[minima_indices[s, 0], minima_indices[s, 1]] + 1j * self.omegaI[
                minima_indices[s, 0], minima_indices[s, 1]]
            eigenvalues.append(eig * self.omega_ref)

        return eigenvalues

    def PlotInverseConditionNumber(self, scale=None, save_filename=None, formatFig=(10, 6), ref_solution=None):
        """
        plots the chi map for the problem modeled. Log scale can be applied to see better the valleys
        """
        x = np.linspace(np.min(self.omegaR), np.max(self.omegaR)) * self.omega_ref
        critical_line = np.zeros(len(x))

        fig, ax = plt.subplots(figsize=formatFig)
        if scale == 'log':
            cs = ax.contourf(self.omegaR * self.omega_ref, self.omegaI * self.omega_ref, self.chi, N_levels,
                             locator=ticker.LogLocator(), cmap=color_map)
        else:
            cs = ax.contourf(self.omegaR * self.omega_ref, self.omegaI * self.omega_ref, np.log(self.chi), N_levels,
                             cmap=color_map)
            # ax.plot(self.eigs.real * self.omega_ref, self.eigs.imag * self.omega_ref, 'ko')
        ax.plot(x, critical_line, '--r')
        ax.set_xlabel(r'$\omega_{R} \quad [rad/s]$')
        ax.set_ylabel(r'$\omega_{I} \quad [rad/s]$')
        ax.set_title(r'$\log \chi$')
        ax.set_xlim([np.min(self.omegaR * self.omega_ref), np.max(self.omegaR * self.omega_ref)])
        cbar = fig.colorbar(cs)
        if ref_solution is not None:
            cs = ax.plot(ref_solution.real, ref_solution.imag, 'ws')
        if save_filename is not None:
            fig.savefig(folder_name + save_filename + '.pdf', bbox_inches='tight')

    def PlotSingularValues(self, scale=None, save_filename=None, formatFig=(15, 6)):
        """
        plot of the singular values max and min, to check if their magnitude is within machine accuracy
        """
        x = np.linspace(np.min(self.omegaR), np.max(self.omegaR)) * self.omega_ref
        critical_line = np.zeros(len(x))
        fig, ax = plt.subplots(1, 2, figsize=formatFig)
        if scale == 'log':
            cs0 = ax[0].contourf(self.omegaR * self.omega_ref, self.omegaI * self.omega_ref, self.sing_value_max, N_levels,
                                 locator=ticker.LogLocator(), cmap=color_map)
            cs1 = ax[1].contourf(self.omegaR * self.omega_ref, self.omegaI * self.omega_ref, self.sing_value_min, N_levels,
                                 locator=ticker.LogLocator(), cmap=color_map)
        else:
            cs0 = ax[0].contourf(self.omegaR * self.omega_ref, self.omegaI * self.omega_ref, self.sing_value_max, N_levels,
                                 cmap=color_map)
            cs1 = ax[1].contourf(self.omegaR * self.omega_ref, self.omegaI * self.omega_ref, self.sing_value_min, N_levels,
                                 cmap=color_map)

        ax[0].plot(x, critical_line, '--r')
        ax[0].set_xlabel(r'$\omega_{R} \quad [rad/s]$')
        ax[0].set_ylabel(r'$\omega_{I} \quad [rad/s]$')
        ax[0].set_title(r'$\sigma_{max}$')
        cbar = fig.colorbar(cs0)
        ax[1].plot(x, critical_line, '--r')
        ax[1].set_xlabel(r'$\omega_{R} \quad [rad/s]$')
        ax[1].set_title(r'$\sigma_{min}$')
        cbar = fig.colorbar(cs1)

        if save_filename is not None:
            fig.savefig(folder_name + save_filename + '.pdf', bbox_inches='tight')

    def ComputeBoundaryNormals(self):
        """
       compute the normal vectors on the hub and shroud nodes
       """
        self.data.ComputeBoundaryNormals()  # method belonging to compressor_grid object

    def ShowNormals(self):
        """
       plots the boundary nodes and the normals
       """
        self.data.ShowNormals()  # method belonging to compressor_grid object

    def build_A_global_matrix(self):
        """
        build the A global matrix, stacking together the A matrices of all the nodes
        """
        self.A_g = np.zeros((self.Q_const.shape[0], self.Q_const.shape[1]), dtype=complex)
        for ii in range(0, self.dataSpectral.nAxialNodes):
            for jj in range(0, self.dataSpectral.nRadialNodes):
                # add all the remaining terms on the diagonal
                diag_block_ij = self.data.dataSet[ii, jj].A
                node_counter = self.data.dataSet[ii, jj].nodeCounter
                row = node_counter * 5
                column = node_counter * 5  # diagonal block
                self.add_to_A_g(diag_block_ij, row, column)

    def build_C_global_matrix(self):
        """
        build the C*j*m/r global matrix, stacking together the C*j*m/r matrices of all the nodes
        """
        self.C_g = np.zeros((self.Q_const.shape[0], self.Q_const.shape[1]), dtype=complex)
        for ii in range(0, self.dataSpectral.nAxialNodes):
            for jj in range(0, self.dataSpectral.nRadialNodes):
                # add all the remaining terms on the diagonal
                diag_block_ij = self.data.dataSet[ii, jj].C
                node_counter = self.data.dataSet[ii, jj].nodeCounter
                row = node_counter * 5
                column = node_counter * 5  # diagonal block
                self.add_to_C_g(diag_block_ij, row, column)

    def build_R_global_matrix(self):
        """
        build the R global matrix
        """
        self.R_g = np.zeros((self.Q_const.shape[0], self.Q_const.shape[1]), dtype=complex)
        for ii in range(0, self.dataSpectral.nAxialNodes):
            for jj in range(0, self.dataSpectral.nRadialNodes):
                # add all the remaining terms on the diagonal
                diag_block_ij = self.data.dataSet[ii, jj].R
                node_counter = self.data.dataSet[ii, jj].nodeCounter
                row = node_counter * 5
                column = node_counter * 5  # diagonal block
                self.add_to_R_g(diag_block_ij, row, column)

    def build_S_global_matrix(self):
        """
        build the S global matrix
        """
        self.S_g = np.zeros((self.Q_const.shape[0], self.Q_const.shape[1]), dtype=complex)
        for ii in range(0, self.dataSpectral.nAxialNodes):
            for jj in range(0, self.dataSpectral.nRadialNodes):
                # add all the remaining terms on the diagonal
                diag_block_ij = self.data.dataSet[ii, jj].S
                node_counter = self.data.dataSet[ii, jj].nodeCounter
                row = node_counter * 5
                column = node_counter * 5  # diagonal block
                self.add_to_S_g(diag_block_ij, row, column)

    def build_Z_global_matrix(self):
        """
        build the Z global matrix, synonym of J. J = Z = (B_d + C + E_d + R)
        """
        self.Z_g = (self.Q_const + self.C_g + self.R_g)

    def apply_boundary_conditions_generalized(self):
        """
        apply the boundary conditions to the system Y*phi = omega*A_g*phi. The coefficients are inserted in Y, while the
        known terms in A_g
        """
        for ii in range(0, self.data.nAxialNodes):
            for jj in range(0, self.data.nRadialNodes):
                marker = self.data.dataSet[ii, jj].marker
                counter = self.data.dataSet[ii, jj].nodeCounter
                row = counter * 5  # 5 equations per node
                if marker == 'inlet':
                    self.apply_bc_condition(row, self.inlet_bc, ii, jj)

                elif marker == 'outlet':
                    self.apply_bc_condition(row, self.outlet_bc, ii, jj)

                elif (marker == 'hub'):
                    self.apply_bc_condition(row, self.hub_bc, ii, jj)

                elif (marker == 'shroud'):
                    self.apply_bc_condition(row, self.shroud_bc, ii, jj)

                elif (marker != 'internal'):
                    raise Exception('Boundary condition unknown. Check the grid markers!')

    def set_boundary_conditions(self, inlet_bc, outlet_bc, hub_bc='euler wall', shroud_bc='euler wall'):
        """
        store in the object the information related to the boundary conditions to use for the problem
        Args:
            inlet_bc: string explaining which boundary condition apply to the inlet points
            outlet_bc: string explaining which boundary condition apply to the outlet points
            hub_bc: string explaining which boundary condition apply to the hub points
            shroud_bc: string explaining which boundary condition apply to the outlet points

        The possible choices for the boundary conditions are:
        'zero pressure' : impose zero pressure fluctuation
        'zero perturbation' : impose zero value to the full perturbation vector
        'euler wall' : impose tangential velocity to the wall normal in that point
        """

        # recognized boundary conditions type
        bc_list = ['zero pressure', 'zero perturbation', 'euler wall', 'compressor inlet', 'compressor outlet']

        if not inlet_bc in bc_list:
            raise ValueError('Incorrect Inlet boundary condition type.')
        if not outlet_bc in bc_list:
            raise ValueError('Incorrect Outlet boundary condition type.')
        if not hub_bc in bc_list:
            raise ValueError('Incorrect Hub boundary condition type.')
        if not shroud_bc in bc_list:
            raise ValueError('Incorrect Shroud boundary condition type.')

        self.inlet_bc = inlet_bc
        self.outlet_bc = outlet_bc
        self.hub_bc = hub_bc
        self.shroud_bc = shroud_bc

        print("+-------------- BOUNDARY CONDITIONS --------------+")
        print("Inlet: %s" % (self.inlet_bc))
        print("Outlet: %s" % (self.outlet_bc))
        print("Hub: %s" % (self.hub_bc))
        print("Shroud: %s" % (self.shroud_bc))
        print("+-------------------------------------------------+")
        print("\n")



    def apply_bc_condition(self, row, condition, ii, jj):
        """
        for the considered grid node, it modifes the 5 governing equations starting from row index,
        which is related to its continuity eq.
        (ii,jj) is the node index, necessary to impose the wall boundary conditions. The considered system at hand is currently:
        (-j*omega*A + Z + S/zita)*tilde{phi}. Therefore BCs are imposed on Z, and A and S must be filled with zeros
        in correspondance of those BCs
        """

        if condition == 'zero pressure':
            # BC for zero pressure perturbation
            self.Z_g[row + 4, :] = np.zeros(self.Z_g[row + 4, :].shape, dtype=complex)
            self.Z_g[row + 4, row + 4] = 1  # zero pressure at that node

            self.A_g[row + 4, :] = np.zeros(self.A_g[row + 4, :].shape, dtype=complex)  # zero row
            self.S_g[row + 4, :] = np.zeros(self.S_g[row + 4, :].shape, dtype=complex)  # zero row

        elif condition == 'zero perturbation':
            # BC for zero pressure perturbation
            self.Z_g[row:row + 5, :] = np.zeros(self.Z_g[row:row + 5, :].shape, dtype=complex)
            self.Z_g[row:row + 5, row:row + 5] = np.eye(5, dtype=complex)

            self.A_g[row:row + 5, :] = np.zeros(self.A_g[row:row + 5, :].shape, dtype=complex)  # zero rows
            self.S_g[row:row + 5, :] = np.zeros(self.S_g[row:row + 5, :].shape, dtype=complex)  # zero rows

        elif condition == 'compressor inlet':
            # BCs are zero for every variable except the pressure at inlet
            self.Z_g[row:row + 4, :] = np.zeros(self.Z_g[row:row + 4, :].shape, dtype=complex)
            self.Z_g[row:row + 4, row:row + 4] = np.eye(4, dtype=complex)

            self.A_g[row:row + 5, :] = np.zeros(self.A_g[row:row + 5, :].shape, dtype=complex)  # zero rows
            self.S_g[row:row + 5, :] = np.zeros(self.S_g[row:row + 5, :].shape, dtype=complex)  # zero rows

        elif condition == 'compressor outlet':
            # BC for zero pressure perturbation
            self.Z_g[row + 4, :] = np.zeros(self.Z_g[row + 4, :].shape, dtype=complex)
            self.Z_g[row + 4, row + 4] = 1  # zero pressure at that node

            self.A_g[row + 4, :] = np.zeros(self.A_g[row + 4, :].shape, dtype=complex)  # zero row
            self.S_g[row + 4, :] = np.zeros(self.S_g[row + 4, :].shape, dtype=complex)  # zero row

        elif condition == 'euler wall':
            # BC for non-penetration condition at the walls, the equation overwritten depends on configs
            if self.substituted_equation == 'ur':
                loc = 1
            elif self.substituted_equation == 'utheta':
                loc = 2
            elif self.substituted_equation == 'uz':
                loc = 3
            else:
                raise ValueError("Subsituted equation parameter not recognized.")

            wall_normal = self.data.dataSet[ii, jj].n_wall

            self.Z_g[row + loc, :] = np.zeros(self.Z_g[row + loc, :].shape, dtype=complex)
            self.Z_g[row + loc, row + 1:row + 4] = wall_normal

            self.A_g[row + loc, :] = np.zeros(self.A_g[row + loc, :].shape, dtype=complex)  # zero known term
            self.S_g[row + loc, :] = np.zeros(self.S_g[row + loc, :].shape, dtype=complex)  # zero known term

        else:
            raise ValueError('unknown boundary condition type')


    def solve_evp_arnoldi(self, omega_search=0, number_search=10, inspect_matrices=False):
        """
        Solve EVP with implicitly restarted Arnoldi Algorithm, with shift-invert strategy
        """

        m = self.harmonic_order
        Omega = self.data.meridional_obj.omega_shaft  # dimensional algebraic omega of the shaft
        omega_ref = self.data.meridional_obj.omega_ref  # dimensional omega of reference
        x_ref = self.data.meridional_obj.x_ref
        u_ref = self.data.meridional_obj.u_ref
        t_ref = x_ref / u_ref
        tau = x_ref / u_ref  # time delay of the body force model (it could also be through flow time)
        sigma = omega_search / omega_ref  # non-dimensional center point of research

        print("+-------------- ARNOLDI EVP SOLVER --------------+")
        print("Circumferential Harmonic Order: %i" %(m))
        print("Shaft Angular Rate: %.2f [rad/s]" %(Omega))
        print("Reference Angular Rate: %.2f [rad/s]" %(omega_ref))
        print("Time Delay Tau: %.6f [s]" % (tau))
        print("Initial Searching Point: %.2f+%.2fj [rad/s]" % (sigma.real, sigma.imag))
        print("Number of Eigenvalues to find: %i" % (number_search))
        print("+------------------------------------------------+")
        print("\n")

        L0 = self.Z_g * (1 + 1j * m * Omega / omega_ref * tau / t_ref) + self.S_g
        L1 = self.A_g * (m * Omega / omega_ref * tau / t_ref - 1j) - 1j * tau / t_ref * self.Z_g
        L2 = -tau / t_ref * self.A_g

        Y1 = np.concatenate((-L0, np.zeros_like(L0)), axis=1)
        Y2 = np.concatenate((np.zeros_like(L0), np.eye(L0.shape[0])), axis=1)
        Y = np.concatenate((Y1, Y2), axis=0)  # Y matrix of EVP problem

        P1 = np.concatenate((L1, L2), axis=1)
        P2 = np.concatenate((np.eye(L0.shape[0]), np.zeros_like(L0)), axis=1)
        P = np.concatenate((P1, P2), axis=0)  # P matrix of EVP problem

        if inspect_matrices:
            plt.figure()
            plt.spy(L0)
            plt.title(r'$\mathbf{L}_{0}$')

            plt.figure()
            plt.spy(L1)
            plt.title(r'$\mathbf{L}_{1}$')

            plt.figure()
            plt.spy(L2)
            plt.title(r'$\mathbf{L}_{2}$')

            plt.figure()
            plt.spy(Y)
            plt.title(r'$\mathbf{Y}$')

            plt.figure()
            plt.spy(P)
            plt.title(r'$\mathbf{P}$')

        print("Transforming generalized EVP in standard one...")
        Y_tilde = np.linalg.inv(Y - sigma * P)
        Y_tilde = np.dot(Y_tilde, P)

        print("Solving standard EVP...")
        self.eigenfreqs, self.eigenmodes = eigs(Y_tilde, k=number_search)
        self.eigenfreqs = sigma + 1 / self.eigenfreqs  # return of the initial shift
        self.eigenfreqs *= omega_ref  # convert to dimensional frequencies
        self.eigenfreqs_df = self.eigenfreqs.imag / omega_ref
        self.eigenfreqs_rs = self.eigenfreqs.real / omega_ref

    def sort_eigensolution(self):
        """
        sort the eigenmodes from the most unstable to the least one
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

    def plot_eigenfrequencies(self, delimit=False, save_filename=None):
        """
        plot the eigenfrequencies obtained with the Arnoldi Method
        """
        fig, ax = plt.subplots(figsize=fig_size)
        ax.scatter(self.eigenfreqs_rs, self.eigenfreqs_df, marker='o', facecolors='red', edgecolors='red',
                   s=marker_size)
        ax.set_xlabel(r'RS [-]')
        ax.set_ylabel(r'DF [-]')
        # ax.legend()
        if delimit:
            ax.set_xlim([-1.5, 1.5])
            ax.set_ylim([-1, 0.5])
        ax.grid(alpha=grid_opacity)
        if save_filename is not None:
            fig.savefig(folder_name + save_filename + '.pdf', bbox_inches='tight')

    def extract_eigenfields(self, n=None):
        """
        from the eigenvectors obtained with Arnoldi Method, extract the first n eigenfields
        """
        if n is None:
            n = len(self.eigenfreqs)
        elif n > len(self.eigenfreqs):
            print("parameter n must be lower than the eigenvector number. n set to max allowed")
            n = len(self.eigenfreqs)

        Nz = self.data.nAxialNodes
        Nr = self.data.nRadialNodes
        self.eigenfields = []
        for mode in range(n):
            eigenfrequency = self.eigenfreqs[mode]
            eigenvector = self.eigenmodes[:, mode]

            rho_eig = []
            ur_eig = []
            ut_eig = []
            uz_eig = []
            p_eig = []
            for i in range(len(eigenvector) // 2):  # remember that the flow state had been doubled in this Alg.
                if (i) % 5 == 0:
                    rho_eig.append(eigenvector[i])
                elif (i - 1) % 5 == 0 and i != 0:
                    ur_eig.append(eigenvector[i])
                elif (i - 2) % 5 == 0 and i != 0:
                    ut_eig.append(eigenvector[i])
                elif (i - 3) % 5 == 0 and i != 0:
                    uz_eig.append(eigenvector[i])
                elif (i - 4) % 5 == 0 and i != 0:
                    p_eig.append(eigenvector[i])
                else:
                    raise ValueError("Not correct indexing for eigenvector retrieval!")

            rho_eig_r = scaled_eigenvector_real(rho_eig, Nz, Nr)
            ur_eig_r = scaled_eigenvector_real(ur_eig, Nz, Nr)
            ut_eig_r = scaled_eigenvector_real(ut_eig, Nz, Nr)
            uz_eig_r = scaled_eigenvector_real(uz_eig, Nz, Nr)
            p_eig_r = scaled_eigenvector_real(p_eig, Nz, Nr)

            self.eigenfields.append(Eigenmode(eigenfrequency, rho_eig_r, ur_eig_r, ut_eig_r, uz_eig_r, p_eig_r))

    def plot_eigenfields(self, n=None, save_filename=None):
        """
        plot the first n eigenmodes structures
        """
        z = self.data.meridional_obj.z_cg
        r = self.data.meridional_obj.r_cg
        Nz = np.shape(z)[0]
        Nr = np.shape(z)[1]
        modes_map = cm.coolwarm

        if n is None:
            n = len(self.eigenfields)
        elif n > len(self.eigenfields):
            print("parameter n must be lower than the eigenfields number. n set to max allowed!")
            n = len(self.eigenfreqs)
        elif n < 1:
            raise ValueError("Select a positive number of modes to show")
        else:
            pass

        imode = 0
        for mode in self.eigenfields[0:n]:
            imode += 1

            plt.figure(figsize=fig_size)
            plt.contourf(z, r, mode.eigen_rho, levels=N_levels_fine, cmap=modes_map)
            plt.xlabel(r'$z$ [-]')
            plt.ylabel(r'$r$ [-]')
            plt.title(r'$\tilde{\rho}_{%i}$' % (imode))
            plt.colorbar()
            if save_filename is not None:
                plt.savefig(folder_name + save_filename + '_rho_%i_%i_%i.pdf' % (Nz, Nr, imode), bbox_inches='tight')

            plt.figure(figsize=fig_size)
            plt.contourf(z, r, mode.eigen_ur, levels=N_levels_fine, cmap=modes_map)
            plt.xlabel(r'$z$ [-]')
            plt.ylabel(r'$r$ [-]')
            plt.title(r'$\tilde{u}_{r,%i}$' % (imode))
            plt.colorbar()
            if save_filename is not None:
                plt.savefig(folder_name + save_filename + '_ur_%i_%i_%i.pdf' % (Nz, Nr, imode), bbox_inches='tight')

            plt.figure(figsize=fig_size)
            plt.contourf(z, r, mode.eigen_utheta, levels=N_levels_fine, cmap=modes_map)
            plt.xlabel(r'$z$ [-]')
            plt.ylabel(r'$r$ [-]')
            plt.title(r'$\tilde{u}_{\theta,%i}$' % (imode))
            plt.colorbar()
            if save_filename is not None:
                plt.savefig(folder_name + save_filename + '_ut_%i_%i_%i.pdf' % (Nz, Nr, imode), bbox_inches='tight')

            plt.figure(figsize=fig_size)
            plt.contourf(z, r, mode.eigen_uz, levels=N_levels_fine, cmap=modes_map)
            plt.xlabel(r'$z$ [-]')
            plt.ylabel(r'$r$ [-]')
            plt.title(r'$\tilde{u}_{z,%i}$' % (imode))
            plt.colorbar()
            if save_filename is not None:
                plt.savefig(folder_name + save_filename + '_uz_%i_%i_%i.pdf' % (Nz, Nr, imode), bbox_inches='tight')

            plt.figure(figsize=fig_size)
            plt.contourf(z, r, mode.eigen_p, levels=N_levels_fine, cmap=modes_map)
            plt.xlabel(r'$z$ [-]')
            plt.ylabel(r'$r$ [-]')
            plt.title(r'$\tilde{p}_{%i}$' % (imode))
            plt.colorbar()
            if save_filename is not None:
                plt.savefig(folder_name + save_filename + '_p_%i_%i_%i.pdf' % (Nz, Nr, imode), bbox_inches='tight')

    def write_results(self, save_filename=None, extension='csv'):
        """
        print information regarding the eigenfrequencies found, in the form of damping factors and rotations speeds
        Possible file types are (csv, pickle).
        csv: write only DF and RS in a csv file, organized in two columns
        pickle: write the full list of eigenfields, which contain frequencies and eigenfunctions
        """
        if save_filename is not None:
            filename = save_filename
        else:
            filename = 'eigenvalues'

        eigenvalue_array = self.eigenfreqs_rs + 1j * self.eigenfreqs_df

        if extension == 'csv':
            with open(folder_name + filename + '.csv', 'w', newline='') as csvfile:
                fieldnames = ['RS', 'DF']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for num in eigenvalue_array:
                    writer.writerow({'RS': num.real, 'DF': num.imag})
        elif extension == 'pickle':
            with open(folder_name + 'eigenfields.pickle', 'wb') as picklefile:
                pickle.dump(self.eigenfields, picklefile)
        else:
            raise ValueError("Incorrect Extension of the output file.")
