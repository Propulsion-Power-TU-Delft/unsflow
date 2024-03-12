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
from Utils.styles import *
from .eigenmode import Eigenmode
from scipy.interpolate import griddata
from scipy.interpolate import Rbf
from Grid.src.functions import compute_picture_size, create_folder
import copy
import os


class SunModel:
    """
    Class used for Sun Model instability prediction based on the data contained in a Grid object containing the node grid.
    The treated perturbation equations are: (-j*omega*A + B*ddr + j*m*C/r + E*ddz + R + S)*Phi' = 0
    
    MATRICES:
        A : coefficient matrix of temporal derivatives
        B : coefficient matrix of radial derivatives
        C : coefficient matrix of circumferential derivatives
        E : coefficient matrix of axial derivatives
        R : coefficient matrix of the known mean flow terms
        S : coefficient matrix of the body force model
    """

    def __init__(self, gridObject, config):
        """
        Instantiate the sun model Object, contaning all the attributes and methods necessary for the instability analysis.
        :param gridObject: is the object contaning all the data, physical and spectral organized in a grid of Node objects.
        :param config: configuration file of the sun model
        """
        self.data = gridObject  # grid object containing also the meridional object with the data
        self.config = config

        self.nPoints = (gridObject.nAxialNodes) * (gridObject.nRadialNodes)

        self.gmma = self.config.get_fluid_gamma()
        print(f"Gamma set to Default Value: {self.gmma}")

        self.substituted_equation = 'utheta'  # decides which equation overwrite with the euler wall condition
        print(f"Default Equation Substitude by Euler Wall: {self.substituted_equation}")

    def set_overwriting_equation_euler_wall(self, equation):
        """
        Select an equation to overwrite with the euler wall. Avilable options: ur, utheta, uz.
        :param equation: specify which equation will be overwritten from the euler wall condition.
        """
        if equation == 'ur':
            self.substituted_equation = 'ur'
            print("Equation to overwrite with Euler Wall condition set to: Radial Momentum!")
        elif equation == 'utheta':
            self.substituted_equation = 'utheta'
            print("Equation to overwrite with Euler Wall condition set to: Tangential Momentum!")
        elif equation == 'uz':
            self.substituted_equation = 'uz'
            print("Equation to overwrite with Euler Wall condition set to: Axial Momentum!")
        else:
            raise ValueError("Not recognized option")
        print()

    def print_normalization_information(self):
        """
        Print information on non-dimensionalization in the sun module. It should provide only ones if the data
        were already normalized in the meridional process.
        """
        print_banner_begin('NORMALIZATION')
        print(f"{'Reference Length [m]:':<{total_chars_mid}}{self.config.get_reference_length():>{total_chars_mid}.2f}")
        print(
            f"{'Reference Velocity [m/s]:':<{total_chars_mid}}{self.config.get_reference_velocity():>{total_chars_mid}.2f}")
        print(
            f"{'Reference Density [kg/m3]:':<{total_chars_mid}}{self.config.get_reference_density():>{total_chars_mid}.2f}")
        print(
            f"{'Reference Pressure [Pa]:':<{total_chars_mid}}{self.config.get_reference_pressure():>{total_chars_mid}.2f}")
        print(f"{'Reference Time [s]:':<{total_chars_mid}}{self.config.get_reference_time():>{total_chars_mid}.6f}")
        print(
            f"{'Reference Omega [rad/s]:':<{total_chars_mid}}{self.config.get_reference_omega():>{total_chars_mid}.2f}")
        print_banner_end()

    def add_shaft_rpm(self, rpm):
        """
        Add the rpm of the shaft, in order to set the reference angular rate of the analysis (omega [rad/s]).
        If the shaft rotest in the negative z-direction, the reference omega is still positive, but omega shaft is
        kept with algebraic sign
        :param rpm: rpm of the shaft, with sign according to z.
        """
        self.omega_shaft = 2 * np.pi * rpm / 60
        self.omega_ref = np.abs(self.omega_shaft)

    def NormalizeData(self):
        """
        Non-dimensionalise the node quantities, if they were not already non-dimensional
        """
        self.data.Normalize(self.config.get_reference_density(), self.config.get_reference_velocity(),
                            self.config.get_reference_length())

    def ComputeSpectralGrid(self):
        """
        It instanties a new grid object which has the computational grid suitable for spectral differentiation, with
        grid poinst located on Gauss-Lobatto points.
        """
        self.dataSpectral = self.data.PhysicalToSpectralData()

    def ShowPhysicalGrid(self, save_filename=None, mode=None):
        """
        It shows the physical grid points, with different colors for the different parts of the domain.
        :param save_filename: specify name if you want to save the figs.
        :param mode: mode used for visualization.
        """
        self.data.ShowGrid(mode=mode)
        plt.title('physical grid')
        plt.xlabel(r'$\hat{z} \quad  [-]$')
        plt.ylabel(r'$\hat{r} \quad  [-]$')
        if save_filename is not None:
            plt.savefig(save_filename + '.pdf', bbox_inches='tight')  # plt.close()

    def ShowSpectralGrid(self, save_filename=None, mode=None):
        """
        It shows the spectral grid points, with different colors for the different parts of the domain.
        :param save_filename: specify name if you want to save the figs.
        :param mode: mode used for visualization.
        """
        self.dataSpectral.ShowGrid(mode=mode)
        plt.title('spectral grid')
        plt.xlabel(r'$\xi \quad  [-]$')
        plt.ylabel(r'$\eta \quad  [-]$')
        if save_filename is not None:
            plt.savefig(save_filename + '.pdf', bbox_inches='tight')  # plt.close()

    def ComputeJacobianPhysical(self, method='rbf', artificial_refinement=False, dx_dz=None, dx_dr=None, dy_dz=None,
                                dy_dr=None):
        """
        It computes the transformation gradients for every grid point, and stores the value at the node level.
        It computes the derivatives on the spectral grid since it is the only one cartesian, and the inverse transformation is
        found by inversion (usgin the Jacobian).
        :param method: method used to interpolate on the operating grid the gradients calculated on the refined grid.
        :param artificial_refinement: True to set artifical refinement method for grid differentiation
        :param dx_dz: analytical transformation. If provided, gradients simply taken from here, not calculated
        :param dx_dr: analytical transformation. If provided, gradients simply taken from here, not calculated
        :param dy_dz: analytical transformation. If provided, gradients simply taken from here, not calculated
        :param dy_dr: analytical transformation. If provided, gradients simply taken from here, not calculated
        """
        routine = self.config.get_grid_transformation_gradient_routine()
        order = self.config.get_grid_transformation_gradient_order()
        print_banner_begin('TRANSFORMATION GRADIENTS')
        print(f"{'Routine Used:':<{total_chars_mid}}{routine:>{total_chars_mid}}")
        print(f"{'Order Used:':<{total_chars_mid}}{order:>{total_chars_mid}}")
        print_banner_end()

        if dx_dz is None and dx_dr is None and dy_dz is None and dy_dr is None:
            if artificial_refinement:
                Z = self.data.meridional_obj.z_cg_fine
                R = self.data.meridional_obj.r_cg_fine
            else:
                Z = self.data.meridional_obj.z_cg
                R = self.data.meridional_obj.r_cg
            Nz_fine = np.shape(Z)[0]
            Nr_fine = np.shape(Z)[1]
            x = GaussLobattoPoints(Nz_fine)
            y = GaussLobattoPoints(Nr_fine)
            Y, X = np.meshgrid(y, x)

            if routine == 'numpy':
                dzdx, dzdy, drdx, drdy = JacobianTransform2(Z, R, X, Y)
            elif routine == 'hard-coded':
                dzdx, dzdy, drdx, drdy = JacobianTransform(Z, R, X, Y)
            elif routine == 'findiff':
                dzdx, dzdy, drdx, drdy = JacobianTransform3(Z, R, X, Y, order=order)
            else:
                raise ValueError('select an available routine for the transformation gradient!')

            if artificial_refinement:
                self.dzdx = self.interpolation_on_original_grid(dzdx, X, Y, method=method)
                self.dzdy = self.interpolation_on_original_grid(dzdy, X, Y, method=method)
                self.drdx = self.interpolation_on_original_grid(drdx, X, Y, method=method)
                self.drdy = self.interpolation_on_original_grid(drdy, X, Y, method=method)
            else:
                self.dzdx, self.dzdy, self.drdx, self.drdy = dzdx, dzdy, drdx, drdy
            self.J = self.dzdx * self.drdy - self.dzdy * self.drdx
            self.dxdz = (1 / self.J) * (self.drdy)
            self.dxdr = (1 / self.J) * (-self.dzdy)
            self.dydz = (1 / self.J) * (-self.drdx)
            self.dydr = (1 / self.J) * (self.dzdx)
        else:
            if dx_dz is not None:
                self.dxdz = dx_dz
                self.dzdx = 1 / dx_dz
            else:
                self.dxdz = np.zeros((self.data.nAxialNodes, self.data.nRadialNodes))
                self.dzdx = np.zeros((self.data.nAxialNodes, self.data.nRadialNodes))
            if dx_dr is not None:
                self.dxdr = dx_dr
                self.drdx = 1 / dx_dr
            else:
                self.dxdr = np.zeros((self.data.nAxialNodes, self.data.nRadialNodes))
                self.drdx = np.zeros((self.data.nAxialNodes, self.data.nRadialNodes))
            if dy_dz is not None:
                self.dydz = dy_dz
                self.dydz = 1 / dy_dz
            else:
                self.dydz = np.zeros((self.data.nAxialNodes, self.data.nRadialNodes))
                self.dzdy = np.zeros((self.data.nAxialNodes, self.data.nRadialNodes))
            if dy_dr is not None:
                self.dydr = dy_dr
                self.drdy = 1 / dy_dr
            else:
                self.dydr = np.zeros((self.data.nAxialNodes, self.data.nRadialNodes))
                self.drdy = np.zeros((self.data.nAxialNodes, self.data.nRadialNodes))
            self.J = self.dzdx * self.drdy - self.dzdy * self.drdx

        for ii in range(0, self.data.nAxialNodes):
            for jj in range(0, self.data.nRadialNodes):
                # add the inverse gradients information to every node
                self.data.dataSet[ii, jj].AddTransformationGradients(self.dzdx[ii, jj], self.dzdy[ii, jj],
                                                                     self.drdx[ii, jj], self.drdy[ii, jj])
                self.data.dataSet[ii, jj].AddJacobian(self.J[ii, jj])

    def ContourTransformation(self, save_filename=None, folder_name=None):
        """
        Show the gradient contours.
        :param save_filename: specify the names if you want to save the figs.
        """
        # plt.figure(figsize=fig_size)
        # plt.contourf(self.dataSpectral.zGrid, self.dataSpectral.rGrid, self.J, levels=N_levels_fine, cmap=color_map)
        # plt.xlabel(r'$\xi \ \mathrm{[-]}$')
        # plt.ylabel(r'$\eta \ \mathrm{[-]}$')
        # plt.title(r'$J$')
        # plt.colorbar()
        # if save_filename is not None:
        #     plt.savefig(folder_name + save_filename + '_J.pdf', bbox_inches='tight')
        #     # plt.close()
        #
        # plt.figure(figsize=fig_size)
        # plt.contourf(self.dataSpectral.zGrid, self.dataSpectral.rGrid, self.dzdx, levels=N_levels_fine, cmap=color_map)
        # plt.xlabel(r'$\xi \ \mathrm{[-]}$')
        # plt.ylabel(r'$\eta \ \mathrm{[-]}$')
        # plt.title(r'$\frac{\partial \hat{z}}{\partial \xi}$')
        # plt.colorbar()
        # if save_filename is not None:
        #     plt.savefig(folder_name + save_filename + '_1.pdf', bbox_inches='tight')
        #     # plt.close()
        #
        # plt.figure(figsize=fig_size)
        # plt.contourf(self.dataSpectral.zGrid, self.dataSpectral.rGrid, self.dzdy, levels=N_levels_fine, cmap=color_map)
        # plt.xlabel(r'$\xi \ \mathrm{[-]}$')
        # plt.ylabel(r'$\eta \ \mathrm{[-]}$')
        # plt.colorbar()
        # plt.title(r'$\frac{\partial \hat{z}}{\partial \eta}$')
        # if save_filename is not None:
        #     plt.savefig(folder_name + save_filename + '_2.pdf', bbox_inches='tight')
        #     # plt.close()
        #
        # plt.figure(figsize=fig_size)
        # plt.contourf(self.dataSpectral.zGrid, self.dataSpectral.rGrid, self.drdx, levels=N_levels_fine, cmap=color_map)
        # plt.xlabel(r'$\xi \ \mathrm{[-]}$')
        # plt.ylabel(r'$\eta \ \mathrm{[-]}$')
        # plt.colorbar()
        # plt.title(r'$\frac{\partial \hat{r}}{\partial \xi}$')
        # if save_filename is not None:
        #     plt.savefig(folder_name + save_filename + '_3.pdf', bbox_inches='tight')
        #     # plt.close()
        #
        # plt.figure(figsize=fig_size)
        # plt.contourf(self.dataSpectral.zGrid, self.dataSpectral.rGrid, self.drdy, levels=N_levels_fine, cmap=color_map)
        # plt.xlabel(r'$\xi \ \mathrm{[-]}$')
        # plt.ylabel(r'$\eta \ \mathrm{[-]}$')
        # plt.colorbar()
        # plt.title(r'$\frac{\partial \hat{r}}{\partial \eta}$')
        # if save_filename is not None:
        #     plt.savefig(folder_name + save_filename + '_4.pdf', bbox_inches='tight')
        #     # plt.close()

        plt.figure()
        plt.contourf(self.data.zGrid, self.data.rGrid, self.dxdr, levels=N_levels, cmap=color_map)
        plt.xlabel(r'$z \ \mathrm{[-]}$')
        plt.ylabel(r'$r \ \mathrm{[-]}$')
        plt.colorbar()
        plt.title(r'$\frac{\partial \xi}{\partial \hat{r}}$')
        plt.gca().set_aspect('equal', adjustable='box')
        if save_filename is not None:
            plt.savefig(folder_name + '/' + save_filename + '_dxi_dr.pdf', bbox_inches='tight')  # plt.close()

        plt.figure()
        plt.contourf(self.data.zGrid, self.data.rGrid, self.dxdz, levels=N_levels, cmap=color_map)
        plt.xlabel(r'$z \ \mathrm{[-]}$')
        plt.ylabel(r'$r \ \mathrm{[-]}$')
        plt.colorbar()
        plt.title(r'$\frac{\partial \xi}{\partial \hat{z}}$')
        plt.gca().set_aspect('equal', adjustable='box')
        if save_filename is not None:
            plt.savefig(folder_name + '/' + save_filename + '_dxi_dz.pdf', bbox_inches='tight')  # plt.close()

        plt.figure()
        plt.contourf(self.data.zGrid, self.data.rGrid, self.dydr, levels=N_levels, cmap=color_map)
        plt.xlabel(r'$z \ \mathrm{[-]}$')
        plt.ylabel(r'$r \ \mathrm{[-]}$')
        plt.colorbar()
        plt.title(r'$\frac{\partial \eta}{\partial \hat{r}}$')
        plt.gca().set_aspect('equal', adjustable='box')
        if save_filename is not None:
            plt.savefig(folder_name + '/' + save_filename + '_deta_dr.pdf', bbox_inches='tight')  # plt.close()

        plt.figure()
        plt.contourf(self.data.zGrid, self.data.rGrid, self.dydz, levels=N_levels, cmap=color_map)
        plt.xlabel(r'$z \ \mathrm{[-]}$')
        plt.ylabel(r'$r \ \mathrm{[-]}$')
        plt.colorbar()
        plt.title(r'$\frac{\partial \eta}{\partial \hat{z}}$')
        plt.gca().set_aspect('equal', adjustable='box')
        if save_filename is not None:
            plt.savefig(folder_name + '/' + save_filename + '_deta_dz.pdf', bbox_inches='tight')  # plt.close()

    # def AddAMatrixToNodes(self):
    #     """
    #     Compute and store at the node level the A matrix. Sun Formulation
    #     """
    #     for ii in range(0, self.data.nAxialNodes):
    #         for jj in range(0, self.data.nRadialNodes):
    #             A = np.eye(5, dtype=complex)
    #
    #             # if data was already non-dimensional, multiply only matrix A times the strouhal number. If the reference
    #             # velocity was found as u_ref = omega_ref * x_ref and t_ref = 1 / omega_ref, automatically the strouhal
    #             # should be 1 by construction. In this case the non-dimensional equations are exactly the same
    #             # of the dimensional ones
    #             strouhal = self.config.get_reference_length() / (self.config.get_reference_velocity() *
    #                                                              self.config.get_reference_time())
    #             A *= strouhal
    #             self.data.dataSet[ii, jj].AddAMatrix(A)

    def AddAMatrixToNodesFrancesco2(self):
        """
        Compute and store at the node level the A matrix. My Formulation.
        """
        for ii in range(0, self.data.nAxialNodes):
            for jj in range(0, self.data.nRadialNodes):
                A = np.eye(5, dtype=complex)
                A[4, 0] = -self.data.meridional_obj.p[ii, jj] * self.gmma / self.data.meridional_obj.rho[ii, jj]

                """ Multiply only matrix A times the strouhal number. If the reference velocity was defined as:
                 u_ref = omega_ref * x_ref and t_ref = 1 / omega_ref, automatically the strouhal
                should be 1 by construction. In this case the non-dimensional equations are exactly the same
                of the dimensional ones."""
                strouhal = self.config.get_reference_length() / (
                            self.config.get_reference_velocity() * self.config.get_reference_time())
                A *= strouhal
                self.data.dataSet[ii, jj].AddAMatrix(A)

    # def AddBMatrixToNodes(self):
    #     """
    #     Compute and store at the node level the B matrix, needed to compute hat{B} later. Sun Formulation.
    #     """
    #     for ii in range(0, self.data.nAxialNodes):
    #         for jj in range(0, self.data.nRadialNodes):
    #             B = np.zeros((5, 5), dtype=complex)
    #             B[0, 0] = self.data.dataSet[ii, jj].ur
    #             B[1, 1] = self.data.dataSet[ii, jj].ur
    #             B[2, 2] = self.data.dataSet[ii, jj].ur
    #             B[3, 3] = self.data.dataSet[ii, jj].ur
    #             B[4, 4] = self.data.dataSet[ii, jj].ur
    #             B[0, 1] = self.data.dataSet[ii, jj].rho
    #             B[1, 4] = 1 / self.data.dataSet[ii, jj].rho
    #             B[4, 1] = self.data.dataSet[ii, jj].p * self.gmma
    #             self.data.dataSet[ii, jj].AddBMatrix(B)

    def AddBMatrixToNodesFrancesco2(self):
        """
        Compute and store at the node level the B matrix, needed to compute hat{B} later. My Formulation.
        """
        for ii in range(0, self.data.nAxialNodes):
            for jj in range(0, self.data.nRadialNodes):
                B = np.eye(5, dtype=complex) * self.data.meridional_obj.ur[ii, jj]
                B[0, 1] = self.data.meridional_obj.rho[ii, jj]
                B[1, 4] = 1 / self.data.meridional_obj.rho[ii, jj]
                B[4, 0] = - self.data.meridional_obj.p[ii, jj] * self.data.meridional_obj.ur[ii, jj] * self.gmma / \
                          self.data.meridional_obj.rho[ii, jj]
                self.data.dataSet[ii, jj].AddBMatrix(B)

    # def AddCMatrixToNodes(self):
    #     """
    #     Compute and store at node level the C matrix, already multiplied by j*m/r. Ready to be used in the final system of eqs.
    #     Sun Formulation.
    #     """
    #     m = self.config.get_circumferential_harmonic_order()
    #     print(f"Circumferential Harmonic Order set to: {m}")
    #
    #     for ii in range(0, self.data.nAxialNodes):
    #         for jj in range(0, self.data.nRadialNodes):
    #             C = np.zeros((5, 5), dtype=complex)
    #             C[0, 0] = self.data.dataSet[ii, jj].ut
    #             C[1, 1] = self.data.dataSet[ii, jj].ut
    #             C[2, 2] = self.data.dataSet[ii, jj].ut
    #             C[3, 3] = self.data.dataSet[ii, jj].ut
    #             C[4, 4] = self.data.dataSet[ii, jj].ut
    #             C[0, 2] = self.data.dataSet[ii, jj].rho
    #             C[2, 4] = 1 / self.data.dataSet[ii, jj].rho
    #             C[4, 2] = self.data.dataSet[ii, jj].p * self.gmma
    #
    #             C = C * 1j * m / self.data.dataSet[ii, jj].r
    #             self.data.dataSet[ii, jj].AddCMatrix(C)

    def AddCMatrixToNodesFrancesco2(self):
        """
        Compute and store at node level the C matrix, already multiplied by j*m/r. Ready to be used in the final system of eqs.
        My Formulation.
        """
        m = self.config.get_circumferential_harmonic_order()
        print(f"Circumferential Harmonic Order set to: {m}")

        for ii in range(0, self.data.nAxialNodes):
            for jj in range(0, self.data.nRadialNodes):
                C = np.eye(5, dtype=complex) * self.data.meridional_obj.ut[ii, jj]
                C[0, 2] = self.data.meridional_obj.rho[ii, jj]
                C[2, 4] = 1 / self.data.meridional_obj.rho[ii, jj]
                C[4, 0] = -self.data.meridional_obj.p[ii, jj] * self.data.meridional_obj.ut[ii, jj] * self.gmma / \
                          self.data.meridional_obj.rho[ii, jj]
                C *= 1j * m / self.data.meridional_obj.r_cg[ii, jj]
                self.data.dataSet[ii, jj].AddCMatrix(C)

    # def AddEMatrixToNodes(self):
    #     """
    #     Compute and store at the node level the E matrix, needed to compute hat{E}. Sun Formulation.
    #     """
    #     for ii in range(0, self.data.nAxialNodes):
    #         for jj in range(0, self.data.nRadialNodes):
    #             E = np.zeros((5, 5), dtype=complex)
    #
    #             E[0, 0] = self.data.dataSet[ii, jj].uz
    #             E[1, 1] = self.data.dataSet[ii, jj].uz
    #             E[2, 2] = self.data.dataSet[ii, jj].uz
    #             E[3, 3] = self.data.dataSet[ii, jj].uz
    #             E[4, 4] = self.data.dataSet[ii, jj].uz
    #
    #             E[0, 3] = self.data.dataSet[ii, jj].rho
    #             E[3, 4] = 1 / self.data.dataSet[ii, jj].rho
    #             E[4, 3] = self.data.dataSet[ii, jj].p * self.gmma
    #
    #             self.data.dataSet[ii, jj].AddEMatrix(E)

    def AddEMatrixToNodesFrancesco2(self):
        """
        Compute and store at the node level the E matrix, needed to compute hat{E}. My Formulation.
        """
        for ii in range(0, self.data.nAxialNodes):
            for jj in range(0, self.data.nRadialNodes):
                E = np.eye(5, dtype=complex) * self.data.meridional_obj.uz[ii, jj]
                E[0, 3] = self.data.meridional_obj.rho[ii, jj]
                E[3, 4] = 1 / self.data.meridional_obj.rho[ii, jj]
                E[4, 0] = -self.data.meridional_obj.p[ii, jj] * self.data.meridional_obj.uz[ii, jj] * self.gmma / \
                          self.data.meridional_obj.rho[ii, jj]
                self.data.dataSet[ii, jj].AddEMatrix(E)

    # def AddRMatrixToNodes(self):
    #     """
    #     Compute and store at the node level the R matrix, ready to be used in the final system of eqs. Sun Formulation.
    #     """
    #     for ii in range(0, self.data.nAxialNodes):
    #         for jj in range(0, self.data.nRadialNodes):
    #             R = np.zeros((5, 5), dtype=complex)
    #             R[0, 0] = self.data.dataSet[ii, jj].dur_dr + self.data.dataSet[ii, jj].duz_dz + (self.data.dataSet[ii, jj].ur
    #                                                                                              / self.data.dataSet[ii, jj].r)
    #             R[0, 1] = self.data.dataSet[ii, jj].rho / self.data.dataSet[ii, jj].r + self.data.dataSet[ii, jj].drho_dr
    #             R[0, 2] = 0
    #             R[0, 3] = self.data.dataSet[ii, jj].drho_dz
    #             R[0, 4] = 0
    #             R[1, 0] = -self.data.dataSet[ii, jj].dp_dr / self.data.dataSet[ii, jj].rho ** 2
    #             R[1, 1] = self.data.dataSet[ii, jj].dur_dr
    #             R[1, 2] = -2 * self.data.dataSet[ii, jj].ut / self.data.dataSet[ii, jj].r
    #             R[1, 3] = self.data.dataSet[ii, jj].dur_dz
    #             R[1, 4] = 0
    #             R[2, 0] = 0
    #             R[2, 1] = self.data.dataSet[ii, jj].dut_dr + self.data.dataSet[ii, jj].ut / self.data.dataSet[ii, jj].r
    #             R[2, 2] = self.data.dataSet[ii, jj].ur / self.data.dataSet[ii, jj].r
    #             R[2, 3] = self.data.dataSet[ii, jj].dut_dz
    #             R[2, 4] = 0
    #             R[3, 0] = -self.data.dataSet[ii, jj].dp_dz / self.data.dataSet[ii, jj].rho ** 2
    #             R[3, 1] = self.data.dataSet[ii, jj].duz_dr
    #             R[3, 2] = 0
    #             R[3, 3] = self.data.dataSet[ii, jj].duz_dz
    #             R[3, 4] = 0
    #             R[4, 0] = 0
    #             R[4, 1] = self.data.dataSet[ii, jj].p * self.gmma / self.data.dataSet[ii, jj].r + self.data.dataSet[ii, jj].dp_dr
    #             R[4, 2] = 0
    #             R[4, 3] = self.data.dataSet[ii, jj].dp_dz
    #             R[4, 4] = self.gmma * (self.data.dataSet[ii, jj].duz_dz + self.data.dataSet[ii, jj].dur_dr +
    #                                    self.data.dataSet[ii, jj].ur / self.data.dataSet[ii, jj].r)
    #             self.data.dataSet[ii, jj].AddRMatrix(R)

    def AddRMatrixToNodesFrancesco2(self):
        """
        Compute and store at the node level the R matrix.
        My version of the equations.
        """
        for ii in range(0, self.data.nAxialNodes):
            for jj in range(0, self.data.nRadialNodes):
                R = np.zeros((5, 5), dtype=complex)

                R[0, 0] = self.data.meridional_obj.dur_dr[ii, jj] + self.data.meridional_obj.duz_dz[ii, jj] + \
                          self.data.meridional_obj.ur[ii, jj] / self.data.meridional_obj.r_cg[ii, jj]
                R[0, 1] = self.data.meridional_obj.rho[ii, jj] / self.data.meridional_obj.r_cg[ii, jj] + \
                          self.data.meridional_obj.drho_dr[ii, jj]
                R[0, 3] = self.data.meridional_obj.drho_dz[ii, jj]

                R[1, 0] = self.data.meridional_obj.dp_dr[ii, jj] / (self.data.meridional_obj.rho[ii, jj] ** 2)
                R[1, 1] = self.data.meridional_obj.dur_dr[ii, jj]
                R[1, 2] = -2 * self.data.meridional_obj.ut[ii, jj] / self.data.meridional_obj.r_cg[ii, jj]
                R[1, 3] = self.data.meridional_obj.dur_dz[ii, jj]

                R[2, 1] = self.data.meridional_obj.dut_dr[ii, jj] + self.data.meridional_obj.ut[ii, jj] / \
                          self.data.meridional_obj.r_cg[ii, jj]
                R[2, 2] = self.data.meridional_obj.ur[ii, jj] / self.data.meridional_obj.r_cg[ii, jj]
                R[2, 3] = self.data.meridional_obj.dut_dz[ii, jj]

                R[3, 0] = self.data.meridional_obj.dp_dz[ii, jj] / (self.data.meridional_obj.rho[ii, jj] ** 2)
                R[3, 1] = self.data.meridional_obj.duz_dr[ii, jj]
                R[3, 3] = self.data.meridional_obj.duz_dz[ii, jj]

                # R[4, 0] = (1 / node.rho) * (node.ur * node.dp_dr + node.uz * node.dp_dz) # first version
                R[4, 0] = -self.gmma / (self.data.meridional_obj.rho[ii, jj] ** 2) * (
                        self.data.meridional_obj.ur[ii, jj] * self.data.meridional_obj.p[ii, jj] *
                        self.data.meridional_obj.drho_dr[ii, jj] + self.data.meridional_obj.uz[ii, jj] *
                        self.data.meridional_obj.p[ii, jj] * self.data.meridional_obj.drho_dz[ii, jj])  # second version
                R[4, 1] = self.data.meridional_obj.dp_dr[ii, jj] - self.data.meridional_obj.p[ii, jj] * \
                          self.data.meridional_obj.drho_dr[ii, jj] * self.gmma / self.data.meridional_obj.rho[ii, jj]
                R[4, 3] = self.data.meridional_obj.dp_dz[ii, jj] - self.gmma / self.data.meridional_obj.rho[ii, jj] * \
                          self.data.meridional_obj.p[ii, jj] * self.data.meridional_obj.drho_dz[ii, jj]
                R[4, 4] = (-self.gmma / self.data.meridional_obj.rho[ii, jj]) * (
                            self.data.meridional_obj.ur[ii, jj] * self.data.meridional_obj.drho_dr[ii, jj] +
                            self.data.meridional_obj.uz[ii, jj] * self.data.meridional_obj.drho_dz[ii, jj])

                self.data.dataSet[ii, jj].AddRMatrix(R)

    def AddSMatrixToNodes(self):
        """
        compute and store at the node level the S matrix, ready to be used in the final system of eqs. The matrix formulation
        depends on the selected body-force model.
        """
        for ii in range(0, self.data.nAxialNodes):
            for jj in range(0, self.data.nRadialNodes):
                S = np.zeros((5, 5), dtype=complex)
                if self.data.meridional_obj.domain == 'rotor' or self.data.meridional_obj.domain == 'stator':
                    S[1, 1] = self.data.meridional_obj.S11[ii, jj]
                    S[1, 2] = self.data.meridional_obj.S12[ii, jj]
                    S[1, 3] = self.data.meridional_obj.S13[ii, jj]

                    S[2, 1] = self.data.meridional_obj.S21[ii, jj]
                    S[2, 2] = self.data.meridional_obj.S22[ii, jj]
                    S[2, 3] = self.data.meridional_obj.S23[ii, jj]

                    S[3, 1] = self.data.meridional_obj.S31[ii, jj]
                    S[3, 2] = self.data.meridional_obj.S32[ii, jj]
                    S[3, 3] = self.data.meridional_obj.S33[ii, jj]

                    if self.data.meridional_obj.domain == 'rotor':
                        S[4, 1] = self.data.meridional_obj.S41[ii, jj]
                        S[4, 2] = self.data.meridional_obj.S42[ii, jj]
                        S[4, 3] = self.data.meridional_obj.S43[ii, jj]
                else:
                    pass

                self.data.dataSet[ii, jj].AddSMatrix(S)

    def AddHatMatricesToNodes(self):
        """
        Compute and store at the node level the hat{B}, hat{E} matrix, needed for following multiplication with the spectral
        differential operators.
        """
        for ii in range(0, self.data.nAxialNodes):
            for jj in range(0, self.data.nRadialNodes):
                Bhat = (1 / self.J[ii, jj]) * (
                            -self.data.dataSet[ii, jj].B * self.dzdy[ii, jj] + self.data.dataSet[ii, jj].E * self.drdy[
                        ii, jj])
                Ehat = (1 / self.J[ii, jj]) * (
                            self.data.dataSet[ii, jj].B * self.dzdx[ii, jj] - self.data.dataSet[ii, jj].E * self.drdx[
                        ii, jj])
                # # # alternative formulation, provides the same results. (Check)
                # Bhat2 = self.data.dataSet[ii, jj].B * self.dxdr[ii, jj] + \
                #        self.data.dataSet[ii, jj].E * self.dxdz[ii, jj]
                # Ehat2 = self.data.dataSet[ii, jj].B * self.dydr[ii, jj] + \
                #        self.data.dataSet[ii, jj].E * self.dydz[ii, jj]
                self.data.dataSet[ii, jj].AddHatMatrices(Bhat, Ehat)

    def ApplySpectralDifferentiation(self, verbose=False):
        """
        This method applies Chebyshev-Gauss-Lobatto differentiation method to hat{B},hat{E}, to express the perturbation
        derivatives as a function of the perturbation at the other nodes. It saves a new global (for all the nodes) matrix Q_const,
        which is part of the global stability matrix. The full dimension is: (nPoints*5,nPoints*5).
        The spectral differentiation formula has been double-checked in the respective debug file.
        :param verbose: print additional info.
        """

        # compute the spectral Matrices for x and y direction with the Bayliss formulation
        Dx = ChebyshevDerivativeMatrixBayliss(self.dataSpectral.z)  # derivative operator in xi
        Dy = ChebyshevDerivativeMatrixBayliss(self.dataSpectral.r)  # derivative operator in eta

        # Q_const is the global matrix storing B and E elements after spectral differentiation.
        self.Q_const = np.zeros((self.nPoints * 5, self.nPoints * 5), dtype=complex)

        # differentiation of a general perturbation vector (for the node (i,j)) along xi  and eta. (formula double-checked)
        for ii in range(0, self.dataSpectral.nAxialNodes):
            for jj in range(0, self.dataSpectral.nRadialNodes):
                B_ij = self.data.dataSet[ii, jj].Bhat.copy()  # Bhat matrix of the ij node
                E_ij = self.data.dataSet[ii, jj].Ehat.copy()  # Ehat matrix of the ij node
                node_counter = self.data.dataSet[ii, jj].nodeCounter

                # xi differentiation. m is in the range of axial nodes, first axis of the matrix
                for m in range(0, self.dataSpectral.nAxialNodes):
                    tmp = Dx[ii, m] * B_ij  # 5x5 matrix to be added to a certain block of Q
                    row = node_counter * 5  # this selects the correct block along i of Q
                    column = self.data.dataSet[m, jj].nodeCounter * 5  # it selects the correct block along j of Q

                    if verbose:
                        print('Node [i,j] = (%i,%i)' % (ii, jj))
                        print('Element along i [m,j] = (%i,%i)' % (m, jj))
                        print('Derivative element [ii,m] = (%i,%i)' % (ii, m))
                        print('[row,col] = (%i,%i)' % (row, column))
                    self.AddToQ_const(tmp, row, column)

                # eta differentiation. n is in the range of radial nodes, second axis of the matrix
                for n in range(0, self.dataSpectral.nRadialNodes):
                    tmp = Dy[jj, n] * E_ij
                    row = node_counter * 5
                    column = (self.data.dataSet[ii, n].nodeCounter) * 5  # it selects the correct block along j of Q
                    if verbose:
                        print('Node [i,j] = (%.1d,%.1d)' % (ii, jj))
                        print('Element along j [i,n] = (%.1d,%.1d)' % (jj, n))
                        print('Derivative element [jj,n] = (%.1d,%.1d)' % (jj, n))
                        print('[row,col] = (%.1d,%.1d)' % (row, column))
                    self.AddToQ_const(tmp, row, column)

    def ApplySpectralDifferentiationKronecker(self, verbose=False):
        """
        This method applies Chebyshev-Gauss-Lobatto differentiation method to hat{B},hat{E}, to express the perturbation
        derivatives as a function of the perturbation at the other nodes. It saves a new global (for all the nodes) matrix Q_const,
        which is part of the global stability matrix. The full dimension is: (nPoints*5,nPoints*5).
        The spectral differentiation formula has been taken from Spectral Methods in Matlab, Trefethen.
        :param verbose: print additional info.
        """

        # Differential operators in 1D
        Dx = ChebyshevDerivativeMatrixBayliss(self.dataSpectral.z)  # derivative operator in xi
        Dy = ChebyshevDerivativeMatrixBayliss(self.dataSpectral.r)  # derivative operator in eta

        # Corresponding operators extended to 2D thanks to Kronecker products.
        Ix = np.eye(self.data.nAxialNodes)
        Iy = np.eye(self.data.nRadialNodes)
        Dx_2d = np.kron(Dx, Iy)
        Dy_2d = np.kron(Ix, Dy)

        # Q_const is the global matrix storing B and E elements after spectral differentiation.
        self.Q_const = np.zeros((self.nPoints * 5, self.nPoints * 5), dtype=complex)

        for ii in range(Dx_2d.shape[0]):
            for jj in range(Dx_2d.shape[1]):
                # indexes on the Q_const matrix. (Times 5 because there are 5 equations per node)
                row = ii*5
                col = jj*5

                # indexes of the nodes in 2D. jj is here the absolute node counter, going from 0 to nx*ny
                inode = jj // self.data.nRadialNodes
                jnode = jj % self.data.nRadialNodes

                B = self.data.dataSet[inode, jnode].Bhat.copy()
                E = self.data.dataSet[inode, jnode].Ehat.copy()
                tmp = B*Dx_2d[ii, jj] + E*Dy_2d[ii, jj]

                self.AddToQ_const(tmp, row, col)


    def apply_finite_differences_on_physical_grid(self):
        """
       This method differentiates directly the problem on the physical grid. It saves a new global (for all the nodes) matrix
       Q_const, which is part of the global stability matrix. The full dimension is: (nPoints*5,nPoints*5).
       The method should only be used for those cases having a cartesian physical grid, otherwise is wrong.
       """

        # the cordinates on the spectral grid directions, necessary for the differentiation matrices Dx and Dy
        z = self.data.zGrid[:, 0]
        dz = z[1] - z[0]
        r = self.data.rGrid[0, :]
        dr = r[1] - r[0]

        # Q_const is the global matrix storing B and E elements after finite differentiation.
        self.Q_const = np.zeros((self.nPoints * 5, self.nPoints * 5), dtype=complex)

        # differentiation of a general perturbation vector (for the node (i,j)) along xi  and eta. (formula double-checked)
        for ii in range(0, self.dataSpectral.nAxialNodes):
            for jj in range(0, self.dataSpectral.nRadialNodes):
                B_ij = self.data.dataSet[ii, jj].B  # B matrix of the ij node
                E_ij = self.data.dataSet[ii, jj].E  # E matrix of the ij node

                if ii == 0 and jj == 0:  # bottom left corner
                    ip = 1
                    im = 0
                    deltaz = 1
                    jp = 1
                    jm = 0
                    deltar = 1
                elif ii == 0 and jj == self.data.nRadialNodes - 1:  # top left corner
                    ip = 1
                    im = 0
                    deltaz = 1
                    jp = 0
                    jm = -1
                    deltar = 1
                elif ii == self.data.nAxialNodes - 1 and jj == 0:  # bottom right corner
                    ip = 0
                    im = -1
                    deltaz = 1
                    jp = 1
                    jm = 0
                    deltar = 1
                elif ii == self.data.nAxialNodes - 1 and jj == self.data.nRadialNodes - 1:  # top right corner
                    ip = 0
                    im = -1
                    deltaz = 1
                    jp = 0
                    jm = -1
                    deltar = 1
                elif ii == 0:  # left internal border
                    ip = 1
                    im = 0
                    deltaz = 1
                    jp = 1
                    jm = -1
                    deltar = 2
                elif ii == self.data.nAxialNodes - 1:  # right internal border
                    ip = 0
                    im = -1
                    deltaz = 1
                    jp = 1
                    jm = -1
                    deltar = 2
                elif jj == 0:  # bottom internal border
                    ip = 1
                    im = -1
                    deltaz = 2
                    jp = 1
                    jm = 0
                    deltar = 1
                elif jj == self.data.nRadialNodes - 1:  # top internal border
                    ip = 1
                    im = -1
                    deltaz = 2
                    jp = 0
                    jm = -1
                    deltar = 1
                else:
                    ip = 1
                    im = -1
                    deltaz = 2
                    jp = 1
                    jm = -1
                    deltar = 2

                # B differentiation
                row = self.data.dataSet[ii, jj].nodeCounter * 5
                colp = self.data.dataSet[ii, jj + jp].nodeCounter * 5
                colm = self.data.dataSet[ii, jj + jm].nodeCounter * 5
                tmp = B_ij / (deltar * dr)
                self.AddToQ_const(tmp, row, colp)
                self.AddToQ_const(-tmp, row, colm)

                # E differentiation
                row = self.data.dataSet[ii, jj].nodeCounter * 5
                colp = self.data.dataSet[ii + ip, jj].nodeCounter * 5
                colm = self.data.dataSet[ii + im, jj].nodeCounter * 5
                tmp = E_ij / (deltaz * dz)
                self.AddToQ_const(tmp, row, colp)
                self.AddToQ_const(-tmp, row, colm)

    def AddToQ_const(self, block, row, column):
        """
        Add 5x5 block to the Qconst matrix, specifying the first element location.
        :param block: 5x5 block to add
        :param row: row index of the first element for positioning.
        :param column: column index of the first element for positioning.
        """
        if (block.dtype != np.complex128 or block.shape != (5, 5)):
            raise TypeError('The block must be a 5x5 complex')
        self.Q_const[row:row + 5, column:column + 5] += block

    # def AddToQ_var(self, block, row, column):
    #     """
    #     Add 5x5 block to the Q_var matrix, specifying the first element location.
    #     :param block: 5x5 block to add
    #     :param row: row index of the first element for positioning.
    #     :param column: column index of the first element for positioning.
    #     """
    #     if (block.dtype!=np.complex128 or block.shape!=(5, 5)):
    #         raise TypeError('The block must be a 5x5 complex')
    #     self.Q_var[row:row + 5, column:column + 5] += block

    # def add_to_Y(self, block, row, column):
    #     """
    #     Add 5x5 block to the Y matrix, specifying the first element location.
    #     :param block: 5x5 block to add
    #     :param row: row index of the first element for positioning.
    #     :param column: column index of the first element for positioning.
    #     """
    #     if (block.dtype!=np.complex128 or block.shape!=(5, 5)):
    #         raise TypeError('The block must be a 5x5 complex')
    #     self.Y[row:row + 5, column:column + 5] += block

    def add_to_A_g(self, block, row, column):
        """
        Add 5x5 block to the A_g matrix, specifying the first element location.
        :param block: 5x5 block to add
        :param row: row index of the first element for positioning.
        :param column: column index of the first element for positioning.
        """
        if (block.dtype != np.complex128 or block.shape != (5, 5)):
            raise TypeError('The block must be a 5x5 complex')
        self.A_g[row:row + 5, column:column + 5] += block

    def add_to_C_g(self, block, row, column):
        """
        Add 5x5 block to the C_g matrix, specifying the first element location.
        :param block: 5x5 block to add
        :param row: row index of the first element for positioning.
        :param column: column index of the first element for positioning.
        """
        if (block.dtype != np.complex128 or block.shape != (5, 5)):
            raise TypeError('The block must be a 5x5 complex')
        self.C_g[row:row + 5, column:column + 5] += block

    def add_to_R_g(self, block, row, column):
        """
        Add 5x5 block to the R_g matrix, specifying the first element location.
        :param block: 5x5 block to add
        :param row: row index of the first element for positioning.
        :param column: column index of the first element for positioning.
        """
        if (block.dtype != np.complex128 or block.shape != (5, 5)):
            raise TypeError('The block must be a 5x5 complex')
        self.R_g[row:row + 5, column:column + 5] += block

    def add_to_S_g(self, block, row, column):
        """
        Add 5x5 block to the S_g matrix, specifying the first element location.
        :param block: 5x5 block to add
        :param row: row index of the first element for positioning.
        :param column: column index of the first element for positioning.
        """
        if (block.dtype != np.complex128 or block.shape != (5, 5)):
            raise TypeError('The block must be a 5x5 complex')
        self.S_g[row:row + 5, column:column + 5] += block

    def ApplyBoundaryConditions(self):
        """
        It applies the correct set of boundary conditions to all the points marked with a boundary marker.
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

    # def AddRemainingMatrices(self, omega):
    #     """
    #     it adds the remaning diagonal block matrices to the full Qtot = Q_const + Q_var. Q_const is constant for every model,
    #     while Q_var depends on omega, which should be already provided as non-dimensional.
    #     :param omega: dicrete omega value for the SVD calculation.
    #     """
    #     # Q_var is composed by A, C, R, S. Q_const is composed by B,E spectrally differentiated. Q_tot is the global sum
    #     print("WARNING: deprecated method")
    #     self.Q_var = np.zeros((self.Q_const.shape),
    #                           dtype=complex)  # variable part of the stability matrix. Instantiated for every new value of omega
    #     for ii in range(0, self.dataSpectral.nAxialNodes):
    #         for jj in range(0, self.dataSpectral.nRadialNodes):
    #             # add all the remaining terms on the diagonal
    #             diag_block_ij = -1j * omega * self.data.dataSet[ii, jj].A + self.data.dataSet[ii, jj].C + self.data.dataSet[
    #                 ii, jj].R + self.data.dataSet[ii, jj].S
    #             node_counter = self.data.dataSet[ii, jj].nodeCounter
    #             row = node_counter * 5
    #             column = node_counter * 5  # diagonal block
    #             self.AddToQ_var(diag_block_ij, row, column)
    #     self.Q_tot = self.Q_const + self.Q_var  # compute the global stability matrix

    def ApplyBCCondition(self, row, condition, wall_normal=np.array([1, 0, 0])):
        """
        for the considered grid node, it modifes the 5 governing equations starting from row index,
        which is related to its continuity eq.
        :param row: row index of equation to overwrite with BC
        :param condition: boundary condition.
        :param wall_normal: normal to the wall (r,theta,z) ref frame.
        """
        print("WARNING: deprecated method.")
        if condition == 'zero perturbation':
            # BC for zero perturbation vector
            self.Q_tot[row:row + 5, :] = np.zeros(self.Q_tot[row:row + 5, :].shape, dtype=complex)  # make it zero first
            self.Q_tot[row:row + 5, row:row + 5] = np.eye(5,
                                                          dtype=complex)  # zero perturbation condition for every flow variable

        elif condition == 'zero pressure':
            # BC for zero pressure perturbation
            self.Q_tot[row + 4, :] = np.zeros(self.Q_tot[row + 1, :].shape, dtype=complex)
            self.Q_tot[row + 4, row + 4] = 1  # zero pressure coefficient at that node

        elif condition == 'euler wall':
            # BC for non-penetration condition at the walls. The variable substituted_equation decides which equation has
            # be overwritten from the boundary condition. In principle anyone of the momentum equations should be ok.

            if self.substituted_equation == 'rho':
                # density equation modified
                print(
                    'Attention, Euler wall condition overwriting continuity equation. It should be done only for testing!')
                self.Q_tot[row, :] = np.zeros(self.Q_tot[row + 3, :].shape,
                                              dtype=complex)  # first make zero the axial velocity equation
                self.Q_tot[row, row + 1:row + 4] = wall_normal  # impose non-penetration condition (u*nr + v*nt + w*nz)

            elif self.substituted_equation == 'ur':
                # radial velocity equation modified
                self.Q_tot[row + 1, :] = np.zeros(self.Q_tot[row + 1, :].shape,
                                                  dtype=complex)  # first make zero the radial velocity equation
                self.Q_tot[row + 1,
                row + 1:row + 4] = wall_normal  # impose non-penetration condition (u*nr + v*nt + w*nz)

            elif self.substituted_equation == 'ut':
                # tangential velocity equation overwritten
                self.Q_tot[row + 2, :] = np.zeros(self.Q_tot[row + 3, :].shape,
                                                  dtype=complex)  # first make zero the tangential velocity equation
                self.Q_tot[row + 2,
                row + 1:row + 4] = wall_normal  # impose non-penetration condition (u*nr + v*nt + w*nz)

            elif self.substituted_equation == 'uz':
                # axial velocity equation modified
                self.Q_tot[row + 3, :] = np.zeros(self.Q_tot[row + 3, :].shape,
                                                  dtype=complex)  # first make zero the axial velocity equation
                self.Q_tot[row + 3,
                row + 1:row + 4] = wall_normal  # impose non-penetration condition (u*nr + v*nt + w*nz)

            elif self.substituted_equation == 'p':
                # pressure equation modified
                print(
                    'Attention, Euler wall condition overwriting pressure equation. It should be done only for testing!')
                self.Q_tot[row + 4, :] = np.zeros(self.Q_tot[row + 4, :].shape,
                                                  dtype=complex)  # first make zero the axial velocity equation
                self.Q_tot[row + 4,
                row + 1:row + 4] = wall_normal  # impose non-penetration condition (u*nr + v*nt + w*nz)
            else:
                raise ValueError('Equation to overwritten not specified!')

        else:
            raise ValueError('Boundary condition unknown')

    def ComputeSVD(self, omega_domain=None, grid_omega=None):
        """
        Compute the SVD for every omega in omega_domain, discretized as in grid_omega. It computes
        every time the part of Q that depends on omega (-j*omega*A), and computes the boundary conditions.
        Then it computes the singular values, and store the inverse of the condition number in the chi attribute. Singular values 
        max and min are also stored for code test.
        :param omega_domain: omega domain of research.
        :param grid_omega: discretization of the domain of research.
        """
        print("WARNING: deprecated method.")
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
                    print('SVD %.1d of %1.d \t (%.1d min remaining)' % (ii * len(omI) + 1 + jj, len(omR) * len(omI),
                                                                        remaining_minutes + 1))  # keep track of the progress

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
        print('Total SVD time: \t %1.d hrs %1.d mins %1.d sec' % (
        hrs, mins, sec))  # self.eigs = np.array(self.FindLocalMinima(self.chi))

    def ComputeSVD2(self, omega_domain=None, grid_omega=None):
        """
        Compute the SVD for every omega in omega_domain, discretized as in grid_omega. It computes
        every time the part of Q that depends on omega (-j*omega*A), and computes the boundary conditions.
        Then it computes the singular values, and store the inverse of the condition number in the chi attribute. Singular values
        max and min are also stored for code test.
        :param omega_domain: omega domain of research.
        :param grid_omega: discretization of the domain of research.
        """
        print("WARNING: deprecated method.")
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
                    print('SVD %.1d of %1.d \t (%.1d min remaining)' % (ii * len(omI) + 1 + jj, len(omR) * len(omI),
                                                                        remaining_minutes + 1))  # keep track of the progress

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

    def ComputeSVDcompressor(self, RS_domain=np.array([-1, 1]), DF_domain=np.array([-1, 1]), grid=np.array([10, 10]),
                             verbose=True):
        """
        Compute the SVD for every omega in omega_domain, discretized as in grid_omega. It computes
        every time the part of Q that depends on omega (-j*omega*A), and computes the boundary conditions.
        Then it computes the singular values, and store the inverse of the condition number in the chi attribute. Singular values 
        max and min are also stored for code test.
        :param RS_domain: domain of research for RS
        :param DF_domain: domain of research for DF
        :param grid: discretization of domain of research
        :param verbose: print some info
        """
        # find the limits of the non-dimensional omega of reserach = omega/omega_ref, where omega real and imaginary are
        # defined starting from the Rotational Speed RS and Damping Factor DF of the instability lobe (m=1)
        # omega_{dimensional}=omega_shaft*RS+1j*U*DF/r where U and r taken where we want.
        # omega_{non-dimensinoal}=omega_{dimensional}/omega_ref
        print("WARNING: deprecated method.")
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
                        print('SVD %.1d of %1.d \t (%.1d min remaining)' % (ii * len(omI) + 1 + jj, len(omR) * len(omI),
                                                                            remaining_minutes + 1))  # keep track of the progress

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

    # def PlotInverseConditionNumberCompressor(self, sing_val=False, scale=None, save_filename=None, formatFig=(10, 6)):
    #     """
    #     Plots the chi map for the problem modeled. Log scale can be applied to see better the valleys.
    #     :param sing_val: if true plots also the singular values contours
    #     :param scale: if 'log' plots the contours with log scale, to improve visibility
    #     :param save_filename: specify names of the figs to save
    #     :param formatFig: figure size of the contours
    #     """
    #     x = np.linspace(self.RS_domain[0], self.RS_domain[1])
    #     y = np.linspace(self.DF_domain[0], self.DF_domain[1])
    #     critical_line = np.zeros(len(x))  # superpose a critical line
    #
    #     fig, ax = plt.subplots(figsize=formatFig)
    #     if scale == 'log':
    #         cs = ax.contourf(self.RS, self.DF, self.chi, N_levels, locator=ticker.LogLocator(), cmap=color_map)
    #     else:
    #         cs = ax.contourf(self.RS, self.DF, self.chi, N_levels, cmap=color_map)
    #     ax.plot(x, critical_line, '--w')
    #     ax.set_xlabel(r'$RS \ \mathrm{[-]}$')
    #     ax.set_ylabel(r'$DF \ \mathrm{[-]}$')
    #     ax.set_title(r'$\chi$')
    #     fig.colorbar(cs)
    #     if save_filename is not None:
    #         fig.savefig(folder_name + save_filename + '.pdf', bbox_inches='tight')
    #         plt.close()
    #
    #     if sing_val:
    #         self.PlotSingularValues()

    def FindLocalMinima(self, field):
        """
        Locate the local minima of a 2D array.
        :param field: 2D array
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

    # def PlotInverseConditionNumber(self, scale=None, save_filename=None, formatFig=(10, 6), ref_solution=None):
    #     """
    #     Plots the chi map for the problem modeled.
    #     :param scale: if 'log' plots the contours with log scale, to improve visibility
    #     :param save_filename: specify names of the figs to save
    #     :param formatFig: figure size of the contours
    #     :param ref_solution: specify analytical solution if available
    #     """
    #     x = np.linspace(np.min(self.omegaR), np.max(self.omegaR)) * self.omega_ref
    #     critical_line = np.zeros(len(x))
    #
    #     fig, ax = plt.subplots(figsize=formatFig)
    #     if scale == 'log':
    #         cs = ax.contourf(self.omegaR * self.omega_ref, self.omegaI * self.omega_ref, self.chi, N_levels,
    #                          locator=ticker.LogLocator(), cmap=color_map)
    #     else:
    #         cs = ax.contourf(self.omegaR * self.omega_ref, self.omegaI * self.omega_ref, np.log(self.chi), N_levels,
    #                          cmap=color_map)
    #         # ax.plot(self.eigs.real * self.omega_ref, self.eigs.imag * self.omega_ref, 'ko')
    #     ax.plot(x, critical_line, '--r')
    #     ax.set_xlabel(r'$\omega_{R} \quad [rad/s]$')
    #     ax.set_ylabel(r'$\omega_{I} \quad [rad/s]$')
    #     ax.set_title(r'$\log \chi$')
    #     ax.set_xlim([np.min(self.omegaR * self.omega_ref), np.max(self.omegaR * self.omega_ref)])
    #     fig.colorbar(cs)
    #     if ref_solution is not None:
    #         ax.plot(ref_solution.real, ref_solution.imag, 'ws')
    #     if save_filename is not None:
    #         fig.savefig(folder_name + save_filename + '.pdf', bbox_inches='tight')
    #         plt.close()

    # def PlotSingularValues(self, scale=None, save_filename=None, formatFig=(15, 6)):
    #     """
    #     Plots the singular values for the problem modeled. Log scale can be applied to see better the valleys.
    #     :param scale: if 'log' plots the contours with log scale, to improve visibility
    #     :param save_filename: specify names of the figs to save
    #     :param formatFig: figure size of the contours
    #     """
    #     x = np.linspace(np.min(self.omegaR), np.max(self.omegaR)) * self.omega_ref
    #     critical_line = np.zeros(len(x))
    #     fig, ax = plt.subplots(1, 2, figsize=formatFig)
    #     if scale == 'log':
    #         cs0 = ax[0].contourf(self.omegaR * self.omega_ref, self.omegaI * self.omega_ref, self.sing_value_max, N_levels,
    #                              locator=ticker.LogLocator(), cmap=color_map)
    #         cs1 = ax[1].contourf(self.omegaR * self.omega_ref, self.omegaI * self.omega_ref, self.sing_value_min, N_levels,
    #                              locator=ticker.LogLocator(), cmap=color_map)
    #     else:
    #         cs0 = ax[0].contourf(self.omegaR * self.omega_ref, self.omegaI * self.omega_ref, self.sing_value_max, N_levels,
    #                              cmap=color_map)
    #         cs1 = ax[1].contourf(self.omegaR * self.omega_ref, self.omegaI * self.omega_ref, self.sing_value_min, N_levels,
    #                              cmap=color_map)
    #
    #     ax[0].plot(x, critical_line, '--r')
    #     ax[0].set_xlabel(r'$\omega_{R} \quad [rad/s]$')
    #     ax[0].set_ylabel(r'$\omega_{I} \quad [rad/s]$')
    #     ax[0].set_title(r'$\sigma_{max}$')
    #     fig.colorbar(cs0)
    #     ax[1].plot(x, critical_line, '--r')
    #     ax[1].set_xlabel(r'$\omega_{R} \quad [rad/s]$')
    #     ax[1].set_title(r'$\sigma_{min}$')
    #     fig.colorbar(cs1)
    #
    #     if save_filename is not None:
    #         fig.savefig(folder_name + save_filename + '.pdf', bbox_inches='tight')
    #         plt.close()

    def ComputeBoundaryNormals(self):
        """
        Compute the normal vectors on the edges of the domain.
        """
        self.data.ComputeBoundaryNormals()

    def ShowNormals(self):
        """
       Plots the boundary nodes and the normals
       """
        self.data.ShowNormals()  # method belonging to compressor_grid object

    def build_A_global_matrix(self):
        """
        Build the A global matrix, stacking together the A matrices of all the nodes.
        """
        self.A_g = np.zeros(self.Q_const.shape, dtype=complex)
        for ii in range(0, self.dataSpectral.nAxialNodes):
            for jj in range(0, self.dataSpectral.nRadialNodes):
                node_counter = self.data.dataSet[ii, jj].nodeCounter
                row = node_counter * 5
                column = node_counter * 5  # diagonal block
                self.add_to_A_g(self.data.dataSet[ii, jj].A, row, column)

    def build_C_global_matrix(self):
        """
        Build the C*j*m/r global matrix, stacking together the C*j*m/r matrices of all the nodes.
        """
        self.C_g = np.zeros(self.Q_const.shape, dtype=complex)
        for ii in range(0, self.dataSpectral.nAxialNodes):
            for jj in range(0, self.dataSpectral.nRadialNodes):
                node_counter = self.data.dataSet[ii, jj].nodeCounter
                row = node_counter * 5
                column = node_counter * 5  # diagonal block
                self.add_to_C_g(self.data.dataSet[ii, jj].C, row, column)

    def build_R_global_matrix(self):
        """
        Build the R global matrix.
        """
        self.R_g = np.zeros(self.Q_const.shape, dtype=complex)
        for ii in range(0, self.dataSpectral.nAxialNodes):
            for jj in range(0, self.dataSpectral.nRadialNodes):
                node_counter = self.data.dataSet[ii, jj].nodeCounter
                row = node_counter * 5
                column = node_counter * 5  # diagonal block
                self.add_to_R_g(self.data.dataSet[ii, jj].R, row, column)

    def build_S_global_matrix(self):
        """
        Build the S global matrix.
        """
        self.S_g = np.zeros(self.Q_const.shape, dtype=complex)
        for ii in range(0, self.dataSpectral.nAxialNodes):
            for jj in range(0, self.dataSpectral.nRadialNodes):
                node_counter = self.data.dataSet[ii, jj].nodeCounter
                row = node_counter * 5
                column = node_counter * 5  # diagonal block
                self.add_to_S_g(self.data.dataSet[ii, jj].S, row, column)

    def build_Z_global_matrix(self):
        """
        Build the Z global matrix, synonym of J. J = Z = (B_d + C + E_d + R), where B_d+E_d=Q_const have been obtained with
        the spectral differentiation method.
        """
        self.Z_g = self.Q_const + self.C_g + self.R_g

    def compute_L_matrices(self, block_i):
        """
        Compute the L0 matrix, defined as L0 = Z_g(1+j*m*Omega*tau)+S_g
        :param block_i: number of the current block
        """
        block_type = self.config.get_blocks_type()[block_i]
        m = self.config.get_circumferential_harmonic_order()

        if block_type == 'unbladed':
            Omega = 0
            tau = 0
        elif block_type == 'stator':
            Omega = 0
            tau = np.mean(self.data.meridional_obj.stream_line_length[-1, :]) / np.mean(
                self.data.meridional_obj.u_meridional[0, :])
        elif block_type == 'rotor':
            Omega = self.config.get_omega_shaft() / self.config.get_reference_omega()
            tau = np.mean(self.data.meridional_obj.stream_line_length[-1, :]) / np.mean(
                self.data.meridional_obj.u_meridional[0, :])
        else:
            raise ValueError('Unknown block type!')

        self.L0 = self.Z_g * (1 + 1j * m * Omega * tau) + self.S_g
        self.L1 = self.A_g * (m * Omega * tau - 1j) - 1j * tau * self.Z_g
        self.L2 = -tau * self.A_g

    def inspect_L_matrices(self, save_filename=None, save_foldername=None):
        """
        Plot the L matrices, to inspect their composition
        """
        if save_filename is not None:
            create_folder(save_foldername)

        fig, ax = plt.subplots(1, 3, figsize=(16, 6))
        ax[0].spy(self.L0)
        ax[1].spy(self.L1)
        ax[2].spy(self.L2)
        ax[0].set_title(r'$L_0$')
        ax[1].set_title(r'$L_1$')
        ax[2].set_title(r'$L_2$')
        plt.savefig(os.path.join(save_foldername, save_filename + '.pdf'), bbox_inches='tight')

    def apply_boundary_conditions_generalized(self, mode='over writing'):
        """
        Apply the boundary conditions for the considered system, whose equations are:
        (-j*omega*A + Z + S/zita)*tilde{phi}. Therefore, BCs are imposed on Z (the only constant matrix), and A and S
        (omega dependent) filled with zeros in correspondance of those BCs.
        """
        self.rows_added = 0
        for ii in range(0, self.data.nAxialNodes):
            for jj in range(0, self.data.nRadialNodes):
                marker = self.data.dataSet[ii, jj].marker
                counter = self.data.dataSet[ii, jj].nodeCounter
                row = counter * 5  # 5 equations per node

                if mode == 'over writing':
                    if marker == 'inlet':
                        self.apply_bc_condition(row, self.inlet_bc, ii, jj)
                    elif marker == 'outlet':
                        self.apply_bc_condition(row, self.outlet_bc, ii, jj)
                    elif marker == 'hub':
                        self.apply_bc_condition(row, self.hub_bc, ii, jj)
                    elif marker == 'shroud':
                        self.apply_bc_condition(row, self.shroud_bc, ii, jj)
                    elif marker != 'internal':
                        raise Exception('Boundary condition unknown. Check the grid markers!')
                elif mode == 'added':
                    if marker == 'inlet':
                        self.add_bc_condition(row, self.inlet_bc, ii, jj)
                    elif marker == 'outlet':
                        self.add_bc_condition(row, self.outlet_bc, ii, jj)
                    elif marker == 'hub':
                        self.add_bc_condition(row, self.hub_bc, ii, jj)
                    elif marker == 'shroud':
                        self.add_bc_condition(row, self.shroud_bc, ii, jj)
                    elif marker != 'internal':
                        raise Exception('Boundary condition unknown. Check the grid markers!')

    def set_boundary_conditions(self):
        """
        Store in the object the information related to the boundary conditions to use for the problem
        """
        self.inlet_bc = self.config.get_inlet_bc()
        self.outlet_bc = self.config.get_outlet_bc()
        self.hub_bc = self.config.get_hub_bc()
        self.shroud_bc = self.config.get_shroud_bc()

        # recognized boundary conditions type
        bc_list = ['zero pressure', 'zero perturbation', 'euler wall', 'compressor inlet', 'compressor outlet',
                   'zero axial velocity', 'free', 'neumann inlet', 'neumann outlet']

        if self.inlet_bc not in bc_list:
            raise ValueError('Incorrect Inlet boundary condition type.')
        if self.outlet_bc not in bc_list:
            raise ValueError('Incorrect Outlet boundary condition type.')
        if self.hub_bc not in bc_list:
            raise ValueError('Incorrect Hub boundary condition type.')
        if self.shroud_bc not in bc_list:
            raise ValueError('Incorrect Shroud boundary condition type.')

        print_banner_begin('BOUNDARY CONDITIONS')
        print(f"{'Inlet Boundary set to:':<{total_chars_mid}}{self.inlet_bc:>{total_chars_mid}}")
        print(f"{'Outlet Boundary set to:':<{total_chars_mid}}{self.outlet_bc:>{total_chars_mid}}")
        print(f"{'Hub Boundary set to:':<{total_chars_mid}}{self.hub_bc:>{total_chars_mid}}")
        print(f"{'Shroud Boundary set to:':<{total_chars_mid}}{self.shroud_bc:>{total_chars_mid}}")
        print_banner_end()

    def apply_bc_condition(self, row, condition, ii, jj):
        """
        For the considered grid node, it modifes the 5 governing equations starting from row index,
        which is related to its continuity eq.
        The considered system at hand is: (L0 + L1*omega + L2*omega**2)*tilde{phi}. Therefore BCs are imposed on L0, since
        they must be respected for every possible value of omega.
        L1 and L2 are then filled in the respective positions with zeros.
        :param row: row index of the equation to modify
        :param condition: type of boundary condition
        :param ii: i-th element of the node grid
        :param jj: j-th element of the node grid
        """

        if condition == 'zero pressure':
            # BC for zero pressure perturbation
            self.L0[row + 4, :] = np.zeros(self.L0[row + 4, :].shape, dtype=complex)
            self.L0[row + 4, row + 4] = 1  # zero pressure at that node

            self.L1[row + 4, :] = np.zeros(self.L1[row + 4, :].shape, dtype=complex)  # zero row
            self.L2[row + 4, :] = np.zeros(self.L2[row + 4, :].shape, dtype=complex)  # zero row

        elif condition == 'zero axial velocity':
            # BC for zero pressure perturbation
            self.L0[row + 3, :] = np.zeros(self.L0[row + 4, :].shape, dtype=complex)
            self.L0[row + 3, row + 3] = 1  # zero pressure at that node

            self.L1[row + 3, :] = np.zeros(self.L1[row + 3, :].shape, dtype=complex)  # zero row
            self.L2[row + 3, :] = np.zeros(self.L2[row + 3, :].shape, dtype=complex)  # zero row

        elif condition == 'free':
            pass

        elif condition == 'zero perturbation':
            # BC for zero pressure perturbation
            self.L0[row:row + 5, :] = np.zeros(self.L0[row:row + 5, :].shape, dtype=complex)
            self.L0[row:row + 5, row:row + 5] = np.eye(5, dtype=complex)

            self.L1[row:row + 5, :] = np.zeros(self.L1[row:row + 5, :].shape, dtype=complex)  # zero rows
            self.L2[row:row + 5, :] = np.zeros(self.L2[row:row + 5, :].shape, dtype=complex)  # zero rows

        elif condition == 'neumann inlet':
            # BC for zero pressure perturbation
            self.L0[row:row + 5, :] = np.zeros(self.L0[row:row + 5, :].shape, dtype=complex)
            node = row // 5  # number of the node
            node_next = node + self.data.nRadialNodes  # number of the next node along the streamline
            row_next = node_next * 5  # equivalent row index for that next node
            self.L0[row:row + 5, row:row + 5] = np.eye(5, dtype=complex)
            self.L0[row:row + 5, row_next:row_next + 5] = -np.eye(5, dtype=complex)

            self.L1[row:row + 5, :] = np.zeros(self.L1[row:row + 5, :].shape, dtype=complex)  # zero rows
            self.L2[row:row + 5, :] = np.zeros(self.L2[row:row + 5, :].shape, dtype=complex)  # zero rows

        elif condition == 'neumann outlet':
            # BC for zero pressure perturbation
            self.L0[row:row + 5, :] = np.zeros(self.L0[row:row + 5, :].shape, dtype=complex)
            node = row // 5  # number of the node
            node_previous = node - self.data.nRadialNodes  # number of the next node along the streamline
            row_previous = node_previous * 5  # equivalent row index for that next node
            self.L0[row:row + 5, row:row + 5] = np.eye(5, dtype=complex)
            self.L0[row:row + 5, row_previous:row_previous + 5] = -np.eye(5, dtype=complex)

            self.L1[row:row + 5, :] = np.zeros(self.L1[row:row + 5, :].shape, dtype=complex)  # zero rows
            self.L2[row:row + 5, :] = np.zeros(self.L2[row:row + 5, :].shape, dtype=complex)  # zero rows

        elif condition == 'compressor inlet':
            # BCs are zero for every variable except the pressure at inlet
            self.L0[row:row + 4, :] = np.zeros(self.L0[row:row + 4, :].shape, dtype=complex)
            self.L0[row:row + 4, row:row + 4] = np.eye(4, dtype=complex)

            self.L1[row:row + 4, :] = np.zeros(self.L1[row:row + 4, :].shape, dtype=complex)  # zero rows
            self.L2[row:row + 4, :] = np.zeros(self.L2[row:row + 4, :].shape, dtype=complex)  # zero rows

        elif condition == 'compressor outlet':
            # BC for zero pressure perturbation
            self.L0[row + 4, :] = np.zeros(self.L0[row + 4, :].shape, dtype=complex)
            self.L0[row + 4, row + 4] = 1  # zero pressure at that node

            self.L1[row + 4, :] = np.zeros(self.L1[row + 4, :].shape, dtype=complex)  # zero row
            self.L2[row + 4, :] = np.zeros(self.L2[row + 4, :].shape, dtype=complex)  # zero row

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

            self.L0[row + loc, :] = np.zeros(self.L0[row + loc, :].shape, dtype=complex)
            self.L0[row + loc, row + 1:row + 4] = wall_normal

            self.L1[row + loc, :] = np.zeros(self.L1[row + loc, :].shape, dtype=complex)  # zero known term
            self.L2[row + loc, :] = np.zeros(self.L2[row + loc, :].shape, dtype=complex)  # zero known term

        else:
            raise ValueError('unknown boundary condition type')

    def add_bc_condition(self, row, condition, ii, jj):
        """
        For the considered grid node, it adds the boundary conditions at the end of the matrix, enlarging the
        dimensions, using the Lagrange Multiplier formulation.
        The considered system at hand is: (L0 + L1*omega + L2*omega**2)*tilde{phi}.
        Therefore BCs are imposed on L0, since they must be respected for every possible value of omega.
        L1 and L2 are then filled in the respective positions with zeros.
        :param row: row index of the equation to modify
        :param condition: type of boundary condition
        :param ii: i-th element of the node grid
        :param jj: j-th element of the node grid
        """

        if condition == 'zero pressure':
            zero_col = np.zeros((self.L0.shape[0], 1))
            new_col = np.zeros((self.L0.shape[0], 1))  # col to add
            zero_row = np.zeros((1, self.L0.shape[0] + 1))
            new_row = np.zeros((1, self.L0.shape[0] + 1))  # row to add

            # set the condition
            new_col[row + 4, 0] = 1
            new_row[0, row + 4] = 1  # zero pressure at the corresponding node

            self.L0 = np.hstack((self.L0, new_col))  # add one zero column to the right of the matrix
            self.L0 = np.vstack((self.L0, new_row))  # add one row below the rectangular matrix, making it square

            # now simply adjust the shape of L1 and L2, making them squares, of the same dimension of L0
            self.L1 = np.hstack((self.L1, zero_col))
            self.L1 = np.vstack((self.L1, zero_row))

            self.L2 = np.hstack((self.L2, zero_col))
            self.L2 = np.vstack((self.L2, zero_row))

            self.rows_added += 1

        # elif condition == 'zero axial velocity':
        #     # BC for zero pressure perturbation
        #     self.L0[row + 3, :] = np.zeros(self.L0[row + 4, :].shape, dtype=complex)
        #     self.L0[row + 3, row + 3] = 1  # zero pressure at that node
        #
        #     self.L1[row + 3, :] = np.zeros(self.L1[row + 3, :].shape, dtype=complex)  # zero row
        #     self.L2[row + 3, :] = np.zeros(self.L2[row + 3, :].shape, dtype=complex)  # zero row
        #
        # elif condition == 'free':
        #     pass
        #
        # elif condition == 'zero perturbation':
        #     # BC for zero pressure perturbation
        #     self.L0[row:row + 5, :] = np.zeros(self.L0[row:row + 5, :].shape, dtype=complex)
        #     self.L0[row:row + 5, row:row + 5] = np.eye(5, dtype=complex)
        #
        #     self.L1[row:row + 5, :] = np.zeros(self.L1[row:row + 5, :].shape, dtype=complex)  # zero rows
        #     self.L2[row:row + 5, :] = np.zeros(self.L2[row:row + 5, :].shape, dtype=complex)  # zero rows
        #
        # elif condition == 'compressor inlet':
        #     # BCs are zero for every variable except the pressure at inlet
        #     self.L0[row:row + 4, :] = np.zeros(self.L0[row:row + 4, :].shape, dtype=complex)
        #     self.L0[row:row + 4, row:row + 4] = np.eye(4, dtype=complex)
        #
        #     self.L1[row:row + 5, :] = np.zeros(self.L1[row:row + 5, :].shape, dtype=complex)  # zero rows
        #     self.L2[row:row + 5, :] = np.zeros(self.L2[row:row + 5, :].shape, dtype=complex)  # zero rows
        #
        # elif condition == 'compressor outlet':
        #     # BC for zero pressure perturbation
        #     self.L0[row + 4, :] = np.zeros(self.L0[row + 4, :].shape, dtype=complex)
        #     self.L0[row + 4, row + 4] = 1  # zero pressure at that node
        #
        #     self.L1[row + 4, :] = np.zeros(self.L1[row + 4, :].shape, dtype=complex)  # zero row
        #     self.L2[row + 4, :] = np.zeros(self.L2[row + 4, :].shape, dtype=complex)  # zero row

        elif condition == 'euler wall':
            # # BC for non-penetration condition at the walls, the equation overwritten depends on configs
            # if self.substituted_equation == 'ur':
            #     loc = 1
            # elif self.substituted_equation == 'utheta':
            #     loc = 2
            # elif self.substituted_equation == 'uz':
            #     loc = 3
            # else:
            #     raise ValueError("Subsituted equation parameter not recognized.")
            #
            wall_normal = self.data.dataSet[ii, jj].n_wall
            #
            # zero_cols = np.zeros((self.L0.shape[0], 3))
            # zero_rows = np.zeros((3, self.L0.shape[0] + 3))
            # new_rows = np.zeros((3, self.L0.shape[0] + 3))
            # new_rows[-1, -4:-1] = wall_normal
            #
            # self.L0 = np.hstack((self.L0, zero_cols))
            # self.L0 = np.vstack((self.L0, new_rows))
            #
            # self.L1 = np.hstack((self.L1, zero_cols))
            # self.L1 = np.vstack((self.L1, zero_rows))
            #
            # self.L2 = np.hstack((self.L2, zero_cols))
            # self.L2 = np.vstack((self.L2, zero_rows))

            zero_col = np.zeros((self.L0.shape[0], 1))  # zero col to adjust the shape
            new_col = np.zeros((self.L0.shape[0], 1))
            zero_row = np.zeros((1, self.L0.shape[0] + 1))  # zero row to adjust the shape
            new_row = np.zeros((1, self.L0.shape[0] + 1))  # row which add the boundary condition on L0 matrix
            new_row[0, row + 1: row + 4] = wall_normal  # zero pressure at the corresponding node
            new_col[row + 1: row + 4, 0] = wall_normal

            self.L0 = np.hstack((self.L0, new_col))  # add one zero column to the right of the matrix
            self.L0 = np.vstack((self.L0, new_row))  # add one row below the rectangular matrix, making it square

            # now simply adjust the shape of L1 and L2, making them squares, of the same dimension of L0
            self.L1 = np.hstack((self.L1, zero_col))
            self.L1 = np.vstack((self.L1, zero_row))

            self.L2 = np.hstack((self.L2, zero_col))
            self.L2 = np.vstack((self.L2, zero_row))

            self.rows_added += 1

        else:
            raise ValueError('unknown boundary condition type')

    # def compute_block_Y_P_matrices(self, inspect_matrices=False):
    #     """
    #     Solve EVP with implicitly restarted Arnoldi Algorithm, with shift-invert strategy.
    #     :param inspect_matrices: plot the structure of the involved matrices, to check sparsity
    #     """
    #
    #     m = self.config.get_circumferential_harmonic_order()
    #     omega_shaft = self.config.get_omega_shaft()
    #     omega_ref = self.config.get_reference_omega()
    #     x_ref = self.config.get_reference_length()
    #     u_ref = self.config.get_reference_velocity()
    #     t_ref = self.config.get_reference_time()
    #     sigma = self.config.get_research_center_omega_eigenvalues() / omega_ref  # non-dimensional center point of research
    #     number_search = self.config.get_research_number_omega_eigenvalues()
    #
    #     print_banner_begin('ARNOLDI SOLVER')
    #     print(f"{'Circumferential Harmonic:':<{total_chars_mid}}{m:>{total_chars_mid}}")
    #     print(f"{'Shaft Angular Speed [rad/s]:':<{total_chars_mid}}{omega_shaft:>{total_chars_mid}.2f}")
    #     print(f"{'Ref. Angular Speed [rad/s]:':<{total_chars_mid}}{omega_ref:>{total_chars_mid}.2f}")
    #     print(f"{'Initial Searching Point [-]:':<{total_chars_mid}}{sigma:>{total_chars_mid}.2f}")
    #     print(f"{'Number of Eigenvalues to Find:':<{total_chars_mid}}{number_search:>{total_chars_mid}}")
    #     print_banner_end()
    #
    #     Y1 = np.concatenate((-self.L0, np.zeros_like(self.L0)), axis=1)
    #     Y2 = np.concatenate((np.zeros_like(self.L0), np.eye(self.L0.shape[0])), axis=1)
    #     self.Y = np.concatenate((Y1, Y2), axis=0)  # Y matrix of EVP problem
    #
    #     P1 = np.concatenate((self.L1, self.L2), axis=1)
    #     P2 = np.concatenate((np.eye(self.L0.shape[0]), np.zeros_like(self.L0)), axis=1)
    #     self.P = np.concatenate((P1, P2), axis=0)  # P matrix of EVP problem
    #
    #     if inspect_matrices:
    #         plt.figure()
    #         plt.spy(self.L0)
    #         plt.title(r'$\mathbf{L}_{0}$')
    #
    #         plt.figure()
    #         plt.spy(self.L1)
    #         plt.title(r'$\mathbf{L}_{1}$')
    #
    #         plt.figure()
    #         plt.spy(self.L2)
    #         plt.title(r'$\mathbf{L}_{2}$')
    #
    #         plt.figure()
    #         plt.spy(self.Y)
    #         plt.title(r'$\mathbf{Y}$')
    #
    #         plt.figure()
    #         plt.spy(self.P)
    #         plt.title(r'$\mathbf{P}$')
    #
    #     # print("Transforming generalized EVP in standard one...")
    #     # Y_tilde = np.linalg.inv(Y - sigma * P)
    #     # Y_tilde = np.dot(Y_tilde, P)
    #     #
    #     # print("Solving standard EVP...")
    #     # self.eigenfreqs, self.eigenmodes = eigs(Y_tilde, k=self.config.get_research_number_omega_eigenvalues())
    #     # self.eigenfreqs = sigma + 1 / self.eigenfreqs  # return of the initial shift
    #     # self.eigenfreqs *= omega_ref  # convert to dimensional frequencies
    #     # self.eigenfreqs_df = self.eigenfreqs.imag / omega_ref
    #     # self.eigenfreqs_rs = self.eigenfreqs.real / omega_ref
    #     # self.sort_eigensolution()

    # def sort_eigensolution(self):
    #     """
    #     Sort the eigenvalues and eigenvectors from the most unstable to the least one.
    #     """
    #     # make copies of the arrays to sort
    #     eigenfreqs = np.copy(self.eigenfreqs)
    #     df = np.copy(self.eigenfreqs_df)
    #     rs = np.copy(self.eigenfreqs_rs)
    #     eigenvectors = np.copy(self.eigenmodes)
    #
    #     # get the sorting indices following descending order of the damping factor
    #     sorted_indices = sorted(range(len(df)), key=lambda i: df[i], reverse=True)
    #
    #     # order the original arrays following the sorting indices
    #     for i in range(len(sorted_indices)):
    #         self.eigenfreqs[i] = eigenfreqs[sorted_indices[i]]
    #         self.eigenfreqs_df[i] = df[sorted_indices[i]]
    #         self.eigenfreqs_rs[i] = rs[sorted_indices[i]]
    #         self.eigenmodes[:, i] = eigenvectors[:, sorted_indices[i]]

    # def plot_eigenfrequencies(self, delimit=False, save_filename=None):
    #     """
    #     Plot the eigenfrequencies obtained with the Arnoldi Method
    #     :param delimit: if true, delimit the plot zone the important one for compressors
    #     :param save_filename: if not None, save figure files
    #     """
    #     fig, ax = plt.subplots(figsize=fig_size)
    #     for mode in self.eigenfields:
    #         # if mode.is_physical:
    #         rs = mode.eigenfrequency.real / self.omega_ref
    #         df = mode.eigenfrequency.imag / self.omega_ref
    #         ax.scatter(rs, df, marker='o', facecolors='red', edgecolors='red',
    #                    s=marker_size)
    #     ax.set_xlabel(r'RS [-]')
    #     ax.set_ylabel(r'DF [-]')
    #     # ax.legend()
    #     if delimit:
    #         ax.set_xlim([-1.5, 1.5])
    #         ax.set_ylim([-1, 0.5])
    #     ax.grid(alpha=grid_opacity)
    #     if save_filename is not None:
    #         fig.savefig(folder_name + save_filename + '.pdf', bbox_inches='tight')
    #         plt.close()

    # def extract_eigenfields(self, n=None):
    #     """
    #     From the eigenvectors obtained with Arnoldi Method, extract the eigenfields (density, velocity, pressure).
    #     The eigensolution should be sorted before applying this method, otherwise the modes are randomly ordered.
    #     :param n: number of eigenfields to extract
    #     """
    #     if n is None:
    #         n = len(self.eigenfreqs)
    #     elif n > len(self.eigenfreqs):
    #         print("parameter n must be lower than the eigenvector number. n set to max allowed")
    #         n = len(self.eigenfreqs)
    #
    #     Nz = self.data.nAxialNodes
    #     Nr = self.data.nRadialNodes
    #     self.eigenfields = []
    #     for mode in range(n):
    #         eigenfrequency = self.eigenfreqs[mode]
    #         eigenvector = self.eigenmodes[:, mode]
    #
    #         rho_eig = []
    #         ur_eig = []
    #         ut_eig = []
    #         uz_eig = []
    #         p_eig = []
    #         for i in range(len(eigenvector) // 2):  # remember that the flow state had been doubled in this Alg.
    #             if (i) % 5 == 0:
    #                 rho_eig.append(eigenvector[i])
    #             elif (i - 1) % 5 == 0 and i != 0:
    #                 ur_eig.append(eigenvector[i])
    #             elif (i - 2) % 5 == 0 and i != 0:
    #                 ut_eig.append(eigenvector[i])
    #             elif (i - 3) % 5 == 0 and i != 0:
    #                 uz_eig.append(eigenvector[i])
    #             elif (i - 4) % 5 == 0 and i != 0:
    #                 p_eig.append(eigenvector[i])
    #             else:
    #                 raise ValueError("Not correct indexing for eigenvector retrieval!")
    #
    #         rho_eig_r = scaled_eigenvector_real(rho_eig, Nz, Nr)
    #         ur_eig_r = scaled_eigenvector_real(ur_eig, Nz, Nr)
    #         ut_eig_r = scaled_eigenvector_real(ut_eig, Nz, Nr)
    #         uz_eig_r = scaled_eigenvector_real(uz_eig, Nz, Nr)
    #         p_eig_r = scaled_eigenvector_real(p_eig, Nz, Nr)
    #
    #         self.eigenfields.append(Eigenmode(eigenfrequency, rho_eig_r, ur_eig_r, ut_eig_r, uz_eig_r, p_eig_r))

    # def plot_eigenfields(self, n=None, save_filename=None):
    #     """
    #     Plot the first n eigenmodes structures.
    #     :param n: specify the first n eigenfunctions to plot
    #     :param save_filename: specify name of the figs to save
    #     """
    #     z = self.data.meridional_obj.z_cg
    #     r = self.data.meridional_obj.r_cg
    #     self.pic_size_blank, self.pic_size_contour = compute_picture_size(z, r)
    #     Nz = np.shape(z)[0]
    #     Nr = np.shape(z)[1]
    #     modes_map = cm.bwr
    #
    #     if n is None:
    #         n = len(self.eigenfields)
    #     elif n > len(self.eigenfields):
    #         print("parameter n must be lower than the eigenfields number. n set to max allowed!")
    #         n = len(self.eigenfreqs)
    #     elif n < 1:
    #         raise ValueError("Select a positive number of modes to show")
    #     else:
    #         pass
    #
    #     imode = 0
    #     for mode in self.eigenfields[0:n]:
    #         imode += 1
    #         # if mode.is_physical:
    #         rs = mode.eigenfrequency.real / self.omega_ref
    #         df = mode.eigenfrequency.imag / self.omega_ref
    #
    #         plt.figure(figsize=self.pic_size_contour)
    #         cnt = plt.contourf(z, r, mode.eigen_rho, levels=N_levels_fine, cmap=modes_map)
    #         for c in cnt.collections:
    #             c.set_edgecolor("face")
    #         plt.xlabel(r'$z$ [-]')
    #         plt.ylabel(r'$r$ [-]')
    #         plt.title(r'$\tilde{\rho}_{%i}: \  \hat{\omega} = [%.2f,%.2f j]$' % (imode, rs, df))
    #         plt.colorbar()
    #         if save_filename is not None:
    #             plt.savefig(folder_name + save_filename + '_rho_%i_%i_%i.pdf' % (Nz, Nr, imode), bbox_inches='tight')
    #             plt.close()
    #
    #         plt.figure(figsize=self.pic_size_contour)
    #         cnt = plt.contourf(z, r, mode.eigen_ur, levels=N_levels_fine, cmap=modes_map)
    #         for c in cnt.collections:
    #             c.set_edgecolor("face")
    #         plt.xlabel(r'$z$ [-]')
    #         plt.ylabel(r'$r$ [-]')
    #         plt.title(r'$\tilde{u}_{r,%i}: \  \hat{\omega} = [%.2f,%.2f j]$' % (imode, rs, df))
    #         plt.colorbar()
    #         if save_filename is not None:
    #             plt.savefig(folder_name + save_filename + '_ur_%i_%i_%i.pdf' % (Nz, Nr, imode), bbox_inches='tight')
    #             plt.close()
    #
    #         plt.figure(figsize=self.pic_size_contour)
    #         cnt = plt.contourf(z, r, mode.eigen_utheta, levels=N_levels_fine, cmap=modes_map)
    #         for c in cnt.collections:
    #             c.set_edgecolor("face")
    #         plt.xlabel(r'$z$ [-]')
    #         plt.ylabel(r'$r$ [-]')
    #         plt.title(r'$\tilde{u}_{\theta,%i}: \  \hat{\omega} = [%.2f,%.2f j]$' % (imode, rs, df))
    #         plt.colorbar()
    #         if save_filename is not None:
    #             plt.savefig(folder_name + save_filename + '_ut_%i_%i_%i.pdf' % (Nz, Nr, imode), bbox_inches='tight')
    #             plt.close()
    #
    #         plt.figure(figsize=self.pic_size_contour)
    #         cnt = plt.contourf(z, r, mode.eigen_uz, levels=N_levels_fine, cmap=modes_map)
    #         for c in cnt.collections:
    #             c.set_edgecolor("face")
    #         plt.xlabel(r'$z$ [-]')
    #         plt.ylabel(r'$r$ [-]')
    #         plt.title(r'$\tilde{u}_{z,%i}: \  \hat{\omega} = [%.2f,%.2f j]$' % (imode, rs, df))
    #         plt.colorbar()
    #         if save_filename is not None:
    #             plt.savefig(folder_name + save_filename + '_uz_%i_%i_%i.pdf' % (Nz, Nr, imode), bbox_inches='tight')
    #             plt.close()
    #
    #         plt.figure(figsize=self.pic_size_contour)
    #         cnt = plt.contourf(z, r, mode.eigen_p, levels=N_levels_fine, cmap=modes_map)
    #         for c in cnt.collections:
    #             c.set_edgecolor("face")
    #         plt.xlabel(r'$z$ [-]')
    #         plt.ylabel(r'$r$ [-]')
    #         plt.title(r'$\tilde{p}_{%i}: \  \hat{\omega} = [%.2f,%.2f j]$' % (imode, rs, df))
    #         plt.colorbar()
    #         if save_filename is not None:
    #             plt.savefig(folder_name + save_filename + '_p_%i_%i_%i.pdf' % (Nz, Nr, imode), bbox_inches='tight')
    #             plt.close()

    # def write_results(self, save_filename=None, extension='csv'):
    #     """
    #     Print information regarding the eigenfrequencies found, in the form of damping factors and rotations speeds
    #     Possible file types are (csv, pickle).
    #     csv: write only DF and RS in a csv file, organized in two columns
    #     pickle: write the full list of eigenfields, which contain frequencies and eigenfunctions, in a single pickle
    #     """
    #     if save_filename is not None:
    #         filename = save_filename
    #     else:
    #         filename = 'eigenvalues'
    #
    #     eigenvalue_array = self.eigenfreqs_rs + 1j * self.eigenfreqs_df
    #
    #     if extension == 'csv':
    #         with open(folder_name + filename + '.csv', 'w', newline='') as csvfile:
    #             fieldnames = ['RS', 'DF']
    #             writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #             writer.writeheader()
    #             for num in eigenvalue_array:
    #                 writer.writerow({'RS': num.real, 'DF': num.imag})
    #     elif extension == 'pickle':
    #         with open(folder_name + 'eigenfields.pickle', 'wb') as picklefile:
    #             pickle.dump(self.eigenfields, picklefile)
    #     else:
    #         raise ValueError("Incorrect Extension of the output file.")

    def interpolation_on_original_grid(self, df, X, Y, method='rbf'):
        """
        Interpolate the gradient field df compute of the fine grid on the original one.
        :param df: gradient 2D array
        :param X: 2D array of the x cordinates of the fine grid
        :param Y: 2D array of the y cordinates of the fine grid
        :param method: method used for the interpolation
        """
        Xnew = self.dataSpectral.zGrid  # original grid
        Ynew = self.dataSpectral.rGrid  # original grid
        if method == 'linear':
            df_new = griddata((X.flatten(), Y.flatten()), df.flatten(), (Xnew, Ynew), method='linear')
        elif method == 'cubic':
            df_new = griddata((X.flatten(), Y.flatten()), df.flatten(), (Xnew, Ynew), method='cubic')
        elif method == 'nearest':
            df_new = griddata((X.flatten(), Y.flatten()), df.flatten(), (Xnew, Ynew), method='nearest')
        elif method == 'rbf':
            rbf_interpolator = Rbf(X, Y, df, function='multiquadric')
            df_new = rbf_interpolator(Xnew, Ynew)
        else:
            raise ValueError("Method unknown")
        return df_new

    def free_dataset_memory(self):
        """
        Release memory that is not needed anymore
        """
        self.data = None
