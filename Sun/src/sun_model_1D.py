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


class SunModel1D:
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

    def __init__(self, duct_obj, config):
        """
        Instantiate the sun model Object, contaning all the attributes and methods necessary for the instability analysis.
        :param duct_obj: is the object contaning all the data
        :param config: configuration file of the sun model
        """
        self.data = duct_obj  # grid object containing also the meridional object with the data
        self.config = config
        self.nPoints = duct_obj.nPoints
        self.gmma = self.config.get_fluid_gamma()
        print(f"Gamma set to Default Value: {self.gmma}")

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

    def compute_A(self):
        """
        Compute and store at the node level the A matrix. My Formulation.
        """
        for ii in range(0, self.nPoints):
            A = np.eye(5, dtype=complex)
            A[4, 0] = -self.data.p[ii] * self.gmma / self.data.rho[ii]

            """ Multiply only matrix A times the strouhal number. If the reference velocity was defined as:
             u_ref = omega_ref * x_ref and t_ref = 1 / omega_ref, automatically the strouhal
            should be 1 by construction. In this case the non-dimensional equations are exactly the same
            of the dimensional ones."""
            strouhal = self.config.get_reference_length() / (
                        self.config.get_reference_velocity() * self.config.get_reference_time())
            A *= strouhal
            self.A[ii*5:ii*5+5, ii*5:ii*5+5] = A

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

    def compute_B(self):
        """
        Compute and store at the node level the B matrix, needed to compute hat{B} later. My Formulation.
        """
        for ii in range(0, self.nPoints):
            B = np.eye(5, dtype=complex) * self.data.ur[ii]
            B[0, 1] = self.data.rho[ii]
            B[1, 4] = 1 / self.data.rho[ii]
            B[4, 0] = - self.data.p[ii] * self.data.ur[ii] * self.gmma / self.data.rho[ii]
            self.B[ii*5:ii*5+5, ii*5:ii*5+5] = B

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

    def compute_C(self):
        """
        Compute and store at node level the C matrix, already multiplied by j*m/r. Ready to be used in the final system of eqs.
        My Formulation.
        """
        m = self.config.get_circumferential_harmonic_order()
        print(f"Circumferential Harmonic Order set to: {m}")

        for ii in range(0, self.nPoints):
            C = np.eye(5, dtype=complex) * self.data.ut[ii]
            C[0, 2] = self.data.rho[ii]
            C[2, 4] = 1 / self.data.rho[ii]
            C[4, 0] = -self.data.p[ii] * self.data.ut[ii] * self.gmma / self.data.rho[ii]
            C *= 1j * m / self.data.r[ii]
            self.C[ii*5:ii*5+5, ii*5:ii*5+5] = C

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

    def compute_E(self, k):
        """
        Compute and store at the node level the E matrix, needed to compute hat{E}. My Formulation.
        """
        for ii in range(0, self.nPoints):
            E = np.eye(5, dtype=complex) * self.data.uz[ii]
            E[0, 3] = self.data.rho[ii]
            E[3, 4] = 1 / self.data.rho[ii]
            E[4, 0] = -self.data.p[ii] * self.data.uz[ii] * self.gmma / self.data.rho[ii]
            E *= 1j*k*E
            self.E[ii*5:ii*5+5, ii*5:ii*5+5] = E

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

    def compute_R(self):
        """
        Compute and store at the node level the R matrix.
        My version of the equations.
        """
        for ii in range(0, self.nPoints):
            R = np.zeros((5, 5), dtype=complex)

            R[0, 0] = self.data.dur_dr[ii] + self.data.duz_dz[ii] + self.data.ur[ii] / self.data.r[ii]
            R[0, 1] = self.data.rho[ii] / self.data.r[ii] + self.data.drho_dr[ii]
            R[0, 3] = self.data.drho_dz[ii]

            R[1, 0] = self.data.dp_dr[ii] / (self.data.rho[ii] ** 2)
            R[1, 1] = self.data.dur_dr[ii]
            R[1, 2] = -2 * self.data.ut[ii] / self.data.r[ii]
            R[1, 3] = self.data.dur_dz[ii]

            R[2, 1] = self.data.dut_dr[ii] + self.data.ut[ii] / \
                      self.data.r[ii]
            R[2, 2] = self.data.ur[ii] / self.data.r[ii]
            R[2, 3] = self.data.dut_dz[ii]

            R[3, 0] = self.data.dp_dz[ii] / (self.data.rho[ii] ** 2)
            R[3, 1] = self.data.duz_dr[ii]
            R[3, 3] = self.data.duz_dz[ii]

            # R[4, 0] = (1 / node.rho) * (node.ur * node.dp_dr + node.uz * node.dp_dz) # first version
            R[4, 0] = -self.gmma / (self.data.rho[ii] ** 2) * (self.data.ur[ii] * self.data.p[ii] *
                    self.data.drho_dr[ii] + self.data.uz[ii] * self.data.p[ii] * self.data.drho_dz[ii])
            R[4, 1] = self.data.dp_dr[ii] - self.data.p[ii] * self.data.drho_dr[ii] * self.gmma / self.data.rho[ii]
            R[4, 3] = self.data.dp_dz[ii] - self.gmma / self.data.rho[ii] * self.data.p[ii] * self.data.drho_dz[ii]
            R[4, 4] = (-self.gmma / self.data.rho[ii]) * (self.data.ur[ii] * self.data.drho_dr[ii] +
                                                          self.data.uz[ii] * self.data.drho_dz[ii])

            self.R[ii*5:ii*5+5, ii*5:ii*5+5] = R

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

    def compute_Bhat(self):
        """
        Compute and store at the node level the hat{B}, hat{E} matrix, needed for following multiplication with the spectral
        differential operators.
        """
        self.Bhat = self.B*2*(self.data.r[0] - self.data.r[-1])

    def ApplySpectralDifferentiation(self, verbose=False):
        """
        This method applies Chebyshev-Gauss-Lobatto differentiation method to Bhat to express the perturbation
        derivatives as a function of the perturbation at the other nodes.
        It saves a new global (for all the nodes) matrix Bd,
        which is part of the global stability matrix. The full dimension is: (nPoints*5,nPoints*5).
        :param verbose: print additional info.
        """

        # compute the spectral Matrices for x and y direction with the Bayliss formulation
        Dxi = ChebyshevDerivativeMatrixBayliss(self.data.r)  # derivative operator in eta

        # Q_const is the global matrix storing B and E elements after spectral differentiation.
        self.Bd = np.zeros_like(self.Bhat)

        # differentiation of a general perturbation vector (for the node (i,j)) along xi
        for ii in range(0, self.nPoints):
            B_ij = self.Bhat[ii*5: ii*5+5, ii*5: ii*5+5]  # Bhat matrix of the i node

            # xi differentiation. m is in the range of axial nodes, first axis of the matrix
            for m in range(0, self.nPoints):
                tmp = Dxi[ii, m] * B_ij  # 5x5 matrix to be added to a certain block of Q
                row = ii * 5  # this selects the correct block along rows
                column = m * 5  # it selects the correct block along cols
                self.Bd[row: row+5, column: column+5] = tmp

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

    def compute_Z(self):
        """
        put in standard EV form: Z*x = omega*A*x
        """
        self.Z = (self.Bd + self.E + self.R + self.C)/1j

    def compute_L0(self, k):
        """
        put in standard EV form: Z*x = omega*A*x
        """
        self.L0 = np.eye(self.nPoints)*4*(self.data.ut/self.data.r)**2 * k**2

    def compute_L1(self):
        """
        put in standard EV form: Z*x = omega*A*x
        """
        self.L1 = np.zeros_like(self.L0)

    def compute_L2(self, k):
        """
        put in standard EV form: Z*x = omega*A*x
        """
        beta = 2/(self.data.r[0]- self.data.r[-1])
        divide_r = np.eye(self.data.nPoints)
        for ii in range(0,self.data.nPoints):
            divide_r[ii, ii] /= self.data.r[ii]

        xi = GaussLobattoPoints(self.data.nPoints)
        D = ChebyshevDerivativeMatrixBayliss(xi)
        D2 = D@D
        self.L2 = beta**2 * D2 + divide_r * beta*D - k**2







    def inspect_L_matrices(self, save_filename=None, save_foldername=None):
        """
        Plot the L matrices, to inspect their composition
        """
        if save_filename is not None:
            create_folder(save_foldername)

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].spy(self.Z)
        ax[1].spy(self.A)
        ax[0].set_title(r'$Z$')
        ax[1].set_title(r'$A$')
        plt.savefig(os.path.join(save_foldername, save_filename + '.pdf'), bbox_inches='tight')

    def apply_boundary_conditions_generalized(self, mode='over writing'):
        """
        Apply the boundary conditions for the considered system, whose equations are:
        (-j*omega*A + Z + S/zita)*tilde{phi}. Therefore, BCs are imposed on Z (the only constant matrix), and A and S
        (omega dependent) filled with zeros in correspondance of those BCs.
        """
        # inlet conditions
        row = 0
        self.apply_bc_condition(row, self.inlet_bc)

        # outlet conditions
        row = (self.nPoints-1)
        self.apply_bc_condition(row, self.outlet_bc)


    def set_boundary_conditions(self):
        """
        Store in the object the information related to the boundary conditions to use for the problem
        """
        self.inlet_bc = self.config.get_inlet_bc()
        self.outlet_bc = self.config.get_outlet_bc()

        # recognized boundary conditions type
        bc_list = ['zero pressure', 'zero perturbation', 'euler wall', 'compressor inlet', 'compressor outlet',
                   'zero axial velocity', 'free', 'neumann inlet', 'neumann outlet', 'zero radial velocity']

        if self.inlet_bc not in bc_list:
            raise ValueError('Incorrect Inlet boundary condition type.')
        if self.outlet_bc not in bc_list:
            raise ValueError('Incorrect Outlet boundary condition type.')

        print_banner_begin('BOUNDARY CONDITIONS')
        print(f"{'Inlet Boundary set to:':<{total_chars_mid}}{self.inlet_bc:>{total_chars_mid}}")
        print(f"{'Outlet Boundary set to:':<{total_chars_mid}}{self.outlet_bc:>{total_chars_mid}}")
        print_banner_end()

    def apply_bc_condition(self, row, condition):
        """
        For the considered grid node, it modifes the 5 governing equations starting from row index,
        which is related to its continuity eq.
        The considered system at hand is: (L0 + L1*omega + L2*omega**2)*tilde{phi}. Therefore BCs are imposed on L0, since
        they must be respected for every possible value of omega.
        L1 and L2 are then filled in the respective positions with zeros.
        :param row: row index of the equation to modify
        :param condition: type of boundary condition
        """

        if condition == 'zero pressure':
            # BC for zero pressure perturbation
            self.Z[row + 4, :] = np.zeros(self.Z[row + 4, :].shape, dtype=complex)
            self.Z[row + 4, row + 4] = 1  # zero pressure at that node

            self.A[row + 4, :] = np.zeros(self.A[row + 4, :].shape, dtype=complex)  # zero row

        # elif condition == 'zero axial velocity':
        #     # BC for zero pressure perturbation
        #     self.L0[row + 3, :] = np.zeros(self.L0[row + 4, :].shape, dtype=complex)
        #     self.L0[row + 3, row + 3] = 1  # zero pressure at that node
        #
        #     self.L1[row + 3, :] = np.zeros(self.L1[row + 3, :].shape, dtype=complex)  # zero row
        #     self.L2[row + 3, :] = np.zeros(self.L2[row + 3, :].shape, dtype=complex)  # zero row

        elif condition == 'zero radial velocity':
            # BC for zero pressure perturbation
            self.L0[row, :] = np.zeros(self.L0[row, :].shape)
            self.L0[row, row] = 1  # zero radial velocity

            self.L1[row, :] = np.zeros(self.L1[row, :].shape)  # zero row
            self.L2[row, :] = np.zeros(self.L2[row, :].shape)

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
        # elif condition == 'neumann inlet':
        #     # BC for zero pressure perturbation
        #     self.L0[row:row + 5, :] = np.zeros(self.L0[row:row + 5, :].shape, dtype=complex)
        #     node = row // 5  # number of the node
        #     node_next = node + self.data.nRadialNodes  # number of the next node along the streamline
        #     row_next = node_next * 5  # equivalent row index for that next node
        #     self.L0[row:row + 5, row:row + 5] = np.eye(5, dtype=complex)
        #     self.L0[row:row + 5, row_next:row_next + 5] = -np.eye(5, dtype=complex)
        #
        #     self.L1[row:row + 5, :] = np.zeros(self.L1[row:row + 5, :].shape, dtype=complex)  # zero rows
        #     self.L2[row:row + 5, :] = np.zeros(self.L2[row:row + 5, :].shape, dtype=complex)  # zero rows
        #
        # elif condition == 'neumann outlet':
        #     # BC for zero pressure perturbation
        #     self.L0[row:row + 5, :] = np.zeros(self.L0[row:row + 5, :].shape, dtype=complex)
        #     node = row // 5  # number of the node
        #     node_previous = node - self.data.nRadialNodes  # number of the next node along the streamline
        #     row_previous = node_previous * 5  # equivalent row index for that next node
        #     self.L0[row:row + 5, row:row + 5] = np.eye(5, dtype=complex)
        #     self.L0[row:row + 5, row_previous:row_previous + 5] = -np.eye(5, dtype=complex)
        #
        #     self.L1[row:row + 5, :] = np.zeros(self.L1[row:row + 5, :].shape, dtype=complex)  # zero rows
        #     self.L2[row:row + 5, :] = np.zeros(self.L2[row:row + 5, :].shape, dtype=complex)  # zero rows
        #
        # elif condition == 'compressor inlet':
        #     # BCs are zero for every variable except the pressure at inlet
        #     self.L0[row:row + 4, :] = np.zeros(self.L0[row:row + 4, :].shape, dtype=complex)
        #     self.L0[row:row + 4, row:row + 4] = np.eye(4, dtype=complex)
        #
        #     self.L1[row:row + 4, :] = np.zeros(self.L1[row:row + 4, :].shape, dtype=complex)  # zero rows
        #     self.L2[row:row + 4, :] = np.zeros(self.L2[row:row + 4, :].shape, dtype=complex)  # zero rows
        #
        # elif condition == 'compressor outlet':
        #     # BC for zero pressure perturbation
        #     self.L0[row + 4, :] = np.zeros(self.L0[row + 4, :].shape, dtype=complex)
        #     self.L0[row + 4, row + 4] = 1  # zero pressure at that node
        #
        #     self.L1[row + 4, :] = np.zeros(self.L1[row + 4, :].shape, dtype=complex)  # zero row
        #     self.L2[row + 4, :] = np.zeros(self.L2[row + 4, :].shape, dtype=complex)  # zero row
        #
        # elif condition == 'euler wall':
        #     # BC for non-penetration condition at the walls, the equation overwritten depends on configs
        #     if self.substituted_equation == 'ur':
        #         loc = 1
        #     elif self.substituted_equation == 'utheta':
        #         loc = 2
        #     elif self.substituted_equation == 'uz':
        #         loc = 3
        #     else:
        #         raise ValueError("Subsituted equation parameter not recognized.")
        #
        #     wall_normal = self.data.dataSet[ii, jj].n_wall
        #
        #     self.L0[row + loc, :] = np.zeros(self.L0[row + loc, :].shape, dtype=complex)
        #     self.L0[row + loc, row + 1:row + 4] = wall_normal
        #
        #     self.L1[row + loc, :] = np.zeros(self.L1[row + loc, :].shape, dtype=complex)  # zero known term
        #     self.L2[row + loc, :] = np.zeros(self.L2[row + loc, :].shape, dtype=complex)  # zero known term

        else:
            raise ValueError('unknown boundary condition type')



    def instantiate_global_matrices(self):
        """
        Build the Full maitrces for the instability problem
        """
        self.A = np.zeros((self.nPoints * 5, self.nPoints * 5), dtype=complex)
        self.B = np.zeros((self.nPoints * 5, self.nPoints * 5), dtype=complex)
        self.C = np.zeros((self.nPoints * 5, self.nPoints * 5), dtype=complex)
        self.E = np.zeros((self.nPoints * 5, self.nPoints * 5), dtype=complex)
        self.R = np.zeros((self.nPoints * 5, self.nPoints * 5), dtype=complex)
