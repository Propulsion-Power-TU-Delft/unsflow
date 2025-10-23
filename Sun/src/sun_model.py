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
from .general_functions import *
from Utils.styles import *
from .eigenmode import Eigenmode
from scipy.interpolate import griddata
from scipy.interpolate import Rbf
import copy
import os
import warnings


class SunModel:
    """
    Class used for Sun Model instability prediction based on the data process from CFD.
    The perturbation equations are: (-j*omega*A + B*ddr + j*m*C/r + E*ddz + R + S)*Phi' = 0
    
    MATRICES:
        A : coefficient matrix of temporal derivatives
        B : coefficient matrix of radial derivatives
        C : coefficient matrix of circumferential derivatives
        E : coefficient matrix of axial derivatives
        R : coefficient matrix of the known mean flow terms
        S : coefficient matrix of the body force model
    """

    def __init__(self, config):
        """
        Instantiate the sun model Object, contaning all the attributes and methods necessary for the instability
        analysis.
        :param gridObject: is the object contaning all the data, physical and spectral.
        :param config: configuration file of the sun model
        """
        self.config = config
        self.blockType = self.config.GetBlockType()
        
        inputFile = self.config.GetInputFile()
        with open(inputFile, 'rb') as file:
            self.inputData = pickle.load(file)
        
        self.inputData = self.CompleteInputData(self.inputData)
        self.inputData = self.NormalizeData(self.inputData)
        
        self.grid = SunGrid(self.inputData)
        self.nStream, self.nSpan = self.inputData['AxialCoord'].shape    
        self.nPoints = (self.nStream) * (self.nSpan)
        self.gmma = self.config.get_fluid_gamma()
        self.wallEquation = self.config.get_euler_wall_equation()
    
    
    def CompleteInputData(self, data):
        """
        Complete the input data with the missing values. If a gradient term is not present, set to zero
        """
        keys = data.keys()
        
        # gradients needed in the code
        gradientFields = [
            'drho_dr', 'drho_dz',
            'dur_dr', 'dur_dz',
            'dut_dr', 'dut_dz',
            'duz_dr', 'duz_dz',
            'dp_dr', 'dp_dz'
        ]
        
        for field in gradientFields:
            if field not in keys:
                data[field] = np.zeros_like(data['AxialCoord'])
        
        return data
    
    def print_normalization_information(self):
        """
        Print information on non-dimensionalization.
        """
        self.ref_length = self.config.get_reference_length()
        self.ref_density = self.config.get_reference_density()
        self.ref_temperature = self.config.get_reference_density()
        self.ref_pressure = self.config.get_reference_pressure()
        self.ref_time = self.config.get_reference_time()
        self.ref_omega = self.config.get_reference_omega()
        
        self.harmonic_order = self.config.get_circumferential_harmonic_order()
        self.omega_ref = self.config.get_reference_omega()
        self.x_ref = self.config.get_reference_length()
        self.u_ref = self.config.get_reference_velocity()
        self.t_ref = self.config.get_reference_time()
        
        total_chars_mid = 30
        print_banner_begin('NORMALIZATION')
        print(f"{'Reference Length [m]:':<{total_chars_mid}}{self.ref_length:>{total_chars_mid}.2f}")
        print(f"{'Reference Velocity [m/s]:':<{total_chars_mid}}{self.u_ref:>{total_chars_mid}.2f}")
        print(f"{'Reference Density [kg/m3]:':<{total_chars_mid}}{self.ref_density:>{total_chars_mid}.2f}")
        print(f"{'Reference Pressure [Pa]:':<{total_chars_mid}}{self.ref_pressure:>{total_chars_mid}.2f}")
        print(f"{'Reference Time [s]:':<{total_chars_mid}}{self.ref_time:>{total_chars_mid}.6f}")
        print(f"{'Reference Omega [rad/s]:':<{total_chars_mid}}{self.ref_omega:>{total_chars_mid}.2f}")
        print_banner_end()
        
        

    def add_shaft_rpm(self, rpm):
        """
        Add the rpm of the shaft, in order to set the reference angular rate of the analysis (omega [rad/s]).
        If the shaft rotates in the negative z-direction, the reference omega is still positive, but omega shaft is
        kept with algebraic sign.
        :param rpm: rotations per minute of the shaft, with sign according to z convetion.
        """
        self.omega_shaft = 2 * np.pi * rpm / 60
        self.omega_ref = np.abs(self.omega_shaft)

    def NormalizeData(self, data):
        """
        Non-dimensionalise the node quantities, if they were not already non-dimensional
        """
        self.print_normalization_information()
        
        totalFields = len(data.keys())
        
        data['AxialCoord'] /= self.ref_length
        data['RadialCoord'] /= self.ref_length
        
        data['Density'] /= self.ref_density
        data['RadialVel'] /= self.u_ref
        data['AxialVel'] /= self.u_ref
        data['TangentialVel'] /= self.u_ref
        data['Pressure'] /= self.ref_pressure
        
        data['drho_dr'] /= (self.ref_density/self.ref_length)
        data['drho_dz'] /= (self.ref_density/self.ref_length)
        data['dur_dr'] /= (self.u_ref/self.ref_length)
        data['dur_dz'] /= (self.u_ref/self.ref_length)
        data['dut_dr'] /= (self.u_ref/self.ref_length)
        data['dut_dz'] /= (self.u_ref/self.ref_length)
        data['duz_dr'] /= (self.u_ref/self.ref_length)
        data['duz_dz'] /= (self.u_ref/self.ref_length)
        data['dp_dr'] /= (self.ref_pressure/self.ref_length)
        data['dp_dz'] /= (self.ref_pressure/self.ref_length)
        
        return data
        

        
    def ComputeSpectralGrid(self):
        """
        It instanties a new grid object which has the computational grid suitable for spectral differentiation, with
        grid poinst located on Gauss-Lobatto points.
        """
        self.gridSpectral = self.grid.PhysicalToSpectralData()

    def ShowPhysicalGrid(self, save_filename=None, mode=None):
        """
        It shows the physical grid points, with different colors for the different parts of the domain.
        :param save_filename: specify name if you want to save the figs.
        :param mode: mode used for visualization.
        """
        self.grid.ShowGrid(mode=mode)
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
        self.gridSpectral.ShowGrid(mode=mode)
        plt.title('spectral grid')
        plt.xlabel(r'$\xi \quad  [-]$')
        plt.ylabel(r'$\eta \quad  [-]$')
        if save_filename is not None:
            plt.savefig(save_filename + '.pdf', bbox_inches='tight')  # plt.close()

    def ComputeJacobianPhysical(self, dx_dz=None, dx_dr=None, dy_dz=None, dy_dr=None):
        """
        It computes the transformation gradients for every grid point, and stores the values in the Nodes. x and y replace
        the xi and eta coordinates of the computational domain.
        It computes the derivatives on the spectral grid since it is the only one cartesian, and the inverse transformation is
        found by inversion (usgin the Jacobian).
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
            Z = self.inputData['AxialCoord']
            R = self.inputData['RadialCoord']
            Nz_fine = np.shape(Z)[0]
            Nr_fine = np.shape(Z)[1]
            x = GaussLobattoPoints(Nz_fine)
            y = GaussLobattoPoints(Nr_fine)
            Y, X = np.meshgrid(y, x)

            if routine == 'numpy':
                dzdx, dzdy, drdx, drdy = JacobianTransform_numpy(Z, R, X, Y)
            elif routine == 'hard-coded':
                print('\nWARNING: you are using hard-coded version of numerical differentiation for the Jacobian of the grid.'
                      'Consider passing to numpy or findiff!\n')
                dzdx, dzdy, drdx, drdy = JacobianTransform_hardcoded(Z, R, X, Y)
            elif routine == 'findiff':
                dzdx, dzdy, drdx, drdy = JacobianTransform_findiff(Z, R, X, Y, order=order)
            else:
                raise ValueError('Unknown method for transformation gradient computation!')

            self.dzdx, self.dzdy, self.drdx, self.drdy = dzdx, dzdy, drdx, drdy
            self.J = self.dzdx * self.drdy - self.dzdy * self.drdx

            # compute also the inverse relations, for validation purposes
            self.dxdz = (1 / self.J) * (self.drdy)
            self.dxdr = (1 / self.J) * (-self.dzdy)
            self.dydz = (1 / self.J) * (-self.drdx)
            self.dydr = (1 / self.J) * (self.dzdx)
        else:
            if dx_dz is not None:
                self.dxdz = dx_dz
                self.dzdx = 1 / dx_dz
            else:
                self.dxdz = np.zeros((self.grid.nStream, self.grid.nSpan))
                self.dzdx = np.zeros((self.grid.nStream, self.grid.nSpan))
            if dx_dr is not None:
                self.dxdr = dx_dr
                self.drdx = 1 / dx_dr
            else:
                self.dxdr = np.zeros((self.grid.nStream, self.grid.nSpan))
                self.drdx = np.zeros((self.grid.nStream, self.grid.nSpan))
            if dy_dz is not None:
                self.dydz = dy_dz
                self.dydz = 1 / dy_dz
            else:
                self.dydz = np.zeros((self.grid.nStream, self.grid.nSpan))
                self.dzdy = np.zeros((self.grid.nStream, self.grid.nSpan))
            if dy_dr is not None:
                self.dydr = dy_dr
                self.drdy = 1 / dy_dr
            else:
                self.dydr = np.zeros((self.grid.nStream, self.grid.nSpan))
                self.drdy = np.zeros((self.grid.nStream, self.grid.nSpan))
            self.J = self.dzdx * self.drdy - self.dzdy * self.drdx

        for ii in range(0, self.nStream):
            for jj in range(0, self.grid.nSpan):
                # add the inverse gradients information to every node
                self.grid.dataSet[ii, jj].AddTransformationGradients(self.dzdx[ii, jj], self.dzdy[ii, jj], self.drdx[ii, jj],
                                                                     self.drdy[ii, jj])
                self.grid.dataSet[ii, jj].AddJacobian(self.J[ii, jj])

    def ContourTransformation(self, save_filename=None, folder_name=None, domain='physical'):
        """
        Show the gradient contours.
        :param save_filename: specify the names if you want to save the figs.
        :param folder_name: folder name
        :param domain: physical, spectral, or all. Decide which gradients to plot.
        """
        if domain not in ['physical', 'spectral', 'all']:
            raise ValueError('Unknown domain parameter!')

        if domain == 'spectral' or domain == 'all':
            plt.figure()
            plt.contourf(self.gridSpectral.zGrid, self.gridSpectral.rGrid, self.J, levels=N_levels, cmap=color_map)
            plt.xlabel(r'$\xi \ \mathrm{[-]}$')
            plt.ylabel(r'$\eta \ \mathrm{[-]}$')
            plt.title(r'$J$')
            plt.colorbar()
            if save_filename is not None:
                plt.savefig(folder_name + '/' + save_filename + '_J_xi_eta.pdf', bbox_inches='tight')

            plt.figure()
            plt.contourf(self.gridSpectral.zGrid, self.gridSpectral.rGrid, self.dzdx, levels=N_levels, cmap=color_map)
            plt.xlabel(r'$\xi \ \mathrm{[-]}$')
            plt.ylabel(r'$\eta \ \mathrm{[-]}$')
            plt.title(r'$\frac{\partial \hat{z}}{\partial \xi}$')
            plt.colorbar()
            if save_filename is not None:
                plt.savefig(folder_name + '/' + save_filename + '_dz_dxi.pdf', bbox_inches='tight')

            plt.figure()
            plt.contourf(self.gridSpectral.zGrid, self.gridSpectral.rGrid, self.dzdy, levels=N_levels, cmap=color_map)
            plt.xlabel(r'$\xi \ \mathrm{[-]}$')
            plt.ylabel(r'$\eta \ \mathrm{[-]}$')
            plt.colorbar()
            plt.title(r'$\frac{\partial \hat{z}}{\partial \eta}$')
            if save_filename is not None:
                plt.savefig(folder_name + '/' + save_filename + '_dz_deta.pdf', bbox_inches='tight')

            plt.figure()
            plt.contourf(self.gridSpectral.zGrid, self.gridSpectral.rGrid, self.drdx, levels=N_levels, cmap=color_map)
            plt.xlabel(r'$\xi \ \mathrm{[-]}$')
            plt.ylabel(r'$\eta \ \mathrm{[-]}$')
            plt.colorbar()
            plt.title(r'$\frac{\partial \hat{r}}{\partial \xi}$')
            if save_filename is not None:
                plt.savefig(folder_name + '/' + save_filename + '_dr_dxi.pdf', bbox_inches='tight')

            plt.figure()
            plt.contourf(self.gridSpectral.zGrid, self.gridSpectral.rGrid, self.drdy, levels=N_levels, cmap=color_map)
            plt.xlabel(r'$\xi \ \mathrm{[-]}$')
            plt.ylabel(r'$\eta \ \mathrm{[-]}$')
            plt.colorbar()
            plt.title(r'$\frac{\partial \hat{r}}{\partial \eta}$')
            if save_filename is not None:
                plt.savefig(folder_name + '/' + save_filename + '_dr_deta.pdf', bbox_inches='tight')

        if domain == 'physical' or domain == 'all':
            plt.figure()
            plt.contourf(self.grid.zGrid, self.grid.rGrid, self.dxdr, levels=N_levels, cmap=color_map)
            plt.xlabel(r'$z \ \mathrm{[-]}$')
            plt.ylabel(r'$r \ \mathrm{[-]}$')
            plt.colorbar()
            plt.title(r'$\frac{\partial \xi}{\partial \hat{r}}$')
            plt.gca().set_aspect('equal', adjustable='box')
            if save_filename is not None:
                plt.savefig(folder_name + '/' + save_filename + '_dxi_dr.pdf', bbox_inches='tight')  # plt.close()

            plt.figure()
            plt.contourf(self.grid.zGrid, self.grid.rGrid, self.dxdz, levels=N_levels, cmap=color_map)
            plt.xlabel(r'$z \ \mathrm{[-]}$')
            plt.ylabel(r'$r \ \mathrm{[-]}$')
            plt.colorbar()
            plt.title(r'$\frac{\partial \xi}{\partial \hat{z}}$')
            plt.gca().set_aspect('equal', adjustable='box')
            if save_filename is not None:
                plt.savefig(folder_name + '/' + save_filename + '_dxi_dz.pdf', bbox_inches='tight')  # plt.close()

            plt.figure()
            plt.contourf(self.grid.zGrid, self.grid.rGrid, self.dydr, levels=N_levels, cmap=color_map)
            plt.xlabel(r'$z \ \mathrm{[-]}$')
            plt.ylabel(r'$r \ \mathrm{[-]}$')
            plt.colorbar()
            plt.title(r'$\frac{\partial \eta}{\partial \hat{r}}$')
            plt.gca().set_aspect('equal', adjustable='box')
            if save_filename is not None:
                plt.savefig(folder_name + '/' + save_filename + '_deta_dr.pdf', bbox_inches='tight')  # plt.close()

            plt.figure()
            plt.contourf(self.grid.zGrid, self.grid.rGrid, self.dydz, levels=N_levels, cmap=color_map)
            plt.xlabel(r'$z \ \mathrm{[-]}$')
            plt.ylabel(r'$r \ \mathrm{[-]}$')
            plt.colorbar()
            plt.title(r'$\frac{\partial \eta}{\partial \hat{z}}$')
            plt.gca().set_aspect('equal', adjustable='box')
            if save_filename is not None:
                plt.savefig(folder_name + '/' + save_filename + '_deta_dz.pdf', bbox_inches='tight')  # plt.close()

    def contour_grid_mapping(self, save_filename=None, folder_name=None):
        """
        Show the grid transformation law z(xi, eta), and r(xi, eta).
        :param save_filename: specify the names if you want to save the figs.
        :param folder_name: folder name
        """
        plt.figure()
        plt.contourf(self.gridSpectral.zGrid, self.gridSpectral.rGrid, self.grid.zGrid, levels=N_levels, cmap=color_map)
        plt.xlabel(r'$\xi \ \mathrm{[-]}$')
        plt.ylabel(r'$\eta \ \mathrm{[-]}$')
        plt.title(r'$z(\xi, \eta)$')
        plt.colorbar()
        if save_filename is not None:
            plt.savefig(folder_name + '/' + save_filename + '_z_xi_eta.pdf', bbox_inches='tight')

        plt.figure()
        plt.contourf(self.gridSpectral.zGrid, self.gridSpectral.rGrid, self.grid.rGrid, levels=N_levels, cmap=color_map)
        plt.xlabel(r'$\xi \ \mathrm{[-]}$')
        plt.ylabel(r'$\eta \ \mathrm{[-]}$')
        plt.title(r'$r(\xi, \eta)$')
        plt.colorbar()
        if save_filename is not None:
            plt.savefig(folder_name + '/' + save_filename + '_r_xi_eta.pdf', bbox_inches='tight')

    def AddAMatrixToNodes_sun(self):
        """
        Compute and store at the node level the A matrix. Sun Formulation
        """
        for ii in range(0, self.grid.nStream):
            for jj in range(0, self.grid.nSpan):
                A = np.eye(5, dtype=complex)

                # if data was already non-dimensional, multiply only matrix A times the strouhal number. If the reference
                # velocity was found as u_ref = omega_ref * x_ref and t_ref = 1 / omega_ref, automatically the strouhal
                # should be 1 by construction. In this case the non-dimensional equations are exactly the same
                # of the dimensional ones
                strouhal = self.config.get_reference_length() / (
                            self.config.get_reference_velocity() * self.config.get_reference_time())
                A *= strouhal
                self.grid.dataSet[ii, jj].AddAMatrix(A)

    def AddAMatrixToNodes_francesco(self):
        """
        Compute and store at the node level the A matrix. My Formulation.
        """
        for ii in range(0, self.grid.nStream):
            for jj in range(0, self.grid.nSpan):
                A = np.eye(5, dtype=complex)
                A[4, 0] = -self.inputData['Pressure'][ii, jj] * self.gmma / self.inputData['Density'][ii, jj]

                """ Multiply only matrix A times the strouhal number. If the reference velocity was defined as:
                 u_ref = omega_ref * x_ref and t_ref = 1 / omega_ref, automatically the strouhal
                should be 1 by construction. In this case the non-dimensional equations are exactly the same
                of the dimensional ones."""
                strouhal = self.config.get_reference_length() / (
                        self.config.get_reference_velocity() * self.config.get_reference_time())
                A *= strouhal
                self.grid.dataSet[ii, jj].AddAMatrix(A)

    def AddBMatrixToNodes_sun(self):
        """
        Compute and store at the node level the B matrix, needed to compute hat{B} later. Sun Formulation.
        """
        for ii in range(0, self.grid.nStream):
            for jj in range(0, self.grid.nSpan):
                B = np.zeros((5, 5), dtype=complex)
                B[0, 0] = self.inputData['RadialVel'][ii, jj]
                B[1, 1] = self.inputData['RadialVel'][ii, jj]
                B[2, 2] = self.inputData['RadialVel'][ii, jj]
                B[3, 3] = self.inputData['RadialVel'][ii, jj]
                B[4, 4] = self.inputData['RadialVel'][ii, jj]
                B[0, 1] = self.inputData['Density'][ii, jj]
                B[1, 4] = 1 / self.inputData['Density'][ii, jj]
                B[4, 1] = self.inputData['Pressure'][ii, jj] * self.gmma
                self.grid.dataSet[ii, jj].AddBMatrix(B)

    def AddBMatrixToNodes_francesco(self):
        """
        Compute and store at the node level the B matrix, needed to compute hat{B} later. My Formulation.
        """
        for ii in range(0, self.grid.nStream):
            for jj in range(0, self.grid.nSpan):
                B = np.eye(5, dtype=complex) * self.inputData['RadialVel'][ii, jj]
                B[0, 1] = self.inputData['Density'][ii, jj]
                B[1, 4] = 1 / self.inputData['Density'][ii, jj]
                B[4, 0] = - self.inputData['Pressure'][ii, jj] * self.inputData['RadialVel'][ii, jj] * self.gmma / \
                          self.inputData['Density'][ii, jj]
                self.grid.dataSet[ii, jj].AddBMatrix(B)

    def AddCMatrixToNodes_sun(self):
        """
        Compute and store at node level the C matrix, already multiplied by j*m/r. Ready to be used in the final system of eqs.
        Sun Formulation.
        """
        m = self.config.get_circumferential_harmonic_order()
        print(f"Circumferential Harmonic Order set to: {m}")

        for ii in range(0, self.grid.nStream):
            for jj in range(0, self.grid.nSpan):
                C = np.zeros((5, 5), dtype=complex)
                C[0, 0] = self.inputData['TangentialVel'][ii, jj]
                C[1, 1] = self.inputData['TangentialVel'][ii, jj]
                C[2, 2] = self.inputData['TangentialVel'][ii, jj]
                C[3, 3] = self.inputData['TangentialVel'][ii, jj]
                C[4, 4] = self.inputData['TangentialVel'][ii, jj]
                C[0, 2] = self.inputData['Density'][ii, jj]
                C[2, 4] = 1 / self.inputData['Density'][ii, jj]
                C[4, 2] = self.inputData['Pressure'][ii, jj] * self.gmma

                C = C * 1j * m / self.inputData['RadialCoord'][ii, jj]
                self.grid.dataSet[ii, jj].AddCMatrix(C)

    def AddCMatrixToNodes_francesco(self):
        """
        Compute and store at node level the C matrix, already multiplied by j*m/r. Ready to be used in the final system of eqs.
        My Formulation.
        """
        m = self.config.get_circumferential_harmonic_order()
        print(f"Circumferential Harmonic Order set to: {m}")

        for ii in range(0, self.grid.nStream):
            for jj in range(0, self.grid.nSpan):
                C = np.eye(5, dtype=complex) * self.inputData['TangentialVel'][ii, jj]
                C[0, 2] = self.inputData['Density'][ii, jj]
                C[2, 4] = 1 / self.inputData['Density'][ii, jj]
                C[4, 0] = -self.inputData['Pressure'][ii, jj] * self.inputData['TangentialVel'][ii, jj] * self.gmma / \
                          self.inputData['Density'][ii, jj]
                C *= 1j * m / self.inputData['RadialCoord'][ii, jj]
                self.grid.dataSet[ii, jj].AddCMatrix(C)

    def AddEMatrixToNodes_sun(self):
        """
        Compute and store at the node level the E matrix, needed to compute hat{E}. Sun Formulation.
        """
        for ii in range(0, self.grid.nStream):
            for jj in range(0, self.grid.nSpan):
                E = np.zeros((5, 5), dtype=complex)

                E[0, 0] = self.inputData['AxialVel'][ii, jj]
                E[1, 1] = self.inputData['AxialVel'][ii, jj]
                E[2, 2] = self.inputData['AxialVel'][ii, jj]
                E[3, 3] = self.inputData['AxialVel'][ii, jj]
                E[4, 4] = self.inputData['AxialVel'][ii, jj]

                E[0, 3] = self.inputData['Density'][ii, jj]
                E[3, 4] = 1 / self.inputData['Density'][ii, jj]
                E[4, 3] = self.inputData['Pressure'][ii, jj] * self.gmma

                self.grid.dataSet[ii, jj].AddEMatrix(E)

    def AddEMatrixToNodes_francesco(self):
        """
        Compute and store at the node level the E matrix, needed to compute hat{E}. My Formulation.
        """
        for ii in range(0, self.grid.nStream):
            for jj in range(0, self.grid.nSpan):
                E = np.eye(5, dtype=complex) * self.inputData['AxialVel'][ii, jj]
                E[0, 3] = self.inputData['Density'][ii, jj]
                E[3, 4] = 1 / self.inputData['Density'][ii, jj]
                E[4, 0] = -self.inputData['Pressure'][ii, jj] * self.inputData['AxialVel'][ii, jj] * self.gmma / \
                          self.inputData['Density'][ii, jj]
                self.grid.dataSet[ii, jj].AddEMatrix(E)

    def AddRMatrixToNodes_sun(self):
        """
        Compute and store at the node level the R matrix, ready to be used in the final system of eqs. Sun Formulation.
        """
        for ii in range(0, self.grid.nStream):
            for jj in range(0, self.grid.nSpan):
                R = np.zeros((5, 5), dtype=complex)
                R[0, 0] = self.inputData['dur_dr'][ii, jj] + self.inputData['duz_dz'][ii, jj] + (
                        self.inputData['RadialVel'][ii, jj] / self.inputData['RadialCoord'][ii, jj])
                R[0, 1] = (self.inputData['Density'][ii, jj] / self.inputData['RadialCoord'][ii, jj] +
                           self.inputData['drho_dr'][ii, jj])
                R[0, 2] = 0
                R[0, 3] = self.inputData['drho_dz'][ii, jj]
                R[0, 4] = 0
                R[1, 0] = -self.inputData['dp_dr'][ii, jj] / self.inputData['Density'][ii, jj] ** 2
                R[1, 1] = self.inputData['dur_dr'][ii, jj]
                R[1, 2] = -2 * self.inputData['TangentialVel'][ii, jj] / self.inputData['RadialCoord'][ii, jj]
                R[1, 3] = self.inputData['dur_dz'][ii, jj]
                R[1, 4] = 0
                R[2, 0] = 0
                R[2, 1] = (self.inputData['dut_dr'][ii, jj] + self.inputData['TangentialVel'][ii, jj] /
                           self.inputData['RadialCoord'][ii, jj])
                R[2, 2] = self.inputData['RadialVel'][ii, jj] / self.inputData['RadialCoord'][ii, jj]
                R[2, 3] = self.inputData['dut_dz'][ii, jj]
                R[2, 4] = 0
                R[3, 0] = -self.inputData['dp_dz'][ii, jj] / self.inputData['Density'][ii, jj] ** 2
                R[3, 1] = self.inputData['duz_dr'][ii, jj]
                R[3, 2] = 0
                R[3, 3] = self.inputData['duz_dz'][ii, jj]
                R[3, 4] = 0
                R[4, 0] = 0
                R[4, 1] = (self.inputData['Pressure'][ii, jj] * self.gmma / self.inputData['RadialCoord'][ii, jj] +
                           self.inputData['dp_dr'][ii, jj])
                R[4, 2] = 0
                R[4, 3] = self.inputData['dp_dz'][ii, jj]
                R[4, 4] = self.gmma * (self.inputData['duz_dz'][ii, jj] + self.inputData['dur_dr'][ii, jj] +
                                       self.inputData['RadialVel'][ii, jj] / self.inputData['RadialCoord'][ii, jj])
                self.grid.dataSet[ii, jj].AddRMatrix(R)

    def AddRMatrixToNodes_francesco(self):
        """
        Compute and store at the node level the R matrix.
        My formulation, last version.
        """
        for ii in range(0, self.grid.nStream):
            for jj in range(0, self.grid.nSpan):
                R = np.zeros((5, 5), dtype=complex)

                R[0, 0] = self.inputData['dur_dr'][ii, jj] + self.inputData['duz_dz'][ii, jj] + \
                          self.inputData['RadialVel'][ii, jj] / self.inputData['RadialCoord'][ii, jj]
                R[0, 1] = self.inputData['Density'][ii, jj] / self.inputData['RadialCoord'][ii, jj] + \
                          self.inputData['drho_dr'][ii, jj]
                R[0, 3] = self.inputData['drho_dz'][ii, jj]

                R[1, 0] = self.inputData['dp_dr'][ii, jj] / (self.inputData['Density'][ii, jj] ** 2)
                R[1, 1] = self.inputData['dur_dr'][ii, jj]
                R[1, 2] = -2 * self.inputData['TangentialVel'][ii, jj] / self.inputData['RadialCoord'][ii, jj]
                R[1, 3] = self.inputData['dur_dz'][ii, jj]

                R[2, 1] = self.inputData['dut_dr'][ii, jj] + self.inputData['TangentialVel'][ii, jj] / \
                          self.inputData['RadialCoord'][ii, jj]
                R[2, 2] = self.inputData['RadialVel'][ii, jj] / self.inputData['RadialCoord'][ii, jj]
                R[2, 3] = self.inputData['dut_dz'][ii, jj]

                R[3, 0] = self.inputData['dp_dz'][ii, jj] / (self.inputData['Density'][ii, jj] ** 2)
                R[3, 1] = self.inputData['duz_dr'][ii, jj]
                R[3, 3] = self.inputData['duz_dz'][ii, jj]

                # R[4, 0] = (1 / node.rho) * (node.ur * node.dp_dr + node.uz * node.dp_dz) # first version
                R[4, 0] = -self.gmma / (self.inputData['Density'][ii, jj] ** 2) * (
                        self.inputData['RadialVel'][ii, jj] * self.inputData['Pressure'][ii, jj] *
                        self.inputData['drho_dr'][ii, jj] + self.inputData['AxialVel'][ii, jj] *
                        self.inputData['Pressure'][ii, jj] * self.inputData['drho_dz'][ii, jj])  # last version
                R[4, 1] = self.inputData['dp_dr'][ii, jj] - self.inputData['Pressure'][ii, jj] * \
                          self.inputData['drho_dr'][ii, jj] * self.gmma / self.inputData['Density'][ii, jj]
                R[4, 3] = self.inputData['dp_dz'][ii, jj] - self.gmma / self.inputData['Density'][ii, jj] * \
                          self.inputData['Pressure'][ii, jj] * self.inputData['drho_dz'][ii, jj]
                R[4, 4] = (-self.gmma / self.inputData['Density'][ii, jj]) * (
                        self.inputData['RadialVel'][ii, jj] * self.inputData['drho_dr'][ii, jj] +
                        self.inputData['AxialVel'][ii, jj] * self.inputData['drho_dz'][ii, jj])

                self.grid.dataSet[ii, jj].AddRMatrix(R)

    def AddSMatrixToNodes(self):
        """
        compute and store at the node level the S matrix, ready to be used in the final system of eqs. The matrix formulation
        depends on the selected body-force model.
        """
        for ii in range(0, self.grid.nStream):
            for jj in range(0, self.grid.nSpan):
                S = np.zeros((5, 5), dtype=complex)
                if self.blockType == 'rotor' or self.blockType == 'stator':
                    S[1, 1] = self.grid.meridional_obj.S11[ii, jj]
                    S[1, 2] = self.grid.meridional_obj.S12[ii, jj]
                    S[1, 3] = self.grid.meridional_obj.S13[ii, jj]
                    S[2, 1] = self.grid.meridional_obj.S21[ii, jj]
                    S[2, 2] = self.grid.meridional_obj.S22[ii, jj]
                    S[2, 3] = self.grid.meridional_obj.S23[ii, jj]
                    S[3, 1] = self.grid.meridional_obj.S31[ii, jj]
                    S[3, 2] = self.grid.meridional_obj.S32[ii, jj]
                    S[3, 3] = self.grid.meridional_obj.S33[ii, jj]

                    if self.blockType == 'rotor':
                        S[4, 1] = self.grid.meridional_obj.S41[ii, jj]
                        S[4, 2] = self.grid.meridional_obj.S42[ii, jj]
                        S[4, 3] = self.grid.meridional_obj.S43[ii, jj]
                else:
                    pass
                S *= -1  # change sign, because the BFM term are brought to the left hand-side of the equation.
                
                self.grid.dataSet[ii, jj].AddSMatrix(S)

    def AddHatMatricesToNodes(self):
        """
        Compute and store at the node level the hat{B}, hat{E} matrix, to express the problem in the computational domain.
        """
        for ii in range(0, self.grid.nStream):
            for jj in range(0, self.grid.nSpan):
                Bhat = (1 / self.J[ii, jj]) * (
                        -self.grid.dataSet[ii, jj].B * self.dzdy[ii, jj] + self.grid.dataSet[ii, jj].E * self.drdy[ii, jj])
                Ehat = (1 / self.J[ii, jj]) * (
                        self.grid.dataSet[ii, jj].B * self.dzdx[ii, jj] - self.grid.dataSet[ii, jj].E * self.drdx[ii, jj])
                # # # alternative formulation, provides the same results. (Check)
                # Bhat2 = self.grid.dataSet[ii, jj].B * self.dxdr[ii, jj] + \
                #        self.grid.dataSet[ii, jj].E * self.dxdz[ii, jj]
                # Ehat2 = self.grid.dataSet[ii, jj].B * self.dydr[ii, jj] + \
                #        self.grid.dataSet[ii, jj].E * self.dydz[ii, jj]
                self.grid.dataSet[ii, jj].AddHatMatrices(Bhat, Ehat)

    def ApplySpectralDifferentiation(self, verbose=False):
        """
        This method applies Chebyshev-Gauss-Lobatto differentiation method to hat{B},hat{E}, to express the perturbation
        derivatives as a function of the perturbation at the other nodes. It saves a new global matrix Q_const,
        which is part of the global stability matrix. The full dimension is: (nPoints*5, nPoints*5).
        The spectral differentiation formula, serial version, has been double-checked.
        :param verbose: print additional info.
        """

        # compute the spectral Matrices for x and y direction with the Bayliss formulation
        Dx = ChebyshevDerivativeMatrixBayliss(self.gridSpectral.z)  # derivative operator in xi
        Dy = ChebyshevDerivativeMatrixBayliss(self.gridSpectral.r)  # derivative operator in eta

        # Q_const is the global matrix storing B and E elements after spectral differentiation.
        self.Q_const = np.zeros((self.nPoints * 5, self.nPoints * 5), dtype=complex)

        # differentiation of a general perturbation vector (for the node (i,j)) along xi  and eta.
        for ii in range(0, self.gridSpectral.nStream):
            for jj in range(0, self.gridSpectral.nSpan):
                B_ij = self.grid.dataSet[ii, jj].Bhat.copy()  # Bhat matrix of the ij node
                E_ij = self.grid.dataSet[ii, jj].Ehat.copy()  # Ehat matrix of the ij node
                node_counter = self.grid.dataSet[ii, jj].nodeCounter

                # xi differentiation. m is in the range of axial nodes, first axis of the matrix
                for m in range(0, self.gridSpectral.nStream):
                    tmp = Dx[ii, m] * B_ij  # 5x5 matrix to be added to a certain block of Q
                    row = node_counter * 5  # this selects the correct block along i of Q
                    column = self.grid.dataSet[m, jj].nodeCounter * 5  # it selects the correct block along j of Q

                    if verbose:
                        print('Node [i,j] = (%i,%i)' % (ii, jj))
                        print('Element along i [m,j] = (%i,%i)' % (m, jj))
                        print('Derivative element [ii,m] = (%i,%i)' % (ii, m))
                        print('[row,col] = (%i,%i)' % (row, column))
                    self.AddToQ_const(tmp, row, column)

                # eta differentiation. n is in the range of radial nodes, second axis of the matrix
                for n in range(0, self.gridSpectral.nSpan):
                    tmp = Dy[jj, n] * E_ij
                    row = node_counter * 5
                    column = self.grid.dataSet[ii, n].nodeCounter * 5  # it selects the correct block along j of Q
                    if verbose:
                        print('Node [i,j] = (%.1d,%.1d)' % (ii, jj))
                        print('Element along j [i,n] = (%.1d,%.1d)' % (jj, n))
                        print('Derivative element [jj,n] = (%.1d,%.1d)' % (jj, n))
                        print('[row,col] = (%.1d,%.1d)' % (row, column))
                    self.AddToQ_const(tmp, row, column)

    def ApplyPhysicalDifferentiation(self, diff_mode='2nd_central', verbose=False):
        """
        Differentiate on the physical grid the perturbation equations in the axial and radial direction.
        :param diff_mode: differentiation mode used
        :param verbose: verbosity
        """
        # Q_const is the global matrix storing B and E elements after spectral differentiation.
        self.Q_const = np.zeros((self.nPoints * 5, self.nPoints * 5), dtype=complex)

        # differentiation of a general perturbation vector (for the node (i,j)) along z  and r.
        for ii in range(0, self.grid.nStream):
            for jj in range(0, self.grid.nSpan):
                B_ij = self.grid.dataSet[ii, jj].Bhat.copy()  # Bhat matrix of the ij node
                E_ij = self.grid.dataSet[ii, jj].Ehat.copy()  # Ehat matrix of the ij node
                node_counter = self.grid.dataSet[ii, jj].nodeCounter

                if diff_mode == '2nd_central':
                    if ii == 0:
                        uw = 0
                        dw = 1
                    elif ii == self.grid.nStream - 1:
                        uw = -1
                        dw = 0
                    else:
                        uw = 1
                        dw = -1
                elif diff_mode == '1st_upwind':
                    raise ValueError('Not implemented yet')
                else:
                    raise ValueError('Not implemented yet')
                tmp = B_ij / (self.gridSpectral.zGrid[ii + dw, jj] - self.gridSpectral.zGrid[ii + uw, jj])
                self.AddToQ_const(-tmp, node_counter * 5, ((ii + uw) * self.grid.nSpan + jj) * 5)
                self.AddToQ_const(+tmp, node_counter * 5, ((ii + dw) * self.grid.nSpan + jj) * 5)

                if diff_mode == '2nd_central':
                    if jj == 0:
                        uw = 0
                        dw = 1
                    elif jj == self.grid.nSpan - 1:
                        uw = -1
                        dw = 0
                    else:
                        uw = 1
                        dw = -1
                elif diff_mode == '1st_upwind':
                    raise ValueError('Not implemented yet')
                else:
                    raise ValueError('Not implemented yet')
                tmp = E_ij / (self.gridSpectral.rGrid[ii, jj + dw] - self.gridSpectral.rGrid[ii, jj + uw])
                self.AddToQ_const(-tmp, node_counter * 5, (ii * self.grid.nSpan + (jj + uw)) * 5)
                self.AddToQ_const(+tmp, node_counter * 5, (ii * self.grid.nSpan + (jj + dw)) * 5)


    def ApplySpectralDifferentiationKronecker(self):
        """
        This method applies Chebyshev-Gauss-Lobatto differentiation method to hat{B},hat{E}, to express the perturbation
        derivatives as a function of the perturbation at the other nodes. It saves a new global (for all the nodes) matrix Q_const,
        which is part of the global stability matrix. The full dimension is: (nPoints*5,nPoints*5).
        The spectral differentiation formula use Kronecker product implementation, and has been taken from -Spectral Methods
         in Matlab, Trefethen-.
        """

        # Differential operators in 1D
        Dx = ChebyshevDerivativeMatrixBayliss(self.gridSpectral.z)  # derivative operator in xi
        Dy = ChebyshevDerivativeMatrixBayliss(self.gridSpectral.r)  # derivative operator in eta

        # Corresponding operators extended to 2D thanks to Kronecker products.
        Ix = np.eye(self.grid.nStream)
        Iy = np.eye(self.grid.nSpan)
        Dx_2d = np.kron(Dx, Iy)
        Dy_2d = np.kron(Ix, Dy)

        # Q_const is the global matrix storing B and E elements after spectral differentiation.
        self.Q_const = np.zeros((self.nPoints * 5, self.nPoints * 5), dtype=complex)

        for ii in range(Dx_2d.shape[0]):
            for jj in range(Dx_2d.shape[1]):
                # indexes on the Q_const matrix. (Times 5 because there are 5 equations per node)
                row = ii * 5
                col = jj * 5

                # indexes of the node in the 2D grid. jj works as the absolute node counter, going from 0 to nx*ny
                inode = jj // self.grid.nSpan
                jnode = jj % self.grid.nSpan

                B = self.grid.dataSet[inode, jnode].Bhat.copy()
                E = self.grid.dataSet[inode, jnode].Ehat.copy()
                tmp = B * Dx_2d[ii, jj] + E * Dy_2d[ii, jj]

                self.AddToQ_const(tmp, row, col)

    def apply_finite_differences_on_physical_grid(self):
        """
        This method differentiates directly the problem on the physical grid. It saves a new global (for all the nodes) matrix
        Q_const, which is part of the global stability matrix. The full dimension is: (nPoints*5,nPoints*5).
        The method should only be used for those cases having a cartesian physical grid, otherwise is wrong.
        """
        warnings.warn("You are using a not validated method, which works only for"
                      "Cartesian physical grids. Consider passing to spectral method")

        # the cordinates on the spectral grid directions, necessary for the differentiation matrices Dx and Dy
        z = self.grid.zGrid[:, 0]
        dz = z[1] - z[0]
        r = self.grid.rGrid[0, :]
        dr = r[1] - r[0]

        # Q_const is the global matrix storing B and E elements after finite differentiation.
        self.Q_const = np.zeros((self.nPoints * 5, self.nPoints * 5), dtype=complex)

        # differentiation of a general perturbation vector (for the node (i,j)) along xi  and eta. (formula double-checked)
        for ii in range(0, self.gridSpectral.nStream):
            for jj in range(0, self.gridSpectral.nSpan):
                B_ij = self.grid.dataSet[ii, jj].B  # B matrix of the ij node
                E_ij = self.grid.dataSet[ii, jj].E  # E matrix of the ij node

                if ii == 0 and jj == 0:  # bottom left corner
                    ip = 1
                    im = 0
                    deltaz = 1
                    jp = 1
                    jm = 0
                    deltar = 1
                elif ii == 0 and jj == self.grid.nSpan - 1:  # top left corner
                    ip = 1
                    im = 0
                    deltaz = 1
                    jp = 0
                    jm = -1
                    deltar = 1
                elif ii == self.grid.nStream - 1 and jj == 0:  # bottom right corner
                    ip = 0
                    im = -1
                    deltaz = 1
                    jp = 1
                    jm = 0
                    deltar = 1
                elif ii == self.grid.nStream - 1 and jj == self.grid.nSpan - 1:  # top right corner
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
                elif ii == self.grid.nStream - 1:  # right internal border
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
                elif jj == self.grid.nSpan - 1:  # top internal border
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
                row = self.grid.dataSet[ii, jj].nodeCounter * 5
                colp = self.grid.dataSet[ii, jj + jp].nodeCounter * 5
                colm = self.grid.dataSet[ii, jj + jm].nodeCounter * 5
                tmp = B_ij / (deltar * dr)
                self.AddToQ_const(tmp, row, colp)
                self.AddToQ_const(-tmp, row, colm)

                # E differentiation
                row = self.grid.dataSet[ii, jj].nodeCounter * 5
                colp = self.grid.dataSet[ii + ip, jj].nodeCounter * 5
                colm = self.grid.dataSet[ii + im, jj].nodeCounter * 5
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

    def ComputeSVD(self, omega_domain=None, grid_omega=None):
        """
        Compute the SVD for every omega in omega_domain, discretized as in grid_omega. It computes
        every time the part of Q that depends on omega (-j*omega*A), and computes the boundary conditions.
        Then it computes the singular values, and store the inverse of the condition number in the chi attribute. Singular values 
        max and min are also stored for code test.
        :param omega_domain: omega domain of research.
        :param grid_omega: discretization of the domain of research.
        """
        warnings.warn("WARNING: deprecated method, don't use it.")
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

    def ComputeSVD2(self, omega_domain=None, grid_omega=None):
        """
        Compute the SVD for every omega in omega_domain, discretized as in grid_omega. It computes
        every time the part of Q that depends on omega (-j*omega*A), and computes the boundary conditions.
        Then it computes the singular values, and store the inverse of the condition number in the chi attribute. Singular values
        max and min are also stored for code test.
        :param omega_domain: omega domain of research.
        :param grid_omega: discretization of the domain of research.
        """
        warnings.warn("WARNING: deprecated method, don't use it.")
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
        warnings.warn("WARNING: deprecated method, don't use it.")
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

    def FindLocalMinima(self, field):
        """
        Locate the local minima of a 2D array.
        :param field: 2D array
        """
        warnings.warn("WARNING: deprecated method, don't use it.")

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

    def ComputeBoundaryNormals(self):
        """
        Compute the normal vectors on the edges of the domain.
        """
        self.grid.ComputeBoundaryNormals()

    def ShowNormals(self):
        """
       Plots the boundary nodes and the normals
       """
        self.grid.ShowNormals()

    def build_A_global_matrix(self):
        """
        Build the A global matrix, stacking together the A matrices of all the nodes.
        """
        self.A_g = np.zeros(self.Q_const.shape, dtype=complex)
        for ii in range(0, self.grid.nStream):
            for jj in range(0, self.grid.nSpan):
                node_counter = self.grid.dataSet[ii, jj].nodeCounter
                row = node_counter * 5
                column = node_counter * 5  # diagonal block
                self.add_to_A_g(self.grid.dataSet[ii, jj].A, row, column)

    def build_C_global_matrix(self):
        """
        Build the C*j*m/r global matrix, stacking together the C*j*m/r matrices of all the nodes.
        """
        self.C_g = np.zeros(self.Q_const.shape, dtype=complex)
        for ii in range(0, self.grid.nStream):
            for jj in range(0, self.grid.nSpan):
                node_counter = self.grid.dataSet[ii, jj].nodeCounter
                row = node_counter * 5
                column = node_counter * 5  # diagonal block
                self.add_to_C_g(self.grid.dataSet[ii, jj].C, row, column)

    def build_R_global_matrix(self):
        """
        Build the R global matrix.
        """
        self.R_g = np.zeros(self.Q_const.shape, dtype=complex)
        for ii in range(0, self.grid.nStream):
            for jj in range(0, self.grid.nSpan):
                node_counter = self.grid.dataSet[ii, jj].nodeCounter
                row = node_counter * 5
                column = node_counter * 5  # diagonal block
                self.add_to_R_g(self.grid.dataSet[ii, jj].R, row, column)

    def build_S_global_matrix(self):
        """
        Build the S global matrix.
        """
        self.S_g = np.zeros(self.Q_const.shape, dtype=complex)
        for ii in range(0, self.grid.nStream):
            for jj in range(0, self.grid.nSpan):
                node_counter = self.grid.dataSet[ii, jj].nodeCounter
                row = node_counter * 5
                column = node_counter * 5  # diagonal block
                self.add_to_S_g(self.grid.dataSet[ii, jj].S, row, column)

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
        try:
            block_type = self.config.get_blocks_type()[block_i]
        except:
            block_type = 'unbladed'
        m = self.config.get_circumferential_harmonic_order()

        if block_type == 'unbladed':
            Omega = 0
            tau = 0
        elif block_type == 'stator':
            Omega = 0
            tau = np.mean(self.grid.meridional_obj.stream_line_length[-1, :]) / np.mean(
                self.grid.meridional_obj.u_meridional[0, :])
        elif block_type == 'rotor':
            Omega = self.config.get_omega_shaft() / self.config.get_reference_omega()
            tau = np.mean(self.grid.meridional_obj.stream_line_length[-1, :]) / np.mean(
                self.grid.meridional_obj.u_meridional[0, :])
        else:
            raise ValueError('Unknown block type!')

        print_banner_begin('BLOCK TYPE')
        print(f"{'Block type:':<{total_chars_mid}}{block_type:>{total_chars_mid}}")
        print(f"{'Block Omega Sun [-]:':<{total_chars_mid}}{Omega:>{total_chars_mid}}")
        print(f"{'Block Tau Sun: [-]':<{total_chars_mid}}{tau:>{total_chars_mid}}")
        print_banner_end()

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
        (L2*omega^2 + L1*omega + L0)*tilde{phi}. Therefore, BCs are imposed on L0 (the only constant matrix), and L1,L2
        (omega dependent) filled with zeros in correspondance of those BCs.
        """
        print('Boundary Condition implementation type: ', mode)
        self.rows_added = 0
        for ii in range(0, self.grid.nStream):
            for jj in range(0, self.grid.nSpan):
                marker = self.grid.dataSet[ii, jj].marker
                counter = self.grid.dataSet[ii, jj].nodeCounter
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
                        if jj == 0 or jj == self.grid.nSpan - 1:
                            print('Corner fix applied')
                            self.apply_bc_condition(row, self.hub_bc, ii, jj)
                    elif marker == 'outlet':
                        self.add_bc_condition(row, self.outlet_bc, ii, jj)
                        if jj == 0 or jj == self.grid.nSpan - 1:
                            print('Corner fix applied')
                            self.apply_bc_condition(row, self.hub_bc, ii, jj)
                    elif marker == 'hub':
                        self.add_bc_condition(row, self.hub_bc, ii, jj)
                    elif marker == 'shroud':
                        self.add_bc_condition(row, self.shroud_bc, ii, jj)
                    elif marker != 'internal':
                        raise Exception('Boundary condition unknown. Check the grid markers!')

    def read_boundary_conditions(self):
        """
        Store in the object the information related to the boundary conditions to use for the problem
        """
        self.inlet_bc = self.config.get_inlet_bc()
        self.outlet_bc = self.config.get_outlet_bc()
        self.hub_bc = self.config.get_hub_bc()
        self.shroud_bc = self.config.get_shroud_bc()

        # recognized boundary conditions type
        bc_list = ['zero pressure', 'zero perturbation', 'euler wall', 'compressor inlet', 'compressor outlet',
                   'zero axial velocity', 'free', 'neumann inlet', 'neumann outlet', 'zero radial velocity']

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
            self.L0[row + 4, row + 4] = 1
            self.L1[row + 4, :] = np.zeros(self.L1[row + 4, :].shape, dtype=complex)  # zero corresponding row
            self.L2[row + 4, :] = np.zeros(self.L2[row + 4, :].shape, dtype=complex)  # zero corresponding row

        elif condition == 'zero axial velocity':
            # BC for zero pressure perturbation
            self.L0[row + 3, :] = np.zeros(self.L0[row + 4, :].shape, dtype=complex)
            self.L0[row + 3, row + 3] = 1
            self.L1[row + 3, :] = np.zeros(self.L1[row + 3, :].shape, dtype=complex)
            self.L2[row + 3, :] = np.zeros(self.L2[row + 3, :].shape, dtype=complex)

        elif condition == 'zero radial velocity':
            # BC for zero pressure perturbation
            self.L0[row + 1, :] = np.zeros(self.L0[row + 1, :].shape, dtype=complex)
            self.L0[row + 1, row + 1] = 1
            self.L1[row + 1, :] = np.zeros(self.L1[row + 1, :].shape, dtype=complex)
            self.L2[row + 1, :] = np.zeros(self.L2[row + 1, :].shape, dtype=complex)

        elif condition == 'free':
            pass

        elif condition == 'zero perturbation':
            self.L0[row:row + 5, :] = np.zeros(self.L0[row:row + 5, :].shape, dtype=complex)
            self.L0[row:row + 5, row:row + 5] = np.eye(5, dtype=complex)

            self.L1[row:row + 5, :] = np.zeros(self.L1[row:row + 5, :].shape, dtype=complex)  # zero correspnding rows
            self.L2[row:row + 5, :] = np.zeros(self.L2[row:row + 5, :].shape, dtype=complex)  # zero corresponding rows

        elif condition == 'neumann inlet':
            self.L0[row:row + 5, :] = np.zeros(self.L0[row:row + 5, :].shape, dtype=complex)
            node = row // 5  # number of the node
            node_next = node + self.grid.nSpan  # number of the next node along the streamline
            row_next = node_next * 5  # equivalent row index for that next node
            self.L0[row:row + 5, row:row + 5] = np.eye(5, dtype=complex)
            self.L0[row:row + 5, row_next:row_next + 5] = -np.eye(5, dtype=complex)
            self.L1[row:row + 5, :] = np.zeros(self.L1[row:row + 5, :].shape, dtype=complex)
            self.L2[row:row + 5, :] = np.zeros(self.L2[row:row + 5, :].shape, dtype=complex)

        elif condition == 'neumann outlet':
            self.L0[row:row + 5, :] = np.zeros(self.L0[row:row + 5, :].shape, dtype=complex)
            node = row // 5
            node_previous = node - self.grid.nSpan
            row_previous = node_previous * 5
            self.L0[row:row + 5, row:row + 5] = np.eye(5, dtype=complex)
            self.L0[row:row + 5, row_previous:row_previous + 5] = -np.eye(5, dtype=complex)
            self.L1[row:row + 5, :] = np.zeros(self.L1[row:row + 5, :].shape, dtype=complex)
            self.L2[row:row + 5, :] = np.zeros(self.L2[row:row + 5, :].shape, dtype=complex)

        elif condition == 'compressor inlet':
            # BCs are zero for every variable except the pressure at inlet
            self.L0[row:row + 4, :] = np.zeros(self.L0[row:row + 4, :].shape, dtype=complex)
            self.L0[row:row + 4, row:row + 4] = np.eye(4, dtype=complex)
            self.L1[row:row + 4, :] = np.zeros(self.L1[row:row + 4, :].shape, dtype=complex)
            self.L2[row:row + 4, :] = np.zeros(self.L2[row:row + 4, :].shape, dtype=complex)

        elif condition == 'compressor outlet':
            # BC for zero pressure perturbation
            self.L0[row + 4, :] = np.zeros(self.L0[row + 4, :].shape, dtype=complex)
            self.L0[row + 4, row + 4] = 1
            self.L1[row + 4, :] = np.zeros(self.L1[row + 4, :].shape, dtype=complex)
            self.L2[row + 4, :] = np.zeros(self.L2[row + 4, :].shape, dtype=complex)

        elif condition == 'euler wall':
            # BC for non-penetration condition at the walls, the equation overwritten depends on the configuration file
            if self.wallEquation.lower() == 'radial velocity':
                loc = 1
            elif self.wallEquation.lower() == 'tangential velocity':
                loc = 2
            elif self.wallEquation.lower() == 'axial velocity':
                loc = 3
            else:
                raise ValueError("Subsituted equation parameter not recognized.")

            wall_normal = self.grid.dataSet[ii, jj].n_wall
            self.L0[row + loc, :] = np.zeros(self.L0[row + loc, :].shape, dtype=complex)
            self.L0[row + loc, row + 1:row + 4] = wall_normal
            self.L1[row + loc, :] = np.zeros(self.L1[row + loc, :].shape, dtype=complex)  # zero known term
            self.L2[row + loc, :] = np.zeros(self.L2[row + loc, :].shape, dtype=complex)  # zero known term

        else:
            raise ValueError('unknown boundary condition type')

    def add_bc_condition(self, row, condition, ii, jj):
        """
        For the considered grid node, it adds the boundary conditions at the end of the matrix, enlarging the
        dimensions, using the Lagrange Multiplier formulation
        (https://biba1632.gitlab.io/code-aster-manuals/docs/reference/r3.03.01.html equation 11.2.2).
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
            new_col = np.zeros((self.L0.shape[0], 1))
            zero_row = np.zeros((1, self.L0.shape[0] + 1))
            new_row = np.zeros((1, self.L0.shape[0] + 1))

            # set the condition. The column must equal the transpose of the added row.
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

        elif condition == 'zero axial velocity':
            raise ValueError('Condition not implemented yet!')

        elif condition == 'free':
            pass

        elif condition == 'zero perturbation':
            raise ValueError('Condition not implemented yet!')

        elif condition == 'compressor inlet':
            raise ValueError('Condition not implemented yet!')

        elif condition == 'compressor outlet':
            raise ValueError('Condition not implemented yet!')

        elif condition == 'euler wall':
            wall_normal = self.grid.dataSet[ii, jj].n_wall
            zero_col = np.zeros((self.L0.shape[0], 1))
            new_col = np.zeros((self.L0.shape[0], 1))
            zero_row = np.zeros((1, self.L0.shape[0] + 1))
            new_row = np.zeros((1, self.L0.shape[0] + 1))
            new_row[0, row + 1: row + 4] = wall_normal
            new_col[row + 1: row + 4, 0] = wall_normal

            self.L0 = np.hstack((self.L0, new_col))
            self.L0 = np.vstack((self.L0, new_row))
            self.L1 = np.hstack((self.L1, zero_col))
            self.L1 = np.vstack((self.L1, zero_row))
            self.L2 = np.hstack((self.L2, zero_col))
            self.L2 = np.vstack((self.L2, zero_row))
            self.rows_added += 1

        else:
            raise ValueError('unknown boundary condition type')

    def compute_block_Y_P_matrices(self, inspect_matrices=False):
        """
        It builds the Y and P matrices for the single block transformed EVP, where the equation is Y*varphi = omega*P*varphi
        :param inspect_matrices: plot the structure of the involved matrices, to check sparsity
        """

        m = self.config.get_circumferential_harmonic_order()
        omega_shaft = self.config.get_omega_shaft()
        omega_ref = self.config.get_reference_omega()
        x_ref = self.config.get_reference_length()
        u_ref = self.config.get_reference_velocity()
        t_ref = self.config.get_reference_time()
        sigma = self.config.get_research_center_omega_eigenvalues() / omega_ref  # non-dimensional center point of research
        number_search = self.config.get_research_number_omega_eigenvalues()

        print_banner_begin('ARNOLDI SOLVER')
        print(f"{'Circumferential Harmonic:':<{total_chars_mid}}{m:>{total_chars_mid}}")
        print(f"{'Shaft Angular Speed [rad/s]:':<{total_chars_mid}}{omega_shaft:>{total_chars_mid}.2f}")
        print(f"{'Ref. Angular Speed [rad/s]:':<{total_chars_mid}}{omega_ref:>{total_chars_mid}.2f}")
        print(f"{'Initial Searching Point [-]:':<{total_chars_mid}}{sigma:>{total_chars_mid}.2f}")
        print(f"{'Number of Eigenvalues to Find:':<{total_chars_mid}}{number_search:>{total_chars_mid}}")
        print_banner_end()

        Y1 = np.concatenate((-self.L0, np.zeros_like(self.L0)), axis=1)
        Y2 = np.concatenate((np.zeros_like(self.L0), np.eye(self.L0.shape[0])), axis=1)
        self.Y = np.concatenate((Y1, Y2), axis=0)  # Y matrix of EVP problem

        P1 = np.concatenate((self.L1, self.L2), axis=1)
        P2 = np.concatenate((np.eye(self.L0.shape[0]), np.zeros_like(self.L0)), axis=1)
        self.P = np.concatenate((P1, P2), axis=0)  # P matrix of EVP problem

        if inspect_matrices:
            plt.figure()
            plt.spy(self.L0)
            plt.title(r'$\mathbf{L}_{0}$')

            plt.figure()
            plt.spy(self.L1)
            plt.title(r'$\mathbf{L}_{1}$')

            plt.figure()
            plt.spy(self.L2)
            plt.title(r'$\mathbf{L}_{2}$')

            plt.figure()
            plt.spy(self.Y)
            plt.title(r'$\mathbf{Y}$')

            plt.figure()
            plt.spy(self.P)
            plt.title(r'$\mathbf{P}$')

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

    def plot_eigenfrequencies(self, delimit=False, save_filename=None, folder_name='pictures'):
        """
        Plot the eigenfrequencies obtained with the Arnoldi Method
        :param delimit: if true, delimit the plot zone the important one for compressors
        :param save_filename: if not None, save figure files
        """
        fig, ax = plt.subplots()
        for mode in self.eigenfields:
            rs = mode.eigenfrequency.real / self.omega_ref
            df = mode.eigenfrequency.imag / self.omega_ref
            ax.scatter(rs, df, marker='o', facecolors='red', edgecolors='red', s=marker_size)
        ax.set_xlabel(r'RS [-]')
        ax.set_ylabel(r'DF [-]')
        if delimit:
            ax.set_xlim([-1.5, 1.5])
            ax.set_ylim([-1, 0.5])
        ax.grid(alpha=grid_opacity)
        if save_filename is not None:
            fig.savefig(folder_name + '/' + save_filename + '.pdf', bbox_inches='tight')

    def extract_eigenfields(self, n=None):
        """
        From the eigenvectors obtained with Arnoldi Method, extract the eigenfields (density, velocity, pressure).
        The eigensolution should be sorted before applying this method, otherwise the modes are randomly ordered.
        :param n: number of eigenfields to extract
        """
        if n is None:
            n = len(self.eigenfreqs)
        elif n > len(self.eigenfreqs):
            print("parameter n must be lower than the eigenvector number. n set to max allowed")
            n = len(self.eigenfreqs)

        Nz = self.grid.nStream
        Nr = self.grid.nSpan
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

    def plot_eigenfields(self, n=None, save_filename=None, folder_name='pictures'):
        """
        Plot the first n eigenmodes structures.
        :param n: specify the first n eigenfunctions to plot
        :param save_filename: specify name of the figs to save
        """
        z = self.grid.meridional_obj.z_cg
        r = self.inputData['RadialCoord']
        self.pic_size_blank, self.pic_size_contour = compute_picture_size(z, r)
        Nz = np.shape(z)[0]
        Nr = np.shape(z)[1]
        modes_map = cm.bwr

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
            # if mode.is_physical:
            rs = mode.eigenfrequency.real / self.omega_ref
            df = mode.eigenfrequency.imag / self.omega_ref

            plt.figure(figsize=self.pic_size_contour)
            cnt = plt.contourf(z, r, mode.eigen_rho, levels=N_levels_fine, cmap=modes_map)
            for c in cnt.collections:
                c.set_edgecolor("face")
            plt.xlabel(r'$z$ [-]')
            plt.ylabel(r'$r$ [-]')
            plt.title(r'$\tilde{\rho}_{%i}: \  \hat{\omega} = [%.2f,%.2f j]$' % (imode, rs, df))
            plt.colorbar()
            if save_filename is not None:
                plt.savefig(folder_name + '/' + save_filename + '_rho_%i_%i_%i.pdf' % (Nz, Nr, imode), bbox_inches='tight')
                plt.close()

            plt.figure(figsize=self.pic_size_contour)
            cnt = plt.contourf(z, r, mode.eigen_ur, levels=N_levels_fine, cmap=modes_map)
            for c in cnt.collections:
                c.set_edgecolor("face")
            plt.xlabel(r'$z$ [-]')
            plt.ylabel(r'$r$ [-]')
            plt.title(r'$\tilde{u}_{r,%i}: \  \hat{\omega} = [%.2f,%.2f j]$' % (imode, rs, df))
            plt.colorbar()
            if save_filename is not None:
                plt.savefig(folder_name + '/' + save_filename + '_ur_%i_%i_%i.pdf' % (Nz, Nr, imode), bbox_inches='tight')
                plt.close()

            plt.figure(figsize=self.pic_size_contour)
            cnt = plt.contourf(z, r, mode.eigen_utheta, levels=N_levels_fine, cmap=modes_map)
            for c in cnt.collections:
                c.set_edgecolor("face")
            plt.xlabel(r'$z$ [-]')
            plt.ylabel(r'$r$ [-]')
            plt.title(r'$\tilde{u}_{\theta,%i}: \  \hat{\omega} = [%.2f,%.2f j]$' % (imode, rs, df))
            plt.colorbar()
            if save_filename is not None:
                plt.savefig(folder_name + '/' + save_filename + '_ut_%i_%i_%i.pdf' % (Nz, Nr, imode), bbox_inches='tight')
                plt.close()

            plt.figure(figsize=self.pic_size_contour)
            cnt = plt.contourf(z, r, mode.eigen_uz, levels=N_levels_fine, cmap=modes_map)
            for c in cnt.collections:
                c.set_edgecolor("face")
            plt.xlabel(r'$z$ [-]')
            plt.ylabel(r'$r$ [-]')
            plt.title(r'$\tilde{u}_{z,%i}: \  \hat{\omega} = [%.2f,%.2f j]$' % (imode, rs, df))
            plt.colorbar()
            if save_filename is not None:
                plt.savefig(folder_name + '/' + save_filename + '_uz_%i_%i_%i.pdf' % (Nz, Nr, imode), bbox_inches='tight')
                plt.close()

            plt.figure(figsize=self.pic_size_contour)
            cnt = plt.contourf(z, r, mode.eigen_p, levels=N_levels_fine, cmap=modes_map)
            for c in cnt.collections:
                c.set_edgecolor("face")
            plt.xlabel(r'$z$ [-]')
            plt.ylabel(r'$r$ [-]')
            plt.title(r'$\tilde{p}_{%i}: \  \hat{\omega} = [%.2f,%.2f j]$' % (imode, rs, df))
            plt.colorbar()
            if save_filename is not None:
                plt.savefig(folder_name + '/' + save_filename + '_p_%i_%i_%i.pdf' % (Nz, Nr, imode), bbox_inches='tight')
                plt.close()


