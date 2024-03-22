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
import os


class SunModelMultiBlock():
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
        self.assemble_physical_grid()

    def assemble_physical_grid(self):
        """
        Stack together the physical grids of the various blocks.
        """
        self.z_grid = self.blocks[0].data.meridional_obj.z_grid
        self.r_grid = self.blocks[0].data.meridional_obj.r_grid
        for block in self.blocks[1:]:
            self.z_grid = np.concatenate((self.z_grid, block.data.meridional_obj.z_grid), axis=0)
            self.r_grid = np.concatenate((self.r_grid, block.data.meridional_obj.r_grid), axis=0)

    def construct_L_global_matrices(self, visual_check=False):
        """
        Construct the global L matrices for the multiblock problem, stacking together along the diagonal the blocks of every
        sub block.
        """
        L0 = [block.L0 for block in self.blocks]
        self.L0 = enlarge_square_matrices(L0)
        if visual_check:
            fig, ax = plt.subplots(1, 4, figsize=(12, 4))
            ax[0].spy(np.abs(L0[0]))
            ax[0].set_title(r'$L0_0$')
            ax[1].spy(np.abs(L0[1]))
            ax[1].set_title(r'$L0_1$')
            ax[2].spy(np.abs(L0[2]))
            ax[2].set_title(r'$L0_2$')
            ax[3].spy(np.abs(self.L0))
            ax[3].set_title(r'$L0_{tot}$')

        L1 = [block.L1 for block in self.blocks]
        self.L1 = enlarge_square_matrices(L1)
        if visual_check:
            fig, ax = plt.subplots(1, 4, figsize=(12, 4))
            ax[0].spy(np.abs(L1[0]))
            ax[0].set_title(r'$L1_0$')
            ax[1].spy(np.abs(L1[1]))
            ax[1].set_title(r'$L1_1$')
            ax[2].spy(np.abs(L1[2]))
            ax[2].set_title(r'$L1_2$')
            ax[3].spy(np.abs(self.L1))
            ax[3].set_title(r'$L1_{tot}$')

        L2 = [block.L2 for block in self.blocks]
        self.L2 = enlarge_square_matrices(L2)
        if visual_check:
            fig, ax = plt.subplots(1, 4, figsize=(12, 4))
            ax[0].spy(np.abs(L2[0]))
            ax[0].set_title(r'$L2_0$')
            ax[1].spy(np.abs(L2[1]))
            ax[1].set_title(r'$L2_1$')
            ax[2].spy(np.abs(L2[2]))
            ax[2].set_title(r'$L2_2$')
            ax[3].spy(np.abs(self.L2))
            ax[3].set_title(r'$L2_{tot}$')

        if any(arg.dtype != np.complex128 for arg in (self.L0, self.L1, self.L2)):
            raise TypeError('The matrices are not complex')

    def apply_matching_conditions(self):
        """
        Apply the matching conditions for the blocks composing the multiblock. At this moment the system is composed by:
        (L0 + L1*omega + L2*omega^2)*x = 0. Since the matching conditions must be guaranteed for any possible omega, they
        are applied on the matrix L0, while the corresponding rows of L1 and L2 are set to 0. In this way the matching
        conditions are guaranteed no matter the value of omega.
        :param mode: method used to implement the same derivatives in the streamwise direction at the matching nodes
        """
        """
        Starting from the second block, the first 5*nspan equations are matched with the last 5*nspan equations of the 
        previous block. Since every node is written for 2 different domains, in one block we implement the same value 
        of the flow variables, while in the following block we impose the same values of the derivatives.
        """
        modes = ['finite difference', 'collocation method']
        mode = self.config.get_boundary_interface_gradient_method()
        print('\nNumerical derivative method at the interfaces between blocks: ', mode)
        if mode not in modes:
            raise ValueError('Uknown differentiation method.')

        # Let's start from the downstream block (index 1), and consider the block itself and the previous (index 0).
        rows_band = self.config.get_spanwise_points() * 5  # number of equations to modify per each block
        eq_counter = self.blocks[0].L0.shape[0]  # this is the equation counter at the end of the first block
        for iblock in range(1, self.number_blocks):

            # previous block rows (where same fluid conditions are implemented)
            self.L0[eq_counter - rows_band:eq_counter, :] = np.zeros_like(self.L0[eq_counter - rows_band:eq_counter, :])
            self.L0[eq_counter - rows_band:eq_counter, eq_counter - rows_band:eq_counter] = np.eye(rows_band)
            self.L0[eq_counter - rows_band:eq_counter, eq_counter:eq_counter + rows_band] = -np.eye(rows_band)

            """following block (where same fluid derivatives are implemented, by means of finite differences for 
            simplicity). Carrying out the finite difference we end up with:
            (phi_up[-1,j]-phi_up[-2,j])/(xi_up[-1,j]-xi_up[-2,j]) + (-phi_dn[1,j]+phi_dn[0,j])/(xi_dn[1,j]-xi_dn[0,j])"""
            if mode == 'finite difference':
                """Carrying out the finite difference we end up with:
                (phi_up[-1,j]-phi_up[-2,j])/(xi_up[-1,j]-xi_up[-2,j]) + 
                (-phi_dn[1,j]+phi_dn[0,j])/(xi_dn[1,j]-xi_dn[0,j])"""
                self.L0[eq_counter:eq_counter + rows_band, :] = np.zeros_like(self.L0[eq_counter:eq_counter + rows_band, :])

                dxi_up = self.blocks[iblock - 1].dataSpectral.zGrid[-1, 0] - self.blocks[iblock - 1].dataSpectral.zGrid[-2, 0]
                dxi_dn = self.blocks[iblock].dataSpectral.zGrid[1, 0] - self.blocks[iblock].dataSpectral.zGrid[0, 0]

                self.L0[eq_counter:eq_counter + rows_band, eq_counter:eq_counter + rows_band] = np.eye(rows_band) / dxi_dn
                self.L0[eq_counter:eq_counter + rows_band, eq_counter - rows_band:eq_counter] = np.eye(rows_band) / dxi_up
                self.L0[eq_counter:eq_counter + rows_band, eq_counter - 2 * rows_band:eq_counter - rows_band] = -np.eye(
                    rows_band) / dxi_up
                self.L0[eq_counter:eq_counter + rows_band, eq_counter + rows_band:eq_counter + 2 * rows_band] = -np.eye(
                    rows_band) / dxi_dn

            elif mode == 'collocation method':
                # Derivatives now expressed through the Chebyshev collocation method.
                self.L0[eq_counter:eq_counter + rows_band, :] = np.zeros_like(self.L0[eq_counter:eq_counter + rows_band, :])

                # previous block
                DX = ChebyshevDerivativeMatrixBayliss(self.blocks[iblock-1].dataSpectral.z)
                Iy = np.eye(self.blocks[iblock-1].data.nRadialNodes)
                DX = np.kron(DX, Iy)
                DX = DX[-self.config.get_spanwise_points():, :]
                DX = np.kron(DX, np.eye(5))
                self.L0[eq_counter:eq_counter + rows_band, eq_counter-DX.shape[1]:eq_counter] = DX

                # next block
                DX = ChebyshevDerivativeMatrixBayliss(self.blocks[iblock].dataSpectral.z)
                Iy = np.eye(self.blocks[iblock].data.nRadialNodes)
                DX = np.kron(DX, Iy)
                DX = DX[0:self.config.get_spanwise_points(), :]
                DX = np.kron(DX, np.eye(5))
                self.L0[eq_counter:eq_counter + rows_band, eq_counter:eq_counter + DX.shape[1]] = -DX


            # make zero all the relevant equations for the L1,L2 matrices
            self.L1[eq_counter - rows_band:eq_counter + rows_band, :] = np.zeros_like(
                self.L1[eq_counter - rows_band:eq_counter + rows_band, :])
            self.L2[eq_counter - rows_band:eq_counter + rows_band, :] = np.zeros_like(
                self.L2[eq_counter - rows_band:eq_counter + rows_band, :])

            eq_counter += self.blocks[iblock].L0.shape[0]

    def compute_P_Y_matrices(self):
        """
        Once the L0,L1,L2 matrices have been modified by the boundary and matching conditions, build the Y and P matrices of the
        equivalent linearized eigenvalue problem.
        """
        Y1 = np.concatenate((-self.L0, np.zeros_like(self.L0)), axis=1)
        Y2 = np.concatenate((np.zeros_like(self.L0), np.eye(self.L0.shape[0])), axis=1)
        self.Y = np.concatenate((Y1, Y2), axis=0)  # Y matrix of EVP problem

        P1 = np.concatenate((self.L1, self.L2), axis=1)
        P2 = np.concatenate((np.eye(self.L0.shape[0]), np.zeros_like(self.L0)), axis=1)
        self.P = np.concatenate((P1, P2), axis=0)  # P matrix of EVP problem

    def solve_evp(self, sort_mode='imaginary decreasing'):
        """
        Solve the EVP using the Arnoldi Algorithm.
        :param sort_mode: specify the criterion on which the eigenfreqencies and modes are sorted. Imaginary decreasing or
        real increasing
        """
        sigma = self.config.get_research_center_omega_eigenvalues() / self.config.get_reference_omega()
        print('Eigenvalues research center: (%.1f, %.1fj) [rad/s]' %(sigma.real*self.config.get_reference_omega(),
                                                                    sigma.imag*self.config.get_reference_omega().imag))
        print("Transforming generalized EVP in standard one...")
        Y_tilde = np.linalg.inv(self.Y - sigma * self.P) @ self.P

        print("Solving standard EVP...")
        self.eigenfreqs, self.eigenmodes = eigs(Y_tilde, k=self.config.get_research_number_omega_eigenvalues())
        self.eigenmodes = self.eigenmodes[0:self.eigenmodes.shape[0] // 2]  # divided by two because of the quadratic extension

        self.eigenfreqs = sigma + 1 / self.eigenfreqs  # return from the initial shift
        self.eigenfreqs *= self.config.get_reference_omega()  # convert to dimensional frequencies
        self.eigenfreqs_df = self.eigenfreqs.imag / self.config.get_reference_omega() / \
                             self.config.get_circumferential_harmonic_order()
        self.eigenfreqs_rs = self.eigenfreqs.real / self.config.get_reference_omega() / \
                             self.config.get_circumferential_harmonic_order()
        self.sort_eigensolution(sort_mode=sort_mode)

    def sort_eigensolution(self, sort_mode='imaginary decreasing'):
        """
        Sort the eigenvalues and eigenvectors following the sort_mode ordering.
        """
        # make copies of the arrays to sort
        eigenfreqs = np.copy(self.eigenfreqs)
        df = np.copy(self.eigenfreqs_df)
        rs = np.copy(self.eigenfreqs_rs)
        eigenvectors = np.copy(self.eigenmodes)

        # get the sorting indices following descending order of the damping factor
        if sort_mode == 'imaginary decreasing':
            sorted_indices = sorted(range(len(df)), key=lambda ii: df[ii], reverse=True)
        elif sort_mode == 'real increasing':
            sorted_indices = sorted(range(len(rs)), key=lambda ii: rs[ii], reverse=False)
        else:
            raise ValueError('Unknown sort mode')

        # order the original arrays following the sorting indices
        for i in range(len(sorted_indices)):
            self.eigenfreqs[i] = eigenfreqs[sorted_indices[i]]
            self.eigenfreqs_df[i] = df[sorted_indices[i]]
            self.eigenfreqs_rs[i] = rs[sorted_indices[i]]
            self.eigenmodes[:, i] = eigenvectors[:, sorted_indices[i]]

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

        Nz = self.z_grid.shape[0]
        Nr = self.z_grid.shape[1]

        self.eigenfields = []
        for mode in range(n):
            eigenfrequency = self.eigenfreqs[mode]
            eigenvector = self.eigenmodes[:, mode]

            rho_eig = []
            ur_eig = []
            ut_eig = []
            uz_eig = []
            p_eig = []
            for i in range(len(eigenvector)):
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

    def plot_eigenfrequencies(self, delimit=None, normalization=True, save_filename=None, save_foldername='pictures'):
        """
        Plot the eigenfrequencies obtained with the Arnoldi Method
        :param delimit: if true, delimit the plot zone the important one for compressors
        :param normalization: if True plots the Damping factor and rotational speed, otherwise it plots the dimensional frequency
        :param save_filename: if not None, save figure files
        :param save_foldername: folder name of the pictures
        """
        fig, ax = plt.subplots()
        if normalization:
            for mode in self.eigenfields:
                # if mode.is_physical:
                rs = mode.eigenfrequency.real / self.config.get_reference_omega() / \
                     self.config.get_circumferential_harmonic_order()
                df = mode.eigenfrequency.imag / self.config.get_reference_omega() / \
                     self.config.get_circumferential_harmonic_order()
                ax.scatter(rs, df, marker='o', facecolors='red', edgecolors='red', s=marker_size)
            ax.set_xlabel(r'RS [-]')
            ax.set_ylabel(r'DF [-]')
        else:
            for mode in self.eigenfields:
                ax.scatter(mode.eigenfrequency.real, mode.eigenfrequency.imag, marker='o', facecolors='none', edgecolors='red',
                           s=marker_size)
            ax.set_xlabel(r'$\omega_R \mathrm{[rad/s]}$')
            ax.set_ylabel(r'$\omega_I \mathrm{[rad/s]}$')

        if delimit is not None:
            ax.set_xlim([delimit[0], delimit[1]])
            ax.set_ylim([delimit[2], delimit[3]])

        ax.grid(alpha=grid_opacity)
        if save_filename is not None:
            fig.savefig(save_foldername + '/' + save_filename + '.pdf', bbox_inches='tight')
            # plt.close()

    def plot_eigenfields(self, n=None, save_filename=None, save_foldername='pictures'):
        """
        Plot the first n eigenmodes structures.
        :param n: specify the first n eigenfunctions to plot
        :param save_filename: specify name of the figs to save
        """
        z = self.z_grid
        r = self.r_grid
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
            rs = mode.eigenfrequency.real / self.config.get_reference_omega()
            df = mode.eigenfrequency.imag / self.config.get_reference_omega()

            plt.figure()
            plt.contourf(z, r, mode.eigen_rho, levels=N_levels, cmap=modes_map)
            plt.xlabel(r'$z$ [-]')
            plt.ylabel(r'$r$ [-]')
            plt.title(r'$\tilde{\rho}_{%i}: \  \hat{\omega} = [%.2f,%.2f j]$' % (imode, rs, df))
            plt.colorbar()
            plt.xticks([])
            plt.yticks([])
            plt.gca().set_aspect('equal', adjustable='box')
            if save_filename is not None:
                plt.savefig(save_foldername + '/' + save_filename + '_rho_%i_%i_%i.pdf' % (Nz, Nr, imode), bbox_inches='tight')

            plt.figure()
            plt.contourf(z, r, mode.eigen_ur, levels=N_levels, cmap=modes_map)
            plt.xlabel(r'$z$ [-]')
            plt.ylabel(r'$r$ [-]')
            plt.title(r'$\tilde{u}_{r,%i}: \  \hat{\omega} = [%.2f,%.2f j]$' % (imode, rs, df))
            plt.colorbar()
            plt.xticks([])
            plt.yticks([])
            plt.gca().set_aspect('equal', adjustable='box')
            if save_filename is not None:
                plt.savefig(save_foldername + '/' + save_filename + '_ur_%i_%i_%i.pdf' % (Nz, Nr, imode), bbox_inches='tight')

            plt.figure()
            plt.contourf(z, r, mode.eigen_utheta, levels=N_levels, cmap=modes_map)
            plt.xlabel(r'$z$ [-]')
            plt.ylabel(r'$r$ [-]')
            plt.title(r'$\tilde{u}_{\theta,%i}: \  \hat{\omega} = [%.2f,%.2f j]$' % (imode, rs, df))
            plt.colorbar()
            plt.xticks([])
            plt.yticks([])
            plt.gca().set_aspect('equal', adjustable='box')
            if save_filename is not None:
                plt.savefig(save_foldername + '/' + save_filename + '_ut_%i_%i_%i.pdf' % (Nz, Nr, imode), bbox_inches='tight')

            plt.figure()
            plt.contourf(z, r, mode.eigen_uz, levels=N_levels, cmap=modes_map)
            plt.xlabel(r'$z$ [-]')
            plt.ylabel(r'$r$ [-]')
            plt.title(r'$\tilde{u}_{z,%i}: \  \hat{\omega} = [%.2f,%.2f j]$' % (imode, rs, df))
            plt.colorbar()
            plt.xticks([])
            plt.yticks([])
            plt.gca().set_aspect('equal', adjustable='box')
            if save_filename is not None:
                plt.savefig(save_foldername + '/' + save_filename + '_uz_%i_%i_%i.pdf' % (Nz, Nr, imode), bbox_inches='tight')

            plt.figure()
            plt.contourf(z, r, mode.eigen_p, levels=N_levels, cmap=modes_map)
            plt.xlabel(r'$z$ [-]')
            plt.ylabel(r'$r$ [-]')
            plt.title(r'$\tilde{p}_{%i}: \  \hat{\omega} = [%.2f,%.2f j]$' % (imode, rs, df))
            plt.colorbar()
            plt.gca().set_aspect('equal', adjustable='box')
            plt.xticks([])
            plt.yticks([])
            plt.quiver(z, r, mode.eigen_uz, mode.eigen_ur)
            if save_filename is not None:
                plt.savefig(save_foldername + '/' + save_filename + '_p_%i_%i_%i.pdf' % (Nz, Nr, imode), bbox_inches='tight')

    def write_results(self, save_filename=None, folder_name='pictures'):
        """
        Print information regarding the eigenfrequencies found, in the form of damping factors and rotations speeds
        Possible file types are (csv, pickle).
        csv: write only DF and RS in a csv file, organized in two columns
        pickle: write the full list of eigenfields, which contain frequencies and eigenfunctions, in a single pickle
        """
        if save_filename is not None:
            filename = save_filename
        else:
            filename = 'eigenvalues'

        eigenvalue_array = self.eigenfreqs_rs + 1j * self.eigenfreqs_df

        # save the csv of the eigenfrequencies already normalized
        with open(folder_name + '/' + filename + '.csv', 'w', newline='') as csvfile:
            fieldnames = ['RS', 'DF']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for num in eigenvalue_array:
                writer.writerow({'RS': num.real, 'DF': num.imag})

        # save the pickle with all the eigenmodes
        with open(folder_name + '/' + 'eigenfields.pickle', 'wb') as picklefile:
            pickle.dump(self.eigenfields, picklefile)

        print('Saved eigenvalues.csv and eigenfields.pickle in: %s', folder_name)
        print_banner_begin('END OF SIMULATION')
        print_banner_end()

    def inspect_L_matrices(self, save_filename=None, save_foldername=None):
        """
        Plot the L matrices of the full system, to inspect their composition
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

    def hist_inspect_L_global_matrices(self):
        """
        Histogram of the values contained in the L matrices
        """
        matrix_dict = {r'$L_0$': self.L0, r'$L_1$': self.L1, r'$L_2$': self.L2}
        for name, matrix in matrix_dict.items():
            real_values = matrix.flatten().real[matrix.flatten().real!=0]
            imag_values = matrix.flatten().imag[matrix.flatten().imag!=0]
            plt.figure()
            plt.hist(real_values, bins=20, color='blue', alpha=0.5, label='real')
            plt.hist(imag_values, bins=20, color='red', alpha=0.5, label='imag')
            plt.legend()
            plt.title(name)
            plt.xlabel('Value')
            plt.ylabel('N elements')
