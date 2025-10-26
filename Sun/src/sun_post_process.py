import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from Sun.src.eigenmode import Eigenmode
from Utils.styles import *
from Sun.src.general_functions import scaled_eigenvector_real

class PostProcessSun():
    """
    Class used for post processing of the results
    """

    def __init__(self, results_filepath):
        with open(results_filepath, 'rb') as f:
            self.data = pickle.load(f)
        
        self.nStream, self.nSpan = self.data['AxialCoords'].shape

    def extract_eigenfields(self):
        """
        From the eigenvectors obtained with Arnoldi Method, extract the eigenfields (density, velocity, pressure).
        The eigensolution should be sorted before applying this method, otherwise the modes are randomly ordered.
        :param n: number of eigenfields to extract
        """

        nModes = len(self.data['Eigenfrequencies'])
        self.eigenfields = []
        for mode in range(nModes):
            eigenfrequency = self.data['Eigenfrequencies'][mode]
            eigenvector = self.data['Eigenmodes'][:, mode]

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

            rho_eig_r = scaled_eigenvector_real(rho_eig, self.nStream, self.nSpan)
            ur_eig_r = scaled_eigenvector_real(ur_eig, self.nStream, self.nSpan)
            ut_eig_r = scaled_eigenvector_real(ut_eig, self.nStream, self.nSpan)
            uz_eig_r = scaled_eigenvector_real(uz_eig, self.nStream, self.nSpan)
            p_eig_r = scaled_eigenvector_real(p_eig, self.nStream, self.nSpan)

            self.eigenfields.append(Eigenmode(eigenfrequency, rho_eig_r, ur_eig_r, ut_eig_r, uz_eig_r, p_eig_r))

    def plot_eigenfrequencies(self, delimit=None, normalization=False, save_filename=None, save_foldername='pictures'):
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
                ax.plot(mode.eigenfrequency.real, mode.eigenfrequency.imag, 'bo', mfc='none')
            ax.set_xlabel(r'$\omega_R \mathrm{[rad/s]}$')
            ax.set_ylabel(r'$\omega_I \mathrm{[rad/s]}$')

        if delimit is not None:
            ax.set_xlim([delimit[0], delimit[1]])
            ax.set_ylim([delimit[2], delimit[3]])

        ax.grid(alpha=grid_opacity)
        if save_filename is not None:
            fig.savefig(save_foldername + '/' + save_filename + '.pdf', bbox_inches='tight')

    def plot_eigenfields(self, n=None, save_filename=None, save_foldername='pictures'):
        """
        Plot the first n eigenmodes structures.
        :param n: specify the first n eigenfunctions to plot
        :param save_filename: specify name of the figs to save
        """
        z = self.data['AxialCoords']
        r = self.data['RadialCoords']
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
            rs = mode.eigenfrequency.real
            df = mode.eigenfrequency.imag

            fig, axs = plt.subplots(2, 3, figsize=(18, 10))
            axs = axs.flatten()

            titles = [
                r'$\tilde{\rho}_{%i}$' % (imode),
                r'$\tilde{u}_{r,%i}$' % (imode),
                r'$\tilde{u}_{\theta,%i}$' % (imode),
                r'$\tilde{u}_{z,%i}$' % (imode),
                r'$\tilde{p}_{%i}$' % (imode),
                r'$\tilde{p}_{%i}$ with $\vec{u}_m$ quiver' % (imode),
            ]

            fields = [
                mode.eigen_rho,
                mode.eigen_ur,
                mode.eigen_utheta,
                mode.eigen_uz,
                mode.eigen_p,
                mode.eigen_p,
            ]

            quiver_flags = [False, False, False, False, False, True]

            for i, (ax, field, title, qflag) in enumerate(zip(axs, fields, titles, quiver_flags)):
                cf = ax.contourf(z, r, field, levels=N_levels, cmap=modes_map)
                ax.set_title(title)
                ax.set_aspect('equal', adjustable='box')
                plt.colorbar(cf, ax=ax)

                # Keep y-label only for first column
                if i % 3 == 0:
                    ax.set_ylabel(r'$r$ [m]')
                else:
                    ax.set_ylabel('')

                # Keep x-label only for bottom row
                if i >= 3:
                    ax.set_xlabel(r'$z$ [m]')
                else:
                    ax.set_xlabel('')

                if qflag:
                    ax.quiver(z, r, mode.eigen_uz, mode.eigen_ur)

            # Hide any unused subplots (if any)
            for i in range(len(fields), len(axs)):
                axs[i].axis('off')

            plt.tight_layout()

            if save_filename is not None:
                plt.savefig(
                    save_foldername + '/' + save_filename + '_modes_%i_%i_%i.pdf' %
                    (self.nStream, self.nSpan, imode),
                    bbox_inches='tight'
                )







