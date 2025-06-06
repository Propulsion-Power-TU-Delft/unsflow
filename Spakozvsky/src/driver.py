import os
from Spakozvsky.src.functions import Shot_Gun
import numpy as np
import matplotlib.pyplot as plt
import pickle
from Spakozvsky.src.functions import Trad_n
from Utils.styles import total_chars, total_chars_mid
from Sun.src.general_functions import print_banner_end, print_banner_begin


class Driver:
    """
    Simulation driver of the Spakovszky Instability Analysis.
    """
    def __init__(self, compressor_type):
        """
        Construct the main driver of the Spakovszky instability model.
        :param compressor_type: Specify if axial or radial
        """
        self.compressor_type = compressor_type
        self.components = []


    def add_component(self, component):
        """
        Add a component to the compression system. Follow streamwise order, from inlet to outlet.
        :param component: block of the component to add
        """
        self.components.append(component)

    def compute_global_Xsys(self, s, n):
        """
        Depending on the blocks added, compute the global transfer function of the problem. Boundary conditions
        will be added later. The Xsys matrix must relate the output waves to input waves (A, B, C).
        :param s: Laplace variable
        :param n: azimuthal harmonic order
        """
        self.number_of_blocks = len(self.components)

        # the last element transfer function must be inverted since we need to relate output to input waves amplitude.
        # the following multiplication are done going downstream
        M = np.linalg.inv(self.components[-1].transfer_function(s, n))
        for i in range(self.number_of_blocks-2, -1, -1):
            H = self.components[i].transfer_function(s, n)
            M = M @ H
        return M

    def set_inlet_boundary_conditions(self):
        """
        Generate the IC matrix.
        """
        print_banner_begin('BOUNDARY CONDITIONS')
        print(f"{'Inlet Boundary Conditions:':<{total_chars_mid}}{'Standard':>{total_chars_mid}}")
        self.IC = np.array([[0, 1, 0],
                            [0, 0, 1]])

    def set_outlet_boundary_conditions(self, bc_type='infinite duct length', *exit_conditions):
        """
        Set the parameters to compute the EC matrix.
        :param bc_type: type of the boundary condition
        :param exit_conditions: tuple storing the information of the outlet.
        """
        self.exit_bc_type = bc_type
        print(f"{'Outlet Boundary Conditions:':<{total_chars_mid}}{self.exit_bc_type:>{total_chars_mid}}")
        if bc_type.lower() == 'finite duct length':
            print('Exit Boundary Condition Type: finite duct length')
            self.exit_uz = exit_conditions[0][0]
            self.exit_ut = exit_conditions[0][1]
            self.exit_z = exit_conditions[0][2]
        elif bc_type.lower() == 'radial plenum discharge':
            print('Exit Boundary Condition Type: radial plenum discharge')
            self.exit_r = exit_conditions[0][0]
            self.exit_ur = exit_conditions[0][1]
            self.exit_ut = exit_conditions[0][2]
            self.exit_Q = 2 * np.pi * self.exit_r * self.exit_ur
            self.exit_GAMMA = 2 * np.pi * self.exit_r * self.exit_ut
        elif bc_type.lower() == 'infinite duct length':
            pass
        else:
            raise ValueError("Boundary condition not recognized")

    def compute_outlet_boundary_conditions(self, s, n):
        """
        Generate the EC matrix.
        :param s: Laplace variable
        :param n: hamronic order
        """
        if self.exit_bc_type=='infinite duct length':
            EC = np.array([[1, 0, 0]])
        elif self.exit_bc_type=='finite duct length':
            EC = np.array([[(-s/n - self.exit_uz - 1j*self.exit_ut)*np.exp(n*self.exit_z),
                           (+s/n - self.exit_uz + 1j*self.exit_ut)*np.exp(-n*self.exit_z),
                            0]])
        elif self.exit_bc_type=='radial plenum discharge':
            Tmat = Trad_n(self.exit_r, self.exit_r, n, s, self.exit_Q, self.exit_GAMMA, 0)
            EC = np.array([[Tmat[2, 0], Tmat[2, 1], Tmat[2, 2]]])
        else:
            raise ValueError("Boundary condition not recognized")
        return EC

    def compute_global_Ysys_determinant(self, s, n):
        """
        From the Xsys and the boundary conditions, generate the Y matrix whose eigenvalues are the poles of the system.
        :param s: Laplace variable
        :param n: harmonic order
        """
        Y = np.concatenate((self.compute_outlet_boundary_conditions(s, n) @ self.compute_global_Xsys(s, n), self.IC), axis=0)
        return np.linalg.det(Y)

    def set_eigenvalues_research_settings(self, domain, grid, attempts, tol):
        """
        Set the numerical settings of the eigenvalues research.
        :param domain: domain of research [omegar_r min, omega_r max, omega_i min, omega_r max]
        :param grid: specify the sub-zones that you want to analyse with the Shot-Gun Method [real sub-blocks, imag sub-blocks]
        :param attempts: numerical parameter of the Shot-Gun Method. Increase to improve probability to get all the poles.
        :param tol: tolerance parameter of the Shot-Gun Method. Threshold to accept an eigenvalue location.
        """
        self.domain = domain
        self.grid = grid
        self.attempts = attempts
        self.tol = tol

    def find_eigenvalues(self, n):
        """
        Find the eigenvalues of the system calling the Shot-Gun method.
        :param n: harmonic order of the research (a single integer, or a list of integers).
        """
        self.poles_dict = {}
        try:
            for nn in n:
                print("Looking for eigenvalues of harmonic: %i" %(nn))
                poles = Shot_Gun(self.compute_global_Ysys_determinant,
                                 self.domain, self.grid, n=nn, attempts=self.attempts, tol=self.tol)
                self.poles_dict[nn] = poles
        except:
            print("Looking for eigenvalues of harmonic: %i" % (n))
            poles = Shot_Gun(self.compute_global_Ysys_determinant,
                             self.domain, self.grid, n=n, attempts=self.attempts, tol=self.tol)
            self.poles_dict[n] = poles


    def plot_eigenvalues(self, domain=None, save_filename=None, save_foldername=None):
        """
        Plot the eigenvalues found in the poles dictionary. The imaginary part is reversed in order to match Spakovsky
        sign conventions (lambda = sigma - j*omega). A positive sigma implies instability of the associated mode.
        :param domain: domain [x_min, x_max, y_min, y_max] for plotting the results
        :param save_filename: specify the file name
        :param save_foldername: specify the folder name
        """
        if save_filename is not None:
            if save_foldername is None:
                save_foldername = 'pictures'
            if not os.path.exists(save_foldername):
                os.makedirs(save_foldername)
        plt.figure()
        for n in self.poles_dict.keys():
            plt.plot(self.poles_dict[n].real, -self.poles_dict[n].imag, 'o', label='n:%i' %(n))

        real_axis_x = np.linspace(self.domain[0], self.domain[1], 100)
        real_axis_y = np.zeros(len(real_axis_x))
        imag_axis_y = np.linspace(self.domain[2], self.domain[3], 100)
        imag_axis_x = np.zeros(len(imag_axis_y))
        plt.plot(real_axis_x, real_axis_y, '--k', linewidth=0.5)
        plt.plot(imag_axis_x, imag_axis_y, '--k', linewidth=0.5)
        if domain is not None:
            plt.xlim([domain[0], domain[1]])
            plt.ylim([domain[2], domain[3]])
        else:
            plt.xlim([self.domain[0], self.domain[1]])
            plt.ylim([self.domain[2], self.domain[3]])
        plt.legend()
        plt.xlabel(r'$\sigma_{n}$')
        plt.ylabel(r'$j \omega_{n}$')
        plt.grid(alpha=0.2)
        if save_filename is not None:
            plt.savefig(save_foldername + '/' + save_filename + '.pdf', bbox_inches = 'tight')


    def store_results_pickle(self, save_filename=None, save_foldername=None):
        """
        Store the Driver object of the analysis. In this way you can re-use methods belonging to the class for plotting, or
        just extract the poles dictionnary for further analysis.
        :param save_filename: specify the file name
        :param save_foldername: specify the folder name
        """
        if save_foldername is None:
            save_foldername = 'results'

        if not os.path.exists(save_foldername):
            os.makedirs(save_foldername)

        if save_filename is None:
            save_filename = 'poles'

        with open(save_foldername + '/' + save_filename + '.pickle', 'wb') as file:
            pickle.dump(self, file)
