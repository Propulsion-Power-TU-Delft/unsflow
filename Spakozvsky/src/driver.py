from Spakozvsky.src.functions import Shot_Gun
import numpy as np
import matplotlib.pyplot as plt


class Driver:
    """
    this class contains the driver of the Spakovszky model instability calculation.
    """
    def __init__(self, compressor_type):
        """
        Construct the main drive of the Spakovszky instability model.
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
        self.IC = np.array([[0, 1, 0],
                            [0, 0, 1]])

    def set_outlet_boundary_conditions(self, bc_type='infinite duct length'):
        """
        Generate the EC matrix.
        """
        if bc_type=='infinite duct length':
            print("Exit Boundary Conditions: infinite duct length")
            self.EC = np.array([[1, 0, 0]])
        else:
            raise ValueError("Boundary condition not recognized")

    def compute_global_Ysys_determinant(self, s, n):
        Y = np.concatenate((self.EC @ self.compute_global_Xsys(s, n), self.IC), axis=0)
        return np.linalg.det(Y)

    def set_eigenvalues_research_settings(self, domain, grid, attempts, tol):
        self.domain = domain
        self.grid = grid
        self.attempts = attempts
        self.tol = tol

    def find_eigenvalues(self, n):
        """
        Find the eigenvalues of the system calling the Shot-Gun method.
        :param n: harmonic order of the research
        """
        self.poles_dict = {}
        try:
            for nn in n:
                poles = Shot_Gun(self.compute_global_Ysys_determinant,
                                 self.domain, self.grid, n=nn, attempts=self.attempts, tol=self.tol)
                self.poles_dict[nn] = poles
        except:
            poles = Shot_Gun(self.compute_global_Ysys_determinant,
                             self.domain, self.grid, n=n, attempts=self.attempts, tol=self.tol)
            self.poles_dict[n] = poles


    def plot_eigenvalues(self, domain, savefilename=None):
        for n in self.poles_dict.keys():
            plt.plot(self.poles_dict[n].real, -self.poles_dict[n].imag, 'o', label='n:%i' %(n))
        real_axis_x = np.linspace(domain[0], domain[1], 100)
        real_axis_y = np.zeros(len(real_axis_x))
        imag_axis_y = np.linspace(domain[2], domain[3], 100)
        imag_axis_x = np.zeros(len(imag_axis_y))
        plt.plot(real_axis_x, real_axis_y, '--k', linewidth=0.5)
        plt.plot(imag_axis_x, imag_axis_y, '--k', linewidth=0.5)
        plt.xlim([domain[0], domain[1]])
        plt.ylim([domain[2], domain[3]])
        plt.legend()
        plt.xlabel(r'$\sigma_{n}$')
        plt.ylabel(r'$j \omega_{n}$')
        plt.title('Root locus')
        plt.grid(alpha=0.2)
        if savefilename is not None:
            plt.savefig(savefilename + '.pdf', bbox_inches = 'tight')
