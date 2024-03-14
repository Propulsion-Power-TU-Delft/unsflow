import numpy as np


class CouetteTaylor1D():
    """
    This class it is the equivalent of a meridional_process object calculated processing CFD data. In this case we simply
    insert the meridional flow fields as 2D array, in order to compute the eigenfrequencies of the annulus duct.
    """

    def __init__(self, rmin, rmax, N, config, mode='default'):
        """
        Build the 2D arrays on the meridional plane, to be compatible with meridional_process object of a compressor.
        The non-dimensionalization procedure is treated later.
        :param rmin: inner radius
        :param rmax: outer radius
        :param config: configuration file
        """
        self.config = config
        self.nRadialNodes = N
        self.nspan = N
        self.nPoints = N
        if mode == 'default':
            self.r = np.linspace(rmin, rmax, N)
        elif mode == 'gauss-lobatto':
            self.r = gauss_lobatto_grid_generation(N, rmin, rmax)
        else:
            raise ValueError('Unrecognized mode of grid generation')
        self.rho = np.zeros_like(self.r)
        self.ur = np.zeros_like(self.r)
        self.ut = np.zeros_like(self.r)
        self.uz = np.zeros_like(self.r)
        self.p = np.zeros_like(self.r)
        self.drho_dr = np.zeros_like(self.r)
        self.drho_dz = np.zeros_like(self.r)
        self.dur_dr = np.zeros_like(self.r)
        self.dur_dz = np.zeros_like(self.r)
        self.dut_dr = np.zeros_like(self.r)
        self.dut_dz = np.zeros_like(self.r)
        self.duz_dr = np.zeros_like(self.r)
        self.duz_dz = np.zeros_like(self.r)
        self.dp_dr = np.zeros_like(self.r)
        self.dp_dz = np.zeros_like(self.r)


    def normalize_data(self):
        """
        given the fundamental quantities, normalize everything
        """
        self.rho_ref = self.config.get_reference_density()
        self.u_ref = self.config.get_reference_velocity()
        self.x_ref = self.config.get_reference_length()
        self.p_ref = self.rho_ref*self.u_ref**2
        self.t_ref = self.config.get_reference_time()
        self.omega_ref = self.config.get_reference_omega()

        self.r /= self.x_ref
        self.rho /= self.rho_ref
        self.ur /= self.u_ref
        self.ut /= self.u_ref
        self.uz /= self.u_ref
        self.p /= self.p_ref

        self.drho_dr /= self.rho_ref/self.x_ref
        self.drho_dz /= self.rho_ref/self.x_ref
        self.dur_dr /= self.u_ref/self.x_ref
        self.dur_dz /= self.u_ref/self.x_ref
        self.dut_dr /= self.u_ref/self.x_ref
        self.dut_dz /= self.u_ref/self.x_ref
        self.duz_dr /= self.u_ref/self.x_ref
        self.duz_dz /= self.u_ref/self.x_ref
        self.dp_dr /= self.p_ref/self.x_ref
        self.dp_dz /= self.p_ref/self.x_ref



def gauss_lobatto_grid_generation(N, x_start, x_end):
    """
    return the array of points distributed following gauss-lobatto structure
    """
    xi = np.zeros(N)
    for ii in range(len(xi)):
        xi[ii] = x_start + (x_end-x_start)*(1-np.cos(np.pi*ii/(N-1)))/2
    return xi
