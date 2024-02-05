import numpy as np


class AnnulusMeridional():
    """
    This class it is the equivalent of a meridional_process object calculated processing CFD data. In this case we simply
    insert the meridional flow fields as 2D array, in order to compute the eigenfrequencies of the annulus duct.
    """

    def __init__(self, zmin, zmax, rmin, rmax, Nz, Nr, rho, u, v, w, p, config, grid_refinement=1):
        """
        Build the 2D arrays on the meridional plane, to be compatible with meridional_process object of a compressor.
        The non-dimensionalization procedure is treated later.
        :param zmin: inlet axial cordinate
        :param zmax: outlet axial cordinate
        :param rmin: inner radius
        :param rmax: outer radius
        :param Nz: streamwise points
        :param Nr: spanwise points
        :param rho: density 2D field
        :param u: radial velocity 2D field
        :param v: tang. velocity 2D field
        :param w: axial velocity 2D field
        :param p: pressure 2D field
        :param config: configuration file
        :param grid_refinement: refinement of the grid, to compute metrics on the finer grid, interpolating later on the coarse.
        """
        self.config = config
        self.nAxialNodes = Nz
        self.nRadialNodes = Nr
        self.nstream = Nz
        self.nspan = Nr
        self.nPoints = Nz * Nr
        self.z = np.linspace(zmin, zmax, Nz)
        self.r = np.linspace(rmin, rmax, Nr)
        self.z_finegrid = np.linspace(zmin, zmax, Nz*grid_refinement)  # for transformation gradient computation
        self.r_finegrid = np.linspace(rmin, rmax, Nr*grid_refinement)
        self.r_grid, self.z_grid = np.meshgrid(self.r, self.z)
        self.r_cg, self.z_cg = np.meshgrid(self.r, self.z)
        self.r_cg_fine, self.z_cg_fine = np.meshgrid(self.r_finegrid, self.z_finegrid)
        self.rho = np.zeros_like(self.z_grid)+rho
        self.ur = np.zeros_like(self.z_grid)+u
        self.ut = np.zeros_like(self.z_grid)+v
        self.uz = np.zeros_like(self.z_grid)+w
        self.p = np.zeros_like(self.z_grid)+p
        self.drho_dr = np.zeros_like(self.rho)
        self.drho_dz = np.zeros_like(self.rho)
        self.dur_dr = np.zeros_like(self.rho)
        self.dur_dz = np.zeros_like(self.rho)
        self.dut_dr = np.zeros_like(self.rho)
        self.dut_dz = np.zeros_like(self.rho)
        self.duz_dr = np.zeros_like(self.rho)
        self.duz_dz = np.zeros_like(self.rho)
        self.dp_dr = np.zeros_like(self.rho)
        self.dp_dz = np.zeros_like(self.rho)
        self.domain = 'unbladed'


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

        self.r_grid /= self.x_ref
        self.z_grid /= self.x_ref
        self.r_cg /= self.x_ref
        self.z_cg /= self.x_ref
        self.r_cg_fine /= self.x_ref
        self.z_cg_fine /= self.x_ref
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
