import numpy as np


class DiffuserMeridional():
    """
    This class it is the equivalent of a meridional_process object calculated processing CFD data. In this case we simply
    insert the meridional flow fields as 2D array, in order to compute the eigenfrequencies of the annulus duct.
    """

    def __init__(self, r1, r2, w, rho, ur, ut, uz, p, dur_dr, dut_dr, grid_refinement=1):
        """
        Build the 2D arrays on the meridional plane, to be compatible with meridional_process object of a compressor.
        The non-dimensionalization procedure is treated later.

        :param grid_refinement: refinement of the grid, to compute metrics on the finer grid, interpolating later on the coarse.
        """
        self.nAxialNodes = np.shape(rho)[0]
        self.nRadialNodes = np.shape(rho)[1]
        self.nstream = self.nAxialNodes
        self.nspan = self.nRadialNodes
        self.nPoints = self.nstream*self.nspan
        self.z = np.linspace(0, w, self.nAxialNodes)
        self.r = np.linspace(r1, r2, self.nRadialNodes)
        self.z_finegrid = np.linspace(0, w, self.nAxialNodes*grid_refinement)  # for transformation gradient computation
        self.r_finegrid = np.linspace(r1, r2, self.nRadialNodes*grid_refinement)
        self.r_grid, self.z_grid = np.meshgrid(self.r, self.z)
        self.r_cg, self.z_cg = np.meshgrid(self.r, self.z)
        self.r_cg_fine, self.z_cg_fine = np.meshgrid(self.r_finegrid, self.z_finegrid)
        self.rho = rho
        self.ur = ur
        self.ut = ut
        self.uz = uz
        self.p = p
        self.drho_dr = np.zeros_like(self.rho)
        self.drho_dtheta = np.zeros_like(self.rho)
        self.drho_dz = np.zeros_like(self.rho)
        self.dur_dr = dur_dr
        self.dur_dtheta = np.zeros_like(self.rho)
        self.dur_dz = np.zeros_like(self.rho)
        self.dut_dr = dut_dr
        self.dut_dtheta = np.zeros_like(self.rho)
        self.dut_dz = np.zeros_like(self.rho)
        self.duz_dr = np.zeros_like(self.rho)
        self.duz_dtheta = np.zeros_like(self.rho)
        self.duz_dz = np.zeros_like(self.rho)
        self.dp_dr = np.zeros_like(self.rho)
        self.dp_dtheta = np.zeros_like(self.rho)
        self.dp_dz = np.zeros_like(self.rho)


    def normalize_data(self, rho_ref, u_ref, x_ref):
        """
        given the fundamental quantities, normalize everything
        """
        self.rho_ref = rho_ref
        self.u_ref = u_ref
        self.x_ref = x_ref
        self.p_ref = rho_ref*u_ref**2
        self.t_ref = x_ref/u_ref
        self.omega_ref = 1/self.t_ref

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
        self.drho_dtheta /= self.rho_ref/self.x_ref
        self.drho_dz /= self.rho_ref/self.x_ref
        self.dur_dr /= self.u_ref/self.x_ref
        self.dur_dtheta /= self.u_ref/self.x_ref
        self.dur_dz /= self.u_ref/self.x_ref
        self.dut_dr /= self.u_ref/self.x_ref
        self.dut_dtheta /= self.u_ref/self.x_ref
        self.dut_dz /= self.u_ref/self.x_ref
        self.duz_dr /= self.u_ref/self.x_ref
        self.duz_dtheta /= self.u_ref/self.x_ref
        self.duz_dz /= self.u_ref/self.x_ref
        self.dp_dr /= self.p_ref/self.x_ref
        self.dp_dtheta /= self.p_ref/self.x_ref
        self.dp_dz /= self.p_ref/self.x_ref
