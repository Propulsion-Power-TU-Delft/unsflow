import numpy as np


class AnnulusMeridional():
    """
    This class it is the equivalent of a meridional_process object calculated processing CFD data. In this case we simply
    insert the meridional flow fields as 2D array, in order to compute the eigenfrequencies of the annulus duct.
    """

    def __init__(self, zmin, zmax, rmin, rmax, Nz, Nr, rho, u, v, w, p, grid_refinement=5):
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
        :param grid_refinement: refinement of the grid, to compute metrics on the finer grid, interpolating later on the coarse.
        """
        self.nAxialNodes = Nz
        self.nRadialNodes = Nr
        self.nstream = Nz
        self.nspan = Nr
        self.nPoints = Nz * Nr
        self.z = np.linspace(zmin, zmax, Nz)
        self.r = np.linspace(rmin, rmax, Nr)
        self.z_finegrid = np.linspace(zmin, zmax, Nz*grid_refinement)  # for transformation gradient computation
        self.r_finegrid = np.linspace(rmin, rmax, Nr*5*grid_refinement)
        self.r_grid, self.z_grid = np.meshgrid(self.r, self.z)
        self.r_cg, self.z_cg = np.meshgrid(self.r, self.z)
        self.r_cg_fine, self.z_cg_fine = np.meshgrid(self.r_finegrid, self.z_finegrid)
        self.rho = rho
        self.ur = u
        self.ut = v
        self.uz = w
        self.p = p
        self.drho_dr = np.zeros_like(self.rho)
        self.drho_dtheta = np.zeros_like(self.rho)
        self.drho_dz = np.zeros_like(self.rho)
        self.dur_dr = np.zeros_like(self.rho)
        self.dur_dtheta = np.zeros_like(self.rho)
        self.dur_dz = np.zeros_like(self.rho)
        self.dut_dr = np.zeros_like(self.rho)
        self.dut_dtheta = np.zeros_like(self.rho)
        self.dut_dz = np.zeros_like(self.rho)
        self.duz_dr = np.zeros_like(self.rho)
        self.duz_dtheta = np.zeros_like(self.rho)
        self.duz_dz = np.zeros_like(self.rho)
        self.dp_dr = np.zeros_like(self.rho)
        self.dp_dtheta = np.zeros_like(self.rho)
        self.dp_dz = np.zeros_like(self.rho)
