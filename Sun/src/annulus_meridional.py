import numpy as np


class AnnulusMeridional():
    """
    This class it is the equivalent of a meridional_process object calculated processing CFD data. In this case we simply
    insert the meridional flow fields as 2D array, in order to compute the eigenfrequencies of the annulus duct.
    """

    def __init__(self, zmin, zmax, rmin, rmax, Nz, Nr, rho, u, v, w, p):
        """
        build the 2D arrays on the meridional plane, to be compatible with meriidonal_process object
        Args:
            zmin: inlet axial cordinate
            zmax: outlet axial cordinate
            rmin: inner radius
            rmax: outer radius
            Nz: streamwise points
            Nr: spanwise points
            rho: density 2D field
            u: radial velocity 2D field
            v: tang. velocity 2D field
            w: axial velocity 2D field
            p: pressure 2D field
        """
        self.nAxialNodes = Nz
        self.nRadialNodes = Nr
        self.nstream = Nz
        self.nspan = Nr
        self.nPoints = Nz * Nr
        self.z = np.linspace(zmin, zmax, Nz)
        self.r = np.linspace(rmin, rmax, Nr)
        self.r_grid, self.z_grid = np.meshgrid(self.r, self.z)
        self.r_cg, self.z_cg = np.meshgrid(self.r, self.z)
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
