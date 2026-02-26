import matplotlib.pyplot as plt
import numpy as np
from unsflow.utils.plot_styles import N_levels, color_map


class DiffuserMeridional():
    """
    This class it is the equivalent of a meridional_process object calculated processing CFD data. In this case we simply
    insert the meridional flow fields as 2D array, in order to compute the eigenfrequencies of the annulus duct.
    """

    def __init__(self, r1, r2, b, Nst, Nsp, rho1, ur1, ut1, uz1, p1, config, mode='gauss-lobatto'):
        """
        Build the 2D arrays of the vaneless diffuser

        :param grid_refinement: refinement of the grid, to compute metrics on the finer grid, interpolating later on the coarse.
        """
        self.config = config
        self.nAxialNodes = Nst
        self.nRadialNodes = Nsp
        self.nstream = self.nAxialNodes
        self.nspan = self.nRadialNodes
        self.nPoints = self.nstream*self.nspan

        r_points = gauss_lobatto_grid_generation(Nst, r1, r2)
        b_points = gauss_lobatto_grid_generation(Nsp, 0, b)

        self.z_grid = np.zeros((Nst, Nsp))
        self.r_grid = np.zeros((Nst, Nsp))

        for ii in range(Nst):
            for jj in range(Nsp):
                self.z_grid[ii, jj] = b_points[jj]
                self.r_grid[ii, jj] = r_points[ii]

        self.r_cg, self.z_cg = self.r_grid, self.z_grid
        plt.figure()
        plt.scatter(self.z_grid, self.r_grid)
        plt.title('title')

        self.ur = ur1*r1/self.r_grid
        self.ut = ut1*r1/self.r_grid

        self.rho = (np.zeros_like(self.ur)+1)*rho1
        self.p = p1 + 0.5*rho1*(ur1**2+ut1**2) - 0.5*self.rho*(self.ur**2+self.ut**2)
        self.uz = np.zeros_like(self.ur)

        self.drho_dr = np.zeros_like(self.rho)
        self.drho_dz = np.zeros_like(self.rho)
        self.dur_dr = -ur1*r1/(self.r_grid**2)
        self.dur_dz = np.zeros_like(self.rho)
        self.dut_dr = -ut1*r1/(self.r_grid**2)
        self.dut_dz = np.zeros_like(self.rho)
        self.duz_dr = np.zeros_like(self.rho)
        self.duz_dz = np.zeros_like(self.rho)
        self.dp_dr = (-self.rho)/(2) * (self.dur_dr+self.dut_dr)
        self.dp_dz = np.zeros_like(self.rho)
        self.domain = 'unbladed'


    def normalize_data(self):
        """
        given the fundamental quantities, normalize everything
        """
        self.rho_ref = self.config.get_reference_density()
        self.u_ref = self.config.get_reference_velocity()
        self.x_ref = self.config.get_reference_length()
        self.p_ref = self.rho_ref * self.u_ref ** 2
        self.t_ref = self.config.get_reference_time()
        self.omega_ref = self.config.get_reference_omega()

        self.r_grid /= self.x_ref
        self.z_grid /= self.x_ref
        self.r_cg /= self.x_ref
        self.z_cg /= self.x_ref
        # self.r_cg_fine /= self.x_ref
        # self.z_cg_fine /= self.x_ref
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

    def contour_fields(self):
        Z = self.z_cg.copy()
        R = self.r_cg.copy()

        plt.figure()
        plt.contourf(Z, R, self.rho, levels=N_levels, cmap=color_map)
        plt.xlabel('z')
        plt.ylabel('r')
        plt.title(r'$\rho$')
        plt.colorbar()

        plt.figure()
        plt.contourf(Z, R, self.ur, levels=N_levels, cmap=color_map)
        plt.xlabel('z')
        plt.ylabel('r')
        plt.title(r'$u_r$')
        plt.colorbar()

        plt.figure()
        plt.contourf(Z, R, self.ut, levels=N_levels, cmap=color_map)
        plt.xlabel('z')
        plt.ylabel('r')
        plt.title(r'$u_{\theta}$')
        plt.colorbar()

        plt.figure()
        plt.contourf(Z, R, self.uz, levels=N_levels, cmap=color_map)
        plt.xlabel('z')
        plt.ylabel('r')
        plt.title(r'$u_{z}$')
        plt.colorbar()

        plt.figure()
        plt.contourf(Z, R, self.p, levels=N_levels, cmap=color_map)
        plt.xlabel('z')
        plt.ylabel('r')
        plt.title(r'$p$')
        plt.colorbar()

        plt.figure()
        plt.contourf(Z, R, np.sqrt(self.ur**2+self.ut**2), levels=N_levels, cmap=color_map)
        plt.xlabel('z')
        plt.ylabel('r')
        plt.title(r'$V$')
        plt.colorbar()

        plt.figure()
        plt.contourf(Z, R, self.dur_dr, levels=N_levels, cmap=color_map)
        plt.xlabel('z')
        plt.ylabel('r')
        plt.title(r'$\partial u_r / \partial r$')
        plt.colorbar()

        plt.figure()
        plt.contourf(Z, R, self.dut_dr, levels=N_levels, cmap=color_map)
        plt.xlabel('z')
        plt.ylabel('r')
        plt.title(r'$\partial u_{\theta} / \partial r$')
        plt.colorbar()

        plt.figure()
        plt.contourf(Z, R, self.dp_dr, levels=N_levels, cmap=color_map)
        plt.xlabel('z')
        plt.ylabel('r')
        plt.title(r'$\partial p / \partial r$')
        plt.colorbar()


def gauss_lobatto_grid_generation(N, x_start, x_end):
    """
    return the array of points distributed following gauss-lobatto structure
    """
    xi = np.zeros(N)
    for ii in range(len(xi)):
        xi[ii] = x_start + (x_end-x_start)*(1-np.cos(np.pi*ii/(N-1)))/2
    return xi