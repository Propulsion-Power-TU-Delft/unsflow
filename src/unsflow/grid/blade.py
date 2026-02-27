import warnings
import matplotlib.pyplot as plt
from numpy import array, sin, cos, tan, pi
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from .functions import cartesian_to_cylindrical, compute_gradient_least_square
from unsflow.utils.formatting import print_banner
from unsflow.utils.formatting import total_chars, total_chars_mid
from unsflow.grid.functions import *
from unsflow.grid.profile import Profile
from unsflow.grid.body_force import BodyForce
from unsflow.utils.plot_styles import *
from scipy import interpolate
import math
import os
import pandas as pd
import pickle
import plotly.graph_objects as go
from scipy.interpolate import bisplrep, bisplev
from shapely.geometry import LineString
from scipy.spatial import KDTree
from unsflow.grid.surface import Surface
from scipy.interpolate import bisplev, bisplrep, griddata
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter





class Blade:

    def __init__(self, config, iblock, iblade, bladeType):
        """
        Class used to model the blade from the file .curve, which is created during blade generation. The usual format for that file is the one of Ansys.

        Parameters
        -----------------------------------

        `config` : configuration object
        
        `iblock` : grid block counter
        
        `iblade`: blade counter
        
        `bladeType`: main or splitter
        """
        print_banner('BLADE %02i %s' %(iblade, bladeType))
        self.config = config
        
        self.x = []
        self.y = []
        self.z = []
        self.blade = []  
        self.profile = []  
        self.mark = []  
        self.leading_edge = []
        self.trailing_edge = []
        
        self.pressureSurface = Surface('Pressure Surface', config)
        self.suctionSurface = Surface('Suction Surface', config)
        self.camberSurface = Surface('Camber Surface', config)
        
        self.iblock = iblock
        self.iblade = iblade
        self.bladeType = bladeType
        self.extrapolationMethod = self.config.get_extrapolation_method()

        self.read_from_curve_file(iblade, iblock)
        print(f"{'Rescale Factor [-]:':<{total_chars_mid}}{self.config.get_coordinates_rescaling_factor():>{total_chars_mid}.3f}")
        print(f"{'Reference Length [m]:':<{total_chars_mid}}{self.config.get_reference_length():>{total_chars_mid}.3f}")
        print(f"{'Blade inlet type:':<{total_chars_mid}}{self.config.get_blade_inlet_type()[iblade]:>{total_chars_mid}}")
        print(f"{'Blade outlet type:':<{total_chars_mid}}{self.config.get_blade_outlet_type()[iblade]:>{total_chars_mid}}")
        print(f"{'Method used for blade camber reconstruction:':<{total_chars_mid}}{self.config.get_blades_camber_reconstruction()[self.iblade]:>{total_chars_mid}}")
        print(f"{'Blade edges extrapolation coefficient:':<{total_chars_mid}}{self.config.get_blade_edges_extrapolation_coefficient()[self.iblade]:>{total_chars_mid}.3f}")
        print(f"{'Camber smoothing coefficient:' :<{total_chars_mid}}{self.config.get_blade_camber_smoothing_coefficient():>{total_chars_mid}.3f}")
        print_banner()


    def read_from_curve_file(self, iblade, iblock):
        if self.bladeType == 'main':
            filepath = self.config.get_blade_curve_filepath()[iblade]
        elif self.bladeType == 'splitter':
            filepath = self.config.get_splitter_blade_curve_filepath()[iblade]
            
        print(f"{'Blade coordinate file:':<{total_chars_mid}}{filepath:>{total_chars_mid}}")

        with open(filepath) as f:
            lines = f.readlines()

        # parsing of turbogrid .curve format (does not consider the splitter blade in the main blade file).
        # The splitter blade must be given in a separate blade file
        parse_key = 'MAIN' 
        for line in lines:
            line = line.strip()
            words_list = line.split()
            if len(words_list) > 0:

                if words_list[0] == '##':                              
                    parse_key = words_list[1].upper()
                elif words_list[0] == '#':                              
                    profile_span = words_list[2]
                elif (len(words_list) == 3 or len(words_list) == 4):    
                    self.x.append(float(words_list[0]))
                    self.y.append(float(words_list[1]))
                    if self.config.invert_axial_coordinates():
                        self.z.append(-float(words_list[2]))
                    else:
                        self.z.append(float(words_list[2]))
                    self.blade.append(parse_key)
                    self.profile.append(profile_span)

                    if len(words_list) == 3:
                        self.mark.append('')                            
                    else:
                        self.mark.append(words_list[-1])                
                else:
                    pass                                               

        self.x = array(self.x, dtype=float)
        self.y = array(self.y, dtype=float)
        self.z = array(self.z, dtype=float)
        self.blade = array(self.blade)
        self.profile = array(self.profile)
        self.mark = array(self.mark)

        rescaling_factor = self.config.get_coordinates_rescaling_factor()
        self.x *= rescaling_factor
        self.y *= rescaling_factor
        self.z *= rescaling_factor
        self.theta = np.arctan2(self.y, self.x)
        self.r = np.sqrt(self.x ** 2 + self.y ** 2)

        idx_main_blade = np.where(self.blade == 'MAIN')
        self.x_main = self.x[idx_main_blade]
        self.y_main = self.y[idx_main_blade]
        self.z_main = self.z[idx_main_blade]
        self.r_main = self.r[idx_main_blade]
        self.theta_main = self.theta[idx_main_blade]

        main_profiles = np.unique(self.profile)
        main_profiles = [int(prof) for prof in main_profiles]
        main_profiles.sort()
        number_profiles = len(main_profiles)
        print(f"{'Number of profiles:':<{total_chars_mid}}{number_profiles:>{total_chars_mid}}")
        
        self.thickness = {}
        self.rc_data, self.thetac_data, self.zc_data, self.thk_data = [], [], [], []
        self.rss_data, self.thetass_data, self.zss_data = [], [], []
        self.rps_data, self.thetaps_data, self.zps_data = [], [], []
        
        profiles_to_plot = [0, number_profiles//4, number_profiles//2, 3*number_profiles//4, number_profiles-1] 
        for i in range(number_profiles):
            idx = np.where((self.profile == str(main_profiles[i])) & (self.blade == 'MAIN'))
            z = self.z_main[idx]
            r = self.r_main[idx]
            theta = self.theta_main[idx]
            x = self.x_main[idx]
            y = self.y_main[idx]

            # spline of the profile in 3D
            splineOrder = self.config.get_blade_profiles_spline_order()[self.iblade]
            tck, u = splprep([x, y, z], k=splineOrder, s=0)
            u_fine = np.linspace(0, 1, 5000)
            spline_points = splev(u_fine, tck)

            # obtain the coordinates of the spline in the blade to blade view
            is_te_cutoff = self.config.cutoff_trailing_edge(iblade)
            r1,t1,m1,z1, r2,t2,m2,z2, rc,tc,mc,zc = self.compute_meridional_coordinate(spline_points, is_te_cutoff)
            
            # distinguish the two sides between pressure and suction
            turningDirection = self.config.get_blade_turning_direction()[iblade]
            deltaTheta = np.mean(t1) - np.mean(t2)
            
            if turningDirection == 'not_relevant':
                z_ps, r_ps, theta_ps, m_ps = z1,r1,t1,m1
                z_ss, r_ss, theta_ss, m_ss = z2,r2,t2,m2
            elif deltaTheta > 0 and turningDirection == 'positive':
                z_ps, r_ps, theta_ps, m_ps = z1,r1,t1,m1
                z_ss, r_ss, theta_ss, m_ss = z2,r2,t2,m2
            elif deltaTheta > 0 and turningDirection == 'negative':
                z_ps, r_ps, theta_ps, m_ps = z2,r2,t2,m2
                z_ss, r_ss, theta_ss, m_ss = z1,r1,t1,m1
            elif deltaTheta < 0 and turningDirection == 'positive':
                z_ps, r_ps, theta_ps, m_ps = z2,r2,t2,m2
                z_ss, r_ss, theta_ss, m_ss = z1,r1,t1,m1
            elif deltaTheta < 0 and turningDirection == 'negative':
                z_ps, r_ps, theta_ps, m_ps = z1,r1,t1,m1
                z_ss, r_ss, theta_ss, m_ss = z2,r2,t2,m2
            else:
                raise ValueError('Unknown turning direction for blade pressure side detection')

            self.camberSurface.add_curve(rc*np.cos(tc), rc*np.sin(tc), zc)
            self.pressureSurface.add_curve(r_ps*np.cos(theta_ps), r_ps*np.sin(theta_ps), z_ps)
            self.suctionSurface.add_curve(r_ss*np.cos(theta_ss), r_ss*np.sin(theta_ss), z_ss)
            
            if i in profiles_to_plot:
                self.plot_b2b_profile(i, m_ps, r_ps*theta_ps, m_ss, r_ss*theta_ss, mc, rc*tc, number_profiles)

        self.camberSurface.bspline_surface_generation()
        self.r_camberSurface, self.theta_camberSurface, self.z_camberSurface = \
            self.camberSurface.get_global_bspline_surface(method='cylindrical')
        
        self.pressureSurface.bspline_surface_generation()
        self.r_psSurface, self.theta_psSurface, self.z_psSurface = \
            self.pressureSurface.get_global_bspline_surface(method='cylindrical')

        self.suctionSurface.bspline_surface_generation()
        self.r_ssSurface, self.theta_ssSurface, self.z_ssSurface = \
            self.suctionSurface.get_global_bspline_surface(method='cylindrical')
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(
            *(compute_cartesian_coords(self.r_psSurface, self.theta_psSurface, self.z_psSurface)), 
            alpha=0.5)
        ax.plot_surface(
            *(compute_cartesian_coords(self.r_ssSurface, self.theta_ssSurface, self.z_ssSurface)), 
            alpha=0.5)
        ax.plot_surface(
            *(compute_cartesian_coords(self.r_camberSurface, self.theta_camberSurface, self.z_camberSurface)), 
            alpha=0.5)
        ax.plot(self.x_main, self.y_main, self.z_main, 'o', color='k', ms=2)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')      
        ax.set_title('Reconstructed blade surfaces')
        ax.set_aspect('equal')
        plt.tight_layout()
        
        self.nr_camberSurface, self.nt_camberSurface, self.nz_camberSurface = \
            self.compute_surface_normal_vectors(
                self.r_camberSurface, 
                self.theta_camberSurface, 
                self.z_camberSurface, 
                coords='cylindrical')
        
        turningDirection = self.config.get_blade_turning_direction()[iblade]
        avgValue = np.mean(self.nt_camberSurface)
        
        if avgValue > 0 and turningDirection == 'positive':
            pass
        elif avgValue < 0 and turningDirection == 'negative':
            pass
        else:
            self.nt_camberSurface = -self.nt_camberSurface
            self.nr_camberSurface = -self.nr_camberSurface
            self.nz_camberSurface = -self.nz_camberSurface
        
    
    def plot_surface_normals(self, x1, x2, x3, nx1, nx2, nx3, coordsFrame, title, interval=10):
        def cartesian_points(r, t, z):
            return r*np.cos(t), r*np.sin(t), z
        
        if coordsFrame.lower()=='cylindric':
            x,y,z = cartesian_points(x1, x2, x3)
            ni,nj = x.shape
            nx = np.zeros((ni,nj))
            ny = np.zeros((ni,nj))
            for i in range(ni):
                for j in range(nj):
                    normal_cyl = np.array([nx1[i,j], nx2[i,j], nx3[i,j]])
                    normal_cart = cylindrical_to_cartesian(x[i,j], y[i,j], z[i,j], normal_cyl)
                    nx[i,j] = normal_cart[0]
                    ny[i,j] = normal_cart[1]
            nz = nx3
        else:
            x,y,z = x1, x2, x3
            nx,ny,nz = nx1, nx2, nx3
        
        arrow_len = (np.max(z)-np.min(z))/3
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x,y,z, alpha=0.5)
        ax.quiver(x[::10], y[::10], z[::10], nx[::10], ny[::10], nz[::10], length=arrow_len, normalize=True, color='r')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')      
        ax.set_title(title)
        ax.set_aspect('equal')
    
    

    def compute_meridional_coordinate(self, spline_points, te_cutoff):
        """
        For the x,y,z points in the spline_points list, compute the associated mprime coordinate (curvilinear abscissa), 
        distinguishing also between the two sides of the blade.
        """
        x,y,z = spline_points
        r = np.sqrt(x**2+y**2)

        if self.config.get_blade_inlet_type()[self.iblade].lower()=='axial':
            le = np.argmin(z)
        else:
            le = np.argmin(r)
        
        if self.config.get_blade_outlet_type()[self.iblade].lower()=='axial':
            te = np.argmax(z)
        else:
            te = np.argmax(r)
        
        if le>te: #swap if the list of points goes from trailing to leading edge
            le_copy = le.copy()
            le = te.copy()
            te = le_copy
            
        x1,y1,z1 = x[le:te+1], y[le:te+1], z[le:te+1]
        x2,y2,z2 = x[te:], y[te:], z[te:]
        x2 = np.concatenate((x2, x[0:le+1]))
        y2 = np.concatenate((y2, y[0:le+1]))
        z2 = np.concatenate((z2, z[0:le+1]))

        def flip_orders(xp,yp,zp):
            # flip the points if they were ordered in from trailing to leading edge
            if zp[0]>zp[-1]:
                return np.flip(xp), np.flip(yp), np.flip(zp)
            else:
                return xp, yp, zp
            
        x1,y1,z1 = flip_orders(x1,y1,z1)
        x2,y2,z2 = flip_orders(x2,y2,z2)

        def compute_mprime_coords(xp, yp, zp):
            """
            Compute the blade to blade coordinates
            """
            rp = np.sqrt(xp**2+yp**2)
            thetap = np.arctan2(yp,xp)
            sp = np.zeros_like(rp)
            for i in range(1,len(sp)):
                dr = rp[i]-rp[i-1]
                dz = zp[i]-zp[i-1]
                dm = np.sqrt(dr**2+dz**2)
                sp[i] = sp[i-1]+dm
            return rp, thetap, sp
        r1,t1,m1 = compute_mprime_coords(x1,y1,z1)
        r2,t2,m2 = compute_mprime_coords(x2,y2,z2)
        
        if te_cutoff:
            mglob = np.concatenate((m1,m2))
            mglob_cut = np.max(mglob*(1-0.005))
            r1,t1,m1,z1 = self.fix_cutoff_trailingedge(r1,t1,m1,z1,mglob_cut)
            r2,t2,m2,z2 = self.fix_cutoff_trailingedge(r2,t2,m2,z2,mglob_cut)

        rglob = np.concatenate((r1,r2))
        tglob = np.concatenate((t1,t2))
        mglob = np.concatenate((m1,m2))
        zglob = np.concatenate((z1,z2))
        
        s_camber = np.linspace(0, np.max(mglob), (len(x1)+len(x2))//2) 
        coeff = np.polyfit(mglob, rglob*tglob, deg=13) 
        rt_camber = np.polyval(coeff, s_camber)
        coeff = np.polyfit(mglob, rglob, deg=13)  
        r_camber = np.polyval(coeff, s_camber)
        theta_camber = rt_camber/r_camber
        coeff = np.polyfit(mglob, zglob, deg=13)  
        z_camber = np.polyval(coeff, s_camber)
        
        # fix the extremes
        r_camber_new, theta_camber_new, z_camber_new, s_camber_new = \
            self.fix_camberline_extremes(r_camber, theta_camber, z_camber, s_camber)

        return r1, t1, m1, z1, r2, t2, m2, z2, r_camber_new, theta_camber_new, s_camber_new, z_camber_new
    
    
    def fix_cutoff_trailingedge(self, r, t, m, z, mcut):
        idx = np.where(m<=mcut)
        return r[idx],t[idx],m[idx],z[idx]
    
    
    def fix_camberline_extremes(self, rc, tc, zc, sc):
        """Approximate the theta of the camber line with a polynomial fitted on a portion of points close to the edges

        """
        npoints = len(rc)
        portion = npoints//35 # 10% of the points
        nportions = 1
        degree = 2
        
        # LE
        sle = sc[0:portion]
        coeffs = np.polyfit(sc[portion:portion+nportions*portion], tc[portion:portion+nportions*portion], deg=degree)
        tle = np.polyval(coeffs, sle)
        snew = np.concatenate((sle, sc[portion:]))
        tnew = np.concatenate((tle, tc[portion:]))
        
        # TE 
        ste = snew[-portion:]
        coeffs = np.polyfit(snew[-portion-nportions*portion:-portion], tc[-portion-nportions*portion:-portion], deg=degree)
        tte = np.polyval(coeffs, ste)
        snew = np.concatenate((snew[0:-portion], ste))
        tnew = np.concatenate((tnew[0:-portion], tte))

        return rc, tnew, zc, sc


    def compute_quantities_on_meridional_grid(self):
        """
        Find the camber information on the blade grid via interpolation of the various functions stored on the camber.
        Check the degree of the polynomial if it is ok.
        """
        self.z_camber = self.z_grid
        self.r_camber = self.r_grid
        
        method = 'linear'
        
        self.theta_ss = robust_griddata_interpolation_with_linear_filler(
            self.z_ssSurface, 
            self.r_ssSurface, 
            self.theta_ssSurface, 
            self.z_grid, 
            self.r_grid,
            method)
        
        self.theta_ps = robust_griddata_interpolation_with_linear_filler(
            self.z_psSurface, 
            self.r_psSurface, 
            self.theta_psSurface, 
            self.z_grid, 
            self.r_grid,
            method)

        self.theta_camber = robust_griddata_interpolation_with_linear_filler(
            self.z_camberSurface, 
            self.r_camberSurface, 
            self.theta_camberSurface, 
            self.z_grid, 
            self.r_grid, 
            method)

        self.thk = self.r_grid*np.abs(self.theta_ps-self.theta_ss)
        try:
            self.thk += self.splitterThickness
        except:
            pass

        Nb = self.config.get_blades_number()[self.iblade]
        self.blockage = 1 - Nb * self.thk / (2*np.pi*self.r_grid)
        
        self.n_camber_r = robust_griddata_interpolation_with_linear_filler(
            self.z_camberSurface,
            self.r_camberSurface,
            self.nr_camberSurface,
            self.z_grid, 
            self.r_grid,
            method)

        self.n_camber_t = robust_griddata_interpolation_with_linear_filler(
            self.z_camberSurface,
            self.r_camberSurface,
            self.nt_camberSurface,
            self.z_grid, 
            self.r_grid,
            method)

        self.n_camber_z = robust_griddata_interpolation_with_linear_filler(
            self.z_camberSurface,
            self.r_camberSurface,
            self.nz_camberSurface,
            self.z_grid, 
            self.r_grid,
            method)


    def add_meridional_grid(self, zgrid, rgrid):
        """
        Add the meridional grid to the blade taken from the block object
        """
        self.z_grid, self.r_grid = zgrid, rgrid


    def compute_meridional_coordinates(self, normalize=False):
        """
        Compute the meridional streamline and spanline lengths from the given z and r grid of the blade.
        
        Parameters
        ----------
        normalize : bool, optional
            If True, the streamline length and spanline length are normalized to lie between 0 and 1.
        """
        self.streamline_length = compute_meridional_streamwise_coordinates(
            self.z_grid, 
            self.r_grid, 
            normalize=normalize)
        self.spanline_length = compute_meridional_spanwise_coordinates(
            self.z_grid, 
            self.r_grid, 
            normalize=normalize)


    def plot_meridional_coordinates(self, save_filename=None):
        """
        plot the streamline length contour
        """
        contour_template(self.z_grid, self.r_grid, self.streamwise_coord, name=r'$\bar{s}_{stw} \ \rm{[-]}$')
        if save_filename is not None:
            plt.savefig(
                self.config.get_pictures_folder_path() + '/' + save_filename + '_streamline_length.pdf', 
                bbox_inches='tight')

        contour_template(self.z_grid, self.r_grid, self.spanwise_coord, name=r'$\bar{s}_{spw} \ \rm{[-]}$')
        if save_filename is not None:
            plt.savefig(
                self.config.get_pictures_folder_path() + '/' + save_filename + '_spanline_length.pdf', 
                bbox_inches='tight')

    def plot_blockage_contour(self, save_filename=None):
        """
        plot the blockage
        """
        contour_template(self.z_grid, self.r_grid, self.blockage, name=r'$b \ \rm{[-]}$')
        if save_filename is not None:
            plt.savefig(
                self.config.get_pictures_folder_path() + '/' + save_filename + '_blockage.pdf', 
                bbox_inches='tight')
        
        contour_template(self.z_grid, self.r_grid, self.thk, name=r'$t_{\theta} \ \rm{[m]}$')
        if save_filename is not None:
            plt.savefig(
                self.config.get_pictures_folder_path() + '/' + save_filename + '_thickness.pdf', 
                bbox_inches='tight')


    def plot_camber_normal_contour(self, save_filename=None):
        """
        plot the camber normal vector contours
        """
        contour_template(self.z_camber, self.r_camber, self.n_camber_r, name=r'$n_r$')
        if save_filename is not None:
            plt.savefig(
                self.config.get_pictures_folder_path() + '/' + save_filename + '_normal_r.pdf', 
                bbox_inches='tight')

        contour_template(self.z_camber, self.r_camber, self.n_camber_t, name=r'$n_{\theta}$')
        if save_filename is not None:
            plt.savefig(
                self.config.get_pictures_folder_path() + '/' + save_filename + '_normal_theta.pdf', 
                bbox_inches='tight')

        contour_template(self.z_camber, self.r_camber, self.n_camber_z, name=r'$n_z$')
        if save_filename is not None:
            plt.savefig(
                self.config.get_pictures_folder_path() + '/' + save_filename + '_normal_z.pdf', 
                bbox_inches='tight')


    def compute_normal_vector_on_point_ij(self, i, j, xgrid, ygrid, zgrid, check=False):
        """
        For a certain point (x,y) on the camber surface z=f(x,y), find the normal vector through vectorial product
        of the vectors connecting streamwise and spanwise points. Preserve the directions to have consistent vectors
        :param i: i index of the point on the blade grid
        :param j: j index of the point on the blade grid
        :param check: if True plots the result
        """
        ni = xgrid.shape[0] - 1  # last element index
        nj = xgrid.shape[1] - 1  # last element index

        # compute versor along the first direction
        if i == ni:
            stream_v = np.array([xgrid[i, j] - xgrid[i - 2, j],
                                 ygrid[i, j] - ygrid[i - 2, j],
                                 zgrid[i, j] - zgrid[i - 2, j]])
        elif i == 0:
            stream_v = np.array([xgrid[i + 2, j] - xgrid[i, j],
                                 ygrid[i + 2, j] - ygrid[i, j],
                                 zgrid[i + 2, j] - zgrid[i, j]])
        else:
            stream_v = np.array([xgrid[i + 1, j] - xgrid[i - 1, j],
                                 ygrid[i + 1, j] - ygrid[i - 1, j],
                                 zgrid[i + 1, j] - zgrid[i - 1, j]])
        stream_v /= np.linalg.norm(stream_v)

        # compute versor along the second direction
        if j == nj:
            span_v = np.array([xgrid[i, j] - xgrid[i, j - 2],
                               ygrid[i, j] - ygrid[i, j - 2],
                               zgrid[i, j] - zgrid[i, j - 2]])
        elif j == 0:
            span_v = np.array([xgrid[i, j + 2] - xgrid[i, j],
                               ygrid[i, j + 2] - ygrid[i, j],
                               zgrid[i, j + 2] - zgrid[i, j]])
        else:
            span_v = np.array([xgrid[i, j + 1] - xgrid[i, j - 1],
                               ygrid[i, j + 1] - ygrid[i, j - 1],
                               zgrid[i, j + 1] - zgrid[i, j - 1]])
        span_v /= np.linalg.norm(span_v)

        # the normal is the vectorial product of the two
        normal = np.cross(stream_v, span_v)
        normal /= np.linalg.norm(normal)

        if check:
            arrow_len = (np.max(zgrid)-np.min(zgrid))/3
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(xgrid, ygrid, zgrid, alpha=0.3)
            ax.set_xlabel(r'$x$')
            ax.set_ylabel(r'$y$')
            ax.set_zlabel(r'$z$')
            ax.set_aspect('equal', adjustable='box')
            ax.quiver(xgrid[i, j], ygrid[i, j], zgrid[i, j], stream_v[0], stream_v[1], stream_v[2], length=arrow_len, color='red')
            ax.quiver(xgrid[i, j], ygrid[i, j], zgrid[i, j], span_v[0], span_v[1], span_v[2], length=arrow_len, color='green')
            ax.quiver(xgrid[i, j], ygrid[i, j], zgrid[i, j], normal[0], normal[1], normal[2], length=arrow_len, color='blue')
        return normal

    def find_inlet_points(self):
        """
        Find the points defining the inlet from the coordinates of the blade points.
        """
        iblade = self.iblade
        self.inlet_z = []
        self.inlet_r = []
        self.profile_types = np.unique(self.profile)
        self.profile_types = sorted(self.profile_types, key=lambda x: float(x.strip('%')))  
        # in order to have span percentages in ascending order

        for span in self.profile_types:  # for each profile
            idx = np.where(np.logical_and(self.profile == span, self.blade == 'MAIN'))
            z = self.z_main[idx]
            r = self.r_main[idx]

            blade_inlet_type = self.config.get_blade_inlet_type()
            if isinstance(blade_inlet_type, list):
                blade_inlet_type = blade_inlet_type[iblade]
            else:
                pass

            if blade_inlet_type == 'axial':
                # leading edge point
                min_z = np.min(z)  # minimum axial cordinate
                min_r_id = np.argmin(z)  # corresponding index for the r cordinate
                min_r = r[min_r_id]
            elif blade_inlet_type == 'radial':
                min_r = np.min(r)
                min_z_id = np.argmin(r)
                min_z = z[min_z_id]
            else:
                raise ValueError('Set a geometry type of the blade leading edge')

            self.inlet_z.append(min_z)
            self.inlet_r.append(min_r)
        self.inlet = np.stack((self.inlet_z, self.inlet_r), axis=1)


    def extract_inlet_points(self, iblade):
        """
        Find the points defining the inlet from the coordinates of the blade points.
        """
        inlet_z, inlet_r = [], []
        for i in range(len(self.leading_edge)):
            inlet_z.append(self.leading_edge[i][0])
            inlet_r.append(self.leading_edge[i][1])
        plt.figure()
        plt.plot(self.z_main, self.r_main, 'o')
        plt.plot(inlet_z, inlet_r, 's')
        self.inlet = np.stack((inlet_z, inlet_r), axis=1)


    def extract_outlet_points(self, iblade):
        """
        Find the points defining the inlet from the coordinates of the blade points.
        """
        outlet_z, outlet_r = [], []
        for i in range(len(self.leading_edge)):
            outlet_z.append(self.trailing_edge[i][0])
            outlet_r.append(self.trailing_edge[i][1])
        plt.figure()
        plt.plot(self.z_main, self.r_main, 'o')
        plt.plot(outlet_z, outlet_r, 's')
        self.outlet = np.stack((outlet_z, outlet_r), axis=1)


    def find_outlet_points(self):
        """
        find the points defining the inlet are taken as
        the points with minimum z cordinates for each profile of the blade.
        """
        iblade = self.iblade
        self.outlet_z = []
        self.outlet_r = []
        self.profile_types = sorted(self.profile_types, key=lambda x: float(x.strip('%')))  # to sort the list in correct way
        # in order to have span percentages in ascending order

        for span in self.profile_types:  # for each profile
            idx = np.where(np.logical_and(self.profile == span, self.blade == 'MAIN'))
            z = self.z_main[idx]
            r = self.r_main[idx]

            blade_outlet_type = self.config.get_blade_outlet_type()
            if isinstance(blade_outlet_type, list):
                blade_outlet_type = blade_outlet_type[iblade]
            else:
                pass

            if blade_outlet_type == 'radial':
                # trailing edge points
                max_r = np.max(r)
                max_z_id = np.argmax(r)
                max_z = z[max_z_id]
            elif blade_outlet_type == 'axial':
                # trailing edge points
                max_z = np.max(z)
                max_r_id = np.argmax(z)
                max_r = r[max_r_id]
            else:
                raise ValueError('Set a geometry type of the blade leading edge')

            self.outlet_z.append(max_z)
            self.outlet_r.append(max_r)
        self.outlet = np.stack((self.outlet_z, self.outlet_r), axis=1)


    def compute_surface_normal_vectors(self, x1, x2, x3, coords):
        """
        Compute the normal to a surface defined by 2d arrays.
        
        coords specify if the input coords are in cylindrical (r,theta,z) or cartesian reference frame (x,y,z)
        """
        if coords == 'cartesian':
            xgrid, ygrid, zgrid = x1, x2, x3
        elif coords == 'cylindrical':
            xgrid = x1 * np.cos(x2)
            ygrid = x1 * np.sin(x2)
            zgrid = x3
        else:
            raise ValueError('coords must be cartesian or cylindrical')

        # Create 2D NumPy array of empty arrays
        ni,nj = xgrid.shape
        cartesianNormals = np.empty(xgrid.shape, dtype=object)
        cylindricNormals = np.empty(xgrid.shape, dtype=object)

        for i in range(ni):
            for j in range(nj):
                cartesianNormals[i, j] = self.compute_normal_vector_on_point_ij(i, j, xgrid, ygrid, zgrid, check=False)

                cylindricNormals[i, j] = cartesian_to_cylindrical(xgrid[i, j],
                                                                  ygrid[i, j],
                                                                  zgrid[i, j],
                                                                  cartesianNormals[i, j])
        
        # reorder the vectors in 2d arrays
        n_camber_r = np.zeros_like(xgrid)
        n_camber_t = np.zeros_like(xgrid)
        n_camber_z = np.zeros_like(xgrid)
        for i in range(ni):
            for j in range(nj):
                n_camber_r[i, j] = cylindricNormals[i, j][0]
                n_camber_t[i, j] = cylindricNormals[i, j][1]
                n_camber_z[i, j] = cylindricNormals[i, j][2]

        if np.mean(n_camber_z)<0: # orient the normal to point towards positive axial coordinates (useful for blades)
            n_camber_z *= -1
            n_camber_r *= -1
            n_camber_t *= -1
        
        return n_camber_r, n_camber_t, n_camber_z


    def show_normal_vectors(self, save_filename=None, folder_name=None):
        """
        Show all the normal vectors on the camber surface.
        :param save_filename: if specified, saves the plots with the given name
        :param folder_name: folder name of the pictures
        """
        self.scale = (np.max(self.z_camber) - np.min(self.z_camber)) / 15
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(self.x_camber, self.y_camber, self.z_camber, alpha=0.5)
        for i in range(0, self.x_camber.shape[0]):
            for j in range(0, self.x_camber.shape[1]):
                ax.quiver(self.x_camber[i, j], self.y_camber[i, j], self.z_camber[i, j], self.normal_vectors[i, j][0],
                          self.normal_vectors[i, j][1], self.normal_vectors[i, j][2], length=self.scale, color='red')
        ax.set_box_aspect([1, 1, 1])
        ax.grid(False)
        surf.set_edgecolor('none')  # Remove edges
        surf.set_linewidth(0.1)  # Set linewidth
        surf.set_antialiased(True)  # Enable antialiasing for smoother edges
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        ax.set_zlabel(r'$z$')
        ax.set_title('normal vectors')
        if save_filename is not None:
            plt.savefig(folder_name + save_filename + '.pdf', bbox_inches='tight')


    def show_streamline_vectors(self, save_filename=None, folder_name=None):
        """
        Show all the streamline vectors on the camber surface.
        :param save_filename: if specified, saves the plots with the given name
        :param folder_name: folder name of the pictures
        """
        fig = plt.figure(figsize=self.picture_size_blank)
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(self.x_camber, self.y_camber, self.z_camber, alpha=0.5)
        for i in range(0, self.x_camber.shape[0]):
            for j in range(0, self.x_camber.shape[1]):
                ax.quiver(
                    self.x_camber[i, j], 
                    self.y_camber[i, j], 
                    self.z_camber[i, j], 
                    self.streamline_vectors[i, j][0],
                    self.streamline_vectors[i, j][1], 
                    self.streamline_vectors[i, j][2], 
                    length=self.scale, 
                    color='green')
        ax.set_box_aspect([1, 1, 1])
        ax.grid(False)
        surf.set_edgecolor('none')  
        surf.set_linewidth(0.1)  
        surf.set_antialiased(True) 
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        ax.set_zlabel(r'$z$')
        ax.set_box_aspect([1, 1, 1])
        ax.set_title('streamline vectors')
        if save_filename is not None:
            plt.savefig(folder_name + save_filename + '.pdf', bbox_inches='tight')


    def show_spanline_vectors(self, save_filename=None, folder_name=None):
        """
        Show all the spanline vectors on the camber surface.
        :param save_filename: if specified, saves the plots with the given name
        :param folder_name: folder name of the pictures
        """
        fig = plt.figure(figsize=self.picture_size_blank)
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(self.x_camber, self.y_camber, self.z_camber, alpha=0.5)
        for i in range(0, self.x_camber.shape[0]):
            for j in range(0, self.x_camber.shape[1]):
                ax.quiver(
                    self.x_camber[i, j], 
                    self.y_camber[i, j], 
                    self.z_camber[i, j], 
                    self.spanline_vectors[i, j][0],
                    self.spanline_vectors[i, j][1], 
                    self.spanline_vectors[i, j][2], 
                    length=self.scale, 
                    color='purple')
        ax.set_box_aspect([1, 1, 1])
        ax.grid(False)
        surf.set_edgecolor('none')  
        surf.set_linewidth(0.1)  
        surf.set_antialiased(True)  
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        ax.set_zlabel(r'$z$')
        ax.set_box_aspect([1, 1, 1])
        ax.set_title('spanline vectors')
        if save_filename is not None:
            plt.savefig(folder_name + '/' + save_filename + '.pdf', bbox_inches='tight')


    def compute_blade_camber_angles(self, convention='neutral'):
        """
        From the normal and streamline vectors of the camber compute:
        -gas_path_angle: gas path angle (angle in the meridional plane between camber-streamline and axial direction)
        -blade_metal_angle: angle between the camber-streamline vector and the meridional direction
        -lean_angle: check the definition on the Lamprakis paper about BFM (2025)
        -blade_blockage: as defined by Kottapalli. 
        :param convention: neutral doesn't care about the sign, but rotation-wise takes positive the angles in the
        direction of rotation
        """
        # First compute the vectors oriented along streamwise and spanwise directions of the blade
        def getCartesianCoords(r,t,z):
            return r*np.cos(t), r*np.sin(t), z
        
        # camber grid vectors
        xCamber, yCamber, zCamber = getCartesianCoords(self.r_grid, self.theta_camber, self.z_grid)
        self.camberStreamVectors = ComputeStreamwiseVectorsToSurface(xCamber, yCamber, zCamber)
        self.camberSpanVectors = ComputeSpanwiseVectorsToSurface(xCamber, yCamber, zCamber)
        
        # meridional grid vectors
        meridionalStreamVectors = ComputeMeridionalVectors(self.z_grid, self.r_grid)
        
        ni,nj = self.z_grid.shape
        self.gas_path_angle = np.zeros_like(self.z_grid)
        self.blade_metal_angle = np.zeros_like(self.z_grid)
        self.blade_lean_angle = np.zeros_like(self.z_grid)
        for i in range(ni):
            for j in range(nj):
                
                # khat is the vector following the streamwise direction on the camber
                khat = cartesian_to_cylindrical(xCamber[i,j], yCamber[i,j], zCamber[i,j], self.camberStreamVectors[i,j,:])
                khat /= np.linalg.norm(khat)
                
                # shat is the vector following the spanwise direction on the camber
                shat = cartesian_to_cylindrical(xCamber[i,j], yCamber[i,j], zCamber[i,j], self.camberSpanVectors[i,j,:])
                shat /= np.linalg.norm(shat)
                
                # mhat is the vector following the meridional grids direction (zero theta components)
                mhat = meridionalStreamVectors[i, j] / np.linalg.norm(meridionalStreamVectors[i, j])
                
                
                # 1) GAS PATH ANGLE. Convention is that is positive only if mhat has positive radial component
                self.gas_path_angle[i, j] = ComputeAngleBetweenVectors(mhat, np.array([0,0,1]))
                if mhat[0] < 0:
                    self.gas_path_angle[i, j] *= -1
                
                
                # 2) LEAN ANGLE. Convention is that is positive when 
                # # compute the n vector (see Lamprakis)
                thetahat = np.array([0.0, 1.0, 0.0])
                nhat = np.cross(thetahat, mhat)
                nhat /= np.linalg.norm(nhat)
                
                # take the one with positive radial component
                if nhat[0] < 0:
                    nhat *= -1
                
                # projection of camber-spanline on the n-theta plane. Remove every component in the meridional direction
                pntheta_s = shat - np.dot(shat, mhat) * mhat
                pntheta_s /= np.linalg.norm(pntheta_s)
                self.blade_lean_angle[i, j] = ComputeAngleBetweenVectors(pntheta_s, nhat)
                if pntheta_s[1] < 0: # convention for the sign of lean angle
                    self.blade_lean_angle[i, j] *= -1

                
                # 3) METAL ANGLE
                # convention for the metal angle is that is positive only if kvector has positive theta component
                self.blade_metal_angle[i, j] = ComputeAngleBetweenVectors(mhat, khat)
                if khat[1] < 0:
                    self.blade_metal_angle[i, j] *= -1


    def show_blade_angles_contour(self, save_filename=None, folder_name=None):
        """
        Contour of the blade angles.
        :param save_filename: if specified, saves the plots with the given name
        """
        contour_template(
            self.z_camber, self.r_camber, 180 / np.pi * self.gas_path_angle, r'$\varphi \quad \mathrm{[deg]}$')
        if save_filename is not None:
            plt.savefig(
                self.config.get_pictures_folder_path() + '/' + save_filename + '_gas_path_angle.pdf', 
                bbox_inches='tight')

        contour_template(
            self.z_camber, self.r_camber, 180 / np.pi * self.blade_metal_angle, r'$\kappa \quad \mathrm{[deg]}$')
        if save_filename is not None:
            plt.savefig(
                self.config.get_pictures_folder_path() + '/' + save_filename + '_blade_metal_angle.pdf', 
                bbox_inches='tight')

        contour_template(
            self.z_camber, self.r_camber, 180 / np.pi * self.blade_lean_angle, r'$\lambda \quad \mathrm{[deg]}$')
        if save_filename is not None:
            plt.savefig(
                self.config.get_pictures_folder_path() + '/' + save_filename + '_blade_lean_angle.pdf', 
                bbox_inches='tight')

    def compute_blade_span_indexes(self, spans, span_len):
        """
        Given a tuple of spans (normalized from 0-hub to 1-tip), return the indexes of the meridional grid 
        as close as possible to those values.
        """
        idx_spans = np.zeros(len(spans), dtype=int)
        span_len = span_len[0, :]
        for ii, span in enumerate(spans):
            idx_spans[ii] = np.argmin(np.abs(span_len-span))
        return idx_spans


    def compute_paraview_grid_points(self, coeff, debug_visual=False):
        """
        compute the grid points on the meridional plane that will be used in the paraview macro. The borders are treated
        in order to avoid the spline to not cross the volume of the .vtu file. This is needed to avoid nans.
        :param coeff: interpolation coefficient for the borders treatment
        """
        self.x_paraview = self.r_camber.copy()
        self.y_paraview = np.zeros_like(self.x_paraview)
        self.z_paraview = self.z_camber.copy()

        # treat the borders
        self.z_paraview[0, :] = self.z_paraview[0, :] + coeff*(self.z_paraview[1, :]-self.z_paraview[0, :])
        self.z_paraview[-1, :] = self.z_paraview[-1, :] + coeff * (self.z_paraview[-2, :] - self.z_paraview[-1, :])
        self.x_paraview[:, 0] = self.x_paraview[:, 0] + coeff * (self.x_paraview[:, 1] - self.x_paraview[:, 0])
        self.x_paraview[:, -1] = self.x_paraview[:, -1] + coeff * (self.x_paraview[:, -2] - self.x_paraview[:, -1])

        if debug_visual:
            plt.figure()
            plt.scatter(self.z_camber, self.r_camber, marker='o', edgecolors='black', facecolors='none')
            plt.scatter(self.z_paraview, self.x_paraview, marker='^', edgecolors='red', facecolors='none')


    def write_paraview_grid_file(self, filename='meridional_grid.csv', foldername='Grid'):
        """
        write the file requireed by Paraview to run the circumferential avg.
        The format of the file generated is:
        istream, ispan, x, y, z
        """
        os.makedirs(foldername, exist_ok=True)
        with open(foldername + '/' + filename, 'w') as file:
            for istream in range(0, self.x_paraview.shape[0]):
                for ispan in range(0, self.x_paraview.shape[1]):
                    file.write(
                        '%i,%i,%.6f,%.6f,%.6f\n' % (istream, ispan, self.x_paraview[istream, ispan],
                                                    self.y_paraview[istream, ispan], self.z_paraview[istream, ispan]))


    def compute_streamwise_meridional_projection_length(self, z1, r1, theta1, z2, r2, theta2):
        """
        Given the coordinates defining the two sides of the blade, compute the associated curvilinear 
        abscissa of their projection on the meridional plane (z,r)
        """
        blade_type = self.config.get_blade_outlet_type()[self.iblade]
        s1 = np.zeros_like(z1)
        s2 = np.zeros_like(z2)

        # leading edge index of the minimum-z coordinate, and bookkeping of the associated curve
        if np.min(z1)<np.min(z2):
            id_LE = np.argmin(z1)
            inlet_line = 1
        else:
            id_LE = np.argmin(z2)
            inlet_line = 2

        # trailing edge index of the last point, and bookkeping of the associated curve
        if blade_type == 'axial':
            if np.max(z1) >= np.max(z2):
                id_TE = np.argmax(z1)
                outlet_line = 1
            else:
                id_TE = np.argmax(z2)
                outlet_line = 2
        elif blade_type == 'radial':
            if np.max(r1) >= np.max(r2):
                id_TE = np.argmax(r1)
                outlet_line = 1
            else:
                id_TE = np.argmax(r2)
                outlet_line = 2
        else:
            raise ValueError('Unknown blade type')

        # generate the curve from leading edge to trailing edge, deciding automatically which data using 
        # thanks to previous bookkeping
        if inlet_line==1 and outlet_line==2:
            zmeridional, rmeridional = z1[id_LE:], r1[id_LE:]
        elif inlet_line==2 and outlet_line==1:
            zmeridional, rmeridional = z2[id_LE:], r2[id_LE:]
        elif inlet_line==1 and outlet_line==1:
            zmeridional, rmeridional = z1[id_LE:id_TE+1], r1[id_LE:id_TE+1]
        elif inlet_line==2 and outlet_line==2:
            zmeridional, rmeridional = z2[id_LE:id_TE+1], r2[id_LE:id_TE+1]
        else:
            raise ValueError('Problem')

        # spline of the projection on the (z,r) plane, and associated curvilinear abscissa length
        zs, rs = compute_2dSpline_curve(zmeridional, rmeridional, 1000)
        sref = np.zeros_like(zs)
        sref[0] = 0
        for iPoint in range(1, len(sref)):
            dz = zs[iPoint] - zs[iPoint-1]
            dr = rs[iPoint] - rs[iPoint-1]
            dl = np.sqrt(dz**2+dr**2)
            sref[iPoint] = sref[iPoint-1] + dl

        def find_projected_length(zp, rp, zl, rl, sl):
            """
            for zp,rp coordinate of the points, find the associated value of curvilinear length on the meridional spline
            """
            length = np.sqrt((zp-zl)**2+(rp-rl)**2)
            index = np.argmin(length)
            return sl[index]

        # for each side point, find the related curvilinear length projection on the meridional plane
        for ii in range(len(z1)):
            s1[ii] = find_projected_length(z1[ii], r1[ii], zs, rs, sref)
        for ii in range(len(z2)):
            s2[ii] = find_projected_length(z2[ii], r2[ii], zs, rs, sref)

        if self.config.get_visual_debug():
            plt.figure()
            plt.plot(z1, r1, 'o', mfc='none', label='blade side 1')
            plt.plot(z2, r2, '^', mfc='none', label='blade side 2')
            plt.plot(zs, rs, mfc='none', label='projection spline')
            plt.xlabel(r'$z$')
            plt.ylabel(r'$r$')
            plt.legend()

        return s1, s2


    def extract_coordinates_from_camber(self, s_camber, s, z, r, theta):
        """
        Extract z,r,theta for the points on the camber, using the values that created the camber in first place
        """
        z_camber, r_camber = np.zeros_like(s_camber), np.zeros_like(s_camber)
        for i in range(len(s_camber)):
            idx = np.argmin((s_camber[i]-s)**2)
            z_camber[i], r_camber[i] = z[idx], r[idx]
        return z_camber, r_camber


    def compute_metal_angle_along_camber(self, xc, yc):
        """
        Compute metal angle considering the camber
        """
        dx, dy = np.zeros_like(xc), np.zeros_like(yc)
        dx[1:-1], dy[1:-1] = xc[2:] - xc[0:-2], yc[2:] - yc[0:-2]
        dx[0], dy[0] = xc[1] - xc[0], yc[1] - yc[0]
        dx[-1], dy[-1] = xc[-1] - xc[-2], yc[-1] - yc[-2]
        x_vers = dx / np.sqrt(dx ** 2 + dy ** 2)
        y_vers = dy / np.sqrt(dx ** 2 + dy ** 2)
        alpha = np.arctan2(y_vers, x_vers)
        return alpha
    
    
    def compute_lean_angle(self):
        normalPlanar = np.sqrt(self.n_camber_t**2+self.n_camber_z**2)
        self.lean_angle = np.arctan2(self.n_camber_r, normalPlanar)
        contour_template(self.z_grid, self.r_grid, self.lean_angle*180/np.pi, r'$\lambda$ [deg]')
    

    def compute_marble_body_force(self):
        """
        Compute the body force density, using the marble thermodynamic approach based on the circumferentially averaged flow field
        """
        ni,nj = self.meridional_fields['Z'].shape
        
        
        self.meridional_fields['Force_Viscous'] = self.compute_marble_loss_force()
        self.meridional_fields['Force_Tangential'] = self.compute_marble_ftheta()

        fp_versor = np.zeros((ni,nj,3))
        for i in range(ni):
            for j in range(nj):
                w = np.array([self.meridional_fields['Velocity_Radial'][i,j],
                              self.meridional_fields['Velocity_Tangential_Relative'][i,j],
                              self.meridional_fields['Velocity_Axial'][i,j]])
                
                fp_versor[i,j,:] = -w/np.linalg.norm(w)

        self.meridional_fields['Force_Viscous_Radial'] = self.meridional_fields['Force_Viscous']*fp_versor[:,:,0]
        self.meridional_fields['Force_Viscous_Tangential'] = self.meridional_fields['Force_Viscous']*fp_versor[:,:,1]
        self.meridional_fields['Force_Viscous_Axial'] = self.meridional_fields['Force_Viscous']*fp_versor[:,:,2]
        
        self.meridional_fields['Force_Inviscid_Tangential'] = self.meridional_fields['Force_Tangential']-self.meridional_fields['Force_Viscous_Tangential']
        self.meridional_fields['Force_Inviscid_Radial'] = np.abs(self.meridional_fields['Force_Inviscid_Tangential'])*np.tan(self.lean_angle)

        self.meridional_fields['Force_Axial'] = self.meridional_fields['Force_Viscous_Axial']-(self.meridional_fields['Force_Tangential']-self.meridional_fields['Force_Viscous_Tangential'])*self.meridional_fields['Velocity_Tangential_Relative']/self.meridional_fields['Velocity_Axial']
        self.meridional_fields['Force_Inviscid_Axial'] = self.meridional_fields['Force_Axial']-self.meridional_fields['Force_Viscous_Axial']

        self.meridional_fields['Force_Inviscid'] = np.sqrt(self.meridional_fields['Force_Inviscid_Axial']**2+
                                                           self.meridional_fields['Force_Inviscid_Radial']**2+
                                                           self.meridional_fields['Force_Inviscid_Tangential']**2)
        
        self.meridional_fields['Force_Radial'] = self.meridional_fields['Force_Viscous_Radial']+self.meridional_fields['Force_Inviscid_Radial']


    def clip_contour(self, fsource, vmin, vmax):
        """
        clip a 2D array between vmin and vmax
        """
        f = fsource.copy()
        ni,nj = f.shape
        for i in range(ni):
            for j in range(nj):
                if f[i,j]<vmin:
                    f[i,j] = vmin
                elif f[i,j]>= vmax:
                    f[i,j]=vmax
                else:
                    pass
        return f


    def compute_marble_ftheta(self, method='local'):
        """
        Compute the global tangential force.
        Method <distributed> spread the gradient of angular momentum linearly from inlet to outlet
        Method <local> uses local gradients of the angular momentum
        """
        self.meridional_fields['Velocity_Meridional'] = np.sqrt(self.meridional_fields['Velocity_Axial']**2 + self.meridional_fields['Velocity_Radial']**2)
        um = self.meridional_fields['Velocity_Meridional'].copy()
        ut = self.meridional_fields['Velocity_Tangential'].copy()
        r = self.meridional_fields['R'].copy()
        z = self.meridional_fields['Z'].copy()
        ftheta = np.zeros_like(um)
        
        if method=='local':
            drut_dz, drut_dr = compute_gradient_least_square(self.meridional_fields['Z'], self.meridional_fields['R'], r*ut)
            contour_template(z, r, r, r'$r$')
            contour_template(z, r, ut, r'$u_{\theta}$')
            contour_template(z, r, r*ut, r'$r u_{\theta}$')
            contour_template(z, r, drut_dz, r'$\partial(r u_{\theta}) / \partial z$')
            contour_template(z, r, drut_dr, r'$\partial(r u_{\theta}) / \partial r$')
            ftheta = 1/r*(drut_dz*self.meridional_fields['Velocity_Axial']+drut_dr*self.meridional_fields['Velocity_Radial'])
        
        elif method=='distributed':
            for j in range(um.shape[1]):
                deltaF = (r[-1,j]*ut[-1,j])-(r[0,j]*ut[0,j])
                deltaM = self.streamline_length[-1,j]-self.streamline_length[0,j]
                ftheta[:,j] = um[:,j]/r[:,j]*deltaF/deltaM
        
        else:
            raise ValueError('Method unknown')
        
        contour_template(z, r, ftheta, r'$f_{\theta}$', vmax=0)
        return ftheta
    

    def compute_marble_loss_force(self):
        floss = np.zeros_like(self.meridional_fields['R'])

        T = self.meridional_fields['Temperature'].copy()
        W = np.sqrt(self.meridional_fields['Velocity_Axial']**2 + self.meridional_fields['Velocity_Radial']**2 + self.meridional_fields['Velocity_Tangential_Relative']**2)
        Um = np.sqrt(self.meridional_fields['Velocity_Axial']**2 + self.meridional_fields['Velocity_Radial']**2)

        for j in range(T.shape[1]):
            deltaS = self.meridional_fields['Entropy'][-1,j]-self.meridional_fields['Entropy'][0,j]
            deltaM = self.streamline_length[-1,j]-self.streamline_length[0,j]

            floss[:,j] = T[:,j]*Um[:,j]/W[:,j]*deltaS/deltaM
        return floss 
    

    def cut_blade_tip(self, clearance_meters):
        """
        Remove every force component in the gap from the shroud described by clearance_meters
        """
        gap = clearance_meters
        self.compute_spanline_length()
        ni,nj = self.meridional_fields['R'].shape

        for i in range(ni):
            for j in range(nj):
                distance = self.spanline_length[i,-1]-self.spanline_length[i,j]
                if distance<=gap:
                    self.meridional_fields['Force_Axial'][i,j] = 0
                    self.meridional_fields['Force_Tangential'][i,j] = 0
                    self.meridional_fields['Force_Radial'][i,j] = 0
    

    def compute_kiwada_body_force(self):
        """
        Compute the force using the relations of Kiwada, for the global force, already decomposed in in its components.
        """
        R = self.meridional_fields['R'].copy()
        Z = self.meridional_fields['Z'].copy()
        B = self.blockage.copy()
        
        dbdz, dbdr = compute_gradient_least_square(Z, R, self.blockage)

        contour_template(Z, R, B, r'$b$')
        contour_template(Z, R, dbdz, r'$\partial_z b$')
        contour_template(Z, R, dbdr, r'$\partial_r b$')

        # axial equation
        dA1dz = compute_gradient_least_square(Z, R, B*self.meridional_fields['A1'])[0]
        dA2dr = compute_gradient_least_square(Z, R, B*R*self.meridional_fields['A2'])[1]
        self.meridional_fields['Force_Axial'] = 1/B*dA1dz+1/B/R*dA2dr

        dR1dz = compute_gradient_least_square(Z, R, B*self.meridional_fields['A2'])[0]
        dR2dr = compute_gradient_least_square(Z,R, B*R*self.meridional_fields['R2'])[1]
        self.meridional_fields['Force_Radial'] = 1/B*dR1dz+1/B/R*dR2dr-self.meridional_fields['R3']/R

        dT1dz = compute_gradient_least_square(Z, R, B*self.meridional_fields['T1'])[0]
        dT2dr = compute_gradient_least_square(Z, R, B*R*self.meridional_fields['T2'])[1]
        self.meridional_fields['Force_Tangential'] = 1/B*dT1dz + 1/B/R*dT2dr + self.meridional_fields['T3']/R

        fmag = np.sqrt(
            self.meridional_fields['Force_Radial']**2 + 
            self.meridional_fields['Force_Tangential']**2 + 
            self.meridional_fields['Force_Axial']**2)

        ni,nj = R.shape
        self.meridional_fields['Force_Viscous'] = np.zeros((ni,nj))
        for i in range(ni):
            for j in range(nj):
                w = np.array([self.meridional_fields['Velocity_Radial'][i,j],
                              self.meridional_fields['Velocity_Tangential_Relative'][i,j],
                              self.meridional_fields['Velocity_Axial'][i,j]])
                fg = np.array([self.meridional_fields['Force_Radial'][i,j],
                               self.meridional_fields['Force_Tangential'][i,j],
                               self.meridional_fields['Force_Axial'][i,j]])
                
                fp_vers = -w/np.linalg.norm(w)
                self.meridional_fields['Force_Viscous'][i,j] = np.dot(fg, fp_vers)
        self.meridional_fields['Force_Inviscid'] = np.sqrt(fmag**2-self.meridional_fields['Force_Viscous']**2)
    

    def cure_hub(self, span_extent, f):
        """
        For f defined on the meridional grid, cure the field within hub and the span extent. 
        Cure means copying from the first acceptable value outside of the span extent.
        """
        gap = span_extent
        self.compute_spanline_length(normalize=True)
        ni,nj = f.shape
        for i in range(ni):
            j = 0
            while self.spanline_length[i,j]<span_extent:
                j_id = j
                j += 1
            f[i,0:j_id] = f[i,j_id]
    
    def cure_shroud(self, span_extent, f):
        """
        For f defined on the meridional grid, cure the field within shroud and the span extent. 
        Cure means copying from the first acceptable value outside of the span extent.
        """
        self.compute_spanline_length(normalize=True)
        ni,nj = f.shape
        for i in range(ni):
            j = nj-1
            while self.spanline_length[i,j]>1-span_extent:
                j_id = j
                j -= 1
            f[i,j_id:] = f[i,j_id]
    
    def extrapolate_camber_vector(self):
        self.n_camber_r = self.extrapolate_2dfield_stream_span(self.z_grid, self.r_grid, self.n_camber_r, stream=True, span=False)
        self.n_camber_t = self.extrapolate_2dfield_stream_span(self.z_grid, self.r_grid, self.n_camber_t, stream=True, span=False)
        self.n_camber_z = self.extrapolate_2dfield_stream_span(self.z_grid, self.r_grid, self.n_camber_z, stream=True, span=False)
        
        self.n_camber_r /= np.sqrt(self.n_camber_r**2+self.n_camber_t**2+self.n_camber_z**2)
        self.n_camber_t /= np.sqrt(self.n_camber_r**2+self.n_camber_t**2+self.n_camber_z**2)
        self.n_camber_z /= np.sqrt(self.n_camber_r**2+self.n_camber_t**2+self.n_camber_z**2)
        
        self.blade_metal_angle = self.extrapolate_2dfield_stream_span(self.z_grid, self.r_grid, self.blade_metal_angle, stream=True, span=False)
    
    
    def extrapolate_2dfield_stream_span(self, zgrid, rgrid, field, stream=True, span=False):
        """
        Extrapolate the 2d field over the last portion close to leading and trailing edge
        """
        normalizedStreamLength = compute_meridional_streamwise_coordinates(zgrid, rgrid, normalize=True)
        coefficient = self.config.get_blade_edges_extrapolation_coefficient()[self.iblade]
        if coefficient==0:
            return field
        
        def LinearExtrapolation(x, y, xnew):
            ynew = np.zeros_like(xnew)
            yprime = np.gradient(y, x, edge_order=2)
            if xnew[0]<=x[0]:
                # left extrapolation
                for i in range(len(xnew)):
                    ynew[i] = y[0]+yprime[0]*(xnew[i]-x[0])
            else:
                for i in range(len(xnew)):
                    ynew[i] = y[-1]+yprime[0]*(xnew[i]-x[-1])
            return ynew
        
        def ExtrapolateDataStream(f):
            ni,nj = f.shape
            for j in range(nj):
                leadingPoints = np.where(normalizedStreamLength[:,j]<=coefficient)
                trailingPoints = np.where(normalizedStreamLength[:,j]>=1-coefficient)
                internalPoints = np.where((normalizedStreamLength[:, j] > coefficient) & (normalizedStreamLength[:, j] < 1 - coefficient))
                
                f[leadingPoints,j] = LinearExtrapolation(normalizedStreamLength[internalPoints,j].flatten(), f[internalPoints,j].flatten(), normalizedStreamLength[leadingPoints,j].flatten())
                f[trailingPoints,j] = LinearExtrapolation(normalizedStreamLength[internalPoints,j].flatten(), f[internalPoints,j].flatten(), normalizedStreamLength[trailingPoints,j].flatten())                
            return f
        
        if stream:
            newField = ExtrapolateDataStream(field)
        
        normalizedSpanLength = compute_meridional_spanwise_coordinates(zgrid, rgrid, normalize=True)
        def ExtrapolateDataSpan(f):
            ni,nj = f.shape
            for i in range(ni):
                hubPoints = np.where(normalizedSpanLength[i,:]<=coefficient)
                shroudPoints = np.where(normalizedSpanLength[i,:]>=1-coefficient)
                internalPoints = np.where((normalizedSpanLength[i,:] > coefficient) & (normalizedSpanLength[i,:] < 1 - coefficient))
                
                f[i, hubPoints] = LinearExtrapolation(normalizedSpanLength[i,internalPoints].flatten(), f[i,internalPoints].flatten(), normalizedSpanLength[i,hubPoints].flatten())
                f[i, shroudPoints] = LinearExtrapolation(normalizedSpanLength[i,internalPoints].flatten(), f[i,internalPoints].flatten(), normalizedSpanLength[i,shroudPoints].flatten())
            return f
        
        if span:
            newField = ExtrapolateDataSpan(newField)
        
        return newField
        
    
    def extract_body_force(self, metal_angle):
        """Given the meridional fields of the body force extraction procedure stored in the pickle at filepath, 
        compute the relevant body forces fields
        """
        self.bodyForce = BodyForce(self.config, self.iblade)
        self.bodyForce.InterpolateCircumferentialAveragedFields(self.z_grid, self.r_grid)
        
        extractionMethod = self.config.get_body_force_extraction_method()
        if extractionMethod.lower() == 'marble':
            self.bodyForce.ComputeBodyForceMarble(self.n_camber_z, self.n_camber_r, self.n_camber_t)
        elif extractionMethod.lower() == 'kiwada':
            self.bodyForce.ComputeBodyForceKiwada(self.blockage)
        else:
            raise ValueError(f"Unknown body force extraction method: {extractionMethod}")
            
        self.bodyForce.HubShroudBodyForceExtrapolation()
        
        calibrationMethod = self.config.get_body_force_calibration_method()
        self.bodyForce.ComputeCalibrationCoefficients(
            calibrationMethod, self.n_camber_z, self.n_camber_r, self.n_camber_t, self.blockage)
        
    
    def interpolate_body_force(self):
        """interpolate the body force fields
        """
        self.bodyForce = BodyForce(self.config, self.iblade)
        
        filepath = self.config.get_blade_body_force_filepath(self.iblade)
        with open(filepath, 'rb') as f:
            blade = pickle.load(f)
        
        # build the three dictionnaries needed
        self.bodyForce.meridionalFields = {}
        self.bodyForce.bodyForceFields = {}
        self.bodyForce.calibrationCoefficients = {}
        
        for key in blade.bodyForce.meridionalFields.keys():
            self.bodyForce.meridionalFields[key] = robust_griddata_interpolation_with_linear_filler(blade.bodyForce.meridionalFields["Axial_Coordinate"], 
                                                                                                    blade.bodyForce.meridionalFields["Radial_Coordinate"], 
                                                                                                    blade.bodyForce.meridionalFields[key], 
                                                                                                    self.z_grid, self.r_grid)
        
        for key in blade.bodyForce.bodyForceFields.keys():
            self.bodyForce.bodyForceFields[key] = robust_griddata_interpolation_with_linear_filler(blade.bodyForce.meridionalFields["Axial_Coordinate"], 
                                                                                                    blade.bodyForce.meridionalFields["Radial_Coordinate"], 
                                                                                                    blade.bodyForce.bodyForceFields[key], 
                                                                                                    self.z_grid, self.r_grid)
        
        for key in blade.bodyForce.calibrationCoefficients.keys():
            self.bodyForce.calibrationCoefficients[key] = robust_griddata_interpolation_with_linear_filler(blade.bodyForce.meridionalFields["Axial_Coordinate"], 
                                                                                                        blade.bodyForce.meridionalFields["Radial_Coordinate"], 
                                                                                                        blade.bodyForce.calibrationCoefficients[key], 
                                                                                                        self.z_grid, self.r_grid)
        
    
    
    def compute_endwalls_gaps(self):
        tip_gap = self.config.get_blade_tip_gap()[self.iblade]
        hub_gap = self.config.get_blade_hub_gap()[self.iblade]
        
        self.bladePresent = np.ones_like(self.z_grid)
        
        spanLength = compute_meridional_spanwise_coordinates(self.z_grid, self.r_grid)
        
        # hub cut
        for i in range(spanLength.shape[0]):
            mask = np.where(spanLength[i,:] <= 0+(hub_gap-1e-12))
            self.bladePresent[i,mask] = 0
        
        # tip cut
        for i in range(spanLength.shape[0]):
            mask = np.where(spanLength[i,:] >= spanLength[i,-1]-(tip_gap-1e-12))
            self.bladePresent[i,mask] = 0
        
        contour_template(self.z_grid, self.r_grid, self.bladePresent, 'bladePresent')
    
    
    def compute_splitter_thickness(self, pressSide, suctSide):
        r_eval = self.r_grid
        z_eval = self.z_grid
        
        def splitter_interpolation(xpoints, ypoints, zpoints, x_evaluation, y_evaluation, filler, method='linear'):
            z_evaluation = griddata((xpoints.flatten(), ypoints.flatten()), zpoints.flatten(), (x_evaluation, y_evaluation), method=method, fill_value=np.nan)
            ni,nj = z_evaluation.shape
            for i in range(ni):
                if np.isnan(z_evaluation[i,0]):
                    z_evaluation[i,0] = z_evaluation[i,1]
                if np.isnan(z_evaluation[i,-1]):
                    z_evaluation[i,-1] = z_evaluation[i,-2]
            
            nan_mask = np.isnan(z_evaluation)
            z_evaluation[nan_mask] = filler
            return z_evaluation
        
        
        thetaPS = splitter_interpolation(pressSide[0], pressSide[1], pressSide[2], z_eval, r_eval, filler=0)
        thetaSS = splitter_interpolation(suctSide[0], suctSide[1], suctSide[2], z_eval, r_eval, filler=0)
        self.splitterThickness = self.r_grid*(np.abs(thetaPS-thetaSS))
    
    
    def infer_body_force(self):
        """interpolate the body force regression matrices onto the blade grid points
        """
        filepath = self.config.get_body_force_inference_path(self.iblade)
        
        with open(filepath, 'rb') as f:
            coeffMatrixInference = pickle.load(f)
        
        streamwiseCoord = compute_meridional_streamwise_coordinates(self.z_grid, self.r_grid, normalize=True)
        spanwiseCoord = compute_meridional_spanwise_coordinates(self.z_grid, self.r_grid, normalize=True)
        
        self.inviscidForceInference, self.viscousForceInference = self.inferBodyForcePolynomials(coeffMatrixInference, (streamwiseCoord, spanwiseCoord))
        
        # now pad the coefficients for those points out of the limits of robust interpolation
        hubLimit = 0.10
        shroudLimit = 0.9
        leLimit = 0.05
        teLimit = 0.95
        ni, nj = streamwiseCoord.shape
        
        # stream direction padding
        for j in range(nj):
            idxOut = np.where(streamwiseCoord[:,j] < leLimit)
            if idxOut[0].size > 0:
                idxRef = idxOut[0].max()+1
                self.inviscidForceInference[idxOut, j, :] = self.inviscidForceInference[idxRef, j, :]
                self.viscousForceInference[idxOut, j, :] = self.viscousForceInference[idxRef, j, :]
            
            idxOut = np.where(streamwiseCoord[:,j] > teLimit)
            if idxOut[0].size > 0:
                idxRef = idxOut[0].min()-1
                self.inviscidForceInference[idxOut, j, :] = self.inviscidForceInference[idxRef, j, :]
                self.viscousForceInference[idxOut, j, :] = self.viscousForceInference[idxRef, j, :]
        
        # span direction padding
        for i in range(ni):
            idxOut = np.where(spanwiseCoord[i,:] < hubLimit)
            if idxOut[0].size > 0:
                idxRef = idxOut[0].max()+1
                self.inviscidForceInference[i, idxOut, :] = self.inviscidForceInference[i, idxRef, :]
                self.viscousForceInference[i, idxOut, :] = self.viscousForceInference[i, idxRef, :]
            
            idxOut = np.where(spanwiseCoord[i,:] > shroudLimit)
            if idxOut[0].size > 0:
                idxRef = idxOut[0].min()-1
                self.inviscidForceInference[i, idxOut, :] = self.inviscidForceInference[i, idxRef, :]
                self.viscousForceInference[i, idxOut, :] = self.viscousForceInference[i, idxRef, :]

        
    def inferBodyForcePolynomials(self, coeffMatrixInference, features):
        streamwiseData = coeffMatrixInference['Streamwise']
        spanwiseData = coeffMatrixInference['Spanwise']
        coeffMatrixInviscid = coeffMatrixInference['coeffMatrixInviscid']
        coeffMatrixViscous = coeffMatrixInference['coeffMatrixViscous']
        
        stream_min_max = coeffMatrixInference['Streamwise_min_max']
        span_min_max = coeffMatrixInference['Spanwise_min_max']
        

        streamwiseEval = features[0]
        spanwiseEval = features[1]
        
        ni,nj = streamwiseEval.shape
        nk = coeffMatrixInviscid.shape[1]
        
        inviscidCoeff = np.zeros((ni,nj,nk))
        viscousCoeff = np.zeros((ni,nj,nk))
                
        #automatic
        for k in range(nk):
            inviscidCoeff[:,:,k] = griddata_interpolation_with_nearest_filler(inverse_rescaling_minmax(streamwiseData, stream_min_max), 
                                                                              inverse_rescaling_minmax(spanwiseData, span_min_max), 
                                                                              coeffMatrixInviscid[:,k], 
                                                                              streamwiseEval, spanwiseEval)
            viscousCoeff[:,:,k] = griddata_interpolation_with_nearest_filler(inverse_rescaling_minmax(streamwiseData, stream_min_max), 
                                                                              inverse_rescaling_minmax(spanwiseData, span_min_max), 
                                                                              coeffMatrixViscous[:,k], 
                                                                              streamwiseEval, spanwiseEval)
                
        return inviscidCoeff, viscousCoeff
    
    
    def getOutputFieldsForMeshBFM(self):
        outputFields = self.config.get_turbo_BFM_mesh_output_fields()
        outputDict = {}
        
        if 'blockage' in outputFields:
            outputDict['blockage'] = self.blockage
            
        if 'camber' in outputFields:
            outputDict['normalAxial'] = self.n_camber_z
            outputDict['normalRadial'] = self.n_camber_r
            outputDict['normalTangential'] = self.n_camber_t
        
        if 'blade_angles' in outputFields:
            
            outputDict['bladeMetalAngle'] = self.blade_metal_angle
            outputDict['bladeCamberCurvature'] = self.camber_curvature
            outputDict['bladeLeanAngle'] = self.blade_lean_angle
            outputDict['bladeGasPathAngle'] = self.gas_path_angle
        
        if 'rpm' in outputFields:
            omega = self.config.get_omega_shaft()[self.iblock]
            outputDict['rpm'] = omega*60/(2*np.pi)+ np.zeros_like(self.z_grid)
        
        if 'stwl' in outputFields:
            stwl = compute_meridional_streamwise_coordinates(self.z_grid, self.r_grid, normalize=False)
            outputDict['streamwiseLength'] = stwl
        
        if 'spwl' in outputFields:
            outputDict['spanwiseLength'] = self.spanwise_normalized_coord
        
        if 'blade_present' in outputFields:
            outputDict['bladePresent'] = self.bladePresent+ np.zeros_like(self.z_grid)
        
        if 'number_blades' in outputFields:
            outputDict['numberBlades'] = self.config.get_blades_number()[self.iblade] + np.zeros_like(self.z_grid)
        
        return outputDict
        
        
    
    def plot_b2b_profile(self, iprofile, x_ps, y_ps, x_ss, y_ss, x_c, y_c, number_profiles):
        plt.figure()
        plt.plot(x_ps, y_ps, '-', color='C0', label='Pressure Side')
        plt.plot(x_ss, y_ss, '-', color='C1', label='Suction Side')
        plt.plot(x_c, y_c, '-.', color='C2', ms=2, label='Camber')
        plt.xlabel(r'$m$ [m]')
        plt.ylabel(r'$r \theta$ [m]')
        plt.legend()
        plt.title(f'Profile {iprofile+1} of {number_profiles}')
        plt.grid(alpha=grid_opacity)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.savefig(
            self.config.get_pictures_folder_path() + '/blade_%i_%s_b2b-profile_%.2f.pdf' 
            %(self.iblade, self.bladeType,(iprofile+1)/number_profiles), 
            bbox_inches='tight')
        
        

                
                



























