#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 16:53:11 2023
@author: F. Neri, TU Delft
"""
from numpy import sqrt
import pandas as pd
from .functions import *


class CfdData:
    """
    Class storing the CFD data, extracted from a CFD software. For the moment only the constructor for csv files
    extracted from Ansys is implemented
    """
    
    
    def __init__(self, filepath, rho_ref, x_ref, rpm_ref, rpm_drag, T_ref, cut_block = None, blade = None,
                 normalize = True, file_type='Ansys .csv', verbose=False, ):
        """
        read the data from the csv file extracted from Ansys CFD-post. rpm_drag is used to compute relative and drag velocities
        If normalize = True, it stores the normalization quantities:
        rho_ref: reference density, for air can be 1.014 [kg/m3]
        omega_ref: angular speed of the shaft [rpm]
        x_ref: tip radius of the blade at leading edge [m]
        T_ref: can be standard temperature [K]
        """
        self.filepath = filepath
        self.omega_shaft = rpm_drag*2*np.pi/60
        self.verbose = verbose
        self.cut_block = cut_block
        self.normalize = normalize
        if self.normalize:
            self.rho_ref = rho_ref
            self.x_ref = x_ref
            self.rpm_ref = np.abs(rpm_ref)
            self.T_ref = T_ref
            self.compute_normalization_quantities()
        if blade is not None:
            self.blade = blade
        self.file_type = file_type
        if self.file_type == 'Ansys .csv':
            if self.verbose:
                print('reading CFD data...')
            self.read_from_ansys_csv(normalize = normalize)
        
        
        
    def read_from_ansys_csv(self, normalize = False):
        """
        read the data from a CSV file extracted from Ansys. Check that all the quantities are stored in the file, 
        as well as the correct names of the variables
        """
        self.data = pd.read_csv(self.filepath, skiprows=range(5))
        self.x = self.data[' X [ m ]'].values
        self.y = self.data[' Y [ m ]'].values
        self.z = self.data[' Z [ m ]'].values
        self.r = sqrt(self.x**2 + self.y**2)
        self.theta = np.arctan2(self.y, self.x)
        self.rho = self.data[' Density [ kg m^-3 ]'].values
        self.ux = self.data[' Velocity in Stn Frame u [ m s^-1 ]'].values
        self.uy = self.data[' Velocity in Stn Frame v [ m s^-1 ]'].values
        self.uz = self.data[' Velocity in Stn Frame w [ m s^-1 ]'].values
        self.p = self.data[' Pressure [ Pa ]'].values
        self.T = self.data[' Temperature [ K ]'].values
        self.s = self.data[' Static Entropy [ J kg^-1 K^-1 ]'].values

        #gradients
        self.drho_dx = self.data[' Density.Gradient X [ kg m^-4 ]'].values
        self.drho_dy = self.data[' Density.Gradient Y [ kg m^-4 ]'].values
        self.drho_dz = self.data[' Density.Gradient Z [ kg m^-4 ]'].values
        self.dux_dx = self.data[' Velocity in Stn Frame u.Gradient X [ s^-1 ]'].values
        self.dux_dy = self.data[' Velocity in Stn Frame u.Gradient Y [ s^-1 ]'].values
        self.dux_dz = self.data[' Velocity in Stn Frame u.Gradient Z [ s^-1 ]'].values
        self.duy_dx = self.data[' Velocity in Stn Frame v.Gradient X [ s^-1 ]'].values
        self.duy_dy = self.data[' Velocity in Stn Frame v.Gradient Y [ s^-1 ]'].values
        self.duy_dz = self.data[' Velocity in Stn Frame v.Gradient Z [ s^-1 ]'].values
        self.duz_dx = self.data[' Velocity in Stn Frame w.Gradient X [ s^-1 ]'].values
        self.duz_dy = self.data[' Velocity in Stn Frame w.Gradient Y [ s^-1 ]'].values
        self.duz_dz = self.data[' Velocity in Stn Frame w.Gradient Z [ s^-1 ]'].values
        self.dp_dx = self.data[' Pressure.Gradient X [ kg m^-2 s^-2 ]'].values
        self.dp_dy = self.data[' Pressure.Gradient Y [ kg m^-2 s^-2 ]'].values
        self.dp_dz = self.data[' Pressure.Gradient Z [ kg m^-2 s^-2 ]'].values
        self.ds_dx = self.data[' Static Entropy.Gradient X [ m s^-2 K^-1 ]'].values
        self.ds_dy = self.data[' Static Entropy.Gradient Y [ m s^-2 K^-1 ]'].values
        self.ds_dz = self.data[' Static Entropy.Gradient Z [ m s^-2 K^-1 ]'].values


        if normalize:
            self.normalize_data()
            if self.verbose:
                print('CFD data normalized')
        else:
            if self.verbose:
                print('CFD data NOT normalized')



    def process_from_ansys_csv(self, cut=False):
        """
        if cut==True, it cuts the domain thanks to the information contained in the cut_block border.
        Then it computes the derived quantities (quantities in cylindrical cordinates)
        """
        if (self.file_type == 'Ansys .csv' and self.cut_block is not None and cut=='True'):
            if self.verbose:
                print('cutting domain...')
            self.cut_domain(self.cut_block.border)

        self.compute_derived_quantities()
        
        
        
    def compute_derived_quantities(self):
        """
        Compute other derived quantities, projecting in the radial and tangential direction when needed
        """

        # velocity magnitude
        self.u_mag = sqrt(self.ux ** 2 + self.uy ** 2 + self.uz ** 2)

        #velocity in cylindrical cordinates
        self.ur, self.ut = project_vector_to_cylindrical(self.ux, self.uy, self.theta)

        # #gradients in cylindrical cordinates
        self.drho_dr, self.drho_dtheta = project_scalar_gradient_to_cylindrical(self.drho_dx, self.drho_dy,
                                                                            self.r, self.theta)
        self.dur_dr, self.dur_dtheta, self.dut_dr, self.dut_dtheta = project_velocity_gradient_to_cylindrical(self.dux_dx,
                                                            self.dux_dy, self.duy_dx, self.duy_dy, self.r, self.theta)
        self.duz_dr, self.duz_dtheta = project_scalar_gradient_to_cylindrical(self.duz_dx, self.duz_dy, self.r, self.theta)
        self.dur_dz = cos(self.theta)*self.dux_dz + sin(self.theta)*self.duy_dz
        self.dut_dz = -sin(self.theta)*self.dux_dz + cos(self.theta)*self.duy_dz
        self.dp_dr, self.dp_dtheta = project_scalar_gradient_to_cylindrical(self.dp_dx, self.dp_dy, self.r, self.theta)
        self.ds_dr, self.ds_dtheta = project_scalar_gradient_to_cylindrical(self.ds_dx, self.ds_dy, self.r, self.theta)

        # relative quantities
        self.ut_drag = self.r * self.omega_shaft  # drag velocity
        self.ut_rel = self.ut - self.ut_drag  # relative velocity
        self.u_mag_rel = sqrt(self.ur ** 2 + self.ut_rel ** 2 + self.uz ** 2)



    def compute_bfm_radial_fields(self):
        """
        computes the 3D fields necessary for the radial body force model, as described in my draft appendix, Radial Machines
        model. The 3D fields described here will be circumferentially averaged. Mu, coefficient of the resistant force
        is calculated directly in the 2D meridional object, because too expensive to be done in 3D.

        To be checked
        """

        # ideal flow direction (following streamwise positions on the camber surface), in cartesian cordinates.
        # No need to save x,y component
        tau_x = np.array(self.streamline_vec)[:, 0]
        tau_y = np.array(self.streamline_vec)[:, 1]
        self.tau_z = np.array(self.streamline_vec)[:, 2]

        # normal flow direction, identified by the camber surface, in cartesian cordinates. No need to save x,y components
        n_x = np.array(self.normal_vec)[:, 0]
        n_y = np.array(self.normal_vec)[:, 1]
        self.n_z = np.array(self.normal_vec)[:, 2]


        # convert in cylindrical cordinates
        self.tau_r, self.tau_t = project_vector_to_cylindrical(tau_x, tau_y, self.theta)
        self.n_r, self.n_t = project_vector_to_cylindrical(n_x, n_y, self.theta)

        # real flow direction in the relative frame of reference. t is a unitary vector
        self.t_r = self.ur / self.u_mag_rel
        self.t_t = self.ut_rel / self.u_mag_rel
        self.t_z = self.uz / self.u_mag_rel

        # k-coeff as described in eq. 10 of article "Flow stability model of ceontrifugal compressors" - Sun et Al. 2016
        self.k = (self.dut_dr+self.uz/self.ur*self.dut_dz+self.ut/self.r) / (self.u_mag_rel*self.tau_t-self.ut_rel)

        # F_{n,theta} as described in eq 9 of the same article
        self.F_ntheta = self.k*self.ur*(self.u_mag_rel*self.tau_t-self.ut_rel)
        self.F_nr = self.F_ntheta * self.n_r / self.n_t
        self.F_nz = self.F_ntheta * self.n_z / self.n_t

        # a1, a2, a3 as defined in equation 16 (corrected by me) of the same article (mu is computed on the meridional grid)
        self.a1 = self.k * ((self.u_mag_rel + self.ur**2/self.u_mag_rel)*self.tau_t - self.ut_rel)
        self.a2 = self.k * self.ur * (self.ut_rel/self.u_mag_rel*self.tau_t -1)
        self.a3 = self.k * self.ur * self.tau_t * self.uz / self.u_mag_rel

        # elements of the normal force perturbation (steady-state) coefficient matrix, as defined in my draft, eq C.20. So
        # this matrix still needs to be multiplied by lambda
        self.Fn_prime_ss_00 = self.n_r / self.n_t * self.a1
        self.Fn_prime_ss_01 = self.n_r / self.n_t * self.a2
        self.Fn_prime_ss_02 = self.n_r / self.n_t * self.a3
        self.Fn_prime_ss_10 = self.a1
        self.Fn_prime_ss_11 = self.a2
        self.Fn_prime_ss_12 = self.a3
        self.Fn_prime_ss_20 = self.n_z / self.n_t * self.a1
        self.Fn_prime_ss_21 = self.n_z / self.n_t * self.a2
        self.Fn_prime_ss_22 = self.n_z / self.n_t * self.a3

        # elements of the resistant force Ft perturbation (steady state) coefficient matrix, as defined in my draft, eq. C.22
        # where the influence on mu has still to be included. Mu will be calculated on the meridional grid and then added in
        # a second moment, since otherwise is too cpu expensive to do in 3D. So, this matrix still needs to be multiplied
        # by mu and lambda
        self.Ft_prime_ss_00 = -2 * self.t_r * self.ur
        self.Ft_prime_ss_01 = -2 * self.t_r * self.ut_rel
        self.Ft_prime_ss_02 = -2 * self.t_r * self.uz
        self.Ft_prime_ss_10 = -2 * self.t_t * self.ur
        self.Ft_prime_ss_11 = -2 * self.t_t * self.ut_rel
        self.Ft_prime_ss_12 = -2 * self.t_t * self.uz
        self.Ft_prime_ss_20 = -2 * self.t_z * self.ur
        self.Ft_prime_ss_21 = -2 * self.t_z * self.ut_rel
        self.Ft_prime_ss_22 = -2 * self.t_z * self.uz



    def cut_domain(self, border):
        """
        cut the cfd domain inside the mesh defined by the block object
        """

        from shapely.geometry import Point
        from shapely.geometry.polygon import Polygon

        # Assuming you have the 'border' list of (z, r) points for the polygon

        # Convert the 'border' list to a Shapely Polygon object
        polygon = Polygon(border)

        # Create a list to store the results of the point-in-polygon check
        idx_cut = []

        # Loop through your points
        for i in range(self.x.shape[0]):
            # print(i)
            point_to_check = Point(self.z[i], self.r[i])
            # Check if the point is inside the polygon
            result = point_to_check.within(polygon)
            idx_cut.append(result)

        # Use boolean indexing to extract the points that are inside the polygon. deletes everything that lies outside
        self.idx_cut = idx_cut
        self.x = self.x[idx_cut]
        self.y = self.y[idx_cut]
        self.z = self.z[idx_cut]
        self.r = self.r[idx_cut]
        self.theta = self.theta[idx_cut]
        self.rho = self.rho[idx_cut]
        self.ux = self.ux[idx_cut]
        self.uy = self.uy[idx_cut]
        self.uz = self.uz[idx_cut]
        self.p = self.p[idx_cut]
        self.T = self.T[idx_cut]
        self.s = self.s[idx_cut]
        self.drho_dx = self.drho_dx[idx_cut]
        self.drho_dy = self.drho_dy[idx_cut]
        self.drho_dz = self.drho_dz[idx_cut]
        self.dux_dx = self.dux_dx[idx_cut]
        self.dux_dy = self.dux_dy[idx_cut]
        self.dux_dz = self.dux_dz[idx_cut]
        self.duy_dx = self.duy_dx[idx_cut]
        self.duy_dy = self.duy_dy[idx_cut]
        self.duy_dz = self.duy_dz[idx_cut]
        self.duz_dx = self.duz_dx[idx_cut]
        self.duz_dy = self.duz_dy[idx_cut]
        self.duz_dz = self.duz_dz[idx_cut]
        self.dp_dx = self.dp_dx[idx_cut]
        self.dp_dy = self.dp_dy[idx_cut]
        self.dp_dz = self.dp_dz[idx_cut]
        self.ds_dx = self.ds_dx[idx_cut]
        self.ds_dy = self.ds_dy[idx_cut]
        self.ds_dz = self.ds_dz[idx_cut]



    def compute_flow_ideal_vectors(self):
        """
        the aim is to find for every point in the dataset, the ideal flow vectors (the equivalent of the normal to the camber
        surface, the ideal streamwise direction, and the ideal spanwise directions). These vectors can be found taking
        the vectors on the camber surface point equivalent to them (same r,z position), and rotating them along z-axis
        of an angle equal to the difference between the point and its projection on the camber surface.
        To speed up the computation, only the points making part of the meridional grid can be considered if the cut method
        was previously applied
        """

        # instantiate empty lists
        self.normal_vec = []
        self.streamline_vec = []
        self.spanline_vec = []
        total_elem = self.x.shape[0]
        if self.verbose:
            print('computing flow directions...')
        for i in range(0, total_elem):
            normal, streamline, spanline = self.compute_flow_directions(i)
            self.normal_vec.append(normal)
            self.streamline_vec.append(streamline)
            self.spanline_vec.append(spanline)



    def compute_flow_directions(self, i):
        """
        for the i element, take the vectors on the equivalent camber surface point, rotate them along z of the
        difference angle between the two
        Args:
            i: element of the dataset

        Returns:
            normal, streamline, spanline: flow vectors for each point
        """

        # select the point in the dataset
        z = self.z[i]
        r = self.r[i]

        # find corresponding point on the camber surface, as the closest one on the camber grid
        camber_point_idx = np.abs((self.blade.r_camber.flatten() - r) ** 2 + (self.blade.z_camber.flatten() - z) ** 2).argmin()

        # retrieve the flow vectors on the camber surface
        normal_direction = self.blade.normal_vectors.flatten()[camber_point_idx]
        stream_direction = self.blade.streamline_vectors.flatten()[camber_point_idx]
        span_direction = self.blade.spanline_vectors.flatten()[camber_point_idx]

        # find the rotation to give to the vectors
        theta = self.theta[i]
        theta_camber = np.arctan2(self.blade.y_camber.flatten()[camber_point_idx],
                                  self.blade.x_camber.flatten()[camber_point_idx])
        delta_theta = theta - theta_camber

        # rotation matrix around the z-axis
        rot_mat = np.array([[np.cos(delta_theta), -np.sin(delta_theta), 0],
                            [np.sin(delta_theta), np.cos(delta_theta), 0],
                            [0, 0, 1]])

        normal = np.matmul(rot_mat, normal_direction)
        stream = np.matmul(rot_mat, stream_direction)
        span = np.matmul(rot_mat, span_direction)

        return normal, stream, span



    def compute_normalization_quantities(self):
        """
        given the fundamental quantities, compute all the reference quantities for following non-dimensionalization
        """
        self.omega_ref = self.rpm_ref * 2*np.pi / 60  # convert to [rad/s]
        self.u_ref = self.omega_ref * self.x_ref  # tip speed of the machine
        self.t_ref = 1 / self.omega_ref  # to be coherent
        self.p_ref = 0.5*self.rho_ref * self.u_ref**2
        self.s_ref = self.u_ref ** 2 / self.T_ref  # reference entropy


    def normalize_data(self):
        """
        normalize everything, in order to increase numerical accuracy
        """

        # non-dimensionalize cordinates
        self.z /= self.x_ref
        self.r /= self.x_ref
        self.omega_shaft /= self.omega_ref

        # normalization of the fields
        self.rho /= self.rho_ref
        self.ux /= self.u_ref
        self.uy /= self.u_ref
        self.uz /= self.u_ref
        self.p /= self.p_ref
        self.T /= self.T_ref
        self.s /= self.s_ref

        # normalization of the gradients
        self.drho_dx /= (self.rho_ref / self.x_ref)
        self.drho_dy /= (self.rho_ref / self.x_ref)
        self.drho_dz /= (self.rho_ref / self.x_ref)
        self.dux_dx /= (self.u_ref / self.x_ref)
        self.dux_dy /= (self.u_ref / self.x_ref)
        self.dux_dz /= (self.u_ref / self.x_ref)
        self.duy_dx /= (self.u_ref / self.x_ref)
        self.duy_dy /= (self.u_ref / self.x_ref)
        self.duy_dz /= (self.u_ref / self.x_ref)
        self.duz_dx /= (self.u_ref / self.x_ref)
        self.duz_dy /= (self.u_ref / self.x_ref)
        self.duz_dz /= (self.u_ref / self.x_ref)
        self.dp_dx /= (self.p_ref / self.x_ref)
        self.dp_dy /= (self.p_ref / self.x_ref)
        self.dp_dz /= (self.p_ref / self.x_ref)
        self.ds_dx /= (self.s_ref / self.x_ref)
        self.ds_dy /= (self.s_ref / self.x_ref)
        self.ds_dz /= (self.s_ref / self.x_ref)
