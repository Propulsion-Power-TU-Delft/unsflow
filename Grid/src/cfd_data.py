#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 16:53:11 2023
@author: F. Neri, TU Delft
"""
import warnings
import pandas as pd
from .functions import *
from Sun.src.general_functions import print_banner_begin, print_banner_end
from Utils.styles import total_chars, total_chars_mid
from numpy import sin, cos, tan, sqrt


class CfdData:
    """
    Class storing the CFD data, extracted from a CFD software. For the moment only the constructor for csv files
    extracted from Ansys is implemented
    """

    def __init__(self, config, blade):
        """
        read the data from the csv file extracted from Ansys CFD-post. rpm_drag is used to compute relative and drag velocities
        If normalize = True, it stores the normalization quantities:
        rho_ref: reference density, for air can be 1.014 [kg/m3]
        omega_ref: angular speed of the shaft [rpm], with algebraic sign
        x_ref: tip radius of the blade at leading edge [m]
        T_ref: can be standard temperature [K]
        All other non-dimensionalization quantities are obtained from these fundamental ones.
        :param config: configuration file
        :param config: blade object
        """
        self.config = config
        if blade is not None:
            self.blade = blade

        if self.config.get_cfd_filetype() == 'Ansys3D':
            self.read_from_ansys_3D_csv()
        elif self.config.get_cfd_filetype() == 'Ansys2D':
            self.read_from_ansys_2D_csv()
        elif self.config.get_cfd_filetype() == 'ParaviewCSV':
            self.read_from_paraview_csv()
        else:
            raise ValueError("File type not recognized.")

        if self.config.get_verbosity():
            print_banner_begin('CFD DATA PROCESSING')
            print(f"{'CFD filepath:':<{total_chars_mid}}{self.config.get_cfd_filepath():>{total_chars_mid}}")
            print(f"{'CFD filetype:':<{total_chars_mid}}{self.config.get_cfd_filetype():>{total_chars_mid}}")
            print(f"{'Shaft Omega [rpm]:':<{total_chars_mid}}{self.config.get_omega_shaft():>{total_chars_mid}.3f}")
            print(f"{'Reference Omega [rpm]:':<{total_chars_mid}}{self.config.get_reference_omega():>{total_chars_mid}.3f}")
            print(f"{'Reference Density [kg/m3]:':<{total_chars_mid}}{self.config.get_reference_density():>{total_chars_mid}.3f}")
            print(f"{'Reference Length [m]:':<{total_chars_mid}}{self.config.get_reference_length():>{total_chars_mid}.3f}")
            print(f"{'Reference Velocity [m/s]:':<{total_chars_mid}}{self.config.get_reference_velocity():>{total_chars_mid}.3f}")
            print(f"{'Reference Pressure [Pa]:':<{total_chars_mid}}{self.config.get_reference_pressure():>{total_chars_mid}.3f}")
            print(f"{'Reference Time [s]:':<{total_chars_mid}}{self.config.get_reference_time():>{total_chars_mid}.6f}")
            print(f"{'Reference Temperature [K]:':<{total_chars_mid}}"
                  f"{self.config.get_reference_temperature():>{total_chars_mid}.3f}")
            print(f"{'Reference Entropy [J/kgK]:':<{total_chars_mid}}{self.config.get_reference_entropy():>{total_chars_mid}.3f}")
            print(f"{'Dataset Normalized:':<{total_chars_mid}}{self.config.get_normalize_data():>{total_chars_mid}}")
            print_banner_end()

    def read_from_ansys_3D_csv(self):
        """
        read the data from a 3D CSV file extracted from Ansys. Check that all the quantities are stored in the file,
        as well as the correct names of the variables.
        """
        data = pd.read_csv(self.config.get_cfd_filepath(), skiprows=range(5))
        self.x = data[' X [ m ]'].values
        self.y = data[' Y [ m ]'].values
        self.z = data[' Z [ m ]'].values
        self.r = sqrt(self.x ** 2 + self.y ** 2)
        self.theta = np.arctan2(self.y, self.x)
        self.rho = data[' Density [ kg m^-3 ]'].values
        self.ur = data[' Velocity Radial [ m s^-1 ]'].values
        self.ut = data[' Velocity in Stn Frame Circumferential [ m s^-1 ]'].values
        self.uz = data[' Velocity Axial [ m s^-1 ]'].values
        self.p = data[' Pressure [ Pa ]'].values
        self.T = data[' Temperature [ K ]'].values
        self.s = data[' Static Entropy [ J kg^-1 K^-1 ]'].values
        self.finite_volume = data[' Volume of Finite Volumes [ m^3 ]'].values

        # gradients
        self.drho_dr = data[' drhodr [ kg m^-4 ]'].values
        self.drho_dz = data[' Density.Gradient Z [ kg m^-4 ]'].values
        self.dur_dr = data[' durdr [ s^-1 ]'].values
        self.dur_dz = data[' Velocity Radial.Gradient Z [ s^-1 ]'].values
        self.dut_dr = data[' dutdr [ s^-1 ]'].values
        self.dut_dz = data[' Velocity in Stn Frame Circumferential.Gradient Z [ s^-1 ]'].values
        self.duz_dr = data[' duzdr [ s^-1 ]'].values
        self.duz_dz = data[' Velocity in Stn Frame w.Gradient Z [ s^-1 ]'].values
        self.dp_dr = data[' dpdr [ kg m^-2 s^-2 ]'].values
        self.dp_dz = data[' Pressure.Gradient Z [ kg m^-2 s^-2 ]'].values
        self.ds_dr = data[' dsdr [ m s^-2 K^-1 ]'].values
        self.ds_dz = data[' Static Entropy.Gradient Z [ m s^-2 K^-1 ]'].values

        if self.config.get_normalize_data():
            self.normalize_data()
            print('CFD data normalized')
        else:
            print('CFD data NOT normalized')

    def read_from_paraview_csv(self):
        """
        read the data from a 3D CSV file extracted from Paraview. Check that all the quantities are stored in the file,
        as well as the correct names of the variables.
        """
        df = pd.read_csv(self.config.get_cfd_filepath())
        data = df.to_dict(orient='list')
        for key, value in data.items():
            data[key] = np.array(value, dtype=float)
            print('%s in dataset' % key)

        self.x = data['Centers_0']
        self.y = data['Centers_1']
        self.z = data['Centers_2']
        self.r = sqrt(self.x ** 2 + self.y ** 2)
        self.theta = np.arctan2(self.y, self.x)
        self.rho = data['Density']
        self.ux = data['Velocity_0']
        self.uy = data['Velocity_1']
        self.uz = data['Velocity_2']
        self.p = data['Pressure']
        self.T = data['Temperature']
        # self.s = data[' Static Entropy [ J kg^-1 K^-1 ]'].values
        self.volume = data['Volume']

        if self.config.get_normalize_data():
            self.normalize_data()
            print('CFD data normalized')
        else:
            print('CFD data NOT normalized')

    def read_from_ansys_2D_csv(self):
        """
        read the data from a 2D CSV file extracted from Ansys, meridionally processed.
        """
        self.data = pd.read_csv(self.config.get_cfd_filepath(), skiprows=range(5))
        self.x = self.data[' X [ m ]'].values
        self.y = self.data[' Y [ m ]'].values
        self.z = self.data[' Z [ m ]'].values
        self.r = sqrt(self.x ** 2 + self.y ** 2)
        self.theta = np.arctan2(self.y, self.x)
        self.rho = self.data[' Density MCA on Meridional Surface [ kg m^-3 ]'].values
        self.ur = self.data[' Velocity Radial MCA on Meridional Surface [ m s^-1 ]'].values
        self.ut = self.data[' Velocity in Stn Frame Circumferential MCA on Meridional Surface [ m s^-1 ]'].values
        self.uz = self.data[' Velocity Axial MCA on Meridional Surface [ m s^-1 ]'].values
        self.p = self.data[' Pressure MCA on Meridional Surface [ Pa ]'].values
        self.T = self.data[' Temperature MCA on Meridional Surface [ K ]'].values
        self.s = self.data[' Static Entropy MCA on Meridional Surface [ J kg^-1 K^-1 ]'].values

        self.drho_dr = self.data[' drhodr MCA on Meridional Surface [ kg m^-4 ]']
        self.drho_dz = self.data[' Density.Gradient Z MCA on Meridional Surface [ kg m^-4 ]']
        self.dur_dr = self.data[' durdr MCA on Meridional Surface [ s^-1 ]']
        self.dur_dz = self.data[' durdz MCA on Meridional Surface [ s^-1 ]']
        self.dut_dr = self.data[' dutdr MCA on Meridional Surface [ s^-1 ]']
        self.dut_dz = self.data[' dutdz MCA on Meridional Surface [ s^-1 ]']
        self.duz_dr = self.data[' duzdr MCA on Meridional Surface [ s^-1 ]']
        self.duz_dz = self.data[' Velocity in Stn Frame w.Gradient Z MCA on Meridional Surface [ s^-1 ]']
        self.dp_dr = self.data[' dpdr MCA on Meridional Surface [ kg m^-2 s^-2 ]']
        self.dp_dz = self.data[' Pressure.Gradient Z MCA on Meridional Surface [ kg m^-2 s^-2 ]']
        self.ds_dr = self.data[' dsdr MCA on Meridional Surface [ m s^-2 K^-1 ]']
        self.ds_dz = self.data[' Static Entropy.Gradient Z MCA on Meridional Surface [ m s^-2 K^-1 ]']

        if self.config.get_normalize_data():
            self.normalize_data()

    def compute_cylindrical_velocities(self):
        """
        Compute velocity in cylindrical cordinates.
        """
        self.ur = self.ux * np.cos(self.theta) + self.uy * np.sin(self.theta)
        self.ut = self.ux * np.cos(self.theta) + self.uy * np.sin(self.theta)

    def compute_derived_quantities(self):
        """
        Compute derived quantities, in particular the vector components in the cylindrical reference frame.
        """
        self.u_mag = sqrt(self.ur ** 2 + self.ut ** 2 + self.uz ** 2)
        self.ut_drag = self.r * self.config.get_omega_shaft()/self.config.get_reference_omega()  # drag velocity
        self.ut_rel = self.ut - self.ut_drag  # relative velocity
        self.u_mag_rel = sqrt(self.ur ** 2 + self.ut_rel ** 2 + self.uz ** 2)


    def compute_bfm_radial_fields(self):
        """
        Computes the 3D fields necessary for the radial body force model, as described in my draft appendix, Radial Machines
        model. The 3D fields described here will be circumferentially averaged. Mu, coefficient of the resistant force
        is calculated directly in the 2D meridional object, because too expensive to be done in 3D.

        The model is still not validated, and could be implemented better directly working on the meridional data.
        """
        warnings.warn("WARNING: BFM method not validated yet, and not ready to be used.")
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
        self.k = (self.dut_dr + self.uz / self.ur * self.dut_dz + self.ut / self.r) / (self.u_mag_rel * self.tau_t - self.ut_rel)

        # F_{n,theta} as described in eq 9 of the same article
        self.F_ntheta = self.k * self.ur * (self.u_mag_rel * self.tau_t - self.ut_rel)
        self.F_nr = self.F_ntheta * self.n_r / self.n_t
        self.F_nz = self.F_ntheta * self.n_z / self.n_t

        # a1, a2, a3 as defined in equation 16 (corrected by me) of the same article (mu is computed on the meridional grid)
        self.a1 = self.k * ((self.u_mag_rel + self.ur ** 2 / self.u_mag_rel) * self.tau_t - self.ut_rel)
        self.a2 = self.k * self.ur * (self.ut_rel / self.u_mag_rel * self.tau_t - 1)
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
        Cut the cfd domain inside the mesh defined by the block object.
        :param border: block object describing the border for the cut.
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
        # self.drho_dx = self.drho_dx[idx_cut]
        # self.drho_dy = self.drho_dy[idx_cut]
        # self.drho_dz = self.drho_dz[idx_cut]
        # self.dux_dx = self.dux_dx[idx_cut]
        # self.dux_dy = self.dux_dy[idx_cut]
        # self.dux_dz = self.dux_dz[idx_cut]
        # self.duy_dx = self.duy_dx[idx_cut]
        # self.duy_dy = self.duy_dy[idx_cut]
        # self.duy_dz = self.duy_dz[idx_cut]
        # self.duz_dx = self.duz_dx[idx_cut]
        # self.duz_dy = self.duz_dy[idx_cut]
        # self.duz_dz = self.duz_dz[idx_cut]
        # self.dp_dx = self.dp_dx[idx_cut]
        # self.dp_dy = self.dp_dy[idx_cut]
        # self.dp_dz = self.dp_dz[idx_cut]
        # self.ds_dx = self.ds_dx[idx_cut]
        # self.ds_dy = self.ds_dy[idx_cut]
        # self.ds_dz = self.ds_dz[idx_cut]

    def compute_flow_ideal_vectors(self):
        """
        For every point in the dataset, find the ideal flow vectors (the equivalent of the normal to the camber
        surface, the ideal streamwise direction, and the ideal spanwise directions). These vectors can be found taking
        the vectors on the camber surface point equivalent to them (same r,z position), and rotating them along z-axis
        of an angle equal to the difference between the point and its projection on the camber surface.
        To speed up the computation, only the points making part of the meridional grid can be considered if the cut method
        was previously applied.
        """
        # instantiate empty lists
        self.normal_vec = []
        self.streamline_vec = []
        self.spanline_vec = []
        total_elem = self.x.shape[0]
        print('computing flow directions...')
        for i in range(0, total_elem):
            normal, streamline, spanline = self.compute_flow_directions(i)
            self.normal_vec.append(normal)
            self.streamline_vec.append(streamline)
            self.spanline_vec.append(spanline)

    def compute_flow_directions(self, i):
        """
        For the i element of the dataset, take the direction vectors on the equivalent camber surface point passing through that
        point, in order to find camber normal and tangential local vectors. Method not validated.
        :param i: i-th point of the dataset
        """
        warnings.warn("Method not validated yet.")
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


    def normalize_data(self):
        """
        Normalize the dataset based on the reference quantities stores in the configuration files.
        """
        self.x /= self.config.get_reference_length()
        self.y /= self.config.get_reference_length()
        self.z /= self.config.get_reference_length()
        self.r /= self.config.get_reference_length()
        self.omega_shaft = self.config.get_omega_shaft() / self.config.get_reference_omega()

        # normalization of the fields
        self.rho /= self.config.get_reference_density()
        self.ux /= self.config.get_reference_velocity()
        self.uy /= self.config.get_reference_velocity()
        self.uz /= self.config.get_reference_velocity()
        self.p /= self.config.get_reference_pressure()
        self.T /= self.config.get_reference_temperature()
        # self.s /= self.config.get_reference_entropy()
        #
        # self.drho_dr /= (self.config.get_reference_density() / self.config.get_reference_length())
        # self.drho_dz /= (self.config.get_reference_density() / self.config.get_reference_length())
        # self.dur_dr /= (self.config.get_reference_velocity() / self.config.get_reference_length())
        # self.dur_dz /= (self.config.get_reference_velocity() / self.config.get_reference_length())
        # self.dut_dr /= (self.config.get_reference_velocity() / self.config.get_reference_length())
        # self.dut_dz /= (self.config.get_reference_velocity() / self.config.get_reference_length())
        # self.duz_dr /= (self.config.get_reference_velocity() / self.config.get_reference_length())
        # self.duz_dz /= (self.config.get_reference_velocity() / self.config.get_reference_length())
        # self.dp_dr /= (self.config.get_reference_pressure() / self.config.get_reference_length())
        # self.dp_dz /= (self.config.get_reference_pressure() / self.config.get_reference_length())
        # self.ds_dr /= (self.config.get_reference_entropy() / self.config.get_reference_length())
        # self.ds_dz /= (self.config.get_reference_entropy() / self.config.get_reference_length())
