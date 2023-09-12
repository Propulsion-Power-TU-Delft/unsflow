#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 18:29:29 2023
@author: F. Neri, TU Delft
"""
import numpy as np
from scipy.spatial import KDTree
from .styles import *
from .functions import cluster_sample_u
from .curve import Curve


class Block:
    """
    this class contains a single block, obtained after trimming the hub and shroud curves where needed.
    """

    def __init__(self, hub_curve, shroud_curve, nstream=10, nspan=10):
        """
        provide the two curve objects related to hub and shroud, and in how many number of points 
        you want to discretize the streamwise and spanwise direction
        """
        self.hub = hub_curve
        self.shroud = shroud_curve
        self.nstream = nstream
        self.nspan = nspan
        self.units = self.hub.units



    def trim_inlet(self, z_trim='span', r_trim='span', units='in'):
        """
        trim the inlet at a trim plane
        """
        self.hub.trim_inlet(z_trim, r_trim)
        self.shroud.trim_inlet(z_trim, r_trim)



    def trim_outlet(self, z_trim='span', r_trim='span', units='in'):
        """
        trim the outlet at a trim plane
        """
        self.hub.trim_outlet(z_trim, r_trim)
        self.shroud.trim_outlet(z_trim, r_trim)



    def spline_of_hub_shroud(self):
        """
        compute hub,shroud splines, that are parameterized from 0 to 1 between the extremes.
        """
        self.hub_trim = Curve(z=self.hub.z_spline, r=self.hub.r_spline, nstream=self.nstream, mode='cordinates')
        self.shroud_trim = Curve(z=self.shroud.z_spline, r=self.shroud.r_spline,
                                 nstream=self.nstream, mode='cordinates')
        # self.hub_trim.plot_spline()
        # self.shroud_trim.plot_spline()



    def spline_of_leading_trailing_edge(self):
        self.inlet = np.concatenate((np.reshape(self.point_hub_inlet, (1, 2)),
                                    self.inlet[1:-1, :],
                                    np.reshape(self.point_shroud_inlet, (1, 2))))

        self.outlet = np.concatenate((np.reshape(self.point_hub_outlet, (1, 2)),
                                     self.outlet[1:-1, :],
                                     np.reshape(self.point_shroud_outlet, (1, 2))))

        self.leading_edge = Curve(z=self.inlet[:, 0], r=self.inlet[:, 1], nstream=self.nspan, mode='cordinates')
        self.trailing_edge = Curve(z=self.outlet[:, 0], r=self.outlet[:, 1], nstream=self.nspan, mode='cordinates')



    def sample_hub_shroud(self, sampling_mode='default'):
        """
        sample the hub and shroud spline (trimmed correctly) with a certain sampling mode
        """
        self.hub_trim.sample(sampling_mode=sampling_mode)
        self.shroud_trim.sample(sampling_mode=sampling_mode)



    def sample_leading_trailing_edges(self, sampling_mode='default'):
        """
        sample the hub and shroud spline (trimmed correctly) with a certain sampling mode
        """
        self.leading_edge.sample(sampling_mode=sampling_mode)
        self.trailing_edge.sample(sampling_mode=sampling_mode)



    def compute_grid_points(self, sampling_mode='default', grid_mode='spanwise', curved_border=True):
        """
        obtain the points in the middle of the channel connecting with a straight line the streamwise points 
        on hub and shroud. Then sample them on the connecting line with a certain sampling algorithm
        """
        if sampling_mode == 'default':
            self.u_span = np.linspace(0, 1, self.nspan)
            self.u_stream = np.linspace(0, 1, self.nstream)
        elif sampling_mode == 'clustering':
            self.u_span = cluster_sample_u(self.nspan)  # this can also be obtained with sigmoid
            self.u_stream = cluster_sample_u(self.nstream)  # this can also be obtained with sigmoid
        self.r_grid_points = np.zeros((self.nstream, self.nspan))
        self.z_grid_points = np.zeros((self.nstream, self.nspan))


        if grid_mode == 'spanwise':
            # algorithm for internal points, connecting points on the hub and shroud along the span direction
            for istream in range(0, self.nstream):
                for ispan in range(0, self.nspan):

                    self.r_grid_points[istream, ispan] = self.hub_trim.r_sample[istream] + self.u_span[ispan] * (
                            self.shroud_trim.r_sample[istream] - self.hub_trim.r_sample[istream])

                    self.z_grid_points[istream, ispan] = self.hub_trim.z_sample[istream] + self.u_span[ispan] * (
                            self.shroud_trim.z_sample[istream] - self.hub_trim.z_sample[istream])

            # now overwrite the points that in reality are taken from the curved leading and trailing edges
            if curved_border:
                self.r_grid_points[0, :] = self.leading_edge.r_sample
                self.r_grid_points[-1, :] = self.trailing_edge.r_sample
                self.z_grid_points[0, :] = self.leading_edge.z_sample
                self.z_grid_points[-1, :] = self.trailing_edge.z_sample

        elif grid_mode == 'streamwise':
            # algorithm for internal points, connecting points on the inlet and trailing edges along the stream direction
            for istream in range(0, self.nstream):
                for ispan in range(0, self.nspan):
                    self.r_grid_points[istream, ispan] = self.leading_edge.r_sample[ispan] + self.u_stream[istream] * (
                            self.trailing_edge.r_sample[ispan] - self.leading_edge.r_sample[ispan])

                    self.z_grid_points[istream, ispan] = self.leading_edge.z_sample[ispan] + self.u_stream[istream] * (
                            self.trailing_edge.z_sample[ispan] - self.leading_edge.z_sample[ispan])

            # now overwrite the points that in reality are taken from the curved leading and trailing edges
            if curved_border:
                self.r_grid_points[:, 0] = self.hub_trim.r_sample
                self.r_grid_points[:, -1] = self.shroud_trim.r_sample
                self.z_grid_points[:, 0] = self.hub_trim.z_sample
                self.z_grid_points[:, -1] = self.shroud_trim.z_sample






    def plot_borders_grid(self):
        """
        it plots only the hub and shroud correctly trimmed, and the original set of points
        """
        plt.figure(figsize=fig_size)
        plt.plot(self.hub.z_spline, self.hub.r_spline, label='hub')
        plt.plot(self.hub_trim.z_sample, self.hub_trim.r_sample, 'o', label='hub sampled')
        plt.plot(self.shroud.z_spline, self.shroud.r_spline, label='shroud')
        plt.plot(self.shroud_trim.z_sample, self.shroud_trim.r_sample, 'o', label='shroud sample')
        plt.legend()
        plt.xlabel(r'$z \ \mathrm{[%s]}$' % (self.units))
        plt.ylabel(r'$r \ \mathrm{[%s]}$' % (self.units))



    def add_inlet_outlet_curves(self, inlet, outlet):
        """
        stores information regarding the inlet and outlet points, in order to compute the leading and trailing splines.
        """
        self.inlet = inlet
        self.outlet = outlet
        self.inlet_curve = Curve(z=inlet[:, 0], r=inlet[:, 1], mode='cordinates', degree_spline=3)
        self.outlet_curve = Curve(z=outlet[:, 0], r=outlet[:, 1], mode='cordinates', degree_spline=3)



    def extend_inlet_outlet_curves(self):
        """
        extend the inlet and outlet curves in order to later find intersections with the hub and shroud
        """
        self.inlet_curve.extend()
        self.outlet_curve.extend()



    def find_intersections(self, tol=1e-4):
        """
        having the hub and shroud curves, it looks for the intersections of these curves with the inlet and outlet points
        """
        hub_curve = np.stack((self.hub.z, self.hub.r), axis=1)
        shroud_curve = np.stack((self.shroud.z, self.shroud.r), axis=1)
        # hub_curve = Curve(hub_curve, )
        # shroud_curve = np.stack((self.shroud.z, self.shroud.r), axis=1)

        inlet_curve = np.stack((self.inlet_curve.z_spline_ext, self.inlet_curve.r_spline_ext), axis=1)
        outlet_curve = np.stack((self.outlet_curve.z_spline_ext, self.outlet_curve.r_spline_ext), axis=1)

        print('If no intersections are found between blade edges and endwalls, it means that the threshold error is too low '
              'for the intersection algorithm')

        self.point_hub_inlet = self.point_intersection(inlet_curve, hub_curve, tol)
        self.point_hub_outlet = self.point_intersection(outlet_curve, hub_curve, tol)
        self.point_shroud_inlet = self.point_intersection(inlet_curve, shroud_curve, tol)
        self.point_shroud_outlet = self.point_intersection(outlet_curve, shroud_curve, tol)

        # plt.figure()
        # plt.scatter(hub_curve[:, 0], hub_curve[:, 1])
        # plt.scatter(shroud_curve[:, 0], shroud_curve[:, 1])
        # plt.scatter(inlet_curve[:, 0], inlet_curve[:, 1])
        # plt.scatter(self.point_hub_inlet[0], self.point_hub_inlet[1])
        # plt.scatter(self.point_shroud_inlet[0], self.point_shroud_inlet[1])
        # plt.scatter(outlet_curve[:, 0], outlet_curve[:, 1])
        # plt.scatter(self.point_hub_outlet[0], self.point_hub_outlet[1])
        # plt.scatter(self.point_shroud_outlet[0], self.point_shroud_outlet[1])


    @staticmethod
    def point_intersection(curve1, curve2, tol=1e-4):
        """
        find and return the intersection between 2 curves. static method because it is bound to the class, not to an instance
        of the class. It could also avoid to specify the self, since it is not used. it could be used with decorator @static
        """
        tree = KDTree(curve1)
        distances, indices = tree.query(curve2)
        intersection_points = curve1[indices[distances < tol]]
        point = np.mean(intersection_points, axis=0)
        return point



    def bladed_zone_trim(self, machine_type = 'radial'):
        """
        trim the block hub and shroud curves at the found intersections with the inlet and outlet curves
        """
        if machine_type == 'radial':
            self.hub.trim_inlet(z_trim=self.point_hub_inlet[0])
            self.hub.trim_outlet(r_trim=self.point_hub_outlet[1])
            self.shroud.trim_inlet(z_trim=self.point_shroud_inlet[0])
            self.shroud.trim_outlet(r_trim=self.point_shroud_outlet[1])
        elif machine_type == 'axial':
            self.hub.trim_inlet(z_trim=self.point_hub_inlet[0])
            self.hub.trim_outlet(z_trim=self.point_hub_outlet[0])
            self.shroud.trim_inlet(z_trim=self.point_shroud_inlet[0])
            self.shroud.trim_outlet(z_trim=self.point_shroud_outlet[0])
        else:
            raise ValueError('Insert a valid machine type')



    def inlet_zone_trim(self):
        """
        trim the inlet block hub and shroud curves at the found intersections with the inlet curves of the blade
        """
        self.hub.trim_outlet(z_trim=self.point_hub_inlet[0])
        self.shroud.trim_outlet(z_trim=self.point_shroud_inlet[0])



    def outlet_zone_trim(self):
        """
        trim the outlet block hub and shroud curves at the found intersections with the trailing edge of the blade
        """
        self.hub.trim_inlet(r_trim=self.point_hub_outlet[1])
        self.shroud.trim_inlet(r_trim=self.point_shroud_outlet[1])



    def compute_double_grid(self):
        """
        compute a secondary grid, using the points that lie in the baricenter of 4 primary grid points
        """
        self.z_grid_centers = np.zeros((self.nstream + 1, self.nspan + 1))
        self.r_grid_centers = np.zeros((self.nstream + 1, self.nspan + 1))

        # internal points
        for istream in range(1, self.nstream):
            for ispan in range(1, self.nspan):
                z_mid_point = 0.25 * (self.z_grid_points[istream, ispan] + self.z_grid_points[istream - 1, ispan] +
                                      self.z_grid_points[istream, ispan - 1] + self.z_grid_points[istream - 1, ispan - 1])

                r_mid_point = 0.25 * (self.r_grid_points[istream, ispan] + self.r_grid_points[istream - 1, ispan] +
                                      self.r_grid_points[istream, ispan - 1] + self.r_grid_points[istream - 1, ispan - 1])

                self.z_grid_centers[istream, ispan] = z_mid_point
                self.r_grid_centers[istream, ispan] = r_mid_point

        # vertices
        self.z_grid_centers[0, 0] = self.z_grid_points[0, 0]
        self.r_grid_centers[0, 0] = self.r_grid_points[0, 0]
        self.z_grid_centers[0, -1] = self.z_grid_points[0, -1]
        self.r_grid_centers[0, -1] = self.r_grid_points[0, -1]
        self.z_grid_centers[-1, -1] = self.z_grid_points[-1, -1]
        self.r_grid_centers[-1, -1] = self.r_grid_points[-1, -1]
        self.z_grid_centers[-1, 0] = self.z_grid_points[-1, 0]
        self.r_grid_centers[-1, 0] = self.r_grid_points[-1, 0]

        # istream = 0 border
        for istream in range(0, 1):
            for ispan in range(1, self.nspan):
                z_mid_point = 0.5 * (self.z_grid_points[istream, ispan] + self.z_grid_points[istream, ispan - 1])
                r_mid_point = 0.5 * (self.r_grid_points[istream, ispan] + self.r_grid_points[istream, ispan - 1])
                self.z_grid_centers[istream, ispan] = z_mid_point
                self.r_grid_centers[istream, ispan] = r_mid_point

        # istream = -1 border
        for istream in range(self.nstream, self.nstream + 1):
            for ispan in range(1, self.nspan):
                z_mid_point = 0.5 * (self.z_grid_points[istream - 1, ispan] + self.z_grid_points[istream - 1, ispan - 1])
                r_mid_point = 0.5 * (self.r_grid_points[istream - 1, ispan] + self.r_grid_points[istream - 1, ispan - 1])
                self.z_grid_centers[istream, ispan] = z_mid_point
                self.r_grid_centers[istream, ispan] = r_mid_point

        # ispan = 0 border
        for istream in range(1, self.nstream):
            for ispan in range(0, 1):
                z_mid_point = 0.5 * (self.z_grid_points[istream, ispan] + self.z_grid_points[istream - 1, ispan])
                r_mid_point = 0.5 * (self.r_grid_points[istream, ispan] + self.r_grid_points[istream - 1, ispan])
                self.z_grid_centers[istream, ispan] = z_mid_point
                self.r_grid_centers[istream, ispan] = r_mid_point

        # ispan = -1 border
        for istream in range(1, self.nstream):
            for ispan in range(self.nspan, self.nspan + 1):
                z_mid_point = 0.5 * (self.z_grid_points[istream, ispan - 1] + self.z_grid_points[istream - 1, ispan - 1])
                r_mid_point = 0.5 * (self.r_grid_points[istream, ispan - 1] + self.r_grid_points[istream - 1, ispan - 1])
                self.z_grid_centers[istream, ispan] = z_mid_point
                self.r_grid_centers[istream, ispan] = r_mid_point



    def plot_full_grid(self, save_filename=None, primary_grid=False, primary_grid_points=False, secondary_grid=False,
                       secondary_grid_points=False, hub_shroud=False, outline=False):
        """
        plot everything of the grid
        """
        self.AR = (np.max(self.r_grid_points) - np.min(self.r_grid_points)) / \
                  (np.max(self.z_grid_points) - np.min(self.z_grid_points))

        self.blade_picture_size = (7, 7*self.AR)

        plt.figure(figsize=self.blade_picture_size)

        # hub and shroud plot
        if hub_shroud:
            plt.plot(self.hub.z_spline, self.hub.r_spline, lw=light_line_width, c='black')
            plt.plot(self.shroud.z_spline, self.shroud.r_spline, lw=light_line_width, c='black')

        # primary grid
        if primary_grid:
            for istream in range(0, self.nstream):
                plt.plot(self.z_grid_points[istream, :], self.r_grid_points[istream, :], lw=light_line_width, c='black')
            for ispan in range(0, self.nspan):
                plt.plot(self.z_grid_points[:, ispan], self.r_grid_points[:, ispan], lw=light_line_width, c='black')
        elif outline:
            plt.plot(self.z_grid_points[0, :], self.r_grid_points[0, :], lw=line_width, label='leading edge')
            plt.plot(self.z_grid_points[-1, :], self.r_grid_points[-1, :], lw=line_width, label='trailing edge')
            plt.plot(self.z_grid_points[:, 0], self.r_grid_points[:, 0], lw=line_width, label='hub')
            plt.plot(self.z_grid_points[:, -1], self.r_grid_points[:, -1], lw=line_width, label='shroud')

        # primary grid points
        if primary_grid_points:
            plt.scatter(self.z_grid_points.flatten(), self.r_grid_points.flatten(),
                        c='black', s=scatter_point_size, label='primary grid nodes')

        # secondary grid
        if secondary_grid:
            for istream in range(0, self.nstream + 1):
                plt.plot(self.z_grid_centers[istream, :], self.r_grid_centers[istream, :], '--b', lw=light_line_width)
            for ispan in range(0, self.nspan + 1):
                plt.plot(self.z_grid_centers[:, ispan], self.r_grid_centers[:, ispan], '--b', lw=light_line_width)

                # secondary grid points
        if secondary_grid_points:
            plt.scatter(self.z_grid_centers.flatten(), self.r_grid_centers.flatten(), c='blue', s=scatter_point_size,
                        label='secondary grid nodes')

        if (primary_grid_points or secondary_grid_points or outline):
            plt.legend()
        plt.xlabel(r'$z \ \mathrm{%s}$' % (self.units))
        plt.ylabel(r'$r \ \mathrm{%s}$' % (self.units))
        plt.title(r'$(%d \times %d)$' % (self.nstream, self.nspan))

        if save_filename is not None:
            plt.savefig(folder_name + save_filename + '.pdf', bbox_inches='tight')



    def find_border(self):
        """
        find the border delimiting the block. it stores the border info as (r,z) column arrays
        """
        border_z = []
        border_r = []

        # append hub cordinates
        border_z.append(self.z_grid_points[0:, 0])
        border_r.append(self.r_grid_points[0:, 0])

        # append outlet cordinates
        border_z.append(self.z_grid_points[-1, 1:])
        border_r.append(self.r_grid_points[-1, 1:])

        # append shroud cordinates
        border_z.append(np.flip(self.z_grid_points[0:-2, -1]))
        border_r.append(np.flip(self.r_grid_points[0:-2, -1]))

        # append inlet cordinates
        border_z.append(np.flip(self.z_grid_points[0, 1:]))
        border_r.append(np.flip(self.r_grid_points[0, 1:]))

        border_z = [item for sublist in border_z for item in sublist]
        border_r = [item for sublist in border_r for item in sublist]

        self.border = np.stack((border_z, border_r), axis=1)

