import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from unsflow.utils.plot_styles import *
from unsflow.grid.functions import cluster_points, elliptic_grid_generation, transfinite_grid_generation, compute_meridional_streamwise_coordinates, contour_template
from .curve import Curve
from unsflow.utils.formatting import print_banner, total_chars, total_chars_mid
from .config import Config
from .body_force import BodyForce
import pickle


class Block:

    def __init__(self, config, iblock):
        self.config = config
        
        splineOrder = self.config.get_boundaries_spline_order()
        self.hub = Curve(config=config, curve_filepath=config.get_hub_curve_filepath(), degree_spline=splineOrder)
        self.shroud = Curve(config=config, curve_filepath=config.get_shroud_curve_filepath(), degree_spline=splineOrder)
        self.nstream = self.config.get_streamwise_points()[iblock]
        self.nspan = self.config.get_spanwise_points()
        self.iblock = iblock
        
        inletZ = np.linspace(self.hub.z[0], self.shroud.z[0], self.nspan)
        inletR = np.linspace(self.hub.r[0], self.shroud.r[0], self.nspan)
        outletZ = np.linspace(self.hub.z[-1], self.shroud.z[-1], self.nspan)
        outletR = np.linspace(self.hub.r[-1], self.shroud.r[-1], self.nspan)
        self.inletLine = np.stack((inletZ, inletR), axis=1)
        self.outletLine = np.stack((outletZ, outletR), axis=1)
        self.theta_camber = np.zeros((self.nstream, self.nspan))
        self.add_bfm_file_arrays()
    
    
    def trim_inlet(self):
        trim_inlet = self.config.trim_inlet()
        if trim_inlet is False:
            return
        else:
            trim_type = self.config.get_blocks_trim_type()[self.iblock]
            if trim_type == 'axial' or trim_type == 'axial-radial':
                self.hub.trim_curve_inlet(z_trim=trim_inlet)
                self.shroud.trim_curve_inlet(z_trim=trim_inlet)
            else:
                self.hub.trim_curve_inlet(r_trim=trim_inlet)
                self.shroud.trim_curve_inlet(r_trim=trim_inlet)
            inletZ = np.linspace(self.hub.z[0], self.shroud.z[0], self.nspan)
            inletR = np.linspace(self.hub.r[0], self.shroud.r[0], self.nspan)
            outletZ = np.linspace(self.hub.z[-1], self.shroud.z[-1], self.nspan)
            outletR = np.linspace(self.hub.r[-1], self.shroud.r[-1], self.nspan)
            self.inletLine = np.stack((inletZ, inletR), axis=1)
            self.outletLine = np.stack((outletZ, outletR), axis=1)
    
    
    def trim_outlet(self):
        trim_outlet = self.config.trim_outlet()
        if trim_outlet is False:
            return
        else:
            trim_type = self.config.get_blocks_trim_type()[self.iblock]
            if trim_type == 'radial' or trim_type == 'axial-radial':
                self.hub.trim_curve_outlet(r_trim=trim_outlet)
                self.shroud.trim_curve_outlet(r_trim=trim_outlet)
            else:
                self.hub.trim_curve_outlet(z_trim=trim_outlet)
                self.shroud.trim_curve_outlet(z_trim=trim_outlet)
            inletZ = np.linspace(self.hub.z[0], self.shroud.z[0], self.nspan)
            inletR = np.linspace(self.hub.r[0], self.shroud.r[0], self.nspan)
            outletZ = np.linspace(self.hub.z[-1], self.shroud.z[-1], self.nspan)
            outletR = np.linspace(self.hub.r[-1], self.shroud.r[-1], self.nspan)
            self.inletLine = np.stack((inletZ, inletR), axis=1)
            self.outletLine = np.stack((outletZ, outletR), axis=1)

    
    def add_bfm_file_arrays(self):
        """
        When the BFM file must be written, generate also the data needed for it
        """
        self.bfmFields = {}
        outputFields = self.config.get_turbo_BFM_mesh_output_fields()
        
        if 'blockage' in outputFields:
            self.bfmFields['blockage'] = np.ones((self.nstream, self.nspan))
            
        if 'camber' in outputFields:
            print('Camber normal vector grid added to the CTurboBFM mesh file')
            self.bfmFields['normalAxial'] = np.zeros((self.nstream, self.nspan))
            self.bfmFields['normalRadial'] = np.zeros((self.nstream, self.nspan))
            self.bfmFields['normalTangential'] = np.zeros((self.nstream, self.nspan))
        
        if 'blade_angles' in outputFields:
            self.bfmFields['bladeMetalAngle'] = np.zeros((self.nstream, self.nspan))
            self.bfmFields['bladeLeanAngle'] = np.zeros((self.nstream, self.nspan))
            self.bfmFields['bladeGasPathAngle'] = np.zeros((self.nstream, self.nspan))
        
        if 'rpm' in outputFields:
            self.bfmFields['rpm'] = np.zeros((self.nstream, self.nspan))
        
        if 'stwl' in outputFields:
            self.bfmFields['streamwiseLength'] = np.zeros((self.nstream, self.nspan))
        
        if 'spwl' in outputFields:
            self.bfmFields['spanwiseLength'] = np.zeros((self.nstream, self.nspan))
        
        if 'blade_present' in outputFields:
            self.bfmFields['bladePresent'] = np.zeros((self.nstream, self.nspan))
        
        if 'number_blades' in outputFields:
            self.bfmFields['numberBlades'] = np.zeros((self.nstream, self.nspan))
      

    def trim_flowpath_inlet(self, z_trim=None, r_trim=None):
        """
        Trim the inlet at a certain location.
        :param z_trim: z cordinate of trim
        :param r_trim: r cordinate of trim
        """
        self.hub.trim_curve_inlet(z_trim, r_trim)
        self.shroud.trim_curve_inlet(z_trim, r_trim)

    def trim_flowpath_outlet(self, z_trim=None, r_trim=None):
        """
        Trim the outlet at a certain location.
        :param z_trim: z cordinate of trim
        :param r_trim: r cordinate of trim
        """
        self.hub.trim_curve_outlet(z_trim, r_trim)
        self.shroud.trim_curve_outlet(z_trim, r_trim)

    def spline_of_hub_shroud(self):
        """
        Compute hub,shroud splines, that are parameterized from 0 to 1 between the extremes.
        """
        self.hub_trim = Curve(z=self.hub.z_spline, r=self.hub.r_spline, mode='cordinates')
        self.shroud_trim = Curve(z=self.shroud.z_spline, r=self.shroud.r_spline, mode='cordinates')

    def spline_of_leading_trailing_edge(self, iblade):
        """
        Make splines of the inlet and outlet border of the domain considered
        """
        def remove_excess_points(topology, point_hub, point_shroud, curve):
            if topology == 'axial':
                condition1 = point_hub[1] < curve[:, 1]
                condition2 = curve[:, 1] < point_shroud[1]
                res = condition1 & condition2
            elif topology == 'radial':
                condition1 = point_hub[0] < curve[:, 0]
                condition2 = curve[:, 0] < point_shroud[0]
                res = condition1 & condition2
            else:
                raise ValueError("Not recognized topology")
            
            curve = curve[res, :]
            return curve

        inlet_type = self.config.get_blade_inlet_type()
        outlet_type = self.config.get_blade_outlet_type()
        if isinstance(inlet_type, list):
            inlet_type = inlet_type[iblade]
        elif isinstance(inlet_type, str):
            pass
        else:
            raise ValueError('Unkown error of inlet type')

        if isinstance(outlet_type, list):
            outlet_type = outlet_type[iblade]
        elif isinstance(outlet_type, str):
            pass
        else:
            raise ValueError('Unkown error of inlet type')


        self.inlet = remove_excess_points(inlet_type, self.point_hub_inlet, self.point_shroud_inlet, self.inlet)
        self.outlet = remove_excess_points(outlet_type, self.point_hub_outlet, self.point_shroud_outlet, self.outlet)

        self.inlet = np.concatenate(
            (np.reshape(self.point_hub_inlet, (1, 2)), self.inlet[1:-1, :], np.reshape(self.point_shroud_inlet, (1, 2))))

        self.outlet = np.concatenate(
            (np.reshape(self.point_hub_outlet, (1, 2)), self.outlet[1:-1, :], np.reshape(self.point_shroud_outlet, (1, 2))))

        self.leading_edge = Curve(z=self.inlet[:, 0], r=self.inlet[:, 1], mode='cordinates')
        self.trailing_edge = Curve(z=self.outlet[:, 0], r=self.outlet[:, 1], mode='cordinates')

    def spline_of_outlet(self):
        # outlet border
        self.outlet = np.concatenate(
            (np.reshape(self.point_hub_inlet, (1, 2)), self.inlet[1:-1, :], np.reshape(self.point_shroud_inlet, (1, 2))))
        self.trailing_edge = Curve(z=self.outlet[:, 0], r=self.outlet[:, 1], mode='cordinates')

        # inlet border
        inlet_z = np.array([self.hub_trim.z[0], self.shroud_trim.z[0]])
        inlet_r = np.array([self.hub_trim.r[0], self.shroud_trim.r[0]])
        self.leading_edge = Curve(z=inlet_z, r=inlet_r, mode='cordinates')

    def spline_of_inlet(self):
        """
        make splines of the inlet border for the outlet block, which coincides with self.Outlet
        At the same time prepare the outlet edge, as a straight line between the final points
        """
        self.inlet = np.concatenate(
            (np.reshape(self.point_hub_outlet, (1, 2)), self.outlet[1:-1, :], np.reshape(self.point_shroud_outlet, (1, 2))))

        self.leading_edge = Curve(z=self.inlet[:, 0], r=self.inlet[:, 1], mode='cordinates')

        outlet_z = np.array([self.hub_trim.z[-1], self.shroud_trim.z[-1]])
        outlet_r = np.array([self.hub_trim.r[-1], self.shroud_trim.r[-1]])

        self.trailing_edge = Curve(z=outlet_z, r=outlet_r, mode='cordinates')

    def sample_hub_shroud(self, sampling_mode='default'):
        """
        Sample correctly the hub and shroud spline, already trimmed properly, with a certain sampling mode.
        :param sampling_mode: type of sampling, default or clustered
        """
        self.hub_trim.sample(self.nstream, sampling_mode=sampling_mode)
        self.shroud_trim.sample(self.nstream, sampling_mode=sampling_mode)

    def sample_inlet_outlet(self, sampling_mode='default'):
        """
        Sample the inlet edge for the outlet block.
        :param sampling_mode: type of sampling, default or clustered
        """
        self.leading_edge.sample(npoints=self.nspan, sampling_mode=sampling_mode)
        self.trailing_edge.sample(npoints=self.nspan, sampling_mode=sampling_mode)

    def compute_grid_points(self, inlet_block=False, outlet_block=False, inlet_meridional_obj=None,
                            outlet_meridional_obj=None, save_animation=False):
        """
        Compute the internal grid points with a certain algorithm, specified by grid_mode.
        :param block_counter: int, needed to select the correct information for every block
        :param inlet_block: if True, disables the grid stretching at inlet
        :param outlet_block: if False disables the grid stretching at outlet
        :param inlet_meridional_obj: provide inlet meridional object if you wish to mantain consistency of the shared nodes
        :param outlet_meridional_obj: provide outlet meridional object if you wish to mantain consistency of the shared nodes
        :param save_animation: if True store the Matrix necessary for the animation of the elliptic grid generation.
        """
        block_counter = self.iblock
        stream_coeff = self.config.get_sigmoid_stream_coefficients()[block_counter]
        span_coeff = self.config.get_sigmoid_span_coefficient()

        if self.config.get_verbosity():
            print_banner('GRID GENERATION OF BLOCK %02d' % block_counter)
            block_type = self.config.get_blocks_trim_type()[self.iblock]
            print(f"{'Trim type:':<{total_chars_mid}}{block_type:>{total_chars_mid}}")
            print(f"{'Grid Generation Mode:':<{total_chars_mid}}{self.config.get_mesh_generation_method():>{total_chars_mid}}")
            if self.config.get_mesh_generation_method().lower() == 'elliptic':
                print(f"{'Orthogonality Constraint:':<{total_chars_mid}}{self.config.get_grid_orthogonality():>{total_chars_mid}}")
            print(f"{'Boundaries spline order:':<{total_chars_mid}}{self.config.get_boundaries_spline_order():>{total_chars_mid}}")
            print(f"{'X Stretching Coefficient:':<{total_chars_mid}}"
                    f"{stream_coeff:>{total_chars_mid}}")
            print(f"{'Y Stretching Coefficient:':<{total_chars_mid}}"
                    f"{span_coeff:>{total_chars_mid}}")
            if inlet_meridional_obj is not None:
                print(f"{'Inlet Object Present:':<{total_chars_mid}}{True:>{total_chars_mid}}")
            if outlet_meridional_obj is not None:
                print(f"{'Outlet Object Present:':<{total_chars_mid}}{True:>{total_chars_mid}}")
            print_banner()

        # handle the case in which some grid cordinates must be copied from adjacent blocks
        if inlet_meridional_obj is not None:
            inlet = np.vstack((inlet_meridional_obj.z_grid[-1, :], inlet_meridional_obj.r_grid[-1, :]))
            fix_inlet = True
        else:
            inlet = np.vstack((self.leading_edge.z_sample, self.leading_edge.r_sample))
            fix_inlet = False
        if outlet_meridional_obj is not None:
            outlet = np.vstack((outlet_meridional_obj.z_grid[0, :], outlet_meridional_obj.r_grid[0, :]))
            fix_outlet = True
        else:
            outlet = np.vstack((self.trailing_edge.z_sample, self.trailing_edge.r_sample))
            fix_outlet = False
        hub = np.vstack((self.hub_trim.z_sample, self.hub_trim.r_sample))
        shroud = np.vstack((self.shroud_trim.z_sample, self.shroud_trim.r_sample))

        if self.config.get_mesh_generation_method() == 'elliptic':
            self.z_grid, self.r_grid = elliptic_grid_generation(inlet, hub, outlet, shroud,
                                                                              self.config.get_grid_orthogonality(),
                                                                              self.config.get_mesh_type(),
                                                                              inlet_block=inlet_block, outlet_block=outlet_block,
                                                                              save_animation=save_animation)
        elif self.config.get_mesh_generation_method().upper() == 'TFI':
            self.z_grid, self.r_grid = transfinite_grid_generation(inlet, hub, outlet, shroud,
                                                                                 self.config.get_blocks_topology()[block_counter],
                                                                                 stream_coeff, span_coeff)


    def add_inlet_outlet_curves(self, inlet, outlet):
        """
        Stores information regarding the inlet and outlet curve points, taken from blade object,
        in order to compute the leading and trailing splines.
        """
        self.inlet = inlet
        self.outlet = outlet
        self.inlet_curve = Curve(z=inlet[:, 0], r=inlet[:, 1], mode='cordinates', degree_spline=1)
        self.outlet_curve = Curve(z=outlet[:, 0], r=outlet[:, 1], mode='cordinates', degree_spline=1)

    def extend_inlet_outlet_curves(self):
        """
        Extend the inlet and outlet curves in order to find the intersections with the hub and shroud curves.
        """
        self.inlet_curve.extend()
        self.outlet_curve.extend()

    def find_intersections(self, tol=1e-8):
        """
        Having the hub and shroud curves, it looks for the intersections of these curves with the inlet and outlet points
        :param tol: tolerance of the algorithm to find intersection. If too small, it doesn't find the correct intersections
        :param visual_check: Set to True to graphically see the linest and the intersections found
        """

        hub_curve = np.stack((self.hub.z, self.hub.r), axis=1)
        shroud_curve = np.stack((self.shroud.z, self.shroud.r), axis=1)
        inlet_curve = np.stack((self.inlet_curve.z_spline_ext, self.inlet_curve.r_spline_ext), axis=1)
        outlet_curve = np.stack((self.outlet_curve.z_spline_ext, self.outlet_curve.r_spline_ext), axis=1)

        self.point_hub_inlet = self.point_intersection(inlet_curve, hub_curve, tol=tol)
        self.point_hub_outlet = self.point_intersection(outlet_curve, hub_curve, tol=tol)
        self.point_shroud_inlet = self.point_intersection(inlet_curve, shroud_curve, tol=tol)
        self.point_shroud_outlet = self.point_intersection(outlet_curve, shroud_curve, tol=tol)

        if self.config.get_visual_debug():
            plt.figure()
            plt.plot(hub_curve[:, 0], hub_curve[:, 1], 'C0', label='hub curve')
            plt.plot(shroud_curve[:, 0], shroud_curve[:, 1], 'C1', label='shroud curve')
            plt.plot(inlet_curve[:, 0], inlet_curve[:, 1], 'C2', label='inlet curve')
            plt.plot(outlet_curve[:, 0], outlet_curve[:, 1], 'C3', label='outlet curve')
            plt.plot(self.point_hub_inlet[0], self.point_hub_inlet[1], 'C4x', ms=10, label='hub_le')
            plt.plot(self.point_shroud_inlet[0], self.point_shroud_inlet[1], 'C5x', ms=10, label='shroud_le')
            plt.plot(self.point_hub_outlet[0], self.point_hub_outlet[1], 'C6x', ms=10, label='hub_te')
            plt.plot(self.point_shroud_outlet[0], self.point_shroud_outlet[1], 'C7x', ms=10, label='shroud_te')
            ax = plt.gca()
            ax.set_aspect('equal')
            plt.legend()

    @staticmethod
    def point_intersection(curve1, curve2, tol):
        """
        find and return the intersection between 2 curves. static method because it is bound to the class, not to an instance
        of the class. It could also avoid to specify the self, since it is not used.
        :param curve1: first curve
        :param curve2: second curve
        :param tol: tolerance threshold for the algorithm. 1e-2 seems like a good value, since at this point the cordinates
        are already non-dimensional
        """
        tree = KDTree(curve1)
        intersection_points = []

        # while loop to make sure the intersection algorithm finds a point
        while len(intersection_points) == 0:
            distances, indices = tree.query(curve2)
            intersection_points = curve1[indices[distances < tol]]
            tol *= 10
        point = np.mean(intersection_points, axis=0)
        return point

    def internal_zone_trim(self):
        """
        Trim the block hub and shroud curves at the found intersections with the inlet and outlet curves.
        :param machine_type: needed to know what kind of cut to apply
        """
        block_type = self.config.get_blocks_trim_type()[self.iblock]
        if block_type.lower() == 'axial-radial':
            self.hub.trim_curve_inlet(z_trim=self.point_hub_inlet[0])
            self.hub.trim_curve_outlet(r_trim=self.point_hub_outlet[1])
            self.shroud.trim_curve_inlet(z_trim=self.point_shroud_inlet[0])
            self.shroud.trim_curve_outlet(r_trim=self.point_shroud_outlet[1])
        elif block_type.lower() == 'radial-axial':
            self.hub.trim_curve_inlet(r_trim=self.point_hub_inlet[1])
            self.hub.trim_curve_outlet(z_trim=self.point_hub_outlet[0])
            self.shroud.trim_curve_inlet(r_trim=self.point_shroud_inlet[1])
            self.shroud.trim_curve_outlet(z_trim=self.point_shroud_outlet[0])
        elif block_type.lower() == 'axial':
            self.hub.trim_curve_inlet(z_trim=self.point_hub_inlet[0])
            self.hub.trim_curve_outlet(z_trim=self.point_hub_outlet[0])
            self.shroud.trim_curve_inlet(z_trim=self.point_shroud_inlet[0])
            self.shroud.trim_curve_outlet(z_trim=self.point_shroud_outlet[0])
        elif block_type.lower() == 'radial':
            self.hub.trim_curve_inlet(r_trim=self.point_hub_inlet[1])
            self.hub.trim_curve_outlet(r_trim=self.point_hub_outlet[1])
            self.shroud.trim_curve_inlet(r_trim=self.point_shroud_inlet[1])
            self.shroud.trim_curve_outlet(r_trim=self.point_shroud_outlet[1])
        else:
            raise ValueError('Insert a valid machine type')

    def inlet_zone_trim(self, mode):
        """
        Trim method for the inlet block. Hub and shroud curves are trimmed at found intersections with the leading edge
        intersections of the blade.
        :param mode: axial or radial, used to distinguish trimming algorithm.
        """
        if mode == 'axial':
            self.hub.trim_curve_outlet(z_trim=self.point_hub_inlet[0])
            self.shroud.trim_curve_outlet(z_trim=self.point_shroud_inlet[0])
        elif mode == 'radial':
            self.hub.trim_curve_inlet(r_trim=self.point_hub_inlet[1])
            self.shroud.trim_curve_inlet(r_trim=self.point_shroud_inlet[1])
        else:
            raise ValueError("Unknown trimming method.")

    def outlet_zone_trim(self, mode):
        """
        Trim method for the outlet block. Hub and shroud curves are trimmed at found intersections with the trailing edge
        intersections of the blade.
        :param mode: axial or radial, used to distinguish trimming algorithm.
        """
        if mode == 'radial':
            self.hub.trim_curve_inlet(r_trim=self.point_hub_outlet[1])
            self.shroud.trim_curve_inlet(r_trim=self.point_shroud_outlet[1])
        elif mode == 'axial':
            self.hub.trim_curve_inlet(z_trim=self.point_hub_outlet[0])
            self.shroud.trim_curve_inlet(z_trim=self.point_shroud_outlet[0])
        else:
            raise ValueError("Unknown trimming method.")

    def compute_dual_grid(self):
        """
        compute the secondary dual grid, using the 4 points that lie in the baricenter of 4 primary grid points
        """
        self.z_grid_dual = np.zeros((self.nstream + 1, self.nspan + 1))
        self.r_grid_dual = np.zeros((self.nstream + 1, self.nspan + 1))

        # internal points
        for istream in range(1, self.nstream):
            for ispan in range(1, self.nspan):
                z_mid_point = 0.25 * (
                        self.z_grid[istream, ispan] + self.z_grid[istream - 1, ispan] + self.z_grid[istream, ispan - 1] +
                        self.z_grid[istream - 1, ispan - 1])

                r_mid_point = 0.25 * (
                        self.r_grid[istream, ispan] + self.r_grid[istream - 1, ispan] + self.r_grid[istream, ispan - 1] +
                        self.r_grid[istream - 1, ispan - 1])

                self.z_grid_dual[istream, ispan] = z_mid_point
                self.r_grid_dual[istream, ispan] = r_mid_point

        # fix the vertices
        self.z_grid_dual[0, 0] = self.z_grid[0, 0]
        self.r_grid_dual[0, 0] = self.r_grid[0, 0]
        self.z_grid_dual[0, -1] = self.z_grid[0, -1]
        self.r_grid_dual[0, -1] = self.r_grid[0, -1]
        self.z_grid_dual[-1, -1] = self.z_grid[-1, -1]
        self.r_grid_dual[-1, -1] = self.r_grid[-1, -1]
        self.z_grid_dual[-1, 0] = self.z_grid[-1, 0]
        self.r_grid_dual[-1, 0] = self.r_grid[-1, 0]

        # istream = 0 border
        for istream in range(0, 1):
            for ispan in range(1, self.nspan):
                z_mid_point = 0.5 * (self.z_grid[istream, ispan] + self.z_grid[istream, ispan - 1])
                r_mid_point = 0.5 * (self.r_grid[istream, ispan] + self.r_grid[istream, ispan - 1])
                self.z_grid_dual[istream, ispan] = z_mid_point
                self.r_grid_dual[istream, ispan] = r_mid_point

        # istream = -1 border
        for istream in range(self.nstream, self.nstream + 1):
            for ispan in range(1, self.nspan):
                z_mid_point = 0.5 * (self.z_grid[istream - 1, ispan] + self.z_grid[istream - 1, ispan - 1])
                r_mid_point = 0.5 * (self.r_grid[istream - 1, ispan] + self.r_grid[istream - 1, ispan - 1])
                self.z_grid_dual[istream, ispan] = z_mid_point
                self.r_grid_dual[istream, ispan] = r_mid_point

        # ispan = 0 border
        for istream in range(1, self.nstream):
            for ispan in range(0, 1):
                z_mid_point = 0.5 * (self.z_grid[istream, ispan] + self.z_grid[istream - 1, ispan])
                r_mid_point = 0.5 * (self.r_grid[istream, ispan] + self.r_grid[istream - 1, ispan])
                self.z_grid_dual[istream, ispan] = z_mid_point
                self.r_grid_dual[istream, ispan] = r_mid_point

        # ispan = -1 border
        for istream in range(1, self.nstream):
            for ispan in range(self.nspan, self.nspan + 1):
                z_mid_point = 0.5 * (self.z_grid[istream, ispan - 1] + self.z_grid[istream - 1, ispan - 1])
                r_mid_point = 0.5 * (self.r_grid[istream, ispan - 1] + self.r_grid[istream - 1, ispan - 1])
                self.z_grid_dual[istream, ispan] = z_mid_point
                self.r_grid_dual[istream, ispan] = r_mid_point

    def plot_full_grid(self, save_filename=None, primary_grid=True, primary_grid_points=False, secondary_grid=False,
                       secondary_grid_points=False, hub_shroud=False, outline=False, grid_centers=False, ticks=True,
                       save_foldername=None):
        """
        Plot the obtained grid.
        :param save_filename: specify path of the figures to be saved (if you want to save).
        :param primary_grid: if True plots the primary grid lines
        :param primary_grid_points: if True plots the primary grid points
        :param secondary_grid: if True plots the secondary grid lines
        :param secondary_grid_points: if True plots the secondary grid points
        :param hub_shroud: if True plots hub and shroud highlighted
        :param outline: if True plots the highlighted outline of the domain
        :param grid_centers: if True plots the grid centers
        :param ticks: if True allows ticks to be shown
        :param save_foldername: folder name to save pictures in
        """


        plt.figure()

        # hub and shroud plot
        if hub_shroud:
            plt.plot(self.hub.z_spline, self.hub.r_spline, lw=light_line_width, c='black')
            plt.plot(self.shroud.z_spline, self.shroud.r_spline, lw=light_line_width, c='black')

        # primary grid
        if primary_grid:
            for istream in range(0, self.nstream):
                plt.plot(self.z_grid[istream, :], self.r_grid[istream, :], lw=light_line_width, c='black')
            for ispan in range(0, self.nspan):
                plt.plot(self.z_grid[:, ispan], self.r_grid[:, ispan], lw=light_line_width, c='black')
        elif outline:
            plt.plot(self.z_grid[0, :], self.r_grid[0, :], lw=line_width, label='leading edge')
            plt.plot(self.z_grid[-1, :], self.r_grid[-1, :], lw=line_width, label='trailing edge')
            plt.plot(self.z_grid[:, 0], self.r_grid[:, 0], lw=line_width, label='hub')
            plt.plot(self.z_grid[:, -1], self.r_grid[:, -1], lw=line_width, label='shroud')

        # primary grid points
        if primary_grid_points:
            plt.scatter(self.z_grid.flatten(), self.r_grid.flatten(), c='black', s=marker_size,
                        label='primary grid nodes')

        # secondary grid
        if secondary_grid:
            for istream in range(0, self.nstream + 1):
                plt.plot(self.z_grid_dual[istream, :], self.r_grid_dual[istream, :], '--r', lw=light_line_width)
            for ispan in range(0, self.nspan + 1):
                plt.plot(self.z_grid_dual[:, ispan], self.r_grid_dual[:, ispan], '--r', lw=light_line_width)

        if grid_centers:
            plt.scatter(self.z_grid, self.r_grid, marker='+', s=marker_size_small, c='black')

        if secondary_grid_points:
            plt.scatter(self.z_grid_dual.flatten(), self.r_grid_dual.flatten(), c='red', s=marker_size,
                        label='secondary grid nodes')

        if primary_grid_points or secondary_grid_points or outline:
            plt.legend()
        plt.xlabel(r'$z \ \mathrm{[m]}$')
        plt.ylabel(r'$r \ \mathrm{[m]}$')
        plt.title(r'$(%d \times %d)$' % (self.nstream, self.nspan))
        ax=plt.gca()
        ax.set_aspect('equal')

        if not ticks:
            plt.xticks([])
            plt.yticks([])  # plt.xlabel('')  # plt.ylabel('')

        if save_filename is not None:
            plt.savefig(self.config.get_pictures_folder_path() + '/' + save_filename + '_%02i_%02i.pdf' %(self.nstream, self.nspan), bbox_inches='tight')

    def find_border(self):
        """
        Find the border delimiting the block. it stores the border info as (r,z) column arrays
        """
        border_z = []
        border_r = []

        # append hub cordinates
        border_z.append(self.z_grid[0:, 0])
        border_r.append(self.r_grid[0:, 0])

        # append outlet cordinates
        border_z.append(self.z_grid[-1, 1:])
        border_r.append(self.r_grid[-1, 1:])

        # append shroud cordinates
        border_z.append(np.flip(self.z_grid[0:-2, -1]))
        border_r.append(np.flip(self.r_grid[0:-2, -1]))

        # append inlet cordinates
        border_z.append(np.flip(self.z_grid[0, 1:]))
        border_r.append(np.flip(self.r_grid[0, 1:]))

        border_z = [item for sublist in border_z for item in sublist]
        border_r = [item for sublist in border_r for item in sublist]

        self.border = np.stack((border_z, border_r), axis=1)

    def show_outline_grid(self):
        """
        Show the outline grid, with the sampling points
        """

        plt.figure()
        plt.plot(self.hub_trim.z_sample, self.hub_trim.r_sample, '-o')
        plt.plot(self.shroud_trim.z_sample, self.shroud_trim.r_sample, '-o')
        plt.plot(self.leading_edge.z_sample, self.leading_edge.r_sample, '-o')
        plt.plot(self.trailing_edge.z_sample, self.trailing_edge.r_sample, '-o')


    def compute_three_dimensional_mesh(self, N_THETA):
        """
        Compute the Three-dimensional mesh X,Y,Z as 3D arrays, structured
        """
        theta = np.linspace(0, 2 * np.pi, N_THETA)
        self.X_mesh = np.zeros((self.nstream, self.nspan, N_THETA))
        self.Y_mesh = np.zeros((self.nstream, self.nspan, N_THETA))
        self.Z_mesh = np.zeros((self.nstream, self.nspan, N_THETA))

        for i in range(self.nstream):
            for j in range(self.nspan):
                for k in range(N_THETA):
                    self.X_mesh[i, j, k] = self.r_grid[i, j] * np.cos(theta[k])
                    self.Y_mesh[i, j, k] = self.r_grid[i, j] * np.sin(theta[k])
                    self.Z_mesh[i, j, k] = self.z_grid[i, j]

    def save_mesh_pickle(self, filepath=None):
        """
        Save the mesh cordinates in a pickle
        """

        mesh = {'x': self.X_mesh, 'y': self.Y_mesh, 'z': self.Z_mesh}

        if filepath == None:
            filepath = 'mesh_%02i_%02i_%2i.pickle' % (self.nstream, self.nspan, self.X_mesh.shape[2])
        with open(filepath, 'wb') as f:
            pickle.dump(mesh, f)

        print(f"Data saved to '{filepath}'")
    

    def addFieldsForBFM(self, fields):
        """
        Add the blade fields to the object fields
        """
        self.bfmFields = {}
        for key in fields.keys():
            self.bfmFields[key] = fields[key]