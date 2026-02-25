import matplotlib.pyplot as plt
from numpy import sqrt
from scipy.interpolate import splprep, splev
from utils.styles import *
from .functions import *


class Curve:

    def __init__(self, config=None, curve_filepath=None, z=None, r=None, mode='filedata', degree_spline=1):

        if mode == 'filedata':
            self.read_from_curve_file(curve_filepath)
            self.config = config
            self.r *= self.config.get_coordinates_rescaling_factor() / self.config.get_reference_length()
            self.z *= self.config.get_coordinates_rescaling_factor() / self.config.get_reference_length()
            if self.config.invert_axial_coordinates():
                self.z = -self.z
            
        elif mode == 'cordinates':
            self.r = r
            self.z = z

        self.r_spline, self.z_spline = self.compute_spline(degree_spline=degree_spline)
        
    def read_from_curve_file(self, filepath):
        self.data = np.loadtxt(filepath)
        self.x = self.data[:, 0]
        self.y = self.data[:, 1]
        self.z = self.data[:, 2]
        self.x, self.y, self.z = eliminate_duplicates(self.x, 
                                                      self.y,
                                                      self.z)
        self.r = sqrt(self.x ** 2 + self.y ** 2)

    def compute_spline(self, u_eval=np.linspace(0, 1, 10000), degree_spline=1):
        self.tck, _ = splprep([self.r, self.z], s=0, k=degree_spline)
        r_spline, z_spline = splev(u_eval, self.tck)
        return r_spline, z_spline

    def extend(self, u_min=-0.25, u_max=1.25, degree_spline=1, num_points=10000):
        u_spline_ext = np.linspace(u_min, u_max, num_points)
        self.r_spline_ext, self.z_spline_ext = self.compute_spline(u_eval=u_spline_ext, degree_spline=degree_spline)

    def trim_curve_inlet(self, z_trim='span', r_trim='span'):
        if r_trim == 'span':
            idx = np.where(self.z_spline >= z_trim)
        elif z_trim == 'span':
            idx = np.where(self.r_spline >= r_trim)
        else:
            raise ValueError("Unknown trim type!")
        
        self.z_spline = self.z_spline[idx]
        self.r_spline = self.r_spline[idx]

    def trim_curve_outlet(self, z_trim='span', r_trim='span'):
        if z_trim == 'span':
            idx = np.where(self.r_spline <= r_trim)
        elif r_trim == 'span':
            idx = np.where(self.z_spline <= z_trim)
        else:
            raise ValueError("Unknown trim type!")
        self.z_spline = self.z_spline[idx]
        self.r_spline = self.r_spline[idx]

    def sample(self, npoints, sampling_mode='default'):
        if sampling_mode == 'default':
            self.u_sample = np.linspace(0, 1, npoints)
        elif sampling_mode == 'clustering':
            self.u_sample = cluster_sample_u(npoints)
        elif sampling_mode == 'clustering_left':
            self.u_sample = cluster_sample_u(npoints, border='left')
        elif sampling_mode == 'clustering_right':
            self.u_sample = cluster_sample_u(npoints, border='right')
        
        self.r_sample, self.z_sample = splev(self.u_sample, self.tck)

    def plot_spline(self):
        plt.figure()
        plt.plot(self.z, self.r, 'o', label='cordinates')
        plt.plot(self.z_spline, self.r_spline, label='B-spline')
        plt.legend()
        plt.xlabel(r'$z \ \mathrm{[%s]}$' % (self.config.get_coordinates_file_units()))
        plt.ylabel(r'$r \ \mathrm{[%s]}$' % (self.config.get_coordinates_file_units()))
