import configparser
import ast
import numpy as np
import os


class Config:
    def __init__(self, config_file='input.ini'):
        self.config_parser = configparser.ConfigParser()
        self.config_parser.read(config_file)

        cwd = os.getcwd()
        print('Configuration file path: %s' % os.path.join(cwd, config_file))
        sw_points = self.get_streamwise_points()
        sp_points = self.get_spanwise_points()
        print('Number of streamwise points: ', sw_points)
        print('Number of spanwise points: ', sp_points)

        self.picture_name_template = self.compute_picture_name_template(config_file)

    def compute_picture_name_template(self, config_file):
        prefix = config_file.split('.')[0]
        streamwise = self.get_streamwise_points()
        spanwise = self.get_spanwise_points()
        for st in streamwise:
            prefix += '_' + str(st)
        prefix += '_' + str(spanwise)
        return prefix

    def get_config_value(self, section, option, default=None):
        """
        Helper method to retrieve a configuration value with a default fallback.
        """
        try:
            return self.config_parser.get(section, option)
        except (configparser.NoSectionError, configparser.NoOptionError):
            return default

    def print_config(self):
        """
        Print the entire configuration.
        """
        for section in self.config_parser.sections():
            print(f"[{section}]")
            for option, value in self.config_parser.items(section):
                print(f"{option} = {value}")
            print()

    def create_attributes(self):
        """
        Dynamically create attributes from the configuration.
        """
        for section in self.config_parser.sections():
            for option, value in self.config_parser.items(section):
                setattr(self, option, value)

    def get_mesh_type(self):
        return str(self.config_parser.get('CFD PROCESSING', 'MESH_TYPE'))

    def get_blocks_number(self):
        return int(self.config_parser.get('CFD PROCESSING', 'BLOCKS_NUMBER'))

    def get_streamwise_points(self):
        value = self.config_parser.get('CFD PROCESSING', 'STREAMWISE_POINTS')
        return ast.literal_eval(value)

    def get_blocks_type(self):
        value = self.config_parser.get('CFD PROCESSING', 'BLOCKS_TYPE')
        return ast.literal_eval(value)

    def get_spanwise_points(self):
        return int(self.config_parser.get('CFD PROCESSING', 'SPANWISE_POINTS'))

    def get_cfd_filepath(self):
        return str(self.config_parser.get('CFD PROCESSING', 'CFD_FILEPATH'))

    def get_cfd_filetype(self):
        return str(self.config_parser.get('CFD PROCESSING', 'CFD_FILETYPE'))

    def get_reference_density(self):
        return float(self.config_parser.get('CFD PROCESSING', 'RHO_REF'))

    def get_reference_length(self):
        return float(self.config_parser.get('CFD PROCESSING', 'X_REF'))

    def get_reference_rpm(self):
        return float(self.config_parser.get('CFD PROCESSING', 'RPM_REF'))

    def get_reference_omega(self):
        return 2 * np.pi * float(self.config_parser.get('CFD PROCESSING', 'RPM_REF')) / 60

    def get_omega_shaft(self):
        return 2 * np.pi * self.get_shaft_rpm() / 60

    def get_reference_velocity(self):
        return self.get_reference_omega() * self.get_reference_length()

    def get_reference_time(self):
        return 1 / self.get_reference_omega()

    def get_reference_pressure(self):
        return self.get_reference_density() * self.get_reference_velocity() ** 2

    def get_reference_entropy(self):
        return self.get_reference_velocity() ** 2 / self.get_reference_temperature()

    def get_reference_temperature(self):
        return float(self.config_parser.get('CFD PROCESSING', 'T_REF'))

    def get_shaft_rpm(self):
        return float(self.config_parser.get('CFD PROCESSING', 'SHAFT_RPM'))

    def get_coordinates_file_units(self):
        return str(self.config_parser.get('CFD PROCESSING', 'COORDINATES_FILE_UNITS'))

    def get_coordinates_rescaling_factor(self):
        units = self.get_coordinates_file_units()
        if units == 'm':
            factor = 1
        elif units == 'cm':
            factor = 1e-2
        elif units == 'mm':
            factor = 1e-3
        elif units == 'in':
            factor = 0.0254
        else:
            raise ValueError('Units not supported')
        return factor

    def get_sigmoid_stream_coefficient(self):
        return int(self.config_parser.get('CFD PROCESSING', 'SIGMOID_STREAM_COEFFICIENT'))

    def get_sigmoid_span_coefficient(self):
        return int(self.config_parser.get('CFD PROCESSING', 'SIGMOID_SPAN_COEFFICIENT'))

    def get_hub_curve_filepath(self):
        return str(self.config_parser.get('CFD PROCESSING', 'HUB_COORDINATES_FILEPATH'))

    def get_shroud_curve_filepath(self):
        return str(self.config_parser.get('CFD PROCESSING', 'SHROUD_COORDINATES_FILEPATH'))

    def get_blade_curve_filepath(self):
        return str(self.config_parser.get('CFD PROCESSING', 'BLADE_COORDINATES_FILEPATH'))

    def get_blade_inlet_type(self):
        return str(self.config_parser.get('CFD PROCESSING', 'BLADE_INLET_TYPE'))

    def get_blade_outlet_type(self):
        return str(self.config_parser.get('CFD PROCESSING', 'BLADE_OUTLET_TYPE'))

    def get_verbosity(self):
        res = self.config_parser.get('CFD PROCESSING', 'VERBOSITY')
        if res.lower() == 'true':
            return True
        else:
            return False

    def get_normalize_data(self):
        res = self.config_parser.get('CFD PROCESSING', 'NORMALIZE_DATA')
        if res.lower() == 'true':
            return True
        else:
            return False

    def get_standard_regression(self):
        res = self.config_parser.get('CFD PROCESSING', 'STANDARD_REGRESSION')
        if res.lower() == 'true':
            return True
        else:
            return False

    def get_disable_body_force(self):
        res = self.config_parser.get('SUN MODEL', 'DISABLE_BODY_FORCE')
        if res.lower() == 'true':
            return True
        else:
            return False

    def get_mesh_generation_method(self):
        return str(self.config_parser.get('CFD PROCESSING', 'MESH_GENERATION_METHOD'))

    def get_grid_orthogonality(self):
        return str(self.config_parser.get('CFD PROCESSING', 'GRID_ORTHOGONALITY'))

    def get_fluid_gamma(self):
        return float(self.config_parser.get('CFD PROCESSING', 'GAMMA_FLUID'))

    def get_cfd_interpolation_method(self):
        return str(self.config_parser.get('CFD PROCESSING', 'CFD_SOLUTION_INTERPOLATION_METHOD'))

    def get_gradient_interpolation_method(self):
        return str(self.config_parser.get('CFD PROCESSING', 'GRADIENT_INTERPOLATION_METHOD'))

    def get_meridional_pickle_filepath(self):
        return str(self.config_parser.get('SUN MODEL', 'MERIDIONAL_PICKLE_FILEPATH'))

    def get_grid_transformation_gradient_routine(self):
        return str(self.config_parser.get('SUN MODEL', 'GRID_TRANSFORMATION_GRADIENT_ROUTINE'))

    def get_grid_transformation_gradient_order(self):
        return int(self.config_parser.get('SUN MODEL', 'GRID_TRANSFORMATION_GRADIENT_ORDER'))

    def get_circumferential_harmonic_order(self):
        return int(self.config_parser.get('SUN MODEL', 'CIRCUMFERENTIAL_HARMONIC_ORDER'))

    def get_normalize_instability_equations(self):
        res = self.config_parser.get('SUN MODEL', 'NORMALIZE_INSTABILITY_EQUATIONS')
        if res.lower() == 'true':
            return True
        else:
            return False

    def get_research_center_omega_eigenvalues(self):
        return complex(self.config_parser.get('SUN MODEL', 'OMEGA_EIGV_RESEARCH_CENTER'))

    def get_research_number_omega_eigenvalues(self):
        return int(self.config_parser.get('SUN MODEL', 'NUMBER_EIGV_RESEARCH'))

    def get_inlet_bc(self):
        return str(self.config_parser.get('SUN MODEL', 'INLET_BC'))

    def get_outlet_bc(self):
        return str(self.config_parser.get('SUN MODEL', 'OUTLET_BC'))

    def get_hub_bc(self):
        return str(self.config_parser.get('SUN MODEL', 'HUB_BC'))

    def get_shroud_bc(self):
        return str(self.config_parser.get('SUN MODEL', 'SHROUD_BC'))

    def get_euler_wall_equation(self):
        return str(self.config_parser.get('SUN MODEL', 'EULER_WALL_EQUATION'))

    def get_boundary_interface_gradient_method(self):
        return str(self.config_parser.get('SUN MODEL', 'BOUNDARY_INTERFACE_GRADIENT_METHOD'))

    def get_clipping_bfm(self):
        res = self.config_parser.get('CFD PROCESSING', 'CLIPPING_BFM')
        if res.lower() == 'true':
            return True
        else:
            return False
