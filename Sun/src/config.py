import configparser
import ast
import numpy as np
import os
from Sun.src.general_functions import print_banner_begin, print_banner_end
from Utils.styles import total_chars, total_chars_mid



class Config:
    def __init__(self, config_file='input.ini'):
        self.config_parser = configparser.ConfigParser()
        self.config_parser.read(config_file)

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

    def get_blocks_type(self):
        value = self.config_parser.get('TURBOMACHINERY DATA', 'BLOCKS_TYPE')
        return ast.literal_eval(value)

    def get_blocks_topology(self):
        value = self.config_parser.get('TURBOMACHINERY DATA', 'BLOCKS_TOPOLOGY')
        values = [str(val.strip(',')) for val in value.split()]
        return values



    def get_cfd_filepath(self):
        return str(self.config_parser.get('CFD PROCESSING', 'CFD_FILEPATH'))

    def get_cfd_filetype(self):
        return str(self.config_parser.get('CFD PROCESSING', 'CFD_FILETYPE'))

    def get_reference_density(self):
        return float(self.config_parser.get('CFD PROCESSING', 'RHO_REF'))

    def get_reference_length(self):
        try:
            return float(self.config_parser.get('CFD PROCESSING', 'X_REF'))
        except:
            return 1.0

    def get_reference_rpm(self):
        return float(self.config_parser.get('CFD PROCESSING', 'RPM_REF'))

    def get_reference_omega(self):
        return 2 * np.pi * float(self.config_parser.get('CFD PROCESSING', 'RPM_REF')) / 60

    def get_omega_shaft(self):
        rpm = self.get_shaft_rpm()
        omega = [2*np.pi*n/60 for n in rpm]
        return omega

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
        values = self.config_parser.get('TURBOMACHINERY DATA', 'SHAFT_RPM')
        values = [float(val.strip(',')) for val in values.split()]
        return values


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

    def get_blade_turning_direction(self):
        value = str(self.config_parser.get('BLADE RECONSTRUCTION', 'BLADE_TURNING_DIRECTION'))
        values = [str(val.strip()) for val in value.split(',')]
        return values
    
    def get_blade_curve_filepath(self):
        value = str(self.config_parser.get('BLADE RECONSTRUCTION', 'BLADE_COORDINATES_FILEPATH'))
        values = [str(val.strip()) for val in value.split(',')]
        return values
    
    def get_splitter_blade_curve_filepath(self):
        value = str(self.config_parser.get('BLADE RECONSTRUCTION', 'SPLITTER_BLADES_COORDINATES_FILEPATH'))
        values = [str(val.strip()) for val in value.split(',')]
        return values
    
    def is_splitter_blade_present(self, iblade):
        try:
            value = str(self.config_parser.get('BLADE RECONSTRUCTION', 'SPLITTER_BLADE_PRESENT'))
            values = [str(val.strip()) for val in value.split(',')]
            return values[iblade]
        except:
            nBlades = self.get_blade_rows_number()
            values = [False for i in range(nBlades)]
            return values[iblade]
        
    
    def get_blade_rows_number(self):
        filepaths = self.get_blade_curve_filepath()
        return len(filepaths)
    
    def get_splitter_blade_rows_number(self):
        try:
            filepaths = self.get_splitter_blade_curve_filepath()
            return len(filepaths)
        except:
            return 0

    def get_blade_inlet_type(self):
        value = str(self.config_parser.get('TURBOMACHINERY DATA', 'BLADE_INLET_TYPE'))
        value = [i.strip(',') for i in value.split()]
        return value

    
    def get_blade_profiles_spline_order(self):
        try:
            value = str(self.config_parser.get('BLADE RECONSTRUCTION', 'BLADE_PROFILES_SPLINE_ORDER'))
            value = [int(i.strip(',')) for i in value.split()]
            return value
        except:
            nBlades = self.get_blade_rows_number()
            values = [1 for i in range(nBlades)]
            return values
    
    def cutoff_trailing_edge(self, iblade):
        try:
            value = str(self.config_parser.get('BLADE RECONSTRUCTION', 'TRAILING_EDGE_CUTOFF'))
            value = [str(i.strip(',')) for i in value.split()]
            val = value[iblade]
            if val.lower()=='yes' or val.lower()=='true':
                return True
            elif val.lower()=='no' or val.lower()=='false':
                return False
            else:
                raise ValueError('TRAILING_EDGE_CUTOFF option can be true or false')
        except:
            return False
    
    def GetBlockType(self):
        try:
            value = str(self.config_parser.get('SUN MODEL', 'BLOCK_TYPE'))
            return value
        except:
            return 'unbladed'

    def get_multiblock_driver_type(self):
        try:
            value = str(self.config_parser.get('GENERAL', 'MULTIBLOCK_DRIVER_TYPE'))
            return value
        except:
            return 'multiblock'
        
    def get_mesh_output_topology(self):
        try:
            value = str(self.config_parser.get('GENERAL', 'OUTPUT_TOPOLOGY'))
            return value
        except:
            return 'axisymmetric'
    
    def get_mesh_periodic_number_points(self):
        value = int(self.config_parser.get('GENERAL', 'PERIODIC_NUMBER_POINTS'))
        return value
    
    def get_mesh_periodicity_theta(self):
        value = float(self.config_parser.get('GENERAL', 'PERIODICITY_THETA'))
        return value*np.pi/180

    def get_blade_outlet_type(self):
        value = str(self.config_parser.get('TURBOMACHINERY DATA', 'BLADE_OUTLET_TYPE'))
        value = [i.strip(',') for i in value.split()]
        return value
    
    def get_spanwise_profiles_paths(self, iblade):
        value = str(self.config_parser.get('BODY FORCE', 'SPANWISE_PROFILES_PATH'))
        value = [i.strip(',') for i in value.split()]
        return value[iblade*2:(iblade+1)*2]

    def get_verbosity(self):
        res = self.config_parser.get('GENERAL', 'VERBOSITY')
        if res.lower() == 'true' or res.lower() == 'yes':
            return True
        else:
            return False
    

    def get_visual_debug(self):
        res = self.config_parser.get('GENERAL', 'VISUAL_DEBUG')
        if res.lower() == 'true' or res.lower() == 'yes':
            return True
        else:
            return False
    

    def get_normalize_coordinates(self):
        
        try:
            res = self.config_parser.get('CFD PROCESSING', 'NORMALIZE_COORDINATES')
            if res.lower() == 'yes':
                return True
            elif res.lower() == 'no':
                return False
            else:
                raise ValueError('Invalid option NORMALIZE_COORDINATES. Specify yes or no.')
        except:
            return False # default option
        

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
        try:
            res = self.config_parser.get('SUN MODEL', 'DISABLE_BODY_FORCE')
            if res.lower() == 'true':
                return True
            else:
                return False
        except:
            return False

    def get_fluid_gamma(self):
        return float(self.config_parser.get('CFD PROCESSING', 'GAMMA_FLUID'))

    def get_cfd_interpolation_method(self):
        return str(self.config_parser.get('CFD PROCESSING', 'CFD_SOLUTION_INTERPOLATION_METHOD'))

    def get_gradient_interpolation_method(self):
        return str(self.config_parser.get('CFD PROCESSING', 'GRADIENT_INTERPOLATION_METHOD'))

    def get_meridional_pickle_filepath(self):
        return str(self.config_parser.get('SUN MODEL', 'MERIDIONAL_PICKLE_FILEPATH'))
    

    def get_pictures_folder_path(self):
        try:
            return str(self.config_parser.get('GENERAL', 'PICTURES_FOLDER_PATH'))
        except:
            return 'Pictures' # default
    
    def get_blade_reconstruction_regression_order(self):
        try:
            return int(self.config_parser.get('CFD PROCESSING', 'BLADE_RECONSTRUCTION_REGRESSION_ORDER'))
        except:
            return 3 # default

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
    
    def GetInputFile(self):
        return str(self.config_parser.get('SUN MODEL', 'INPUT_FILE'))

    def get_shroud_bc(self):
        return str(self.config_parser.get('SUN MODEL', 'SHROUD_BC'))

    def get_euler_wall_equation(self):
        return str(self.config_parser.get('SUN MODEL', 'EULER_WALL_EQUATION'))

    def get_boundary_interface_gradient_method(self):
        return str(self.config_parser.get('SUN MODEL', 'BOUNDARY_INTERFACE_GRADIENT_METHOD'))


    def get_blades_camber_reconstruction(self):
        values = str(self.config_parser.get('TURBOMACHINERY DATA', 'BLADES_CAMBER_RECONSTRUCTION'))
        values = [str(i.strip(',')) for i in values.split()]
        return values
    

    def get_blades_number(self):
        values = str(self.config_parser.get('TURBOMACHINERY DATA', 'BLADES_NUMBER'))
        blades = [int(i.strip(',')) for i in values.split()]
        return blades

    def get_rotation_factors(self):
        value = str(self.config_parser.get('TURBOMACHINERY DATA', 'ROTATION_FACTORS'))
        if len(value.split()) > 1:
            factors = [int(i.strip(',')) for i in value.split()]
        else:
            factors = int(value)
        return factors
    
    def get_bfm_rotational_speeds(self):
        value = self.config_parser.get('BFM DATA', 'BFM_ROTATIONAL_SPEEDS')
        components = value.strip("[]").split(", ")
        float_values = [float(value) for value in components]
        float_array = np.array(float_values)
        return float_array

    
    def get_output_type(self):
        try:
            values = self.config_parser.get('GENERAL', 'OUTPUT_TYPE')
            values = [str(val.strip(',')) for val in values.split()]
            return values
        except:
            return ['none'] # default
    

    def get_output_data_folder(self):
        try:
            return str(self.config_parser.get('GENERAL', 'OUTPUT_DATA_FOLDER'))
        except:
            return 'Output' # default
    
