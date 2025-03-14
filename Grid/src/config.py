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
        
        print_banner_begin('CONFIGURATION FILE')
        print(f"{'Configuration file: ':<{total_chars_mid}}{config_file:>{total_chars_mid}}")
        pic_folder = self.get_pictures_folder_path()
        os.makedirs(pic_folder, exist_ok=True)
        # print(f"{'Output folder for the pictures: ':<{total_chars_mid}}{pic_folder:>{total_chars_mid}}")
        # sw_points = self.get_streamwise_points()
        # sp_points = self.get_spanwise_points()
        # print(f"{'Number of streamwise points: ':<{total_chars_mid}}{str(sw_points):>{total_chars_mid}}")
        # print(f"{'Number of spanwise points: ':<{total_chars_mid}}{sp_points:>{total_chars_mid}}")
        # print(f"{'Driver type: ':<{total_chars_mid}}{self.get_multiblock_driver_type():>{total_chars_mid}}")
        print_banner_end('')
        

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
        value = self.config_parser.get('GRID GENERATION', 'STREAMWISE_POINTS')
        vals = [int(val.strip(',')) for val in value.split()]
        vals = np.array(vals, dtype=int)
        return vals

    def get_blocks_type(self):
        value = self.config_parser.get('TURBOMACHINERY DATA', 'BLOCKS_TYPE')
        return ast.literal_eval(value)

    def get_blocks_topology(self):
        value = self.config_parser.get('TURBOMACHINERY DATA', 'BLOCKS_TOPOLOGY')
        values = [str(val.strip(',')) for val in value.split()]
        return values

    def get_spanwise_points(self):
        return int(self.config_parser.get('GRID GENERATION', 'SPANWISE_POINTS'))

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

    def get_coordinates_file_units(self):
        return str(self.config_parser.get('GRID GENERATION', 'COORDINATES_FILE_UNITS'))

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

    def get_sigmoid_stream_coefficients(self):
        value = str(self.config_parser.get('GRID GENERATION', 'SIGMOID_STREAM_COEFFICIENTS'))
        value = [float(val.strip(',')) for val in value.split()]
        return value

    def get_sigmoid_span_coefficient(self):
        return float(self.config_parser.get('GRID GENERATION', 'SIGMOID_SPAN_COEFFICIENT'))

    def get_hub_curve_filepath(self):
        return str(self.config_parser.get('GRID GENERATION', 'HUB_COORDINATES_FILEPATH'))

    def get_shroud_curve_filepath(self):
        return str(self.config_parser.get('GRID GENERATION', 'SHROUD_COORDINATES_FILEPATH'))

    def get_blade_curve_filepath(self):
        value = str(self.config_parser.get('BLADE RECONSTRUCTION', 'BLADE_COORDINATES_FILEPATH'))
        values = [str(val.strip()) for val in value.split(',')]
        return values
    
    def get_blade_rows_number(self):
        filepaths = self.get_blade_curve_filepath()
        return len(filepaths)

    def get_blade_inlet_type(self):
        value = str(self.config_parser.get('TURBOMACHINERY DATA', 'BLADE_INLET_TYPE'))
        value = [i.strip(',') for i in value.split()]
        return value


    def get_blocks_trim_type(self):
        value = str(self.config_parser.get('GRID GENERATION', 'BLOCKS_TRIM_TYPE'))
        value = [str(i.strip(',')) for i in value.split()]
        return value
    
    def get_blade_profiles_spline_order(self):
        value = str(self.config_parser.get('BLADE RECONSTRUCTION', 'BLADE_PROFILES_SPLINE_ORDER'))
        value = [int(i.strip(',')) for i in value.split()]
        return value
    
    def get_boundaries_spline_order(self):
        try:
            value = int(self.config_parser.get('GRID GENERATION', 'BOUNDARIES_SPLINE_ORDER'))
        except:
            value = 3 # default
        return value
    
    def get_multiblock_driver_type(self):
        try:
            value = str(self.config_parser.get('GENERAL', 'MULTIBLOCK_DRIVER_TYPE'))
            return value
        except:
            return 'multiblock'

    def get_blade_outlet_type(self):
        value = str(self.config_parser.get('TURBOMACHINERY DATA', 'BLADE_OUTLET_TYPE'))
        value = [i.strip(',') for i in value.split()]
        return value

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
        res = self.config_parser.get('SUN MODEL', 'DISABLE_BODY_FORCE')
        if res.lower() == 'true':
            return True
        else:
            return False

    def get_mesh_generation_method(self):
        return str(self.config_parser.get('GRID GENERATION', 'MESH_GENERATION_METHOD'))

    def get_grid_orthogonality(self):
        return str(self.config_parser.get('GRID GENERATION', 'GRID_ORTHOGONALITY'))

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
            return 7 # default

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

    
    def invert_axial_coordinates(self):
        try:
            res = self.config_parser.get('GENERAL', 'INVERT_AXIAL_COORDINATES')
            if res.lower() == 'yes':
                return True
            elif res.lower() == 'no':
                return False
            else:
                raise ValueError('Invalid option INVERT_AXIAL_COORDINATES. Specify yes or no.')
        except:
            return False
    
    def get_extrapolation_method(self):
        choices = ['nearest', 'linear']
        
        try:
            res = self.config_parser.get('GENERAL', 'EXTRAPOLATION_METHOD').lower()            
        except:
            res = 'linear'
        
        if res not in choices:
            raise ValueError(f'Invalid option EXTRAPOLATION_METHOD. Possible options are {choices}.')
        
        return res
    
    def get_machine_name(self):
        try:
            return str(self.config_parser.get('TURBOMACHINERY DATA', 'MACHINE_NAME'))
        except:
            return 'Machine' # default
    
    
    def get_output_type(self):
        try:
            values = self.config_parser.get('GENERAL', 'OUTPUT_TYPE')
            values = [str(val.strip(',')) for val in values.split()]
            return values
        except:
            return ['none'] # default
    
    def get_turbo_BFM_mesh_output_fields(self):
        values = self.config_parser.get('GENERAL', 'TURBO_BFM_MESH_OUTPUT_FIELDS').lower()
        values = [str(val.strip(',')) for val in values.split()]
        return values
    
    
    def get_blade_edges_extrapolation_coefficient(self):
        try:
            value = self.config_parser.get('BLADE RECONSTRUCTION', 'BLADE_EDGES_EXTRAPOLATION_COEFFICIENT')
            vals = [float(val.strip(',')) for val in value.split()]
            vals = np.array(vals, dtype=float)
        except:
            nblades = self.get_blade_rows_number()
            vals = np.zeros(nblades)
            print('Default extrapolation blade coefficient: 0')
        return vals
    
    def get_blade_camber_smoothing_coefficient(self):
        try:
            value = float(self.config_parser.get('BLADE RECONSTRUCTION', 'BLADE_CAMBER_SMOOTHING_COEFFICIENT'))
        except:
            value = 0.0
        return value
    
    def get_output_data_folder(self):
        try:
            return str(self.config_parser.get('GENERAL', 'OUTPUT_DATA_FOLDER'))
        except:
            return 'Output' # default
    
    def perform_body_force_reconstruction(self):
        try:
            val = str(self.config_parser.get('BODY FORCE', 'PERFORM_BODY_FORCE_RECONSTRUCTION')).lower()
            if val == 'yes' or val == 'true':
                return True
            else:
                return False
        except:
            return False # default
    
    def get_circumferential_average_type(self):
        choices = ['raw', 'density', 'axial_momentum']
        
        try:
            res =  str(self.config_parser.get('BODY FORCE', 'CIRCUMFERENTIAL_AVERAGE_TYPE')).lower()
        except:
            res = 'raw' # default
        
        if res not in choices:
            raise ValueError(f'Invalid option CIRCUMFERENTIAL_AVERAGE_TYPE. Possible options are {choices}.')
        
        return res
    
    def get_body_force_extraction_method(self):
        choices = ['marble', 'kiwada']
        
        try:
            res =  str(self.config_parser.get('BODY FORCE', 'EXTRACTION_METHOD')).lower()
        except:
            res = 'marble' # default
        
        if res not in choices:
            raise ValueError(f'Invalid option EXTRACTION_METHOD. Possible options are {choices}.')
        
        return res

    def get_circumferential_average_folder_path(self):
        return str(self.config_parser.get('BODY FORCE', 'CIRCUMFERENTIAL_AVERAGE_FOLDER_PATH'))
    
    
    def cut_body_force_blade_tip_extension(self):
        return float(self.config_parser.get('BODY FORCE', 'CUT_BLADE_TIP_EXTENSION'))
    
    def get_bladed_CFD_solver_type(self):
        return str(self.config_parser.get('BODY FORCE', 'BLADED_CFD_SOLVER')).lower()
    
    def hub_shroud_body_force_extrapolation_span_extent(self):
        return float(self.config_parser.get('BODY FORCE', 'HUB_SHROUD_BODY_FORCE_EXTRAPOLATION_SPAN_EXTENT'))
    
    def get_body_force_blade_name(self):
        return str(self.config_parser.get('BODY FORCE', 'BLADE_NAME'))
    
    def get_extracted_body_forces_path(self):
        value = self.config_parser.get('BODY FORCE', 'EXTRACTED_BODY_FORCES_PATH')
        vals = [str(val.strip(',')) for val in value.split()]
        vals = np.array(vals, dtype=str)
        return vals
    
        


