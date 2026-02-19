import configparser
import ast
import numpy as np
import os
from sun.src.general_functions import print_banner_begin, print_banner_end
from utils.styles import total_chars, total_chars_mid



class Config:
    def __init__(self, config_file='input.ini'):
        self.config_parser = configparser.ConfigParser()
        self.config_parser.read(config_file)
        self.print_config()

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
        print_banner_begin('CONFIGURATION FILE')
        for section in self.config_parser.sections():
            print(f"[{section}]")
            for option, value in self.config_parser.items(section):
                print(f"{option} = {value}")
            print()
        print_banner_end('')
        print()

    def create_attributes(self):
        """
        Dynamically create attributes from the configuration.
        """
        for section in self.config_parser.sections():
            for option, value in self.config_parser.items(section):
                setattr(self, option, value)

    def get_blocks_type(self):
        value = self.config_parser.get('SUN MODEL', 'BLOCKS_TYPE')
        return ast.literal_eval(value)

    def get_reference_density(self):
        return float(self.config_parser.get('SUN MODEL', 'RHO_REF'))

    def get_reference_length(self):
        try:
            return float(self.config_parser.get('SUN MODEL', 'X_REF'))
        except:
            return 1.0

    def get_reference_rpm(self):
        return float(self.config_parser.get('SUN MODEL', 'RPM_REF'))

    def get_reference_omega(self):
        return 2 * np.pi * float(self.config_parser.get('SUN MODEL', 'RPM_REF')) / 60

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
        return float(self.config_parser.get('SUN MODEL', 'T_REF'))
    
    def GetBlockType(self):
        try:
            value = str(self.config_parser.get('SUN MODEL', 'BLOCK_TYPE'))
            return value
        except:
            return 'unbladed'
        
    def get_normalize_data(self):
        res = self.config_parser.get('SUN MODEL', 'NORMALIZE_DATA')
        if res.lower() == 'true':
            return True
        elif res.lower() == 'false':
            return False
        else:
            return True

    def get_disable_body_force(self):
        try:
            res = self.config_parser.get('SUN MODEL', 'DISABLE_BODY_FORCE')
            if res.lower() == 'true':
                return True
            else:
                return False
        except:
            return False
        
    def get_result_name(self):
        try:
            res = self.config_parser.get('SUN MODEL', 'RESULT_FILENAME')
            return res
        except:
            return 'results'

    def get_fluid_gamma(self):
        return float(self.config_parser.get('SUN MODEL', 'GAMMA_FLUID'))    

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
    
    def GetInputGridFiles(self):
        values = str(self.config_parser.get('SUN MODEL', 'INPUT_GRID_FILES'))
        values = [str(i.strip(',')) for i in values.split()]
        return values
    
    def getNumberOfBlocks(self):
        inputfiles = self.GetInputGridFiles()
        return len(inputfiles)

    def get_shroud_bc(self):
        return str(self.config_parser.get('SUN MODEL', 'SHROUD_BC'))

    def get_euler_wall_equation(self):
        return str(self.config_parser.get('SUN MODEL', 'EULER_WALL_EQUATION'))

    def get_boundary_interface_gradient_method(self):
        return str(self.config_parser.get('SUN MODEL', 'BOUNDARY_INTERFACE_GRADIENT_METHOD'))
    