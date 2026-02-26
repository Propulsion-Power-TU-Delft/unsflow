import configparser
import ast
import numpy as np
import os
from unsflow.utils.formatting import print_banner
from unsflow.utils.formatting import total_chars, total_chars_mid



class Config:
    def __init__(self, config_file='input.ini'):
        self.config_parser = configparser.ConfigParser()
        self.config_parser.read(config_file)
        
        print_banner('CONFIGURATION FILE')
        print(f"{'Configuration file: ':<{total_chars_mid}}{config_file:>{total_chars_mid}}")
        print_banner('')
        

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

    
    
    

    def get_omega(self):
        rpm = float(self.config_parser.get('COMPRESSION SYSTEM', 'RPM'))
        return 2*np.pi*rpm/60
    
    def get_reference_radius(self):
        return float(self.config_parser.get('COMPRESSION SYSTEM', 'REFERENCE_RADIUS'))
    
    def get_sound_speed(self):
        return float(self.config_parser.get('COMPRESSION SYSTEM', 'SOUND_SPEED'))
    
    def get_plenum_volume(self):
        return float(self.config_parser.get('COMPRESSION SYSTEM', 'PLENUM_VOLUME'))
    
    def get_inlet_duct_diameter(self):
        return float(self.config_parser.get('COMPRESSION SYSTEM', 'INLET_DUCT_DIAMETER'))
    
    def get_inlet_duct_area(self):
        d = self.get_inlet_duct_diameter()
        return np.pi*d**2/4
    
    def get_inlet_duct_length(self):
        return float(self.config_parser.get('COMPRESSION SYSTEM', 'INLET_DUCT_LENGTH'))
    
    def get_throttle_duct_diameter(self):
        return float(self.config_parser.get('COMPRESSION SYSTEM', 'THROTTLE_DUCT_DIAMETER'))
    
    def get_throttle_duct_area(self):
        d = self.get_throttle_duct_diameter()
        return np.pi*d**2/4
    
    def get_throttle_duct_length(self):
        return float(self.config_parser.get('COMPRESSION SYSTEM', 'THROTTLE_DUCT_LENGTH'))
    
    def get_max_time(self):
        return float(self.config_parser.get('SIMULATION', 'MAX_TIME'))
    
    def get_valve_coefficient(self):
        return float(self.config_parser.get('COMPRESSION SYSTEM', 'K_VALVE'))


    def get_flow_coeffs(self):
        values = self.config_parser.get('COMPRESSOR', 'FLOW_COEFFS')
        values = [float(val.strip(',')) for val in values.split()]
        return values
    
    def get_work_coeffs(self):
        values = self.config_parser.get('COMPRESSOR', 'WORK_COEFFS')
        values = [float(val.strip(',')) for val in values.split()]
        return values
    
    def get_reference_speed(self):
        omega = self.get_omega()
        r_ref = self.get_reference_radius()
        return omega*r_ref

    def get_unstalled_characteristic_params(self):
        H =  float(self.config_parser.get('COMPRESSOR', 'H_CHAR'))
        W =  float(self.config_parser.get('COMPRESSOR', 'W_CHAR'))
        psi_c_0 =  float(self.config_parser.get('COMPRESSOR', 'PSI_C_0_CHAR'))
        return H, W, psi_c_0
    
    def get_moore_greitzer_time_lag(self):
        return  float(self.config_parser.get('MOORE GREITZER', 'A_LAG'))
    
    def get_moore_greitzer_m_param(self):
        return  float(self.config_parser.get('MOORE GREITZER', 'M_PARAM'))
