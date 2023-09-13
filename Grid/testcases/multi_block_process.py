#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 11:41:53 2023
@author: F. Neri, TU Delft

"""
import time
import matplotlib.pyplot as plt
import sys

sys.path.append('../../Grid')
import Grid
import pickle
import numpy as np

# Specify the path to your pickle file
inlet_file_path = 'data/meta/nasa_rotor_config_04_inlet_10_10.pickle'
blade_file_path = 'data/meta/nasa_rotor_config_04_30_10.pickle'
outlet_file_path = 'data/meta/nasa_rotor_config_04_outlet_15_10.pickle'

with open(inlet_file_path, 'rb') as file:
    inlet = pickle.load(file)

with open(blade_file_path, 'rb') as file:
    blade = pickle.load(file)

with open(outlet_file_path, 'rb') as file:
    outlet = pickle.load(file)

obj = Grid.src.meridional_process_group.MeridionalProcessGroup()
obj.add_to_group(inlet)
obj.add_to_group(blade)
obj.add_to_group(outlet)
obj.contour()
plt.show()

