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
grid_inlet = '20_20'
grid_blade = '20_20'
grid_outlet = '30_20'

inlet_file_path = 'data/meta/nasa_rotor_config_01_inlet_'+grid_inlet+'.pickle'
blade_file_path = 'data/meta/nasa_rotor_config_01_blade_'+grid_blade+'.pickle'
outlet_file_path = 'data/meta/nasa_rotor_config_01_outlet_'+grid_outlet+'.pickle'

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
obj.assemble_fields()
obj.gauss_filtering()
# obj.contour_fields(save_filename='inlet_%s_blade_%s_outlet_%s' % (grid_inlet, grid_blade, grid_outlet))
obj.show_grid(save_filename='inlet_%s_blade_%s_outlet_%s' % (grid_inlet, grid_blade, grid_outlet))
obj.assemble_field_gradients()
obj.gauss_filtering_gradients()
# obj.contour_field_gradients(save_filename='inlet_%s_blade_%s_outlet_%s' % (grid_inlet, grid_blade, grid_outlet))

obj.plot_averaged_fluxes(field='rho', save_filename='flux_rho_%s_%s_%s' %(grid_inlet, grid_blade, grid_outlet))
obj.plot_averaged_fluxes(field='ur', save_filename='flux_ur_%s_%s_%s' %(grid_inlet, grid_blade, grid_outlet))
obj.plot_averaged_fluxes(field='ut', save_filename='flux_ut_%s_%s_%s' %(grid_inlet, grid_blade, grid_outlet))
obj.plot_averaged_fluxes(field='uz', save_filename='flux_uz_%s_%s_%s' %(grid_inlet, grid_blade, grid_outlet))
obj.plot_averaged_fluxes(field='p', save_filename='flux_p_%s_%s_%s' %(grid_inlet, grid_blade, grid_outlet))
obj.plot_averaged_fluxes(field='T', save_filename='flux_T_%s_%s_%s' %(grid_inlet, grid_blade, grid_outlet))
obj.plot_averaged_fluxes(field='s', save_filename='flux_s_%s_%s_%s' %(grid_inlet, grid_blade, grid_outlet))
obj.plot_averaged_fluxes(field='p_tot', save_filename='flux_p_tot_%s_%s_%s' %(grid_inlet, grid_blade, grid_outlet))
obj.plot_averaged_fluxes(field='T_tot', save_filename='flux_T_tot_%s_%s_%s' %(grid_inlet, grid_blade, grid_outlet))
obj.compute_performance()
obj.print_performance()

obj.store_pickle(file_name='inlet_%s_blade_%s_outlet_%s' % (grid_inlet, grid_blade, grid_outlet))
# plt.show()

