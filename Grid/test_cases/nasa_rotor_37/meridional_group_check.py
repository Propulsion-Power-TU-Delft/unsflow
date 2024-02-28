import time
import matplotlib.pyplot as plt
import pickle
import numpy as np

with open('data/pickle/nasar37_p130_20_20_40_20.pickle', 'rb') as file:
    obj = pickle.load(file)

obj.contour_fields()
obj.contour_field_gradients()
obj.contour_bfm_matrices()
obj.plot_averaged_fluxes(field='rho')
obj.plot_averaged_fluxes(field='ur')
obj.plot_averaged_fluxes(field='ut')
obj.plot_averaged_fluxes(field='uz')
obj.plot_averaged_fluxes(field='p')
obj.plot_averaged_fluxes(field='T')
obj.plot_averaged_fluxes(field='s')
obj.plot_averaged_fluxes(field='p_tot')
obj.plot_averaged_fluxes(field='T_tot')
obj.plot_averaged_fluxes(field='M')
obj.plot_averaged_fluxes(field='M_rel')
obj.compute_performance()
obj.print_performance()
obj.print_memory_info()
plt.show()
