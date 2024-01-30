import time
import matplotlib.pyplot as plt
import pickle
import numpy as np

with open('data/pickle/nasar37_p130_20_20_40_20.pickle', 'rb') as file:
    obj = pickle.load(file)

# obj.contour_fields()
# obj.contour_field_gradients()
# obj.contour_bfm_matrices()

blade = obj.group[1]

plt.figure()
plt.scatter(blade.block.z_grid_cg, blade.block.r_grid_cg)
plt.scatter(blade.block.z_grid_centers,  blade.block.r_grid_centers)
# blade.compute_meridional_area()
blade.block.compute_area_elements()
blade.block.area_elements[0, 0].plot_area_element()
blade.block.area_elements[0, 0].line_elements[0].plot_line_element()

plt.show()
