import numpy as np
import matplotlib.pyplot as plt
from Grid.src.functions import *
import pickle


#### Test Data, basic data onto which things are interpolated
with open('../data/NASA_LSCC_data.pik', 'rb') as file:
    data = pickle.load(file)

xgrid = data.blocks[1].z_grid_points
ygrid = data.blocks[1].r_grid_points

xgridDual, ygridDual = compute_dual_grid(xgrid, ygrid)

plt.figure()
plt.plot(xgrid, ygrid, 'bo')
plt.plot(xgridDual, ygridDual, 'rx')



plt.show()

