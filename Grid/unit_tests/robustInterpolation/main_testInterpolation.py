import numpy as np
import matplotlib.pyplot as plt
from Grid.src.functions import *
import pickle

#### test 2d function to compare results
def functionTest(x,y):
    return np.sin(4*x)*np.cos(3*y**2)/(x**2+y**2)

#### Test Data, basic data onto which things are interpolated
with open('NASA_LSCC_data.pik', 'rb') as file:
    data = pickle.load(file)
xData = data.multiBlockGrid.z_grid_points
yData = data.multiBlockGrid.r_grid_points
zData = functionTest(xData,yData)

### Evaluation Data
with open('NASA_LSCC_eval.pik', 'rb') as file:
    data = pickle.load(file)
xEval = data.multiBlockGrid.z_grid_points
yEval = data.multiBlockGrid.r_grid_points
zEval = robust_griddata_interpolation_with_linear_filler(xData, yData, zData, xEval, yEval)

#### Plotting
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
contour1 = ax[0].contourf(xData, yData, zData, levels=20, cmap=color_map)
fig.colorbar(contour1, ax=ax[0])
ax[0].set_title(r'Data')
contour2 = ax[1].contourf(xEval, yEval, zEval, levels=20, cmap=color_map)
fig.colorbar(contour2, ax=ax[1])
ax[1].set_title(r'Interpolation')

for axx in ax:
    axx.set_aspect('equal')

plt.show()

