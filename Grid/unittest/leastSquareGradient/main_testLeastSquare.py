import numpy as np
import matplotlib.pyplot as plt
from grid.src.functions import *
import pickle

#### test 2d function to compare results
# def functionTest(x,y):
#     f = np.sin(4*x)*np.cos(3*y**2)
#     dfdx = 4*np.cos(4*x)*np.cos(3*y**2)
#     dfdy = -6*y*np.sin(4*x)*np.sin(3*y**2)
#     return f, dfdx, dfdy

def functionTest(x, y):
    f = np.exp(-x**2 - y**2) * np.sin(2*np.pi*x) * np.cos(2*np.pi*y)
    
    # Partial derivatives (analytical)
    dfdx = np.exp(-x**2 - y**2) * (
        2*np.pi*np.cos(2*np.pi*x)*np.cos(2*np.pi*y) - 2*x*np.sin(2*np.pi*x)*np.cos(2*np.pi*y)
    )
    
    dfdy = np.exp(-x**2 - y**2) * (
        -2*np.pi*np.sin(2*np.pi*x)*np.sin(2*np.pi*y) - 2*y*np.sin(2*np.pi*x)*np.cos(2*np.pi*y)
    )
    
    return f, dfdx, dfdy

#### Test Data, basic data onto which things are interpolated
with open('../data/NASA_LSCC_data.pik', 'rb') as file:
    data = pickle.load(file)
xData = data.multiBlockgrid.z_grid_points
yData = data.multiBlockgrid.r_grid_points
zData, dzdxData, dzdyData = functionTest(xData,yData)

### Compute the gradient
dzdx, dzdy = compute_gradient_least_square(xData, yData, zData, enlargeDomain=True)


#### Plotting

# Compute min/max from analytical derivatives
dzdx_min, dzdx_max = np.min(dzdxData), np.max(dzdxData)
dzdy_min, dzdy_max = np.min(dzdyData), np.max(dzdyData)

# Define common levels
nLevels = 20
dzdx_levels = np.linspace(dzdx_min, dzdx_max, nLevels)
dzdy_levels = np.linspace(dzdy_min, dzdy_max, nLevels)

fig, ax = plt.subplots(2, 2, figsize=(12, 10))

# Analytical
contour1 = ax[0,0].contourf(xData, yData, dzdxData, levels=dzdx_levels, cmap=color_map)
fig.colorbar(contour1, ax=ax[0,0])
ax[0,0].set_title(r'$\partial_x f$ analytical')

contour2 = ax[1,0].contourf(xData, yData, dzdyData, levels=dzdy_levels, cmap=color_map)
fig.colorbar(contour2, ax=ax[1,0])
ax[1,0].set_title(r'$\partial_y f$ analytical')

# Numerical (use same levels!)
contour3 = ax[0,1].contourf(xData, yData, dzdx, levels=dzdx_levels, cmap=color_map)
fig.colorbar(contour3, ax=ax[0,1])
ax[0,1].set_title(r'$\partial_x f$ numerical')

contour4 = ax[1,1].contourf(xData, yData, dzdy, levels=dzdy_levels, cmap=color_map)
fig.colorbar(contour4, ax=ax[1,1])
ax[1,1].set_title(r'$\partial_y f$ numerical')

plt.show()

