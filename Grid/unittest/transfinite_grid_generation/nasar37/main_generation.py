import numpy as np
import matplotlib.pyplot as plt
import Grid
from grid.src.functions import transfinite_grid_generation, eriksson_stretching_function_initial, eriksson_stretching_function_final, eriksson_stretching_function_both
import pickle

""" 
test case to debug the elliptic grid generation method. Define a figure delimited by 4 curved
borders, and see how it goes
"""



# parameteric picture
L = 10
R = 1.5

pickle_mesh = 'nasar37_blade/mesh_21_21_26.pickle'
with open(pickle_mesh, 'rb') as f:
    data = pickle.load(f)

Z_mesh = data['z'][:,:,0]
X_mesh = data['x'][:,:,0]
Y_mesh = data['y'][:,:,0]

R_mesh = np.sqrt(X_mesh**2 + Y_mesh**2)

nx = Z_mesh.shape[0]
ny = Z_mesh.shape[1]



"""  left border  """
x = Z_mesh[0,:]
y = R_mesh[0,:]
c_left = np.array((x, y))

"""  bottom border  """
x = Z_mesh[:,0]
y = R_mesh[:,0]
c_bottom = np.array((x, y))

"""  right border  """
x = Z_mesh[-1,:]
y = R_mesh[-1,:]
c_right = np.array((x, y))

"""  top border  """
x = Z_mesh[:,-1]
y = R_mesh[:,-1]
c_top = np.array((x, y))

# GENERATION


X, Y = transfinite_grid_generation(c_left, c_bottom, c_right, c_top, 'internal', 3, 3)

plt.figure()
for i in range(nx):
    plt.plot(X[i, :], Y[i, :], 'k', lw=0.5)
for j in range(ny):
    plt.plot(X[:, j], Y[:, j], 'k', lw=0.5)

plt.show()
