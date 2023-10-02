import numpy as np
import matplotlib.pyplot as plt
import Grid
from Grid.src.functions import elliptic_grid_generation

""" 
test case to debug the elliptic grid generation method. Define a figure delimited by 4 curved
borders, and see how it goes
"""

nx = 30
ny = 20

# parameteric picture
L = 2
H = 2

# left border
x = np.linspace(0, L, nx)
y = np.linspace(0, -H, nx)
c_left = np.array((x,
                   y))

# bottom border
x = L+np.linspace(0, L, ny)
y = -H + np.linspace(0, H, ny)
c_bottom = np.array((x,
                   y))

# right border
x = 2*L+np.linspace(0, -L, nx)
y = np.linspace(0, H, nx)
c_right = np.array((x,
                   y))

# top border
x = np.linspace(0, L, ny)
y = np.linspace(0, H, ny)
c_top = np.array((x,
                   y))


orthogonality = True
x_stretching = 'sigmoid'
y_stretching = 'sigmoid'
X, Y = elliptic_grid_generation(c_left, c_bottom, c_right, c_top, orthogonality=orthogonality, sigmoid_coeff_x=8,
                                sigmoid_coeff_y=8, x_stretching=x_stretching, y_stretching=y_stretching, tol=1e-3,
                                save_filename='orth_%s__xstr_%s__ystr_%s' %(orthogonality, x_stretching, y_stretching))
