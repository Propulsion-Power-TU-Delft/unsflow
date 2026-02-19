import numpy as np
import matplotlib.pyplot as plt
import Grid
from grid.src.functions import elliptic_grid_generation

""" 
test case to debug the elliptic grid generation method. Define a figure delimited by 4 curved
borders, and see how it goes
"""

nx = 30
ny = 20

# parameteric picture
L = 2
H = 1

# left border
c_left = np.array((np.zeros(ny),
                   np.linspace(0, H, ny)))

# bottom border
c_bottom = np.array((np.linspace(0, L, nx),
                     np.zeros(nx)))

# right border
c_right = np.array((np.zeros(ny) + L,
                    np.linspace(0, H, ny)))

# top border
c_top = np.array((np.linspace(0, L, nx),
                  np.zeros(nx) + H))


orthogonality = True
x_stretching = 'sigmoid'
y_stretching = 'sigmoid'
X, Y = elliptic_grid_generation(c_left, c_bottom, c_right, c_top, orthogonality=orthogonality, sigmoid_coeff_x=8,
                                sigmoid_coeff_y=8, x_stretching=x_stretching, y_stretching=y_stretching, tol=1e-3,
                                save_filename='orth_%s__xstr_%s__ystr_%s' %(orthogonality, x_stretching, y_stretching))
