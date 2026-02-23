import numpy as np
import matplotlib.pyplot as plt
from grid.src.functions import elliptic_grid_generation

""" 
test case to debug the elliptic grid generation method. Define a figure delimited by 4 curved
borders, and see how it goes
"""

nx = 25
ny = 25

# parameteric picture
L = 10
R = 1.5

"""  left border  """
x = np.zeros(ny)
y = np.linspace(0, L, ny)
c_left = np.array((x, y))

"""  bottom border  """
theta = np.linspace(0, 2*np.pi, nx)
x = np.linspace(0, L, nx)
y = 0.3*R*np.sin(theta)
c_bottom = np.array((x, y))

"""  right border  """
x = np.zeros(ny)+L
y = np.linspace(0, L+R, ny)
c_right = np.array((x, y))

"""  top border  """
theta = np.linspace(0, np.pi/2, nx)
x = np.linspace(0, L, nx)
y = L+R*np.sin(theta) + 0.5*np.sin(5*theta)
c_top = np.array((x, y))


orthogonality = True
x_stretching = 'sigmoid'
y_stretching = 'sigmoid'


X, Y = elliptic_grid_generation(c_left, c_bottom, c_right, c_top, orthogonality=orthogonality, sigmoid_coeff_x=9,
                                x_stretching=x_stretching, y_stretching=y_stretching, tol=1e-3, sigmoid_coeff_y=9,
                                save_filename='orth_%s__xstr_%s__ystr_%s' %(orthogonality, x_stretching, y_stretching),
                                guardian=False)

