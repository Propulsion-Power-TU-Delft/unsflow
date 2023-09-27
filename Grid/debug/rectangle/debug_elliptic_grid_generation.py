import numpy as np
import matplotlib.pyplot as plt
import Grid
from Grid.src.functions import elliptic_grid_generation

""" 
test case to debug the elliptic grid generation method. Define a figure delimited by 4 curved
borders, and see how it goes
"""

nx = 30
ny = 30

# parameteric picture
L = 1
H = 1

"""  left border  """
c_left = np.array((np.zeros(ny),
                   np.linspace(0,H,ny)))

"""  bottom border  """
c_bottom = np.array((np.linspace(0, L, nx),
                     np.zeros(nx)))

"""  right border  """
c_right = np.array((np.zeros(ny)+L,
                   np.linspace(0,H,ny)))

"""  top border  """
c_top = np.array((np.linspace(0, L, nx),
                     np.zeros(nx)+H))



X, Y = elliptic_grid_generation(c_left, c_bottom, c_right, c_top, orthogonality=True,
                                x_stretching=True, y_stretching=True, tol=1e-3,
                                save_filename='y_stretch_quadratic')
