import numpy as np
import matplotlib.pyplot as plt
from Grid.src.functions import *
import pickle

with open('../data/NASA_LSCC_data.pik', 'rb') as file:
    data = pickle.load(file)

x = data.multiBlockGrid.z_grid_points
y = data.multiBlockGrid.r_grid_points

xNew = enlarge_domain_array(x)
yNew = enlarge_domain_array(y)

plt.figure()
plt.scatter(x, y, marker='x', label='original points')
plt.scatter(xNew, yNew, marker='.', label='enlarged points')
plt.legend()
plt.gca().set_aspect('equal')
plt.show()

