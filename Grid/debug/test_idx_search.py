import numpy as np
from Grid.src.functions import *

x = np.linspace(1, 100, 1000)
y = x**2

x_try = 1.74
y_try = None

x_try, y_try = find_point_on_border(x_try, y_try, x, y)
print('try')