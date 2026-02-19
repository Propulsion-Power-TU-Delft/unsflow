import numpy as np
import matplotlib.pyplot as plt
from grid.src.functions import *

xData = np.linspace(0, 1, 100)
yData = np.linspace(0, 1, 100)
xData, yData = np.meshgrid(xData, yData, indexing='ij')
zData = np.sin(2 * np.pi * xData) * np.cos(2 * np.pi * yData)

xEval = np.linspace(-1e-1, 1+1e-1, 50)
yEval = np.linspace(-1e-1, 1+1e-1, 50)
xEval, yEval = np.meshgrid(xEval, yEval, indexing='ij')

zEval = griddata_interpolation_with_nearest_filler(xData, yData, zData, xEval, yEval)
zEval2 = griddata_interpolation_with_linear_extrapolation(xData, yData, zData, xEval, yEval)


contour_template(xData, yData, zData, 'z', save_filename='reference_data.pdf')
contour_template(xEval, yEval, zEval, 'z', save_filename='linear_interpolation_nearest_extrapolation.pdf')
contour_template(xEval, yEval, zEval2, 'z', save_filename='linear_interpolation_linear_stream-span_extrapolation.pdf')

plt.show()