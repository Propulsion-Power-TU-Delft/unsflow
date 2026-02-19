import pickle
import numpy as np
import matplotlib.pyplot as plt
from grid.src.functions import contour_template, compute_meridional_spanwise_coordinates 

avgTypes = ['massFlowAvg']

files  = ['meridionalFields_%s.pik' % avgType for avgType in avgTypes] 

fieldsToPlot = ['Velocity_Radial', 'Velocity_Tangential', 'Velocity_Axial', 'Density (kg/m³)','Pressure (Pa)']

for i,file in enumerate(files):
    with open(file, 'rb') as f:
        fields = pickle.load(f)

    for key in fields.keys():
        if key in fieldsToPlot:
            contour_template(fields['Axial_Coordinate'], fields['Radial_Coordinate'], fields[key], name=key)
   
plt.show()