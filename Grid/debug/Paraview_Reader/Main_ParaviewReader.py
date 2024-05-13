import numpy as np
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy
import matplotlib.pyplot as plt

volumedata = np.loadtxt('cell_volumes.csv', skiprows=1, delimiter=',')


# Specify the file path to your VTU file
file_path = "NasaLSCC-BFM.vtu"

# Create a reader object
reader = vtk.vtkXMLUnstructuredGridReader()
reader.SetFileName(file_path)
reader.Update()

# Get the output data
data = reader.GetOutput()

# Get point coordinates
points = data.GetPoints()
point_coordinates = vtk_to_numpy(points.GetData())
dict = {'x': point_coordinates[:, 0], 'y': point_coordinates[:, 1], 'z': point_coordinates[:, 2]}

# Access point data (variables associated with points)
point_data = data.GetPointData()

# Get the number of point data arrays
num_arrays = point_data.GetNumberOfArrays()

# Iterate through each point data array
for i in range(num_arrays):
    array_name = point_data.GetArrayName(i)
    array = point_data.GetArray(array_name)
    dict[array_name] = vtk_to_numpy(array)



plt.show()


