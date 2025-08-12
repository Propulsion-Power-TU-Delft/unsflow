# trace generated using paraview version 5.10.1
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 10

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'XML Unstructured Grid Reader'
flowvtu = XMLUnstructuredGridReader(registrationName='flow.vtu', FileName=['/Users/fneri/Documents/PhD/unsflow/Grid/debug/BFM_InputFileGeneration/NasaLSCC/RunTest_Su2_NoSlipWalls/flow.vtu'])
flowvtu.PointArrayStatus = ['Density', 'Momentum', 'Energy', 'Nu_Tilde', 'Pressure', 'Temperature', 'Mach', 'Pressure_Coefficient', 'Laminar_Viscosity', 'Skin_Friction_Coefficient', 'Heat_Flux', 'Y_Plus', 'Residual_Density', 'Residual_Momentum', 'Residual_Energy', 'Residual_Nu_Tilde', 'Eddy_Viscosity', 'n', 'b', 'blockage_gradient', 'body_force_factor', 'rotation_factor', 'blade_count', 'W', 'BF']

# Properties modified on flowvtu
flowvtu.TimeArray = 'None'

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')

# show data in view
flowvtuDisplay = Show(flowvtu, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
flowvtuDisplay.Representation = 'Surface'
flowvtuDisplay.ColorArrayName = [None, '']
flowvtuDisplay.SelectTCoordArray = 'None'
flowvtuDisplay.SelectNormalArray = 'None'
flowvtuDisplay.SelectTangentArray = 'None'
flowvtuDisplay.OSPRayScaleArray = 'BF'
flowvtuDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
flowvtuDisplay.SelectOrientationVectors = 'BF'
flowvtuDisplay.ScaleFactor = 0.01486000046133995
flowvtuDisplay.SelectScaleArray = 'BF'
flowvtuDisplay.GlyphType = 'Arrow'
flowvtuDisplay.GlyphTableIndexArray = 'BF'
flowvtuDisplay.GaussianRadius = 0.0007430000230669976
flowvtuDisplay.SetScaleArray = ['POINTS', 'BF']
flowvtuDisplay.ScaleTransferFunction = 'PiecewiseFunction'
flowvtuDisplay.OpacityArray = ['POINTS', 'BF']
flowvtuDisplay.OpacityTransferFunction = 'PiecewiseFunction'
flowvtuDisplay.DataAxesGrid = 'GridAxesRepresentation'
flowvtuDisplay.PolarAxes = 'PolarAxesRepresentation'
flowvtuDisplay.ScalarOpacityUnitDistance = 0.006739942549997341
flowvtuDisplay.OpacityArrayName = ['POINTS', 'BF']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
flowvtuDisplay.ScaleTransferFunction.Points = [-1368114.75, 0.0, 0.5, 0.0, 15927004.0, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
flowvtuDisplay.OpacityTransferFunction.Points = [-1368114.75, 0.0, 0.5, 0.0, 15927004.0, 1.0, 0.5, 0.0]

# reset view to fit data
renderView1.ResetCamera(False)

# update the view to ensure updated data information
renderView1.Update()

# create a new 'Calculator'
calculator1 = Calculator(registrationName='Calculator1', Input=flowvtu)
calculator1.Function = ''

# Properties modified on calculator1
calculator1.ResultArrayName = 'Velocity'
calculator1.Function = 'Momentum/Density'

# show data in view
calculator1Display = Show(calculator1, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
calculator1Display.Representation = 'Surface'
calculator1Display.ColorArrayName = [None, '']
calculator1Display.SelectTCoordArray = 'None'
calculator1Display.SelectNormalArray = 'None'
calculator1Display.SelectTangentArray = 'None'
calculator1Display.OSPRayScaleArray = 'BF'
calculator1Display.OSPRayScaleFunction = 'PiecewiseFunction'
calculator1Display.SelectOrientationVectors = 'Velocity'
calculator1Display.ScaleFactor = 0.01486000046133995
calculator1Display.SelectScaleArray = 'BF'
calculator1Display.GlyphType = 'Arrow'
calculator1Display.GlyphTableIndexArray = 'BF'
calculator1Display.GaussianRadius = 0.0007430000230669976
calculator1Display.SetScaleArray = ['POINTS', 'BF']
calculator1Display.ScaleTransferFunction = 'PiecewiseFunction'
calculator1Display.OpacityArray = ['POINTS', 'BF']
calculator1Display.OpacityTransferFunction = 'PiecewiseFunction'
calculator1Display.DataAxesGrid = 'GridAxesRepresentation'
calculator1Display.PolarAxes = 'PolarAxesRepresentation'
calculator1Display.ScalarOpacityUnitDistance = 0.006739942549997341
calculator1Display.OpacityArrayName = ['POINTS', 'BF']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
calculator1Display.ScaleTransferFunction.Points = [-1368114.75, 0.0, 0.5, 0.0, 15927004.0, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
calculator1Display.OpacityTransferFunction.Points = [-1368114.75, 0.0, 0.5, 0.0, 15927004.0, 1.0, 0.5, 0.0]

# hide data in view
Hide(flowvtu, renderView1)

# update the view to ensure updated data information
renderView1.Update()

# set scalar coloring
ColorBy(calculator1Display, ('POINTS', 'Velocity', 'Magnitude'))

# rescale color and/or opacity maps used to include current data range
calculator1Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
calculator1Display.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'Velocity'
velocityLUT = GetColorTransferFunction('Velocity')

# get opacity transfer function/opacity map for 'Velocity'
velocityPWF = GetOpacityTransferFunction('Velocity')

# set scalar coloring
ColorBy(calculator1Display, ('POINTS', 'Velocity', 'X'))

# rescale color and/or opacity maps used to exactly fit the current data range
calculator1Display.RescaleTransferFunctionToDataRange(False, False)

# Update a scalar bar component title.
UpdateScalarBarsComponentTitle(velocityLUT, calculator1Display)

# set scalar coloring
ColorBy(calculator1Display, ('POINTS', 'Velocity', 'Y'))

# rescale color and/or opacity maps used to exactly fit the current data range
calculator1Display.RescaleTransferFunctionToDataRange(False, False)

# Update a scalar bar component title.
UpdateScalarBarsComponentTitle(velocityLUT, calculator1Display)

# set scalar coloring
ColorBy(calculator1Display, ('POINTS', 'Velocity', 'Z'))

# rescale color and/or opacity maps used to exactly fit the current data range
calculator1Display.RescaleTransferFunctionToDataRange(False, False)

# Update a scalar bar component title.
UpdateScalarBarsComponentTitle(velocityLUT, calculator1Display)

# set scalar coloring
ColorBy(calculator1Display, ('POINTS', 'Velocity', 'Magnitude'))

# rescale color and/or opacity maps used to exactly fit the current data range
calculator1Display.RescaleTransferFunctionToDataRange(False, False)

# Update a scalar bar component title.
UpdateScalarBarsComponentTitle(velocityLUT, calculator1Display)

# rename source object
RenameSource('Velocity', calculator1)

# create a new 'Calculator'
calculator1_1 = Calculator(registrationName='Calculator1', Input=calculator1)
calculator1_1.Function = ''

# Properties modified on calculator1_1
calculator1_1.ResultArrayName = 'Radius'
calculator1_1.Function = 'sqrt(coordsX^2+coordsY^2)'

# show data in view
calculator1_1Display = Show(calculator1_1, renderView1, 'UnstructuredGridRepresentation')

# get color transfer function/color map for 'Radius'
radiusLUT = GetColorTransferFunction('Radius')

# get opacity transfer function/opacity map for 'Radius'
radiusPWF = GetOpacityTransferFunction('Radius')

# trace defaults for the display properties.
calculator1_1Display.Representation = 'Surface'
calculator1_1Display.ColorArrayName = ['POINTS', 'Radius']
calculator1_1Display.LookupTable = radiusLUT
calculator1_1Display.SelectTCoordArray = 'None'
calculator1_1Display.SelectNormalArray = 'None'
calculator1_1Display.SelectTangentArray = 'None'
calculator1_1Display.OSPRayScaleArray = 'Radius'
calculator1_1Display.OSPRayScaleFunction = 'PiecewiseFunction'
calculator1_1Display.SelectOrientationVectors = 'Velocity'
calculator1_1Display.ScaleFactor = 0.01486000046133995
calculator1_1Display.SelectScaleArray = 'Radius'
calculator1_1Display.GlyphType = 'Arrow'
calculator1_1Display.GlyphTableIndexArray = 'Radius'
calculator1_1Display.GaussianRadius = 0.0007430000230669976
calculator1_1Display.SetScaleArray = ['POINTS', 'Radius']
calculator1_1Display.ScaleTransferFunction = 'PiecewiseFunction'
calculator1_1Display.OpacityArray = ['POINTS', 'Radius']
calculator1_1Display.OpacityTransferFunction = 'PiecewiseFunction'
calculator1_1Display.DataAxesGrid = 'GridAxesRepresentation'
calculator1_1Display.PolarAxes = 'PolarAxesRepresentation'
calculator1_1Display.ScalarOpacityFunction = radiusPWF
calculator1_1Display.ScalarOpacityUnitDistance = 0.006739942549997341
calculator1_1Display.OpacityArrayName = ['POINTS', 'Radius']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
calculator1_1Display.ScaleTransferFunction.Points = [0.17522644167100926, 0.0, 0.5, 0.0, 0.2566635884695172, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
calculator1_1Display.OpacityTransferFunction.Points = [0.17522644167100926, 0.0, 0.5, 0.0, 0.2566635884695172, 1.0, 0.5, 0.0]

# hide data in view
Hide(calculator1, renderView1)

# show color bar/color legend
calculator1_1Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# rename source object
RenameSource('Radius', calculator1_1)

# create a new 'Calculator'
calculator1_2 = Calculator(registrationName='Calculator1', Input=calculator1_1)
calculator1_2.Function = ''

# Properties modified on calculator1_2
calculator1_2.ResultArrayName = 'Theta'
calculator1_2.Function = 'atan2(coordsX, coordsY)'

# show data in view
calculator1_2Display = Show(calculator1_2, renderView1, 'UnstructuredGridRepresentation')

# get color transfer function/color map for 'Theta'
thetaLUT = GetColorTransferFunction('Theta')

# get opacity transfer function/opacity map for 'Theta'
thetaPWF = GetOpacityTransferFunction('Theta')

# trace defaults for the display properties.
calculator1_2Display.Representation = 'Surface'
calculator1_2Display.ColorArrayName = ['POINTS', 'Theta']
calculator1_2Display.LookupTable = thetaLUT
calculator1_2Display.SelectTCoordArray = 'None'
calculator1_2Display.SelectNormalArray = 'None'
calculator1_2Display.SelectTangentArray = 'None'
calculator1_2Display.OSPRayScaleArray = 'Theta'
calculator1_2Display.OSPRayScaleFunction = 'PiecewiseFunction'
calculator1_2Display.SelectOrientationVectors = 'Velocity'
calculator1_2Display.ScaleFactor = 0.01486000046133995
calculator1_2Display.SelectScaleArray = 'Theta'
calculator1_2Display.GlyphType = 'Arrow'
calculator1_2Display.GlyphTableIndexArray = 'Theta'
calculator1_2Display.GaussianRadius = 0.0007430000230669976
calculator1_2Display.SetScaleArray = ['POINTS', 'Theta']
calculator1_2Display.ScaleTransferFunction = 'PiecewiseFunction'
calculator1_2Display.OpacityArray = ['POINTS', 'Theta']
calculator1_2Display.OpacityTransferFunction = 'PiecewiseFunction'
calculator1_2Display.DataAxesGrid = 'GridAxesRepresentation'
calculator1_2Display.PolarAxes = 'PolarAxesRepresentation'
calculator1_2Display.ScalarOpacityFunction = thetaPWF
calculator1_2Display.ScalarOpacityUnitDistance = 0.006739942549997341
calculator1_2Display.OpacityArrayName = ['POINTS', 'Theta']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
calculator1_2Display.ScaleTransferFunction.Points = [1.5533430324337338, 0.0, 0.5, 0.0, 1.5707963267948966, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
calculator1_2Display.OpacityTransferFunction.Points = [1.5533430324337338, 0.0, 0.5, 0.0, 1.5707963267948966, 1.0, 0.5, 0.0]

# hide data in view
Hide(calculator1_1, renderView1)

# show color bar/color legend
calculator1_2Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# Properties modified on calculator1_2
calculator1_2.Function = 'atan(coordsY/ coordsY)'

# update the view to ensure updated data information
renderView1.Update()

# Rescale transfer function
thetaLUT.RescaleTransferFunction(0.0, 1.5707963267948966)

# Rescale transfer function
thetaPWF.RescaleTransferFunction(0.0, 1.5707963267948966)

# Rescale transfer function
thetaLUT.RescaleTransferFunction(0.0, 0.7853981633974483)

# Rescale transfer function
thetaPWF.RescaleTransferFunction(0.0, 0.7853981633974483)

# set active source
SetActiveSource(flowvtu)

# set active source
SetActiveSource(calculator1_2)

# Properties modified on thetaLUT
thetaLUT.NumberOfTableValues = 10

# change representation type
calculator1_2Display.SetRepresentationType('Surface With Edges')

# Properties modified on calculator1_2
calculator1_2.Function = 'atan2(coordsY, coordsX)'

# update the view to ensure updated data information
renderView1.Update()

# Rescale transfer function
thetaLUT.RescaleTransferFunction(0.0, 0.01745329436116288)

# Rescale transfer function
thetaPWF.RescaleTransferFunction(0.0, 0.01745329436116288)

# change representation type
calculator1_2Display.SetRepresentationType('Surface')

# Properties modified on calculator1_2
calculator1_2.Function = 'atan2(coordsY, coordsX)*180/pi'

# update the view to ensure updated data information
renderView1.Update()

# Properties modified on calculator1_2
calculator1_2.Function = 'atan2(coordsY, coordsX)*180/3.14'

# update the view to ensure updated data information
renderView1.Update()

# Rescale transfer function
thetaLUT.RescaleTransferFunction(0.0, 1.0005073200666617)

# Rescale transfer function
thetaPWF.RescaleTransferFunction(0.0, 1.0005073200666617)

# Properties modified on calculator1_2
calculator1_2.Function = 'atan2(coordsY, coordsX)'

# update the view to ensure updated data information
renderView1.Update()

# Rescale transfer function
thetaLUT.RescaleTransferFunction(0.0, 0.01745329436116288)

# Rescale transfer function
thetaPWF.RescaleTransferFunction(0.0, 0.01745329436116288)

# rename source object
RenameSource('Theta', calculator1_2)

# create a new 'Calculator'
calculator1_3 = Calculator(registrationName='Calculator1', Input=calculator1_2)
calculator1_3.Function = ''

# Properties modified on calculator1_3
calculator1_3.ResultArrayName = 'ur'
calculator1_3.Function = 'Velocity[0]'

# show data in view
calculator1_3Display = Show(calculator1_3, renderView1, 'UnstructuredGridRepresentation')

# get color transfer function/color map for 'ur'
urLUT = GetColorTransferFunction('ur')

# get opacity transfer function/opacity map for 'ur'
urPWF = GetOpacityTransferFunction('ur')

# trace defaults for the display properties.
calculator1_3Display.Representation = 'Surface'
calculator1_3Display.ColorArrayName = ['POINTS', 'ur']
calculator1_3Display.LookupTable = urLUT
calculator1_3Display.SelectTCoordArray = 'None'
calculator1_3Display.SelectNormalArray = 'None'
calculator1_3Display.SelectTangentArray = 'None'
calculator1_3Display.OSPRayScaleArray = 'ur'
calculator1_3Display.OSPRayScaleFunction = 'PiecewiseFunction'
calculator1_3Display.SelectOrientationVectors = 'Velocity'
calculator1_3Display.ScaleFactor = 0.01486000046133995
calculator1_3Display.SelectScaleArray = 'ur'
calculator1_3Display.GlyphType = 'Arrow'
calculator1_3Display.GlyphTableIndexArray = 'ur'
calculator1_3Display.GaussianRadius = 0.0007430000230669976
calculator1_3Display.SetScaleArray = ['POINTS', 'ur']
calculator1_3Display.ScaleTransferFunction = 'PiecewiseFunction'
calculator1_3Display.OpacityArray = ['POINTS', 'ur']
calculator1_3Display.OpacityTransferFunction = 'PiecewiseFunction'
calculator1_3Display.DataAxesGrid = 'GridAxesRepresentation'
calculator1_3Display.PolarAxes = 'PolarAxesRepresentation'
calculator1_3Display.ScalarOpacityFunction = urPWF
calculator1_3Display.ScalarOpacityUnitDistance = 0.006739942549997341
calculator1_3Display.OpacityArrayName = ['POINTS', 'ur']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
calculator1_3Display.ScaleTransferFunction.Points = [-61.02663526532329, 0.0, 0.5, 0.0, 67.51661267353121, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
calculator1_3Display.OpacityTransferFunction.Points = [-61.02663526532329, 0.0, 0.5, 0.0, 67.51661267353121, 1.0, 0.5, 0.0]

# hide data in view
Hide(calculator1_2, renderView1)

# show color bar/color legend
calculator1_3Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# Rescale transfer function
urLUT.RescaleTransferFunction(-58.59379005503733, 67.5166126735312)

# Rescale transfer function
urPWF.RescaleTransferFunction(-58.59379005503733, 67.5166126735312)



# Properties modified on calculator1_3
calculator1_3.Function = 'Velocity[0]*cos(Theta)+Velocity[1]*sin(Theta)'

# update the view to ensure updated data information
renderView1.Update()

# Rescale transfer function
urLUT.RescaleTransferFunction(-61.02681730572883, 67.5166126735312)

# Rescale transfer function
urPWF.RescaleTransferFunction(-61.02681730572883, 67.5166126735312)

# rename source object
RenameSource('Ur', calculator1_3)

# Properties modified on calculator1_3
calculator1_3.ResultArrayName = 'Ur'

# update the view to ensure updated data information
renderView1.Update()

# create a new 'Calculator'
calculator1_4 = Calculator(registrationName='Calculator1', Input=calculator1_3)
calculator1_4.Function = ''

# set active source
SetActiveSource(calculator1_3)

# set active source
SetActiveSource(calculator1_4)

# Properties modified on calculator1_4
calculator1_4.ResultArrayName = 'Ut'
calculator1_4.Function = '-Velocity[0]*sin(Theta)+Velocity[1]*cos(Theta)'

# show data in view
calculator1_4Display = Show(calculator1_4, renderView1, 'UnstructuredGridRepresentation')

# get color transfer function/color map for 'Ut'
utLUT = GetColorTransferFunction('Ut')

# get opacity transfer function/opacity map for 'Ut'
utPWF = GetOpacityTransferFunction('Ut')

# trace defaults for the display properties.
calculator1_4Display.Representation = 'Surface'
calculator1_4Display.ColorArrayName = ['POINTS', 'Ut']
calculator1_4Display.LookupTable = utLUT
calculator1_4Display.SelectTCoordArray = 'None'
calculator1_4Display.SelectNormalArray = 'None'
calculator1_4Display.SelectTangentArray = 'None'
calculator1_4Display.OSPRayScaleArray = 'Ut'
calculator1_4Display.OSPRayScaleFunction = 'PiecewiseFunction'
calculator1_4Display.SelectOrientationVectors = 'Velocity'
calculator1_4Display.ScaleFactor = 0.01486000046133995
calculator1_4Display.SelectScaleArray = 'Ut'
calculator1_4Display.GlyphType = 'Arrow'
calculator1_4Display.GlyphTableIndexArray = 'Ut'
calculator1_4Display.GaussianRadius = 0.0007430000230669976
calculator1_4Display.SetScaleArray = ['POINTS', 'Ut']
calculator1_4Display.ScaleTransferFunction = 'PiecewiseFunction'
calculator1_4Display.OpacityArray = ['POINTS', 'Ut']
calculator1_4Display.OpacityTransferFunction = 'PiecewiseFunction'
calculator1_4Display.DataAxesGrid = 'GridAxesRepresentation'
calculator1_4Display.PolarAxes = 'PolarAxesRepresentation'
calculator1_4Display.ScalarOpacityFunction = utPWF
calculator1_4Display.ScalarOpacityUnitDistance = 0.006739942549997341
calculator1_4Display.OpacityArrayName = ['POINTS', 'Ut']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
calculator1_4Display.ScaleTransferFunction.Points = [-249.5223520782826, 0.0, 0.5, 0.0, 0.004079029552798374, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
calculator1_4Display.OpacityTransferFunction.Points = [-249.5223520782826, 0.0, 0.5, 0.0, 0.004079029552798374, 1.0, 0.5, 0.0]

# hide data in view
Hide(calculator1_3, renderView1)

# show color bar/color legend
calculator1_4Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# Rescale transfer function
utLUT.RescaleTransferFunction(-249.52104910129293, 0.0011176199857443336)

# Rescale transfer function
utPWF.RescaleTransferFunction(-249.52104910129293, 0.0011176199857443336)

# rename source object
RenameSource('Ut', calculator1_4)

# set active source
SetActiveSource(flowvtu)

# set active source
SetActiveSource(calculator1)

# set active source
SetActiveSource(calculator1_1)

# set active source
SetActiveSource(calculator1_2)

# set active source
SetActiveSource(calculator1_3)

# set active source
SetActiveSource(calculator1_4)

# create a new 'Calculator'
calculator1_5 = Calculator(registrationName='Calculator1', Input=calculator1_4)
calculator1_5.Function = ''

# Properties modified on calculator1_5
calculator1_5.ResultArrayName = 'Wt'

# Properties modified on calculator1_5
calculator1_5.Function = 'Ut - (-17189*2*3.1415)/60*Radius'

# update the view to ensure updated data information
renderView1.Update()

# set scalar coloring
ColorBy(calculator1_5Display, ('POINTS', 'Wt'))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(utLUT, renderView1)

# rescale color and/or opacity maps used to include current data range
calculator1_5Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
calculator1_5Display.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'Wt'
wtLUT = GetColorTransferFunction('Wt')

# get opacity transfer function/opacity map for 'Wt'
wtPWF = GetOpacityTransferFunction('Wt')

# Rescale transfer function
wtLUT.RescaleTransferFunction(91.77703313085357, 461.9879870449751)

# Rescale transfer function
wtPWF.RescaleTransferFunction(91.77703313085357, 461.9879870449751)

# rename source object
RenameSource('Wt', calculator1_5)

#================================================================
# addendum: following script captures some of the application
# state to faithfully reproduce the visualization during playback
#================================================================

# get layout
layout1 = GetLayout()

#--------------------------------
# saving layout sizes for layouts

# layout/tab size in pixels
layout1.SetSize(1153, 804)

#-----------------------------------
# saving camera placements for views

# current camera placement for renderView1
renderView1.CameraPosition = [0.3080448804973473, 0.16718759278733597, -0.08890364198949548]
renderView1.CameraFocalPoint = [0.20511012894401753, -0.000977319713208369, 0.016734200472336997]
renderView1.CameraViewUp = [0.8800460961737158, -0.45652316918680436, 0.1307878610767811]
renderView1.CameraParallelScale = 0.08476198572684579

#--------------------------------------------
# uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).