import time
import matplotlib.pyplot as plt
import pickle
import numpy as np
from Grid.src.config import Config
from Grid.src.multiblock_grid_driver import MultiBlockGridDriver
from Grid.src.functions import contour_template

# SETTINGS
configuration_file = 'input.ini'
config = Config(configuration_file)
driver = MultiBlockGridDriver(config)
driver.GenerateGrid()
driver.ComputeBladesData()
driver.AssembleMultiBlockGrid()
driver.SaveOutput()

bf = driver.blades[0].bodyForce
omega = 12657*np.pi*2/60
r2 = bf.meridionalFields['Radial_Coordinate'][0,-1]
factor = omega**2 * r2
contour_template(bf.meridionalFields['Axial_Coordinate'], bf.meridionalFields['Radial_Coordinate'], bf.bodyForceFields['Force_Inviscid']/factor, name=r'$f_n / f_{n,ref}$', vmin=0, vmax=2.5, save_filename='finviscid', folder_name='Pics_MagriniComparison')
contour_template(bf.meridionalFields['Axial_Coordinate'], bf.meridionalFields['Radial_Coordinate'], bf.bodyForceFields['Force_Viscous']/factor, name=r'$f_p / f_{n,ref}$', save_filename='fviscous', folder_name='Pics_MagriniComparison')
contour_template(bf.meridionalFields['Axial_Coordinate'], bf.meridionalFields['Radial_Coordinate'], bf.meridionalFields['Velocity_Axial'], name=r'$u_z \ \rm{[m/s]}$', save_filename='uz', folder_name='Pics_MagriniComparison', vmin=0)
contour_template(bf.meridionalFields['Axial_Coordinate'], bf.meridionalFields['Radial_Coordinate'], bf.meridionalFields['Velocity_Radial'], name=r'$u_r \ \rm{[m/s]}$', save_filename='ur', folder_name='Pics_MagriniComparison')
contour_template(bf.meridionalFields['Axial_Coordinate'], bf.meridionalFields['Radial_Coordinate'], -bf.meridionalFields['Velocity_Tangential'], name=r'$u_{\theta} \ \rm{[m/s]}$', save_filename='ut_neg', folder_name='Pics_MagriniComparison')
contour_template(bf.meridionalFields['Axial_Coordinate'], bf.meridionalFields['Radial_Coordinate'], -bf.meridionalFields['Relative_Flow_Angle']*180/np.pi, name=r'$\beta \ \rm{[deg]}$', save_filename='beta', folder_name='Pics_MagriniComparison', vmax=85)

pressure = bf.meridionalFields['Pressure']
temperature = bf.meridionalFields['Temperature']
mach = bf.meridionalFields['Mach']
totalPressure = pressure*(1+(0.4)/2*mach**2)**(1.4/0.4)
totalTemperature = temperature*(1+(0.4)/2*mach**2)
contour_template(bf.meridionalFields['Axial_Coordinate'], bf.meridionalFields['Radial_Coordinate'], totalPressure/101325, name=r'$\beta_{tt} \ \rm{[-]}$', save_filename='prtt', folder_name='Pics_MagriniComparison')
contour_template(bf.meridionalFields['Axial_Coordinate'], bf.meridionalFields['Radial_Coordinate'], totalTemperature/288.15, name=r'$\tau_{tt} \ \rm{[-]}$', save_filename='trtt', folder_name='Pics_MagriniComparison')

plt.show()