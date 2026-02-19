import time
import matplotlib.pyplot as plt
import pickle
import numpy as np
from grid.src.config import Config
from grid.src.multiblock_grid_driver import MultiBlockGridDriver
from grid.src.functions import contour_template

# SETTINGS
configuration_file = 'input.ini'
config = Config(configuration_file)
driver = MultiBlockGridDriver(config)
driver.GenerateGrid()
driver.ComputeBladesData()
driver.AssembleMultiBlockGrid()
driver.SaveOutput()

plt.show()