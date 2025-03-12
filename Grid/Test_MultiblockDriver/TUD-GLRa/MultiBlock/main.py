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

plt.show()