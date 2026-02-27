import time
import matplotlib.pyplot as plt
import pickle
import numpy as np
from unsflow.grid.config import Config
from unsflow.grid.multiblock_grid_driver import MultiBlockGridDriver

# SETTINGS
configuration_file = 'input.ini'
config = Config(configuration_file)
driver = MultiBlockGridDriver(config)
driver.GenerateGrid()
driver.ComputeBladesData()
driver.AssembleMultiBlockGrid()
driver.SaveOutput()

plt.show()