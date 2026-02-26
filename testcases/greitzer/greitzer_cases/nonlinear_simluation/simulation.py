import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.optimize import fsolve
from unsflow.utils.plot_styles import *
from unsflow.greitzer.greitzer import Greitzer
from unsflow.greitzer.config import Config


config = Config('input.ini')
greitzer = Greitzer(config)
greitzer.solveGreitzerSystem()
greitzer.plotTemporalEvolutionGreitzer(save_filename='unstable')
greitzer.plotTrajectoryGreitzer(save_filename='unstable')
greitzer.savePickle(save_filename='unstable')


plt.show()







