import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.optimize import fsolve
from utils.styles import *
from greitzer.src.greitzer import Greitzer
from greitzer.src.config import Config


config = Config('input.ini')
greitzer = Greitzer(config)
greitzer.solveMooreGreitzerSystem()
greitzer.plotTemporalEvolutionMooreGreitzer(save_filename='unstable')
greitzer.plotTrajectoryMooreGreitzer(save_filename='unstable')
greitzer.savePickle(save_filename='unstable')


plt.show()







