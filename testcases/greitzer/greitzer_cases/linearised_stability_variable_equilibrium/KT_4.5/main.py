import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.optimize import fsolve
from unsflow.greitzer.greitzer import Greitzer
from unsflow.greitzer.greitzer import Config


config = Config('input.ini')
greitzer = Greitzer(config)
greitzer.computeLinearizedStabilityMap()
greitzer.plotStabilityMap('map')
greitzer.savePickle('results')


plt.show()