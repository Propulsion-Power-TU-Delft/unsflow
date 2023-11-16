from distutils.core import setup
from distutils.extension import Extension
import Spakozvsky
import Grid
import Sun
import os
import platform


setup(name='unsflow',
      version='1.0.0',
      license='MIT',
      description='Unsflow is a software for compressor instabilities analysis and prediction',
      author='Francesco Neri',
      packages=['unsflow'])
