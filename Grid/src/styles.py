#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 12:45:00 2023
@author: F. Neri, TU Delft

common styles used throughout the code for pictures
"""
import matplotlib.pyplot as plt
import os 
from matplotlib import cm


plt.rc('text')      
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rcParams['font.size'] = 14

# pictures folder
folder_name = 'pictures/'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# meta-data folder
folder_meta_data_default = 'data/meta/'
if not os.path.exists(folder_meta_data_default):
    os.makedirs(folder_meta_data_default)

# default size of the figures
fig_size = (10,6)

# Set grid opacity:
grid_opacity = 0.3

# Set font size for axis ticks:
font_axes = 16

# Set font size for axis labels:
font_labels = 24

# Set font size for
font_annotations = 20

# Set font size for plot title:
font_title = 24

# Set font size for plotted text:
font_text = 16

# Set font size for legend entries:
font_legend = 16

# Set font size for colorbar axis label:
font_colorbar = 24

# Set font size for colorbar axis ticks:
font_colorbar_axes = 18

# Set marker size for all line markers:
marker_size = 50

# Set the scale for marker size plotted in the legend entries:
marker_scale_legend = 1

# Set point size for all scatter plots:
scatter_point_size = 10

# Set line width for all line plots:
line_width = 2

# Set light-line width for all line plots:
light_line_width = 0.5

#set number of levels in contourf plots
N_levels = 75
N_levels_2 = 100

#set colormap for contourf plots
color_map = cm.coolwarm
# color_map = cm.viridis
# color_map = cm.inferno