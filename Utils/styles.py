#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 12:45:00 2023
@author: F. Neri, TU Delft

common styles used throughout the code for pictures
"""
from matplotlib import cm

# default size of the figures
fig_size_single = (7, 5)
fig_size_12 = (12, 5)
fig_size_13 = (15, 4)
fig_size_22 = (12, 10)
fig_size_3 = (15, 13)

# Set grid opacity:
grid_opacity = 0.2

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
marker_size_big = 10
marker_size = 7.5
marker_size_small = 2
scatter_point_size = 10

# Set the scale for marker size plotted in the legend entries:
marker_scale_legend = 1

# Set line width for all line plots:
heavy_line_width = 3
line_width = 1
medium_line_width = 2
light_line_width = 0.5

# set number of levels in contourf plots
N_levels = 25
N_levels_medium = 50
N_levels_fine = 100
N_fine = 100

# set colormap for contourf plots
color_map = cm.viridis

# number of chars for the banners
total_chars = 100
total_chars_mid = total_chars//2
