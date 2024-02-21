#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 09:29:59 2023
@author: F. Neri, TU Delft
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from Utils.styles import *

# Get a list of all directories in the current directory
directories = [d for d in os.listdir('.') if os.path.isdir(d)]
res = {}
# Loop through each directory
for directory in directories:
    log_file_path = os.path.join(directory, 'log.txt')
    if os.path.exists(log_file_path):
        with open(log_file_path, 'r') as file:
            # Read lines from the file and extract the last word
            last_line = None
            for line in file:
                # Split each line into words and take the last word
                last_line = line
            # Print the last word
            if last_line:
                tot_time = last_line.split()[-2]
                print(f"Total time in simulation '{directory}': {tot_time} sec.")
                res[directory] = float(tot_time)
            else:
                print(f"No content found in log.txt in folder '{directory}'")
            print()  # Add a newline for separation

# Extract keys and values from the dictionary
sorted_res = {k: res[k] for k in sorted(res)}

keys = list(sorted_res.keys())
values = list(sorted_res.values())

# Create a bar plot
plt.figure(figsize=(8, 5))
plt.bar(keys, values)

# Add labels and title
plt.ylabel('CPU time [s]', fontsize=font_labels)
plt.xticks(rotation=45, ha="right", rotation_mode="anchor", fontsize=font_axes)
plt.yticks(fontsize=font_axes)
plt.yscale('log')
plt.grid(alpha=grid_opacity)
plt.savefig('duct_cpu_times.pdf', bbox_inches='tight')

number_points = []
for key in keys:
    number_points.append(int(key[0:2]) * int(key[3:5]))

nodes = np.array(number_points)
time = np.array(values)

from scipy.optimize import curve_fit
# Define the custom function for time as a function of nodes
def time_func(nodes, A, B):
    return A**(B*nodes)-1

# Fit the custom function to the data using curve_fit
params, covariance = curve_fit(time_func, nodes, time)

# Extract the fitted parameters
A_fit, B_fit = params

# Generate the fitted curve using the fitted parameters
nodes_test = np.linspace(1, 1250)
fitted_time = time_func(nodes_test, A_fit, B_fit)

# Plot the original data and the fitted curve
plt.figure()
plt.scatter(nodes, time, s=100, edgecolors='black', facecolors='C0', linewidth=2, label='data')
plt.plot(nodes_test, fitted_time, '--r', label=r'curve fit', lw=medium_line_width)
plt.xlabel('Nodes [-]', fontsize=font_labels)
plt.ylabel('CPU time [s]', fontsize=font_labels)
plt.xticks(fontsize=font_axes)
plt.yticks(fontsize=font_axes)
plt.grid(alpha=grid_opacity)
plt.xlabel('Nodes [-]')
plt.ylabel('Time [s]')
plt.legend(fontsize=font_legend)
plt.savefig('duct_cpu_times_nodes.pdf', bbox_inches='tight')





nodes_test = np.linspace(1000, 5000)
fitted_time = time_func(nodes_test, A_fit, B_fit)

# Plot the original data and the fitted curve
plt.figure()
plt.plot(nodes_test, fitted_time/3600, '--r', label=r'curve fit', lw=medium_line_width)
plt.xlabel('Nodes [-]', fontsize=font_labels)
plt.ylabel('CPU time [s]', fontsize=font_labels)
plt.xticks(fontsize=font_axes)
plt.yticks(fontsize=font_axes)
plt.grid(alpha=grid_opacity)
plt.xlabel('Nodes [-]')
plt.ylabel('Time [hrs]')
plt.yscale('log')
plt.legend(fontsize=font_legend)
plt.savefig('duct_cpu_times_extrapolation.pdf', bbox_inches='tight')
# plt.figure()
# plt.scatter(nodes, time, s=100, edgecolors='black', facecolors='C0', linewidth=2)
# plt.xlabel('Nodes [-]', fontsize=font_labels)
# plt.ylabel('CPU time [s]', fontsize=font_labels)
# plt.xticks(fontsize=font_axes)
# plt.yticks(fontsize=font_axes)
# # plt.yscale('log')
# plt.grid(alpha=grid_opacity)
# plt.xlabel('Nodes')
# plt.ylabel('Time')
# plt.savefig('duct_cpu_times_nodes.pdf', bbox_inches='tight')

# Show plot
plt.show()
