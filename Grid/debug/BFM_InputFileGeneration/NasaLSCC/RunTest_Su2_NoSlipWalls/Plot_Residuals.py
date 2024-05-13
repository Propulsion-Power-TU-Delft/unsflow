import numpy as np
import matplotlib.pyplot as plt
from Utils.styles import *

file_path = 'history_BFM.csv'
header = np.genfromtxt(file_path, delimiter=',', max_rows=1, dtype=str)
with open(file_path, "r") as file:
    for i, line in enumerate(file):
        if i == 0:
            elements = line.strip().split(',')
            header = [element.strip('""') for element in elements if element.strip()]
            break


data = np.genfromtxt(file_path, delimiter=',', skip_header=1)

def reset_to_zero(f):
    return f-f[0]

plt.figure()
for i in range(3, len(header)):
    plt.plot(data[:, 2], reset_to_zero(data[:, i]), label=header[i])
plt.xlabel('iterations')
plt.ylabel('residuals')
plt.grid(alpha=0.3)
plt.legend()
plt.savefig('residuals.pdf', bbox_inches='tight')

plt.show()
