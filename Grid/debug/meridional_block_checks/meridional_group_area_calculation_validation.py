import time
import matplotlib.pyplot as plt
import pickle
import numpy as np

pickle_filepaths = ['../../testcases/nasa_rotor_37/data/pickle/nasar37_p130_5_3_5_3.pickle',
                    '../../testcases/nasa_rotor_37/data/pickle/nasar37_p130_5_5_5_5.pickle',
                    '../../testcases/nasa_rotor_37/data/pickle/nasar37_p130_10_10_20_10.pickle',
                    '../../testcases/nasa_rotor_37/data/pickle/nasar37_p130_20_20_40_20.pickle',
                    '../../testcases/nasa_rotor_37/data/pickle/nasar37_p130_10_30_20_30.pickle',
                    '../../testcases/nasa_rotor_37/data/pickle/nasar37_p130_10_40_20_40.pickle',
                    '../../testcases/nasa_rotor_37/data/pickle/nasar37_p130_10_50_20_50.pickle',
                    '../../testcases/nasa_rotor_37/data/pickle/nasar37_p130_5_60_5_60.pickle',
                    '../../testcases/nasa_rotor_37/data/pickle/nasar37_p130_5_70_5_70.pickle',
                    '../../testcases/nasa_rotor_37/data/pickle/nasar37_p130_5_80_5_80.pickle']

objs = []
for filepath in pickle_filepaths[0:-2]:
    with open(filepath, 'rb') as file:
        objs.append(pickle.load(file))

blades = [block.group[1] for block in objs]

for blade in blades:
    blade.block.compute_total_area()
    # blade.block.plot_areas_distribution()

tot_area = [blade.block.area_total for blade in blades]
nodes = [blade.block.z_grid_cg.shape[1]*blade.block.z_grid_cg.shape[0] for blade in blades]

# area convergence
plt.figure()
plt.plot(nodes, tot_area, '-o', label = 'triangulation')
plt.xlabel('Elements')
plt.xticks(nodes)
plt.ylabel(r'$\hat{A}$')
plt.title('Area Convergence')
plt.xscale('log')
plt.legend()
plt.grid(alpha=0.2)

# area convergence relative errors to respect to the last
plt.figure()
plt.plot(nodes, 100*(tot_area-tot_area[-1])/tot_area[-1], '-o')
plt.xlabel('Elements')
plt.xticks(nodes)
plt.ylabel(r'$\varepsilon \ [\%$]')
plt.title('Relative Error')
plt.xscale('log')
plt.grid(alpha=0.2)


plt.show()
