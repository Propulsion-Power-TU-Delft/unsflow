import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm

grid_opacity = 0.3

font_axes = 16
font_labels = 24
font_annotations = 20
font_title = 24
font_text = 16
font_legend = 16
font_colorbar = 24
font_colorbar_axes = 18

marker_size_big = 12.5
marker_size = 5
marker_size_small = 2

heavy_line_width = 3
line_width = 1
medium_line_width = 2
light_line_width = 0.5

N_levels = 25
N_levels_medium = 50
N_levels_fine = 100
color_map = cm.turbo


# # Uncomment the following to have latex output
# font_family = 'serif'
# font_name = 'Computer Modern'
# plt.rc('text', usetex=True)
# plt.rc('font', family=font_family)
# plt.rc('font', serif=font_name)

plt.rc('font', size=font_text)            
plt.rc('axes', titlesize=font_title)       
plt.rc('axes', labelsize=font_labels)      
plt.rc('xtick', labelsize=font_axes)      
plt.rc('ytick', labelsize=font_axes)      
plt.rc('legend', fontsize=font_legend)      
mpl.rcParams['figure.max_open_warning'] = 100
