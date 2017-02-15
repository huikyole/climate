import numpy as np
from pickle import load,dump
from draw_JPDF import draw_JPDF
from matplotlib import *

import ocw.metrics as metrics

histo2d_GPM = load(open('GPM_JPDF_example.pickle'))
histo2d_W24 = load(open('WRF24_JPDF_example.pickle'))

overlap = metrics.calc_histogram_overlap(histo2d_GPM, histo2d_W24)

plot_level = np.array([-21, -3, -1, -0.5, -0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2, 0.5, 1, 3, 21])
cbar_ticks = [-2, -0.5, -0.1, 0.1, 0.5, 2]
cbar_label = [str(i) for i in cbar_ticks]
draw_JPDF(plot_data=np.transpose(histo2d_W24 - histo2d_GPM), plot_level=plot_level, titles='overlap %d ' %overlap +r'%',
          x_ticks=[0.5, 4.5, 9.5, 14.5, 19.5, 23.5], x_names=['1','5','10','15','20','24'],
          y_ticks=np.arange(9), y_names=['0.1','0.2','0.5','1.0','2.0','5.0','10','20','30'],
          output_file='GPM_WRF24_JJPDF_comparison', cmap=cm.BrBG, 
          cbar_ticks=cbar_ticks, cbar_label=cbar_label)
