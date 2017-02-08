from matplotlib import *
from matplotlib.pyplot import *
from matplotlib.colors import from_levels_and_colors, BoundaryNorm
from pylab import *
import numpy as np
from matplotlib import rcParams

def draw_JPDF (plot_data, plot_level, titles, x_ticks, x_names,y_ticks,y_names, output_file, cmap = cm.YlGn, 
               cbar_ticks=[0.01, 0.10, 0.5, 2, 5, 25], cbar_label=['0.01', '0.10', '0.5', '2', '5', '25']):
    '''
    data - a numpy array of data to plot (nY, nX)
    plot_level - levels to plot
    titles - array of titles (nT)
    cbar_title - title of color bar
    x_ticks - x values where tick makrs are located
    x_names - labels for the ticks on x-axis (nX)
    y_ticks - y values where tick makrs are located
    y_names - labels for the ticksy-axis (nY)
    output_file - name of output png file
    '''

    if cmap=='diff':
        cmap = DarkBlueDarkRed(16)

    fig=figure()
    sb = fig.add_subplot(111)
    dimY, dimX = plot_data.shape
    plot_data2 = np.zeros([dimY,dimX])

    rcParams['axes.labelsize'] = 8
    rcParams['xtick.labelsize'] = 8
    rcParams['ytick.labelsize'] = 8
    for iy in range(dimY):
         for ix in range(dimX):
             if plot_data[iy,ix] <= np.min(plot_level):
                 plot_data2[iy,ix] = -1.
             else:
                 plot_data2[iy,ix] = plot_level[np.where(plot_level <= plot_data[iy,ix])].max()
    sb.set_xticks(x_ticks)
    for iy in range(dimY):
         for ix in range(dimX):
             if plot_data[iy,ix] <= np.min(plot_level):
                 plot_data2[iy,ix] = -1.
             else:
                 plot_data2[iy,ix] = plot_level[np.where(plot_level <= plot_data[iy,ix])].max()
    sb.set_xticks(x_ticks)
    sb.set_xticklabels(x_names)
    sb.set_yticks(y_ticks)
    sb.set_yticklabels(y_names)
    title(titles)
    norm = BoundaryNorm(plot_level[1:], cmap.N)
    cmap.set_under('w')
    a=sb.pcolor(plot_data2, edgecolors = 'k', linewidths=0.5, cmap = cmap, norm = norm)
    a.cmap.set_under('w')
    sb.set_xlabel('Spell duration [hrs]')
    sb.set_ylabel('Peak rainfall [mm/hr]')
    cax = fig.add_axes([0.2, -0.06, 0.6, 0.02])
    cbar = colorbar(a, cax=cax, orientation = 'horizontal', extend='both')
    cbar.set_ticks(cbar_ticks)

    cbar.set_ticklabels(cbar_label)
    plt.show()
    fig.savefig(output_file, dpi=600,bbox_inches='tight')

