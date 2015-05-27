#Apache OCW lib immports
from ocw.dataset import Bounds
import ocw.plotter as plotter
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import yaml
import operator

LAT_MIN = -45.0 
LAT_MAX = 42.24
LON_MIN = -24.0
LON_MAX = 60.0 

#map_boundary = Bounds(LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)

# Load a subregion boundary information in yaml format
subregions= yaml.load(open('cordex_AF.yaml'))

# sort the subregion by region names and make a list
subregions= sorted(subregions.items(),key=operator.itemgetter(0))

plotter.draw_subregions(subregions, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, fname='cordex_AF_subregions')


'''
fig = plt.figure()
ax =fig.add_subplot(1,1,1)
m = Basemap(ax=ax, projection='cyl',llcrnrlat = LAT_MIN, urcrnrlat = LAT_MAX,
            llcrnrlon = LON_MIN, urcrnrlon = LON_MAX, resolution = 'l')
m.drawcoastlines(linewidth=0.75)
m.drawcountries(linewidth=0.75)
m.etopo()

for subregion in subregions:
    plotter.draw_screen_poly(subregion[1], m, 'r')
    plt.annotate(subregion[0],xy=(0.5*(subregion[1][2]+subregion[1][3]), 0.5*(subregion[1][0]+subregion[1][1])), ha='center',va='center', fontsize=14)
plt.show()
'''
