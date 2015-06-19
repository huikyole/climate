#Apache OCW lib immports
import ocw.data_source.local as local
import ocw.plotter as plotter
import ocw.utils as utils

# Python libraries
import matplotlib.pyplot as plt
import yaml
import operator

# Load a subregion boundary information in yaml format
subregions= yaml.load(open('cordex_AF_subregions.yaml'))

# sort the subregion by region names and make a list
subregions= sorted(subregions.items(),key=operator.itemgetter(0))

cru_pr = local.load_file('/home/huikyole/climate/examples/Jinwon_paper/cordex_AF_precipitation.nc',
                                     variable_name = 'CRU', lat_name='lat', lon_name='lon')  
lat_min = cru_pr.lats.min()
lat_max = cru_pr.lats.max()
lon_min = cru_pr.lons.min()
lon_max = cru_pr.lons.max()

fig = plt.figure()

ax = fig.add_subplot(1,2,1)
ax = plotter.draw_subregions(ax, subregions, lat_min, lat_max, lon_min, lon_max, title='(a) 21 subreions')

# calculate annual precipitation climatology of CRU data
annually_mean, total_mean = utils.calc_climatology_year(cru_pr)

ax = fig.add_subplot(1,2,2)
ax, cs = plotter.draw_contour_map(ax, total_mean*86400., cru_pr.lats, cru_pr.lons, cmap = 'RdBu',clevs = range(7), title='(b) The annual prec. climatology', extend='max')

# add colorbar
cax = fig.add_axes([0.95,0.2, 0.01, 0.6])
cbar = fig.colorbar(cs, cax=cax)
cbar.ax.set_ylabel('[mm/day]')

plt.show()
fig.savefig('Figure01',dpi=600,bbox_inches='tight')
