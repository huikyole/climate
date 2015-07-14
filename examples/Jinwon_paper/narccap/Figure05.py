#Apache OCW lib immports
import ocw.data_source.local as local
#import ocw.plotter as plotter
import ocw.utils as utils
import ocw.evaluation as evaluation
import ocw.metrics as metrics

# Python libraries
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap 
import string

#def make_axes(parent, **kw): 
# from http://matplotlib.1069221.n5.nabble.com/colorbar-td15992.html

data_file = '/home/huikyole/climate/examples/Jinwon_paper/narccap_tas.nc'

obs_dataset = local.load_file(data_file, variable_name = 'CRU', lat_name='lat', lon_name='lon')  
_, obs_dataset.values = utils.calc_climatology_year(obs_dataset)

variable_names = ['CRCM','ECP2','HRM3','MM5I','RCM3','WRFG','ENS-models']
target_datasets = []
for variable_name in variable_names:
    target_dataset = local.load_file(data_file, variable_name=variable_name, lat_name='lat', lon_name='lon')
    _, target_dataset.values = utils.calc_climatology_year(target_dataset)
    target_dataset.values = target_dataset.values+273.15
    target_datasets.append(target_dataset)


# Using OCW
#determine the metrics
mean_bias = metrics.Bias()

#create the Evaluation object
RCMs_to_CRU_evaluation = evaluation.Evaluation(obs_dataset, # Reference dataset for the evaluation
                                    # list of target datasets for the evaluation
                                    target_datasets,
                                    # 1 or more metrics to use in the evaluation
                                    [mean_bias])

RCMs_to_CRU_evaluation.run()

nmodel = len(variable_names)
ny, nx = obs_dataset.values.shape

rcm_bias = ma.zeros([nmodel, ny, nx])

for imodel in np.arange(nmodel):
    rcm_bias[imodel,:,:] = RCMs_to_CRU_evaluation.results[imodel][0]

# Without using OCW
nmodel = len(variable_names)
ny, nx = obs_dataset.values.shape

rcm_bias2= ma.zeros([nmodel,ny,nx]) 
for imodel in np.arange(nmodel):
    rcm_bias2[imodel,:] = target_datasets[imodel].values - obs_dataset.values

fig = plt.figure()

lat_min = obs_dataset.lats.min()
lat_max = obs_dataset.lats.max()
lon_min = obs_dataset.lons.min()
lon_max = obs_dataset.lons.max()

string_list = list(string.ascii_lowercase) 
ax = fig.add_subplot(4,2,1)
m = Basemap(ax=ax, projection ='cyl', llcrnrlat = lat_min, urcrnrlat = lat_max,
            llcrnrlon = lon_min, urcrnrlon = lon_max, resolution = 'l', fix_aspect=False)
lons, lats = np.meshgrid(obs_dataset.lons, obs_dataset.lats)

x,y = m(lons, lats)

m.drawcoastlines(linewidth=1)
m.drawcountries(linewidth=1)
m.drawstates(linewidth=0.5, color='w')
max = m.contourf(x,y,obs_dataset.values-273.15,levels =np.arange(7)*5, cmap='RdBu_r')
ax.annotate('(a) \n CRU',xy=(-124,25))
cax = fig.add_axes([0.07, 0.72, 0.01, 0.18])
plt.colorbar(max, cax = cax) 

for imodel in np.arange(nmodel):
    ax = fig.add_subplot(4,2,2+imodel)
    m = Basemap(ax=ax, projection ='cyl', llcrnrlat = lat_min, urcrnrlat = lat_max,
            llcrnrlon = lon_min, urcrnrlon = lon_max, resolution = 'l', fix_aspect=False)
    m.drawcoastlines(linewidth=1)
    m.drawcountries(linewidth=1)
    m.drawstates(linewidth=0.5, color='w')
    max = m.contourf(x,y,rcm_bias[imodel,:],levels =[-5,-2.5,-0.5,0,0.5,2.5,5], extend='both', cmap='RdBu_r')
    ax.annotate('('+string_list[imodel+1]+')  \n '+variable_names[imodel],xy=(-124,25))

cax = fig.add_axes([0.91, 0.1, 0.015, 0.8])
plt.colorbar(max, cax = cax) 

plt.subplots_adjust(hspace=0.01,wspace=0.05)

plt.show()
fig.savefig('Figure05',dpi=600,bbox_inches='tight')
