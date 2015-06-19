#Apache OCW lib immports
import ocw.data_source.local as local
import ocw.plotter as plotter
import ocw.utils as utils
import ocw.evaluation as evaluation
import ocw.metrics as metrics

# Python libraries
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import operator
import string

data_file = '/home/huikyole/climate/examples/cordex_AF_paper/cordex_AF_precipitation.nc'

obs_dataset = local.load_file(data_file, variable_name = 'CRU', lat_name='lat', lon_name='lon')  
_, obs_dataset.values = utils.calc_climatology_year(obs_dataset)

variable_names = ['CNRM-ARPEGE51','DMI-HIRHAM','ICTP-REGCM3','IES-CCLM','KNMI-RACMO2.2b','MPI-REMO','SMHI-RCA35','UC-WRF311','UCT-PRECIS','UQAM-CRCM5', 'ENS-models']
target_datasets = []
for variable_name in variable_names:
    target_dataset = local.load_file(data_file, variable_name=variable_name, lat_name='lat', lon_name='lon')
    _, target_dataset.values = utils.calc_climatology_year(target_dataset)
    target_datasets.append(target_dataset)


# Currently (June/04/2015), our metrics do not properly process missing values.
'''
#determine the metrics
mean_bias = metrics.Bias()

#create the Evaluation object
RCMs_to_CRU_evaluation = evaluation.Evaluation(obs_dataset, # Reference dataset for the evaluation
                                    # list of target datasets for the evaluation
                                    target_datasets,
                                    # 1 or more metrics to use in the evaluation
                                    [mean_bias])

RCMs_to_CRU_evaluation.run()

rcm_bias = ma.squeeze(ma.array(RCMs_to_CRU_evaluation.results))
'''
nmodel = len(variable_names)
ny, nx = obs_dataset.values.shape
rcm_bias = ma.zeros([nmodel,ny,nx]) 
for imodel in np.arange(nmodel):
    rcm_bias[imodel,:] = target_datasets[imodel].values - obs_dataset.values

fig = plt.figure()

lat_min = obs_dataset.lats.min()
lat_max = obs_dataset.lats.max()
lon_min = obs_dataset.lons.min()
lon_max = obs_dataset.lons.max()

string_list = list(string.ascii_lowercase) 
for ivar, variable_name in enumerate(variable_names[:-1]):
    plot_title = '('+string_list[ivar]+') '+variable_name[0:3] 
    ax = fig.add_subplot(4,5,ivar+1)

    ax, cs = plotter.draw_contour_map(ax, rcm_bias[ivar,:,:]*86400., obs_dataset.lats, obs_dataset.lons, cmap = 'RdBu',clevs = np.arange(7)-3., title=plot_title, extend='both')

# add colorbar
cax = fig.add_axes([0.95,0.53, 0.01, 0.4])
cbar = fig.colorbar(cs, cax=cax)
cbar.ax.set_ylabel('[mm/day]')
plt.subplots_adjust(hspace=0.4)

ax = fig.add_subplot(2, 2, 3)
spatial_mean_of_obs_and_models = np.zeros([nmodel+1]) 
spatial_mean_of_obs_and_models[0] = utils.calc_area_weighted_spatial_average(obs_dataset)
for imodel in np.arange(nmodel):
    spatial_mean_of_obs_and_models[imodel+1] = utils.calc_area_weighted_spatial_average(target_datasets[imodel])
spatial_mean_of_obs_and_models = spatial_mean_of_obs_and_models*86400.
data_names=['CRU']
for variable_name in variable_names:
    data_names.append(variable_name[0:3])
ax = plotter.draw_barchart(ax, spatial_mean_of_obs_and_models, data_names, xlabel='mm/day', title='(k) area weighted mean prec.') 

taylor_data = np.zeros([nmodel,2])
for imodel in np.arange(nmodel):
    taylor_data[imodel,0] = ma.std(target_datasets[imodel].values)/ma.std(obs_dataset.values)
    taylor_data[imodel,1] = ma.corrcoef(obs_dataset.values.flatten(), target_datasets[imodel].values.flatten())[0,1]

plotter.draw_taylor_diagram(fig, 224, taylor_data, data_names[1:], data_names[0], ptitle='(l)',legend_font_size=5,pos='best',frameon=False, ncol=1)


plt.show()
fig.savefig('Figure02',dpi=600,bbox_inches='tight')
