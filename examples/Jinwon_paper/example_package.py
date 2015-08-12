#Apache OCW lib immports
import ocw.data_source.local as local
import ocw.plotter as plotter
import ocw.utils as utils
from ocw.evaluation import Evaluation
import ocw.metrics as metrics

# Python libraries
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap 
import string

def Map_plot_bias_of_multiyear_climatology(obs_dataset, obs_name, model_datasets, model_names,
                                      file_name, row, column):
    '''Draw maps of observed multi-year climatology and biases of models"'''

    # calculate climatology of observation data
    obs_clim = utils.calc_temporal_mean(obs_dataset)
    # determine the metrics
    map_of_bias = metrics.TemporalMeanBias()

    # create the Evaluation object
    bias_evaluation = Evaluation(obs_dataset, # Reference dataset for the evaluation
                                 model_datasets, # list of target datasets for the evaluation
                                 [map_of_bias, map_of_bias])

    # run the evaluation (bias calculation)
    bias_evaluation.run() 

    rcm_bias = bias_evaluation.results[0]

    fig = plt.figure()

    lat_min = obs_dataset.lats.min()
    lat_max = obs_dataset.lats.max()
    lon_min = obs_dataset.lons.min()
    lon_max = obs_dataset.lons.max()

    string_list = list(string.ascii_lowercase) 
    ax = fig.add_subplot(row,column,1)
    m = Basemap(ax=ax, projection ='cyl', llcrnrlat = lat_min, urcrnrlat = lat_max,
            llcrnrlon = lon_min, urcrnrlon = lon_max, resolution = 'l', fix_aspect=False)
    lons, lats = np.meshgrid(obs_dataset.lons, obs_dataset.lats)

    x,y = m(lons, lats)

    m.drawcoastlines(linewidth=1)
    m.drawcountries(linewidth=1)
    m.drawstates(linewidth=0.5, color='w')
    max = m.contourf(x,y,obs_clim,levels = plotter._nice_intervals(obs_dataset.values, 10), extend='both',cmap='PuOr')
    ax.annotate('(a) \n' + obs_name,xy=(lon_min, lat_min))
    cax = fig.add_axes([0.02, 1.-float(1./row), 0.01, 1./row*0.6])
    plt.colorbar(max, cax = cax) 
    clevs = plotter._nice_intervals(rcm_bias, 11)
    for imodel in np.arange(len(model_datasets)):
        ax = fig.add_subplot(row, column,2+imodel)
        m = Basemap(ax=ax, projection ='cyl', llcrnrlat = lat_min, urcrnrlat = lat_max,
                llcrnrlon = lon_min, urcrnrlon = lon_max, resolution = 'l', fix_aspect=False)
        m.drawcoastlines(linewidth=1)
        m.drawcountries(linewidth=1)
        m.drawstates(linewidth=0.5, color='w')
        max = m.contourf(x,y,rcm_bias[imodel,:],levels = clevs, extend='both', cmap='RdBu_r')
        ax.annotate('('+string_list[imodel+1]+')  \n '+model_names[imodel],xy=(lon_min, lat_min))

    cax = fig.add_axes([0.91, 0.1, 0.015, 0.8])
    plt.colorbar(max, cax = cax) 

    plt.subplots_adjust(hspace=0.01,wspace=0.05)

    plt.show()
    fig.savefig(file_name,dpi=600,bbox_inches='tight')

def Taylor_diagram_spatial_pattern_of_multiyear_climatology(obs_dataset, obs_name, model_datasets, model_names,
                                      file_name):

    # calculate climatological mean fields
    obs_dataset.values = utils.calc_temporal_mean(obs_dataset)
    for dataset in model_datasets:
        dataset.values = utils.calc_temporal_mean(dataset)

    # Metrics (spatial standard deviation and pattern correlation)
    # determine the metrics
    taylor_diagram = metrics.SpatialPatternTaylorDiagram()

    # create the Evaluation object
    taylor_evaluation = Evaluation(obs_dataset, # Reference dataset for the evaluation
                                 model_datasets, # list of target datasets for the evaluation
                                 [taylor_diagram])

    # run the evaluation (bias calculation)
    taylor_evaluation.run() 

    taylor_data = taylor_evaluation.results[0]

    plotter.draw_taylor_diagram(taylor_data, model_names, obs_name, file_name, pos='upper right',frameon=False)
