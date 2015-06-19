#Apache OCW lib immports
import ocw.dataset_processor as dsp
import ocw.data_source.local as local
import ocw.utils as utils
from ocw.dataset import Bounds
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import yaml
from glob import glob
import operator
from dateutil import parser
from datetime import datetime
import os
import sys

config_file = str(sys.argv[1])

""" Step 1: Read configuration file """
print 'Reading the configuration file ', config_file
config = yaml.load(open(config_file))
#start_time = parser.parse(config['start_time'])
#end_time = parser.parse(config['end_time'])
start_time = datetime.strptime(config['start_time'].strftime('%Y%m%d'),'%Y%m%d')
end_time = datetime.strptime(config['end_time'].strftime('%Y%m%d'),'%Y%m%d')
#end_time = config['end_time']

min_lat, max_lat, delta_lat = config['regrid']['spatial_regrid_lats']
nlat = (max_lat - min_lat)/delta_lat+1
min_lon, max_lon, delta_lon = config['regrid']['spatial_regrid_lons']
nlon = (max_lon - min_lon)/delta_lon+1

bounds = Bounds(min_lat, max_lat, min_lon, max_lon, start_time, end_time)

""" Step 2: Load the reference data """
ref_data_info = config['datasets']['reference']
print 'Loading observation datasets, variable:',ref_data_info['variable']
if ref_data_info['data_source'] == 'local':
    ref_datasets, ref_names = local.load_multiple_files(ref_data_info)
    
    for idata,dataset in enumerate(ref_datasets):
        ref_datasets[idata] = dsp.subset(bounds,dataset)
else:
    print ' '
    # TO DO: support RCMED and ESGF

# number of reference (observation) datasets
nref = len(ref_datasets)
print 'Number of observation datasets:',nref
for ref_name in ref_names:
    print ref_name

""" Step 3: Load model NetCDF Files into OCW Dataset Objects """
model_data_info = config['datasets']['targets']
print 'Loading model datasets, variable:',model_data_info['variable']
if model_data_info['data_source'] == 'local':
    model_datasets, model_names = local.load_multiple_files(model_data_info)
    for idata,dataset in enumerate(model_datasets):
        model_datasets[idata] = dsp.subset(bounds,dataset)
else:
    print ' '
    # TO DO: support RCMED and ESGF
# Add a multi-model ensemble
model_datasets.append(dsp.ensemble(model_datasets))
model_names.append('ENS-models')
# number of models
nmodel = len(model_datasets)
print 'Number of model datasets:',nmodel
for model_name in model_names:
    print model_name

""" Step 4: Spatial regriding of the reference datasets """
new_lat = np.linspace(min_lat, max_lat, nlat)
new_lon = np.linspace(min_lon, max_lon, nlon)
print 'Regridding datasets: ', config['regrid']
for idata,dataset in enumerate(ref_datasets):
    ref_datasets[idata] = dsp.spatial_regrid(dataset, new_lat, new_lon)
for idata,dataset in enumerate(model_datasets):
    model_datasets[idata] = dsp.spatial_regrid(dataset, new_lat, new_lon)

""" Step 5: Generate subregion average and standard deviation """
# sort the subregion by region names and make a list
subregions= sorted(config['subregions'].items(),key=operator.itemgetter(0))

# number of subregions
nsubregion = len(subregions)


print 'Calculating spatial averages and standard deviations of ',str(nsubregion),' subregions'

ref_subregion_mean, ref_subregion_std, subregion_array = utils.calc_subregion_area_mean_and_std(ref_datasets, subregions) 
model_subregion_mean, model_subregion_std, subregion_array = utils.calc_subregion_area_mean_and_std(model_datasets, subregions) 

""" Step 6: Write a netCDF file """
print 'Writing a netcdf file: ',config['workdir']+config['output_netcdf_filename']
dsp.write_netcdf_multiple_datasets_with_subregions(ref_datasets, ref_names, model_datasets, model_names,
                                                   path=config['workdir']+config['output_netcdf_filename'],
                                                   subregions=subregions, subregion_array = subregion_array, ref_subregion_mean=ref_subregion_mean, ref_subregion_std=ref_subregion_std,
                                                   model_subregion_mean=model_subregion_mean, model_subregion_std=model_subregion_std)





