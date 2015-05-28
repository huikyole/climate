#Apache OCW lib immports
import ocw.dataset_processor as dsp
import ocw.data_source.local as local
import ocw.data_source.rcmed as rcmed
import ocw.utils as utils
from ocw.dataset import Bounds
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import yaml
from glob import glob
import operator
import datetime


""" Step 1: Download CRU precipitation data and load the data into OCW Dataset Object """
start_time = datetime.datetime(1991,1,1)
end_time = datetime.datetime(2007,12,31)

min_lat = -45.0
max_lat = 42.24
min_lon = -24.0
max_lon = 60.0

bounds = Bounds(min_lat, max_lat, min_lon, max_lon, start_time, end_time)

cru31_dataset_original = local.load_file('/nas/share1-hp/jinwonki/data/obs/cru3.1/d/GLOBAL_CRU_CTL_CRU-TS31_MM_0.5deg_1901-2009_pr.nc','pr')
cru31_dataset = dsp.subset(bounds, cru31_dataset_original)




""" Step 2: Load Local NetCDF Files into OCW Dataset Objects """
filenames = glob('/share4/jinwonki/data/cordex-af/*pr.nc')

# number of models
nmodel = len(filenames)

model_datasets = []
for filename in filenames:
    dataset_original = local.load_file(filename, "pr")
    model_datasets.append(dsp.subset(bounds, dataset_original)) 

""" Step 3: Load Subregion YAML file """
# Load a subregion boundary information in yaml format
subregions= yaml.load(open('cordex_AF_subregions.yaml'))

# sort the subregion by region names and make a list
subregions= sorted(subregions.items(),key=operator.itemgetter(0))

# number of subregions
nsubregion = len(subregions)

""" Step 4: Calculate climatological annual cycle from observation and model datasets in each subregion"""
obs_t_series = np.zeros([12, nsubregion])
model_t_series = np.zeros([12, nsubregion, nmodel+1])

obs_t_series = np.mean(utils.reshape_monthly_to_annually(utils.calc_subregion_area_mean(cru31_dataset, subregions)), axis=0)
for imodel in np.arange(nmodel):
    nt = model_datasets[imodel].times.size
    model_t_series[:,:,imodel]=np.mean((utils.calc_subregion_area_mean(model_datasets[imodel], subregions)).reshape([nt/12,12,nsubregion]), axis=0)
model_t_series[:,:,-1] = np.mean(model_t_series[:,:,:-1], axis=2)    

model_t_series = model_t_series * 3600.*24
obs_t_series = obs_t_series * 3600.*24

fig = plt.figure()

rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 7

region = [1, 3, 6, 8, 10, 9, 11, 12, 13, 15]

for ii in np.arange(10):
    ax=fig.add_subplot(5,2,ii+1)
    ax.plot(obs_t_series[:,region[ii]], color='r',lw=3) 
    ax.plot(model_t_series[:,region[ii], -1], color='b',lw=3) 
    for imodel in np.arange(nmodel):
        ax.plot(model_t_series[:,region[ii],imodel], linestyle='--',color='grey',lw=1) 
    ax.set_xlim([-0.2,11.2])
    ax.set_xticks(np.arange(12))
    ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
    ax.set_title(subregions[region[ii]][0])

fig.subplots_adjust(hspace=0.7)    
fig.savefig('cordex_AF_precipitation_time-series_subregions',dpi=600,bbox_inches='tight')






