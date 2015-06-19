#Apache OCW lib immports
import ocw.dataset_processor as dsp
import ocw.data_source.local as local
import ocw.data_source.rcmed as rcmed
import ocw.utils as utils
from ocw.dataset import Bounds
import ocw.plotter as plotter

from netCDF4 import Dataset
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import yaml
from glob import glob
import operator
import datetime

data_file = '/home/huikyole/climate/examples/cordex_AF_paper/cordex_AF_precipitation.nc'


# Reading data
f = Dataset(data_file)

obs_mean = np.squeeze(f.variables['obs_subregion_mean'][:])
obs_std = np.squeeze(f.variables['obs_subregion_std'][:])

model_mean = f.variables['model_subregion_mean'][:]
nmodel, nt, nsubregion = model_mean.shape

model_names0= ['CNRM-ARPEGE51','DMI-HIRHAM','ICTP-REGCM3','IES-CCLM','KNMI-RACMO2.2b','MPI-REMO','SMHI-RCA35','UC-WRF311','UCT-PRECIS','UQAM-CRCM5', 'ENS-models']

model_names=[]
for model_name in model_names0:
    model_names.append(model_name[0:3])

# Define climatological mean time series
obs_mean_clim = np.zeros([12, nsubregion])
obs_std_clim = np.zeros([12, nsubregion])
model_mean_clim = np.zeros([nmodel, 12, nsubregion])

# Calculate climatological mean time series
for imonth in np.arange(12):
    t_index = np.arange(18)*12+imonth
    obs_mean_clim[imonth,:] = np.mean(obs_mean[t_index,:], axis=0)*86400.
    obs_std_clim[imonth,:] = np.mean(obs_std[t_index,:], axis=0)*86400.
    model_mean_clim[:,imonth,:] = np.mean(model_mean[:,t_index,:], axis=1)*86400.

fig = plt.figure()

rcParams['xtick.labelsize'] = 4
rcParams['ytick.labelsize'] = 3.5

region = [1, 3, 6, 8, 10, 9, 11, 12, 13, 15]

y_min = np.repeat(0,10)
y_max = np.array([1.5, 1.0, 12, 12, 12, 6, 10, 10, 4, 4])

for ii in np.arange(10):
    ax=fig.add_subplot(10,2,ii+3)
    ax.plot(obs_mean_clim[:,region[ii]], color='r',lw=2) 
    #ax.plot(obs_mean_clim[:,region[ii]]+obs_std_clim[:,region[ii]], color='g',lw=3) 
    #ax.plot(obs_mean_clim[:,region[ii]]-obs_std_clim[:,region[ii]], color='g',lw=3) 
    ax.plot(model_mean_clim[-1, :,region[ii]], color='b',lw=2) 
    for imodel in np.arange(nmodel):
        ax.plot(model_mean_clim[imodel,:,region[ii]], linestyle='--',color='grey',lw=0.5) 
    ax.set_xlim([-0.2,11.2])
    ax.set_ylim([y_min[ii], y_max[ii]])
    ax.set_xticks(np.arange(12))
    ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
    ax.set_title('R%02d' %(region[ii]+1), fontsize=7)

# calculate the RMSE normalized by observed temporal standard deviation
rmse = np.zeros([nsubregion, nmodel])
for imodel in np.arange(nmodel):
    for isubregion in np.arange(nsubregion):
        rmse[isubregion, imodel] = np.sqrt(np.mean((model_mean_clim[imodel,:,isubregion]-obs_mean_clim[:,isubregion])**2.))/np.mean(obs_mean_clim[:,isubregion])
region_name = []
for isubregion in np.arange(nsubregion):
    region_name.append('R%02d' %(isubregion+1))

ax = fig.add_subplot(3,2,5)
ax, cs = plotter.draw_portrait_diagram(ax,rmse, collabels=model_names, rowlabels=region_name, ptitle='Normalized RMSE \n (% CRU annual clim.)', cmap='RdBu_r', clevs=np.arange(11)*0.1) 
cax = fig.add_axes([0.49, 0.1, 0.01, 0.18])
cbar = fig.colorbar(cs,cax=cax)

# calculate the correlation coefficient
corr = np.zeros([nsubregion, nmodel])
for imodel in np.arange(nmodel):
    for isubregion in np.arange(nsubregion):
        corr[isubregion, imodel] = np.corrcoef(model_mean_clim[imodel,:,isubregion],obs_mean_clim[:,isubregion])[0,1]

ax = fig.add_subplot(3,2,6)
ax, cs = plotter.draw_portrait_diagram(ax,corr, collabels=model_names, rowlabels=np.repeat(' ', nsubregion), ptitle='Correlation coefficient (r)', cmap='RdBu', clevs=np.arange(12)*0.05+0.45)
cax = fig.add_axes([0.91, 0.1, 0.01, 0.18])
cbar = fig.colorbar(cs,cax=cax)

plt.show()
fig.subplots_adjust(hspace=0.7)    
fig.savefig('Figure05',dpi=600,bbox_inches='tight')






