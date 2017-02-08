from mpl_toolkits.basemap import Basemap, cm
import matplotlib.pyplot as plt
import ocw.data_source.local as local
import numpy.ma as ma
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111)
m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
projection='lcc',lat_1=33,lat_2=45,lon_0=-95,resolution='c')
#m.drawcoastlines()
#m.drawstates()
#m.drawcountries()

m.bluemarble()

GPM_dataset = local.load_GPM_IMERG_files(file_path='./GPM_2015_summer/', filename_pattern=['*20150601*.HDF5'])
x,y = m(GPM_dataset.lons, GPM_dataset.lats)
m.contourf(x,y,np.sum(GPM_dataset.values, axis=0), levels=np.append(np.arange(50)+1,100), alpha=0.7)
m.colorbar(location='right')
plt.show()
fig.savefig('GPM_data_20150601',dpi=600,bbox_inches='tight')
