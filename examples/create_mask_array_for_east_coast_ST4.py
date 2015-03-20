import numpy as np
import numpy.ma as ma
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap
import ocw.data_source.local as local
import ocw.utils as utils

file_path = '/storage/users/obs4Mips/ST4_prec/'
filename_pattern = 'ST4.2006072019.01h.nc'
#dataset = local.load_files(file_path=file_path, filename_pattern=filename_pattern, variable_name='A_PCP_GDS5_SFC_acc1h',\
#                           lat_name='g5_lat_0',lon_name='g5_lon_1')
nc_file = Dataset(file_path + filename_pattern)
output_name = '/home/huikyole/work/Downscaling/shapefile_mask/stage4_CA_mask.nc'

lats = nc_file.variables['g5_lat_0'][:]
lons = nc_file.variables['g5_lon_1'][:]

map_read = Basemap()
map_read.readshapefile('./shape/usa_states','usa_states')

regions = []

# Search targeting regions

#target_states = ['DC','WV','DE','MD','PA','NJ','NY','NH','MA','CT','RI','ME','VT']
target_states = ['CA']

for iregion, region_info in enumerate(map_read.usa_states_info):
    if region_info['st'] in target_states:
        regions.append(map_read.usa_states[iregion])

# create a mask_array

mask_array = utils.mask_grid_points_using_shape_file(lats, lons, regions)

# Write the mask_array to a netCDF file    
out_file = Dataset(output_name, 'w', format='NETCDF4')

ny, nx = mask_array.shape

# Create attribute dimensions
lat_dim = out_file.createDimension('y', ny)
lon_dim = out_file.createDimension('x', nx)

# Create variables
out_file.createVariable('lat', 'f8', ('y','x'))
out_file.createVariable('lon', 'f8', ('y','x'))
value = out_file.createVariable('mask_east_coast', 'I4', ('y','x'))
value.description = 'Grid points inside the 13 East Coast states: 0, otherwise: 1. 13 States are ME, NH, VT, MA, NY, RI, CT, NH, DE, MD, PA, WV, DC'

out_file.variables['lat'][:]=lats
out_file.variables['lon'][:]=lons
out_file.variables['mask_east_coast'][:] = mask_array

out_file.close()

