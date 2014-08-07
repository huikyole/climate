# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import re
import netCDF4
from ocw.dataset import Dataset
import numpy
import numpy.ma as ma
from datetime import timedelta ,datetime
import calendar
import string

import ocw.utils as utils

LAT_NAMES = ['x', 'rlat', 'rlats', 'lat', 'lats', 'latitude', 'latitudes']
LON_NAMES = ['y', 'rlon', 'rlons', 'lon', 'lons', 'longitude', 'longitudes']
TIME_NAMES = ['time', 'times', 'date', 'dates', 'julian']


def _get_netcdf_variable_name(valid_var_names, netcdf, netcdf_var):
    '''Return valid variable from given netCDF object.

    Looks for an occurrence of a valid_var_name in the netcdf variable data.
    If multiple possible matches are found a ValueError is raised. If no
    matching variable names are found a Value is raised.

    :param valid_var_names: The possible variable names to search for in 
        the netCDF object.
    :type valid_var_names: List of Strings
    :param netcdf: The netCDF object in which to check for valid_var_names.
    :type netcdf: netcdf4.Dataset
    :param netcdf_var: The relevant variable name to search over in the 
        netcdf object.

    :returns: The variable from valid_var_names that it locates in 
        the netCDF object.

    :raises: ValueError
    '''

    # Check for valid variable names in netCDF value variable dimensions
    dimensions = netcdf.variables[netcdf_var].dimensions
    dims_lower = [dim.encode().lower() for dim in dimensions]

    intersect = list(set(valid_var_names).intersection(dims_lower))

    if len(intersect) == 1:
        index = dims_lower.index(intersect[0])
        dimension_name = dimensions[index].encode()

        possible_vars = []
        for var in netcdf.variables.keys():
            var_dimensions = netcdf.variables[var].dimensions

            if len(var_dimensions) != 1:
                continue

            if var_dimensions[0].encode() == dimension_name:
                possible_vars.append(var)

        if len(possible_vars) == 1:
            return possible_vars[0]

    # Check for valid variable names in netCDF variable names
    variables = netcdf.variables.keys()
    vars_lower = [var.encode().lower() for var in variables]

    intersect = list(set(valid_var_names).intersection(vars_lower))

    if len(intersect) == 1:
        index = vars_lower.index(intersect[0])
        return variables[index]

    # If we couldn't find a single matching valid variable name, we're
    # unable to load the file properly.
    error = (
        "Unable to locate a single matching variable name in NetCDF object. "
    )
    raise ValueError(error)


def _get_lat_name(variable_names):
    '''Find the latitude variable name

    :param variable_names: List of netCDF variables' name
    :type variable_names: List

    :returns: Latitude variable name
    :rtype: String
    '''

    common_name = set(['lat', 'lats', 'latitude', 'latitudes']).intersection(variable_names)
    if len(common_name) !=1:
        err = "Unable to autodetect latitude variable name."
        raise ValueError(err)
    else:
        lat_variable_name = common_name.pop()

    return lat_variable_name


def _get_lon_name(variable_names):
    '''Find the longitude variable name

    :param variable_names: List of netCDF variables' name
    :type variable_names: List

    :returns: Longitude variable name
    :rtype: String
    '''

    common_name = set(['lon', 'lons', 'longitude', 'longitudes']).intersection(variable_names)
    if len(common_name) !=1:
        err = "Unable to autodetect longitude variable name."
        raise ValueError(err)
    else:
        lon_variable_name = common_name.pop()

    return lon_variable_name


def _get_time_name(variable_names):
    '''Find the time variable name.

    :param: variableNameList: List of netCDF variables' name
    :type variable_names: List

    :returns: Time variable name
    :rtype: String
    '''

    common_name = set(['time', 'times', 'date', 'dates', 'julian']).intersection(variable_names)

    if len(common_name) !=1:
        err = "Unable to autodetect time variable name. These option(s) found: {0} ".format([each for each in common_name])
        raise ValueError(err)
    else:
        time_variable_name = common_name.pop()

    return time_variable_name


def _get_level_name(variable_names):
    '''Find the level variable name.

    :param variable_names: List of netCDF variables' name
    type variable_names: List

    :returns: Level variable name
    :rtype: String
    '''

    level_variable_name = None
    common_name = set(['lev', 'level', 'levels', 'height', 'heights', 'elev', 'elevation', 'elevations']).intersection(variable_names)
    if len(common_name) !=1:
        pass
    else:
        level_variable_name = common_name.pop()

    return level_variable_name


def _get_value_name(possible_value_name):
    '''Find the value variable name.

    :param possible_value_name: List of all value variable names
    :type possible_value_name: List

    :returns: Value variable name
    :rtype: String
    '''

    if len(possible_value_name) == 1:
        value_variable_name = possible_value_name[0]
    else:
        err = "The given value variable name does not match with existing variables name."
        raise ValueError(err)

    return value_variable_name


def load_file(file_path, variable_name):
    '''Load netCDF file, get the all variables name and get the data.

    :param file_path: NetCDF directory with file name
    :type file_path: String
    :param variable_name: The given (by user) value variable name
    :type variable_name: String

    :returns: An OCW Dataset object containing the requested parameter data.
    :rtype: ocw.dataset.Dataset object

    :raises: ValueError
    '''

    try:
        netcdf = netCDF4.Dataset(file_path, mode='r')
    except:
        err = "The given file cannot be loaded (Only netCDF file can be supported)."
        raise ValueError(err)

    lat_name = _get_netcdf_variable_name(LAT_NAMES, netcdf, variable_name)
    lon_name = _get_netcdf_variable_name(LON_NAMES, netcdf, variable_name)
    time_name = _get_netcdf_variable_name(TIME_NAMES, netcdf, variable_name)

    #lat_variable_name = _get_lat_name(variable_names)
    #lon_variable_name = _get_lon_name(variable_names)
    #time_variable_name = _get_time_name(variable_names)
    #level_variable_name = _get_level_name(variable_names)


    # Check returned variable dimensions. lats, lons, and times should be 1D
    #
    # Check dimensions of the values
    # if != 3
    #   find the indices for lat, lon, time
    #   strip out everything else by select 1st of possible options
    #
    # Check the order of the variables
    # if not correct order (times, lats, lons)
    #    reorder as appropriate
    #
    # Make new dataset object

    '''
    if variable_name in variable_names:
        value_variable_name = variable_name
    else:
        possible_value_name = list(set(variable_names) - set([lat_variable_name, lon_variable_name, time_variable_name, level_variable_name]))
        value_variable_name = _get_value_name(possible_value_name)
    '''
    lats = netcdf.variables[lat_name][:]    
    lons = netcdf.variables[lon_name][:]
    time_raw_values = netcdf.variables[time_name][:]
    times = utils.decode_time_values(netcdf, time_name)
    times = numpy.array(times)
    values = ma.array(netcdf.variables[variable_name][:])


    if len(values.shape) == 4:
        #value_dimensions_names = list(netcdf.variables[variable_name].dimensions)
        value_dimensions_names = [dim_name.encode() for dim_name in netcdf.variables[variable_name].dimensions]
        lat_lon_time_var_names = [lat_name, lon_name, time_name]
        level_index = value_dimensions_names.index(list(set(value_dimensions_names) - set(lat_lon_time_var_names))[0])
        if level_index == 0:
            values = values [0,:,:,:]
        elif level_index == 1:
            values = values [:,0,:,:]
        elif level_index == 2:
            values = values [:,:,0,:]
        else:
            values = values [:,:,:,0]

    return Dataset(lats, lons, times, values, variable_name)
