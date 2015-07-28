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

import datetime
import urllib
from os import path

import numpy as np

import ocw.data_source.local as local
import ocw.data_source.rcmed as rcmed
from ocw.dataset import Bounds as Bounds
import ocw.dataset_processor as dsp
import ocw.evaluation as evaluation
import ocw.metrics as metrics
import ocw.plotter as plotter
import ocw.utils as utils

# Season 
# ex) MONTH_START =1 & MONTH_END=12 : annual mean bias
# ex) MONTH_START =12 & MONTH_END=2 : DJF mean bias
MONTH_START = 1
MONTH_END = 12
# Two Local Files 
MODEL = "/nas/share4-cf/jinwonki/data/cordex-af/AFRICA_IES-CCLM_CTL_ERAINT_MM_50km_1989-2008_pr.nc"
CRU="/nas/share1-hp/jinwonki/data/obs/cru3.1/d/GLOBAL_CRU_CTL_CRU-TS31_MM_0.5deg_1901-2009_pr.nc"  # reference observation
# Filename for the output image/plot (without file extension)
OUTPUT_PLOT = "cru_31_pr_knmi_africa_bias_annual"

# Download necessary NetCDF file if not present
#if path.exists(MODEL):
#    pass
#else:
#    urllib.urlretrieve(FILE_LEADER + MODEL, MODEL)

""" Step 1: Load Local NetCDF File into OCW Dataset Objects """
print("Loading %s into an OCW Dataset Object" % (MODEL,))
knmi_dataset = local.load_file(MODEL, "pr")
print("KNMI_Dataset.values shape: (times, lats, lons) - %s \n" % (knmi_dataset.values.shape,))
cru31_dataset = local.load_file(CRU, "pr")
print("CRU_Dataset.values shape: (times, lats, lons) - %s \n" % (knmi_dataset.values.shape,))


'''
""" Step 2: Fetch an OCW Dataset Object from the data_source.rcmed module """
print("Working with the rcmed interface to get CRU3.1 Daily-Max Temp")
metadata = rcmed.get_parameters_metadata()

cru_31 = [m for m in metadata if m['parameter_id'] == "39"][0]

""" The RCMED API uses the following function to query, subset and return the 
raw data from the database:

rcmed.parameter_dataset(dataset_id, parameter_id, min_lat, max_lat, min_lon, 
                        max_lon, start_time, end_time)

The first two required params are in the cru_31 variable we defined earlier
"""
# Must cast to int since the rcmed api requires ints
dataset_id = int(cru_31['dataset_id'])
parameter_id = int(cru_31['parameter_id'])
'''

print("We are going to use the Model to constrain the Spatial Domain")
#  The spatial_boundaries() function returns the spatial extent of the dataset
print("The KNMI_Dataset spatial bounds (min_lat, max_lat, min_lon, max_lon) are: \n"
      "%s\n" % (knmi_dataset.spatial_boundaries(), ))
print("The KNMI_Dataset spatial resolution (lat_resolution, lon_resolution) is: \n"
      "%s\n\n" % (knmi_dataset.spatial_resolution(), ))
min_lat, max_lat, min_lon, max_lon = knmi_dataset.spatial_boundaries()

print("Calculating the Maximum Overlap in Time for the datasets")

cru31_dataset = dsp.normalize_dataset_datetimes(cru31_dataset, 'monthly')
knmi_dataset = dsp.normalize_dataset_datetimes(knmi_dataset, 'monthly')
cru_start, cru_end = cru31_dataset.time_range()
knmi_start, knmi_end = knmi_dataset.time_range()
# Grab the Max Start Time
start_time = max([cru_start, knmi_start])
# Grab the Min End Time
end_time = min([cru_end, knmi_end])
print("Overlap computed to be: %s to %s" % (start_time.strftime("%Y-%m"),
                                          end_time.strftime("%Y-%m")))


""" Step 3: Resample Datasets so they are the same shape """
print("CRU31_Dataset.values shape: (times, lats, lons) - %s" % (cru31_dataset.values.shape,))
print("KNMI_Dataset.values shape: (times, lats, lons) - %s" % (knmi_dataset.values.shape,))

# Create a Bounds object to use for subsetting
new_bounds = Bounds(min_lat, max_lat, min_lon, max_lon, start_time, end_time)
cru31_dataset = dsp.subset(new_bounds, cru31_dataset)
knmi_dataset = dsp.subset(new_bounds, knmi_dataset)

# Temporaly subset both observation and model datasets
cru31_dataset = dsp.temporal_subset(MONTH_START, MONTH_END, cru31_dataset)
knmi_dataset = dsp.temporal_subset(MONTH_START, MONTH_END, knmi_dataset)

print("CRU31_Dataset.values shape: (times, lats, lons) - %s" % (cru31_dataset.values.shape,))
print("KNMI_Dataset.values shape: (times, lats, lons) - %s \n" % (knmi_dataset.values.shape,))

print("Temporally Rebinning the Datasets to a Single Timestep")
# Temporaly subset both observation and model datasets
cru31_dataset = dsp.temporal_subset(MONTH_START, MONTH_END, cru31_dataset)
knmi_dataset = dsp.temporal_subset(MONTH_START, MONTH_END, knmi_dataset)

print("KNMI_Dataset.values shape: %s" % (knmi_dataset.values.shape,))
print("CRU31_Dataset.values shape: %s \n\n" % (cru31_dataset.values.shape,))
 
""" Spatially Regrid the Dataset Objects to a 1/2 degree grid """
# Using the bounds we will create a new set of lats and lons on 0.5 degree step
new_lons = np.arange(min_lon, max_lon, 0.5)
new_lats = np.arange(min_lat, max_lat, 0.5)
 
# Spatially regrid datasets using the new_lats, new_lons numpy arrays
print("Spatially Regridding the KNMI_Dataset...")
knmi_dataset = dsp.spatial_regrid(knmi_dataset, new_lats, new_lons)
print("Spatially Regridding the CRU31_Dataset...")
cru31_dataset = dsp.spatial_regrid(cru31_dataset, new_lats, new_lons)
print("Final shape of the KNMI_Dataset:%s" % (knmi_dataset.values.shape, ))
print("Final shape of the CRU31_Dataset:%s" % (cru31_dataset.values.shape, ))
 
""" Step 5: Checking grid pointw with missing data from observation and models."""
cru31_dataset, knmi_dataset = dsp.mask_missing_data([cru31_dataset, knmi_dataset])

""" Step 6: Calculate seasonal mean climatology """
cru31_dataset.values = utils.calc_temporal_mean(cru31_dataset)
knmi_dataset.values = utils.calc_temporal_mean(knmi_dataset)

""" Step 7: Checking and converting variable units """
cru31_dataset = dsp.variable_unit_conversion(cru31_dataset)
knmi_dataset = dsp.variable_unit_conversion(knmi_dataset)

""" Step 8:  Build a Metric to use for Evaluation - Bias for this example """
# You can build your own metrics, but OCW also ships with some common metrics
print("Setting up a Bias metric to use for evaluation")
bias = metrics.Bias()

""" Step 9: Create an Evaluation Object using Datasets and our Metric """
# The Evaluation Class Signature is:
# Evaluation(reference, targets, metrics, subregions=None)
# Evaluation can take in multiple targets and metrics, so we need to convert
# our examples into Python lists.  Evaluation will iterate over the lists
print("Making the Evaluation definition")
bias_evaluation = evaluation.Evaluation(knmi_dataset, [cru31_dataset], [bias])
print("Executing the Evaluation using the object's run() method")
bias_evaluation.run()
 
""" Step 10: Make a Plot from the Evaluation.results """
# The Evaluation.results are a set of nested lists to support many different
# possible Evaluation scenarios.
#
# The Evaluation results docs say:
# The shape of results is (num_metrics, num_target_datasets) if no subregion
# Accessing the actual results when we have used 1 metric and 1 dataset is
# done this way:
print("Accessing the Results of the Evaluation run")
results = bias_evaluation.results[0][0]
 
# From the bias output I want to make a Contour Map of the region
print("Generating a contour map using ocw.plotter.draw_contour_map()")
 
lats = new_lats
lons = new_lons
fname = OUTPUT_PLOT
gridshape = (1, 1)  # Using a 1 x 1 since we have a single Bias for the full time range
plot_title = "PR Bias of KNMI Compared to CRU 3.1 (%s - %s)" % (start_time.strftime("%Y"), end_time.strftime("%Y"))
sub_titles = ["start month %02d - end month %02d" %(MONTH_START, MONTH_END)]
 
plotter.draw_contour_map(results, lats, lons, fname,
                         gridshape=gridshape, ptitle=plot_title, 
                         subtitles=sub_titles)
