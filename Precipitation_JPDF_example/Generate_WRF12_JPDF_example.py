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

from ocw.dataset import Bounds
import ocw.data_source.local as local
import ocw.dataset_processor as dsp
import ocw.metrics as metrics
from draw_JPDF import draw_JPDF

import numpy as np
import numpy.ma as ma
from pickle import load, dump

"""
This is an example of calculating the joint probability distribution function of rainfall intensity and duration for the Northern Great Plains using NU-WRF simulation with 24-km resolution for 06/01/2002
""" 

""" Step 1: Load the NU-WRF datasets """

WRF_dataset = local.load_WRF_2d_files_RAIN(file_path='./data_WRF12/', filename_pattern=['wrf2dout*'])

""" Step 2: Load the spatial filter (Bukovsky region mask) """

Bukovsky_mask = Bounds(boundary_type='user',
                                    user_mask_file='Bukovsky_regions.nc',
                                    mask_variable_name='Bukovsky',
                                    longitude_name='lon',
                                    latitude_name='lat')  

""" Step 3: Spatial subset the WRF data (for Northern Great Plains, user_mask_values=[10]) """

WRF_dataset_filtered = dsp.subset(WRF_dataset, Bukovsky_mask, user_mask_values=[10])

""" Step 4: Analyze the wet spells """

spell_duration, peak_rainfall, total_rainfall = metrics.wet_spell_analysis(WRF_dataset_filtered.values, threshold=0.1, nyear=1, dt=0.5)

""" Step 5: Calculate the joint PDF(JPDF) of spell_duration and peak_rainfall """

histo2d = metrics.calc_joint_histogram(data_array1=spell_duration, data_array2=peak_rainfall, 
                                       bins_for_data1=np.append(np.arange(25)+0.5,[48.5,120.5]),
                                       bins_for_data2=[0.1,0.2,0.5,1.0,2.0,5.0,10,20,30]) 

histo2d = histo2d/np.sum(histo2d)*100.

""" Step 6: Save and visualize the JPDF """

dump(histo2d, open('WRF12_JPDF_example.pickle','wb'))

plot_level = np.array([0., 0.01,0.05, 0.1, 0.2, 0.5,1,2,3,5,10,25])
draw_JPDF(plot_data=np.transpose(histo2d), plot_level=plot_level, titles='', 
          x_ticks=[0.5, 4.5, 9.5, 14.5, 19.5, 23.5], x_names=['1','5','10','15','20','24'],
          y_ticks=np.arange(9), y_names=['0.1','0.2','0.5','1.0','2.0','5.0','10','20','30'], 
          output_file='WRF12_JPDF_example')
