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

'''
Classes:
    DatasetLoader - Generate OCW Dataset objects from a variety of sources.
'''

import ocw.data_source.local as local
import ocw.data_source.esgf as esgf
import ocw.data_source.rcmed as rcmed
import ocw.data_source.dap as dap

import types

class DatasetLoader:
    '''Generate OCW Dataset objects from a variety of sources.'''

    def __init__(self, loader_names, loader_options1, loader_options2, loader_options3, data_types='obs'):

        '''Generate a list of N OCW Dataset objects, [dataset1, dataset2, ..., datasetN], from a variety of sources.
        
        loader_options information for the loader in loader_names in dictionary
        form. For example:
        `` 
        >>> loader_names = ['rcmed','load_file']
        >>> loader_options1 = {'data_source':'rcmed', 'name':'cru', 'dataset_id':10, 'parameter_id':34}
        >>> loader_options2 = {'path': './data/TRMM_v7_3B43_1980-2010.nc,'variable':'pcp'}
        >>> data_types = 'obs'
        >>> obs_data_loaders = DatasetLoader(loader_names, loader_options1, loader_options2, data_types)
        ``

        As shown in the first example, the dictionary for each keyword argument
        should contain a data source and parameters specific to the loader for
        that data source. Once the configuration is entered, the datasets may be
        loaded using:
        ``
        >>> obs_data_loader.load_datasets()
        >>> obs_datasets = obs_data_loader.datasets

        The loader_names can be module names in ocw/data_source/local.py or esgf, rcmed and dap.
        * ``'load_file'`` - A single dataset file in a local directory
        * ``'local_dataset_from_multiple_netcdf_files'`` - A single dataset split accross multiple files in a
                              local directory
        * ``'local_multiple_files'`` - Multiple datasets in a local directory
        * ``'esgf'`` - Download the dataset from the Earth System Grid
                       Federation
        * ``'rcmed'`` - Download the dataset from the Regional Climate Model
                        Evaluation System Database
        * ``'dap'`` - Download the dataset from an OPeNDAP URL

        Users who wish to download datasets from sources not described above
        may define their own custom dataset loader function and incorporate it
        as follows:
        >>> loader.add_source_loader('my_source_name', my_loader_func)

        :param data_types: (Optional) information about the loaded datasets, 'obs' or 'model'
        :type data_types: :mod:`string`

        :raises KeyError: If an invalid argument is passed to a data source
        loader function.
        '''
        self.loader_names = loader_names
        self.num_of_loaders = len(loader_names)
        self.loader_options = []
        for ii in np.arange(self.num_of_loaders):
            self.loader_options.append(locals()['loader_options%d' %(ii+1)]) 

    def add_source_loader(self, loader_name, loader_option):
        '''
        Add a custom source loader.

        :param loader_name: Reference to a custom defined function.
        :type source_name: :mod:`string`

        :param loader_option: Arguments for the loader_name
        :type loader_option: :mod:`dict`
        '''
        self.loader_names.append(loader_name)
        self.loader_options.append(loader_option)

    def load_datasets(self):
        '''
        Loads the datasets from the given loader configurations.
        '''
        # Load the datasets
        self.datasets= []
        for ii in np.arange(self.num_of_loaders):
            objects = self._load(self.loader_names[ii], self.loader_options[ii]) 
            if type(objects) == types.InstanceType:
                self.datasets.append(objects)
            if type(objects) == types.ListType:
                self.datasets.extend(i for i in objects)

    def _load(self, loader_name, loader_option):
        '''
        Generic dataset loading method.
        '''
        # Find the correct loader function for the given data source
        if loader_name == 'esgf':
            loader_func = getattr(esgf, 'load_dataset')
        elif loader_name == 'rcmed':
            loader_func = getattr(rcmed, 'parameter_dataset')
        else:      
            loader_func = getattr(local, loader_name)

        # The remaining kwargs should be specific to the loader
        output = loader_func(**loader_option)

        return output
