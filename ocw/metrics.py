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
    Metric - Abstract Base Class from which all metrics must inherit.
'''

from abc import ABCMeta, abstractmethod
import ocw.utils as utils
import numpy
from scipy import stats

class Metric(object):
    '''Base Metric Class'''
    __metaclass__ = ABCMeta


class UnaryMetric(Metric):
    '''Abstract Base Class from which all unary metrics inherit.'''
    __metaclass__ = ABCMeta

    @abstractmethod
    def run(self, target_dataset):
        '''Run the metric for a given target dataset.

        :param target_dataset: The dataset on which the current metric will
            be run.
        :type target_dataset: ocw.dataset.Dataset object

        :returns: The result of evaluating the metric on the target_dataset.
        '''


class BinaryMetric(Metric):
    '''Abstract Base Class from which all binary metrics inherit.'''
    __metaclass__ = ABCMeta

    @abstractmethod
    def run(self, ref_dataset, target_dataset):
        '''Run the metric for the given reference and target datasets.

        :param ref_dataset: The Dataset to use as the reference dataset when
            running the evaluation.
        :type ref_dataset: ocw.dataset.Dataset object
        :param target_dataset: The Dataset to use as the target dataset when
            running the evaluation.
        :type target_dataset: ocw.dataset.Dataset object

        :returns: The result of evaluation the metric on the reference and 
            target dataset.
        '''


class Bias(BinaryMetric):
    '''Calculate the bias between a reference and target dataset.'''

    def run(self, ref_dataset, target_dataset):
        '''Calculate the bias between a reference and target dataset.

        .. note::
           Overrides BinaryMetric.run()

        :param ref_dataset: The reference dataset to use in this metric run.
        :type ref_dataset: ocw.dataset.Dataset object
        :param target_dataset: The target dataset to evaluate against the
            reference dataset in this metric run.
        :type target_dataset: ocw.dataset.Dataset object

        :returns: The difference between the reference and target datasets.
        :rtype: Numpy Array
        '''
        return ref_dataset.values - target_dataset.values


class TemporalStdDev(UnaryMetric):
    '''Calculate the standard deviation over the time.'''

    def run(self, target_dataset):
        '''Calculate the temporal std. dev. for a datasets.

        .. note::
           Overrides UnaryMetric.run()

        :param target_dataset: The target_dataset on which to calculate the 
            temporal standard deviation.
        :type target_dataset: ocw.dataset.Dataset object

        :returns: The temporal standard deviation of the target dataset
        :rtype: Numpy Array
        '''
        return target_dataset.values.std(axis=0, ddof=1)


class StdDevRatio(BinaryMetric):
    '''Calculate the standard deviation ratio between two datasets.'''

    def run(self, ref_dataset, target_dataset):
        '''Calculate the standard deviation ratio.

        .. note::
            Overrides BinaryMetric.run()

        :param ref_dataset: The reference dataset to use in this metric run.
        :type ref_dataset: ocw.dataset.Dataset object
        :param target_dataset: The target dataset to evaluate against the
            reference dataset in this metric run.
        :type target_dataset: ocw.dataset.Dataset object

        :returns: The standard deviation ratio of the reference and target
        '''
        return target_dataset.values.std() / ref_dataset.values.std()


class PatternCorrelation(BinaryMetric):
    '''Calculate the correlation coefficient between two datasets'''

    def run(self, ref_dataset, target_dataset):
        '''Calculate the correlation coefficient between two dataset.

        .. note::
           Overrides BinaryMetric.run()

        :param ref_dataset: The reference dataset to use in this metric run.
        :type ref_dataset: ocw.dataset.Dataset object
        :param target_dataset: The target dataset to evaluate against the
            reference dataset in this metric run.
        :type target_dataset: ocw.dataset.Dataset object

        :returns: The correlation coefficient between a reference and target dataset.
        '''
        # stats.pearsonr returns correlation_coefficient, 2-tailed p-value
        # We only care about the correlation coefficient
        # Docs at http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
        return stats.pearsonr(ref_dataset.values.flatten(), target_dataset.values.flatten())[0]


class MeanBias(BinaryMetric):
    '''Calculate the bias averaged over time.'''

    def run(self, ref_dataset, target_dataset, absolute=False):
        '''Calculate the bias averaged over time.

        .. note::
           Overrides BinaryMetric.run()

        :param ref_dataset: The reference dataset to use in this metric run.
        :type ref_dataset: ocw.dataset.Dataset object
        :param target_dataset: The target dataset to evaluate against the
            reference dataset in this metric run.
        :type target_dataset: ocw.dataset.Dataset object

        :returns: The mean bias between a reference and target dataset over time.
        '''

        diff = ref_dataset.values - target_dataset.values
        if absolute:
            diff = abs(diff)
        mean_bias = diff.mean(axis=0)

        return mean_bias
