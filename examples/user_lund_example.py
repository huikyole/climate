import ast
import datetime
from operator import itemgetter
from os import path
import urllib

import numpy as np

from ocw.dataset import Dataset, Bounds
import ocw.data_source.local as local
import ocw.data_source.rcmed as rcmed
import ocw.dataset_processor as dsp
import ocw.evaluation as evaluation
import ocw.metrics as metrics
import ocw.plotter as plotter

CORDEX_AF_TAS = "AFRICA_UQAM-CRCM5_CTL_ERAINT_MM_50km_1989-2008_tas.nc"
CORDEX_AF_PR = "AFRICA_KNMI-RACMO2.2b_CTL_ERAINT_MM_50km_1989-2008_pr.nc"
EU_CORDEX_PR = "pr_EUR-44i_ECMWF-ERAINT_evaluation_r1i1p1_KNMI-RACMO22E_v1_mon.nc"
EU_CORDEX_TAS = "tas_EUR-44i_ECMWF-ERAINT_evaluation_r1i1p1_KNMI-RACMO22E_v1_mon.nc"
CRU_31_PR = "GLOBAL_CRU_CTL_CRU-TS31_MM_0.5deg_1980-2008_pr.nc"
CRU_31_TAS = "GLOBAL_CRU_CTL_CRU-TS31_MM_0.5deg_1980-2008_tas.nc"
TRMM_PR = "TRMM_199801-201212_pr.nc"

# Configure your evaluation here. The evaluation bounds are determined by
# the lat/lon/time values that are set here. If you set the lat/lon values
# outside of the range of the datasets' values you will get an error. Your
# start/end time values should be in 12 month intervals due to the metrics
# being used.
#
# Africa Evaluation Settings
ref_dataset = local.load_file(CORDEX_AF_TAS, "tas")
ref_dataset.name = "cordex_af_tas"
target_dataset = local.load_file(CRU_31_TAS, "tas")
target_dataset.name = "cru_31_tas"
LAT_MIN = -40
LAT_MAX = 40
LON_MIN = -20
LON_MAX = 55
START = datetime.datetime(1999, 1, 1)
END = datetime.datetime(2000, 12, 1)
SEASON_MONTH_START = 1
SEASON_MONTH_END = 12

EVAL_BOUNDS = Bounds(LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, START, END)

# Normalize the time values of our datasets so they fall on expected days
# of the month. For example, monthly data will be normalized so that:
# 15 Jan 2014, 15 Feb 2014 => 1 Jan 2014, 1 Feb 2014
ref_dataset = dsp.normalize_dataset_datetimes(ref_dataset, "monthly")
target_dataset = dsp.normalize_dataset_datetimes(target_dataset, "monthly")

# Subset down the evaluation datasets to our selected evaluation bounds.
target_dataset = dsp.subset(EVAL_BOUNDS, target_dataset)
ref_dataset = dsp.subset(EVAL_BOUNDS, ref_dataset)

# Do a monthly temporal rebin of the evaluation datasets.
target_dataset = dsp.temporal_rebin(target_dataset, datetime.timedelta(days=30))
ref_dataset = dsp.temporal_rebin(ref_dataset, datetime.timedelta(days=30))

# Spatially regrid onto a 1 degree lat/lon grid within our evaluation bounds.
new_lats = np.arange(LAT_MIN, LAT_MAX, 1.0)
new_lons = np.arange(LON_MIN, LON_MAX, 1.0)
target_dataset = dsp.spatial_regrid(target_dataset, new_lats, new_lons)
ref_dataset = dsp.spatial_regrid(ref_dataset, new_lats, new_lons)

# Load the datasets for the evaluation.
mean_bias = metrics.MeanBias()
# These versions of the metrics require seasonal bounds prior to running
# the metrics. You should set these values above in the evaluation
# configuration section.
spatial_std_dev_ratio = metrics.SeasonalSpatialStdDevRatio(month_start=SEASON_MONTH_START, month_end=SEASON_MONTH_END)
pattern_correlation = metrics.SeasonalPatternCorrelation(month_start=SEASON_MONTH_START, month_end=SEASON_MONTH_END)

# Create our example evaluation.
example_eval = evaluation.Evaluation(ref_dataset, # Reference dataset for the evaluation
                                    # 1 or more target datasets for the evaluation
                                    [target_dataset],
                                    # 1 ore more metrics to use in the evaluation
                                    [mean_bias, spatial_std_dev_ratio, pattern_correlation])
example_eval.run()

plotter.draw_contour_map(example_eval.results[0][0],
                         new_lats,
                         new_lons,
                         'lund_example_time_averaged_bias',
                         gridshape=(1, 1),
                         ptitle='Time Averaged Bias')

spatial_stddev_ratio = example_eval.results[0][1]
# Pattern correlation results are a tuple, so we need to index and grab
# the component we care about.
spatial_correlation = example_eval.results[0][2][0]
taylor_data = np.array([[spatial_stddev_ratio], [spatial_correlation]]).transpose()

plotter.draw_taylor_diagram(taylor_data,
                            [target_dataset.name],
                            ref_dataset.name,
                            fname='lund_example_taylor_diagram',
                            frameon=False)
