#!/usr/bin/env python
# coding: utf-8

import json
import sys
from dwitracts.main import DwiTracts, process_tsa_subject
import dwitracts.plot as plot

if len(sys.argv) < 3:
    print('Usage: run_single_subject_tsa.py <config_file> <subject_id>')
    sys.exit(1)

config_file = sys.argv[1]
subject     = sys.argv[2]

print('Config:  {0}'.format(config_file))
print('Subject: {0}'.format(subject))

with open(config_file, 'r') as f:
    params = json.load(f)

verbose = params['general']['verbose']
clobber = params['general']['clobber']
debug   = params['general']['debug']

# Initialize with the full subjects list so tract templates are loaded correctly
my_dwi = DwiTracts(params)
print('Initialising...')
assert my_dwi.initialize()

# Run TSA regression for the single subject only
print('Running TSA regression for subject {0}...'.format(subject))
failures = process_tsa_subject(subject, my_dwi, verbose=verbose, debug=debug)
if failures < 0:
    print('ERROR: process_tsa_subject returned {0}. Check DWI inputs for {1}.'.format(failures, subject))
    sys.exit(1)
elif failures > 0:
    print('WARNING: {0} tract failure(s) for subject {1}.'.format(failures, subject))
else:
    print('TSA regression complete for {0}.'.format(subject))

# Regenerate group mean TSA images (loads existing per-subject betas — fast)
print('Regenerating mean TSA images across all subjects...')
assert my_dwi.generate_mean_tsa_images(verbose=verbose)

# Replot TSA histograms and save stats CSVs
params_plot = {}
params_plot['axis_font']          = 18
params_plot['ticklabel_font']     = 12
params_plot['title_font']         = 18
params_plot['show_labels']        = False
params_plot['show_title']         = False
params_plot['dimensions_tracts']  = (50, 40)
params_plot['dimensions_all']     = (50, 40)
params_plot['dpi_tracts']         = 150
params_plot['dpi_all']            = 300
params_plot['num_bins']           = 20
params_plot['kde']                = True
params_plot['stat']               = 'density'
params_plot['xlim']               = [-0.25, 0.75]
params_plot['xticks']             = [-0.25, 0.0, 0.25, 0.50, 0.75]
params_plot['color']              = '#2b3ad1'
params_plot['image_format']       = 'pdf'

print('Plotting TSA histograms and saving stats...')
for threshold in [0.1, 0.3, 0.5]:
    stats = plot.plot_tsa_histograms(params_plot, my_dwi, tract_names=None,
                                     threshold=threshold, verbose=verbose, clobber=clobber)
    out_csv = '{0}/stats_tsa_{1:02d}.csv'.format(my_dwi.tracts_dir, round(threshold * 100))
    stats.to_csv(out_csv)
    print('  Saved {0}'.format(out_csv))

print('Done.')
