#!/usr/bin/env python

# Runs ProbtrackX on a list of subjects, by submitting jobs to the local
# scheduler. The job file to be run must be called probtrackx_job.sh
# (although this can be specified in the configuration file), and must set
# the scheduler arguments (queue, walltime, memory, etc.) and call run_probtrackx.py.
#
# Requires that BedpostX has already been run for all subjects.
#
# ROI networks and other parameters must be specified in a JSON file.

# Arguments:
# Arg1: configuration file

# Read a list of subjects, and submit parallel ProbtrackX jobs for each one
import csv
import sys
import os
import shutil
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline_state import resolve_correction_method
from run_probtrackx import load_config

cwd = os.getcwd()

config_file = sys.argv[1]

config = load_config(config_file)
config_gen = config['general']
config_ptx = config['probtrackx']
config_sched = config['scheduler']

subjects_file = config_gen['subjects_file']

deriv_dir = '{0}/{1}'.format(config_gen['root_dir'], config_gen['deriv_dir'])

temp_dir = config_gen['temp_dir']
if not os.path.isdir(temp_dir):
    os.makedirs(temp_dir)

subjects = []
with open(subjects_file) as subj_file:
    reader = csv.reader(subj_file)
    for row in reader:
        if len(row) > 0:
            subjects.append(row[0])

print('Processing probtrackx: found {0} subjects'.format(len(subjects)))

for subject in subjects:

    if config_gen['verbose']:
        print('Subject {0}...'.format(subject))

    # Subject-specific paths
    # Output is subject-first
    session = config_gen['session']
    if len(session) > 0:
        session = '{0}/'.format(session)
    dwi_dirname = config_gen.get('dwi_dirname', 'dwi')
    if dwi_dirname == 'auto':
        # Corrected-eddy pipeline: resolve dwi_topup/dwi_fugue/dwi_bdp the
        # same way run_probtrackx.py itself does -- this loop only needs it
        # to pre-create/clobber-check the right probtrackX output dir before
        # submitting, but getting it wrong here silently creates a stray
        # derived/$sess/dwi/probtrackX/... tree that doesn't match where the
        # actual job writes its output.
        sess_rel = '{0}{1}/{2}'.format(config_gen['prefix'], subject, session)
        method = resolve_correction_method(sess_rel)
        if method is None:
            print('No dwi_<method> dir found for auto-resolution [{0}], skipping'.format(subject))
            continue
        dwi_dirname = 'dwi_{0}'.format(method)
    subj_dir = '{0}/{1}{2}/{3}{4}' \
                        .format(deriv_dir, config_gen['prefix'], subject, session, dwi_dirname)
    probtrackx_dir = '{0}/probtrackX/{1}'.format(subj_dir, config_ptx['network_name'])

    # Remove existing directory if clobber
    if not os.path.isdir(probtrackx_dir):
        os.makedirs(probtrackx_dir)
    else:
        if config_gen['clobber']:
            shutil.rmtree(probtrackx_dir)
            os.makedirs(probtrackx_dir)

    # Submit to scheduler (unless otherwise specified)
    if config_sched['submit']:

        cmd = '{0} {1} {2} {3}'.format(config_sched['command'], config_ptx['job_script'], \
                                           subject, config_file)

        if config_gen['dryrun'] or config_gen['verbose']:
            print(cmd)

        if not config_gen['dryrun']:
            os.system(cmd)

    # Run in serial
    else:
        cmd = './{0} {1} {2}'.format(config_ptx['job_script'], \
                                         subject, config_file)
        if not config_gen['dryrun']:
            os.system(cmd)
        else:
            print(cmd)
