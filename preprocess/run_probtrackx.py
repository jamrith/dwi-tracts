#!/usr/bin/env python

# Runs ProbtrackX on a single subject for all ROIs
# Requires that BedpostX has already been run for this subject
# ROIs and other parameters must be specified in a JSON file
# Use GPU version by setting "use_gpu": true in the probtrackx config section

# Command line arguments to this script:
# Arg1: subject ID
# Arg2: configuration file

import subprocess
import sys
import os
import csv
import shutil
from subprocess import Popen
import json

def main():

    global config

    with open(sys.argv[2], 'r') as myfile:
        json_string=myfile.read()

    config = json.loads(json_string)
    config_gen = config['general']

    is_dryrun = config_gen['dryrun']

    # Run this subject
    subject = sys.argv[1]

    append = ''
    if is_dryrun:
        append = ' [DRY RUN]'

    print('Processing subject {0}{1}'.format(subject, append))
    if probtrackx_subject(subject):
        print('Finished subject {0}'.format(subject))
    else:
        print('Subject {0} failed'.format(subject))

def run_fsl(cmd):

    global config
    is_dryrun = config['general']['dryrun']

    if is_dryrun:
        return ''
    else:
        sp = Popen(cmd, shell=True, stderr=subprocess.PIPE)
        out, err = sp.communicate()
        return err

def build_mask_command(fsl_bin, mask_paths, output_path):
    if not mask_paths:
        return None
    cmd = '{0}fslmaths {1}'.format(fsl_bin, mask_paths[0])
    for mask_path in mask_paths[1:]:
        cmd = '{0} -add {1}'.format(cmd, mask_path)
    return '{0} {1}'.format(cmd, output_path)

def generate_mask(fsl_bin, mask_paths, output_path, verbose, subject, label):
    cmd = build_mask_command(fsl_bin, mask_paths, output_path)
    if not cmd:
        return False
    if verbose:
        print(cmd)
    err = run_fsl(cmd)
    if err:
        print('\tError creating {0} mask [{1}]: {2}'.format(label, subject, err))
        return None
    return True

def run_probtrackx_for_seed(seed, targets, exclusion, net_dir, rois_dir,
                             bedpostx_dir, xfm_img, invxfm_img,
                             fsl_bin, probtrackx_cmd, config_ptx, config_gen, subject):
    """Run probtrackx from one seed toward a list of targets with given exclusions.

    Output goes to net_dir/seed/.
    Returns True on success, False/None on failure.
    """
    roi_file = '{0}/{1}.nii.gz'.format(rois_dir, seed)
    roi_list = '{0}/others_{1}.txt'.format(net_dir, seed)
    stop_img  = '{0}/{1}_others.nii.gz'.format(net_dir, seed)
    avoid_img = '{0}/{1}_avoid.nii.gz'.format(net_dir, seed)

    target_mask_paths = []
    with open(roi_list, 'w') as listout:
        for target in targets:
            mask_path = '{0}/{1}.nii.gz'.format(rois_dir, target)
            listout.write('{0}\n'.format(mask_path))
            target_mask_paths.append(mask_path)

    stop_built = generate_mask(fsl_bin, target_mask_paths, stop_img,
                               config_gen['verbose'], subject, 'stop')
    if stop_built is None:
        return False
    if not stop_built:
        print('\tNo targets for seed {0}; skipping.'.format(seed))
        return True

    exclusion_paths = ['{0}/exclusion_masks/{1}.nii.gz'.format(rois_dir, mask)
                       for mask in exclusion]
    avoid_built = generate_mask(fsl_bin, exclusion_paths, avoid_img,
                                config_gen['verbose'], subject, 'exclusion')
    if avoid_built is None:
        return False
    avoid_arg = ' --avoid={0}'.format(avoid_img) if avoid_built else ''

    cmd_pre = ('{0}{1} -V 0 --distthresh={2} --sampvox={3} --forcedir --opd --opathdir '
               '-x {4} -l --onewaycondition -c {5} --nsteps={6} --steplength={7} '
               '--nsamples={8} --fibthresh={9} --s2tastext '
               '--xfm={10} --invxfm={11} -s {12}/merged -m {12}/nodif_brain_mask'
               .format(fsl_bin, probtrackx_cmd,
                       config_ptx['distthresh'], config_ptx['sampvox'], roi_file,
                       config_ptx['cthr'], config_ptx['nsteps'], config_ptx['steplength'],
                       config_ptx['nsamples'], config_ptx['fibthresh'],
                       xfm_img, invxfm_img, bedpostx_dir))

    cmd_net = (' --stop={0}{2} -V 0 --waypoints={0} --waycond=OR --omatrix2 '
               ' --target2={0} --os2t --targetmasks={1} --otargetpaths'
               .format(stop_img, roi_list, avoid_arg))

    out_dir = '{0}/{1}'.format(net_dir, seed)

    for extra, label in [('', ''), (' -o FreeTracking', ' [FreeTracking]')]:
        cmd = '{0} {1} --pd --dir={2}{3}'.format(cmd_pre, cmd_net, out_dir, extra)
        if config_gen['verbose']:
            print(cmd)
        err = run_fsl(cmd)
        if err:
            print('\tError running ProbtrackX{0} [{1}]: {2}'.format(label, subject, err))
            return False

    print('\tDone tracking for seed {0} [{1}]'.format(seed, subject))

    cleanup = [stop_img, roi_list]
    if avoid_built:
        cleanup.append(avoid_img)
    for p in cleanup:
        try:
            os.remove(p)
        except OSError:
            pass

    return True


def probtrackx_subject(subject):
    # Generate probabilistic streamlines between all pairs of ROIs

    global config

    # Get configs
    config_gen = config['general']
    config_bpx = config['bedpostx']
    config_ptx = config['probtrackx']

    if config_gen['verbose']:
        print(config_ptx)

    fsl_bin = config_gen['fsl_bin']

    subj = '{0}{1}'.format(config_gen['prefix'], subject)

    deriv_dir = '{0}/{1}'.format(config_gen['root_dir'], config_gen['deriv_dir'])

    # Subject-specific paths
    session = config_gen['session']
    if len(session) > 0:
        session = '{0}/'.format(session)
    subj_dir = '{0}/{1}{2}/{3}dwi' \
                        .format(deriv_dir, config_gen['prefix'], subject, session)

    bedpostx_dir = '{0}/bedpostX'.format(subj_dir)
    bedpostx_done = os.path.exists('{0}/xfms/eye.mat'.format(bedpostx_dir))
    probtrackx_dir = '{0}/probtrackX/{1}'.format(subj_dir, config_ptx['network_name'])
    rois_dir = config_ptx['roi_dir']

    if not os.path.isdir(probtrackx_dir):
        os.makedirs(probtrackx_dir)
    else:
        if config_gen['clobber']:
            shutil.rmtree(probtrackx_dir)
            os.makedirs(probtrackx_dir)

    invxfm_img = '{0}/reg3G/FA_warp2Mean3G.nii.gz'.format(subj_dir)
    xfm_img = '{0}/reg3G/Mean3G_warp2FA.nii.gz'.format(subj_dir)

    # Check whether BedpostX output exists, otherwise fail
    if not bedpostx_done:
        print('No BedpostX output exists at {0}. Skipping subject. [{1}]'.format(bedpostx_dir, subject))
        return False

    if not os.path.exists(xfm_img) or not os.path.exists(invxfm_img):
        print('No warp images (reg3G) exist. Skipping subject. [{0}]'.format(subject))
        return False

    use_gpu = config_ptx.get('use_gpu', False)
    probtrackx_cmd = 'probtrackx2_gpu' if use_gpu else 'probtrackx2'

    # JSON network list: one probtrackx job per network entry per seed (bidirectional)
    if config_ptx['roi_list'].endswith('.json'):
        with open(config_ptx['roi_list'], 'r') as myfile:
            networks = json.loads(myfile.read())['networks']

        for net_idx, net in enumerate(networks):
            seeds     = net.get('seeds', [])
            targets   = net.get('targets', [])
            exclusion = net.get('exclusion', [])
            net_name  = net.get('name', 'net{0:02d}'.format(net_idx))

            # All networks share the same flat output dir so that downstream
            # scripts find {seed}/target_paths_{target}.nii.gz regardless of
            # which network produced them.  The net_name is used only for logging.
            net_dir = probtrackx_dir

            print('\tNetwork: {0}  seeds={1}  targets={2}'.format(
                net_name, seeds, targets))

            # Forward: each seed → all targets
            for seed in seeds:
                ok = run_probtrackx_for_seed(
                    seed, targets, exclusion, net_dir,
                    rois_dir, bedpostx_dir, xfm_img, invxfm_img,
                    fsl_bin, probtrackx_cmd, config_ptx, config_gen, subject)
                if not ok:
                    return False

            # Reverse: each target → all seeds
            for target in targets:
                ok = run_probtrackx_for_seed(
                    target, seeds, exclusion, net_dir,
                    rois_dir, bedpostx_dir, xfm_img, invxfm_img,
                    fsl_bin, probtrackx_cmd, config_ptx, config_gen, subject)
                if not ok:
                    return False

    # CSV ROI list: original all-pairs behaviour (no per-network exclusions)
    else:
        rois = []
        with open(config_ptx['roi_list'], 'r') as roi_file:
            reader = csv.reader(roi_file)
            for row in reader:
                rois.append(row[0])

        net_dir = probtrackx_dir
        for roi in rois:
            targets = [r for r in rois if r != roi]
            ok = run_probtrackx_for_seed(
                roi, targets, [], net_dir,
                rois_dir, bedpostx_dir, xfm_img, invxfm_img,
                fsl_bin, probtrackx_cmd, config_ptx, config_gen, subject)
            if not ok:
                return False

    return True

if __name__ == '__main__':
    main()
