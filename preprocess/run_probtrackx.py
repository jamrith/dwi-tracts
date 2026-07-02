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


# -- Native-ROI helpers -------------------------------------------------------

def roi_name(entry):
    """Return the ROI name from a network entry (plain string or native-ROI dict)."""
    return entry['roi'] if isinstance(entry, dict) else entry

def roi_space(entry):
    """Return 'template' (default) or 'native'."""
    if isinstance(entry, dict):
        return entry.get('space', 'template')
    return 'template'

def _crop_anterior(mask_path, fraction):
    """Zero out the posterior (1-fraction) of occupied slices in the slab (dim 2).

    Determines anterior direction from the sign of affine[1, 2] (Y/A-P component
    of the through-slab axis): negative means increasing slice index is more
    posterior, so we keep the first `fraction` of occupied slices.
    """
    import nibabel as nib
    import numpy as np

    img  = nib.load(mask_path)
    data = img.get_fdata().copy()

    occupied = np.any(data > 0, axis=(0, 1))
    slices   = np.where(occupied)[0]
    if len(slices) == 0:
        return

    first, last = int(slices[0]), int(slices[-1])
    n_keep = int(np.ceil((last - first + 1) * fraction))

    if img.affine[1, 2] < 0:
        # increasing slice index = more posterior: keep first n_keep slices
        data[:, :, first + n_keep:] = 0
    else:
        # increasing slice index = more anterior: keep last n_keep slices
        data[:, :, :last - n_keep + 1] = 0

    nib.save(nib.Nifti1Image(data, img.affine, img.header), mask_path)


def make_native_roi_in_template(entry, name, subj_dwi_dir, native_roi_dir,
                                 template_ref, fsl_bin, config_gen):
    """Warp an ASHS label into Mean3G template space and return the output path.

    Chain (all transforms pre-existing per subject):
      ASHS seg (T2/tse space)
        --(inverse of mind/s0_to_t2.mat)-->  DWI space   [flirt, NN]
        --(reg3G/FA_warp2Mean3G.nii.gz)   -->  Mean3G     [applywarp, NN]

    Returns the output path on success, None on failure.
    """
    verbose     = config_gen['verbose']
    ashs_side   = entry['ashs_side']
    labels      = entry['ashs_labels']
    anterior_fr = entry.get('anterior_fraction', None)

    subj_root = os.path.dirname(subj_dwi_dir)
    seg_file  = '{0}/ashs/final/ashs_{1}_lfseg_corr_usegray.nii.gz'.format(subj_root, ashs_side)
    s0_to_t2  = '{0}/mind/s0_to_t2.mat'.format(subj_root)
    fa_ref    = '{0}/dti_FA.nii.gz'.format(subj_dwi_dir)
    fa_warp   = '{0}/reg3G/FA_warp2Mean3G.nii.gz'.format(subj_dwi_dir)

    # Cached inversion of s0_to_t2 (shared across all native ROIs for this subject)
    t2_to_s0  = '{0}/t2_to_s0.mat'.format(native_roi_dir)
    tmp_t2    = '{0}/{1}_t2.nii.gz'.format(native_roi_dir, name)
    tmp_dwi   = '{0}/{1}_dwi.nii.gz'.format(native_roi_dir, name)
    out_path  = '{0}/{1}.nii.gz'.format(native_roi_dir, name)

    for f in [seg_file, s0_to_t2, fa_ref, fa_warp, template_ref]:
        if not os.path.isfile(f):
            print('\tNative ROI {0}: missing input {1}'.format(name, f))
            return None

    # Step 0: binarise each requested label, then OR them together in T2 space
    label_masks = []
    for lbl in labels:
        tmp_lbl = '{0}/{1}_lbl{2}.nii.gz'.format(native_roi_dir, name, lbl)
        cmd = '{0}fslmaths {1} -thr {2} -uthr {2} -bin {3}'.format(
            fsl_bin, seg_file, lbl, tmp_lbl)
        if verbose:
            print(cmd)
        err = run_fsl(cmd)
        if err:
            print('\tError extracting label {0} for {1}: {2}'.format(lbl, name, err))
            return None
        label_masks.append(tmp_lbl)

    err = run_fsl(build_mask_command(fsl_bin, label_masks, tmp_t2))
    for f in label_masks:
        try:
            os.remove(f)
        except OSError:
            pass
    if err:
        print('\tError combining labels for {0}: {1}'.format(name, err))
        return None

    # Optional: restrict to anterior fraction of the hippocampal slab
    if anterior_fr is not None:
        _crop_anterior(tmp_t2, anterior_fr)

    # Step 1: invert s0_to_t2 once per subject (cached)
    if not os.path.isfile(t2_to_s0):
        cmd = 'convert_xfm -omat {0} -inverse {1}'.format(t2_to_s0, s0_to_t2)
        if verbose:
            print(cmd)
        err = run_fsl(cmd)
        if err:
            print('\tError inverting s0_to_t2.mat: {0}'.format(err))
            return None

    # Step 2: T2 -> DWI (nearest-neighbour preserves label integrity)
    cmd = ('{0}flirt -in {1} -ref {2} -applyxfm -init {3} '
           '-interp nearestneighbour -out {4}'.format(
               fsl_bin, tmp_t2, fa_ref, t2_to_s0, tmp_dwi))
    if verbose:
        print(cmd)
    err = run_fsl(cmd)
    if err:
        print('\tError warping {0} to DWI space: {1}'.format(name, err))
        return None

    # Step 3: DWI -> Mean3G (nearest-neighbour, exact template grid)
    cmd = ('{0}applywarp --in={1} --ref={2} --warp={3} --interp=nn --out={4}'.format(
        fsl_bin, tmp_dwi, template_ref, fa_warp, out_path))
    if verbose:
        print(cmd)
    err = run_fsl(cmd)
    if err:
        print('\tError warping {0} to Mean3G space: {1}'.format(name, err))
        return None

    # Ensure binary after resampling
    run_fsl('{0}fslmaths {1} -bin {1}'.format(fsl_bin, out_path))

    for f in [tmp_t2, tmp_dwi]:
        try:
            os.remove(f)
        except OSError:
            pass

    if verbose:
        print('\tNative ROI {0}: generated in Mean3G space -> {1}'.format(name, out_path))

    return out_path


def resolve_roi_path(entry, subj_dwi_dir, rois_dir, native_roi_dir,
                     template_ref, fsl_bin, config_gen):
    """Return the Mean3G-grid mask path for this network entry.

    Template entries return the static rois_dir path unchanged.
    Native entries generate a per-subject ASHS-derived mask on the fly.
    Returns None if the native mask cannot be built (caller should skip subject).
    """
    name = roi_name(entry)
    if roi_space(entry) == 'template':
        return '{0}/{1}.nii.gz'.format(rois_dir, name)

    out = '{0}/{1}.nii.gz'.format(native_roi_dir, name)
    if os.path.isfile(out) and not config_gen['clobber']:
        return out

    return make_native_roi_in_template(
        entry, name, subj_dwi_dir, native_roi_dir, template_ref, fsl_bin, config_gen)


# -- Core tracking ------------------------------------------------------------

def run_probtrackx_for_seed(seed, seed_path, targets, target_paths, exclusion, net_dir,
                             rois_dir, bedpostx_dir, xfm_img, invxfm_img,
                             fsl_bin, probtrackx_cmd, config_ptx, config_gen, subject):
    """Run probtrackx from one seed toward a list of targets with given exclusions.

    seed / targets are ROI names (used for output directory and file naming).
    seed_path / target_paths are the resolved mask paths (template or native).
    Output goes to net_dir/seed/.
    Returns True on success, False/None on failure.
    """
    roi_list  = '{0}/others_{1}.txt'.format(net_dir, seed)
    stop_img  = '{0}/{1}_others.nii.gz'.format(net_dir, seed)
    avoid_img = '{0}/{1}_avoid.nii.gz'.format(net_dir, seed)

    target_mask_paths = []
    with open(roi_list, 'w') as listout:
        for target_path in target_paths:
            listout.write('{0}\n'.format(target_path))
            target_mask_paths.append(target_path)

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
                       config_ptx['distthresh'], config_ptx['sampvox'], seed_path,
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

        # native_rois config block is required only when any entry has space: "native"
        config_native = config_ptx.get('native_rois', {})
        template_ref  = config_native.get('template_ref', None)

        # Per-subject directory for native ROI masks resampled into Mean3G space
        native_roi_dir = '{0}/native_rois/{1}'.format(subj_dir, config_ptx['network_name'])

        for net_idx, net in enumerate(networks):
            seeds     = net.get('seeds', [])
            targets   = net.get('targets', [])
            exclusion = net.get('exclusion', [])
            net_name  = net.get('name', 'net{0:02d}'.format(net_idx))

            # Check that a template_ref is configured if any entry needs it
            all_entries = seeds + targets
            has_native  = any(roi_space(e) == 'native' for e in all_entries)
            if has_native and not template_ref:
                print('ERROR: network {0} has native ROIs but probtrackx.native_rois.template_ref '
                      'is not set in the config. Skipping subject.'.format(net_name))
                return False
            if has_native and not os.path.isdir(native_roi_dir):
                os.makedirs(native_roi_dir)

            # All networks share the same flat output dir so that downstream
            # scripts find {seed}/target_paths_{target}.nii.gz regardless of
            # which network produced them.  The net_name is used only for logging.
            net_dir = probtrackx_dir

            print('\tNetwork: {0}  seeds={1}  targets={2}'.format(
                net_name, [roi_name(s) for s in seeds], [roi_name(t) for t in targets]))

            # Resolve target paths once (shared across all seeds in this network)
            resolved_targets = []
            for t_entry in targets:
                t_path = resolve_roi_path(
                    t_entry, subj_dir, rois_dir, native_roi_dir,
                    template_ref, fsl_bin, config_gen)
                if t_path is None:
                    print('\tCould not resolve target {0} for subject {1}. Skipping.'.format(
                        roi_name(t_entry), subject))
                    return False
                resolved_targets.append(t_path)
            target_names = [roi_name(t) for t in targets]

            # Forward: each seed → all targets
            for seed_entry in seeds:
                seed_nm   = roi_name(seed_entry)
                seed_path = resolve_roi_path(
                    seed_entry, subj_dir, rois_dir, native_roi_dir,
                    template_ref, fsl_bin, config_gen)
                if seed_path is None:
                    print('\tCould not resolve seed {0} for subject {1}. Skipping.'.format(
                        seed_nm, subject))
                    return False
                ok = run_probtrackx_for_seed(
                    seed_nm, seed_path, target_names, resolved_targets,
                    exclusion, net_dir, rois_dir, bedpostx_dir, xfm_img, invxfm_img,
                    fsl_bin, probtrackx_cmd, config_ptx, config_gen, subject)
                if not ok:
                    return False

            # Reverse: each target → all seeds
            resolved_seeds = []
            for s_entry in seeds:
                s_path = resolve_roi_path(
                    s_entry, subj_dir, rois_dir, native_roi_dir,
                    template_ref, fsl_bin, config_gen)
                if s_path is None:
                    return False
                resolved_seeds.append(s_path)
            seed_names = [roi_name(s) for s in seeds]

            for t_entry, t_path in zip(targets, resolved_targets):
                t_nm = roi_name(t_entry)
                ok = run_probtrackx_for_seed(
                    t_nm, t_path, seed_names, resolved_seeds,
                    exclusion, net_dir, rois_dir, bedpostx_dir, xfm_img, invxfm_img,
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
            other_rois  = [r for r in rois if r != roi]
            other_paths = ['{0}/{1}.nii.gz'.format(rois_dir, r) for r in other_rois]
            roi_path    = '{0}/{1}.nii.gz'.format(rois_dir, roi)
            ok = run_probtrackx_for_seed(
                roi, roi_path, other_rois, other_paths, [], net_dir,
                rois_dir, bedpostx_dir, xfm_img, invxfm_img,
                fsl_bin, probtrackx_cmd, config_ptx, config_gen, subject)
            if not ok:
                return False

    return True

if __name__ == '__main__':
    main()
