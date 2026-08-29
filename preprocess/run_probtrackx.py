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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline_state import resolve_correction_method_robust

def load_config(path):
    """Loads a JSON config, following one optional "_base" key: a sibling
    file to load first, whose sections this file's sections then override
    (one level deep -- each section like "general"/"probtrackx" is a flat
    key/value dict, so a shallow per-section merge is sufficient). Lets
    near-duplicate configs (e.g. LC-native vs Fornix-native probtrackx
    runs, identical apart from roi_list/network_name) share one base file
    instead of being fully copy-pasted."""
    with open(path, 'r') as f:
        config = json.loads(f.read())
    base_rel = config.pop('_base', None)
    if not base_rel:
        return config
    base = load_config(os.path.join(os.path.dirname(path), base_rel))
    merged = {}
    for section in set(base) | set(config):
        merged[section] = dict(base.get(section, {}), **config.get(section, {}))
    return merged

def main():

    global config

    config = load_config(sys.argv[2])
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

def _crop_anterior(mask_path, fraction, keep='anterior'):
    """Zero out a fraction of occupied slices in the slab (dim 2), keeping
    either the anterior or the posterior end.

    Determines anterior direction from the sign of affine[1, 2] (Y/A-P component
    of the through-slab axis): negative means increasing slice index is more
    posterior. `keep='anterior'` (default, original behaviour) keeps the
    anterior `fraction` of occupied slices; `keep='posterior'` keeps the
    posterior `fraction` instead (added for UF's posterior-BA35 seed --
    same slice-counting logic, just keeping the other end of the slab).
    """
    import nibabel as nib
    import numpy as np

    if keep not in ('anterior', 'posterior'):
        raise ValueError("keep must be 'anterior' or 'posterior', got {0!r}".format(keep))

    img  = nib.load(mask_path)
    data = img.get_fdata().copy()

    occupied = np.any(data > 0, axis=(0, 1))
    slices   = np.where(occupied)[0]
    if len(slices) == 0:
        return

    first, last = int(slices[0]), int(slices[-1])
    n_keep = int(np.ceil((last - first + 1) * fraction))

    # increasing_is_posterior: whether increasing slice index runs toward
    # the posterior of the subject (affine[1,2] < 0), else it runs anterior
    increasing_is_posterior = img.affine[1, 2] < 0
    keep_low_end = (keep == 'anterior') == increasing_is_posterior
    # keep_low_end True  -> keep slices [first, first+n_keep)
    # keep_low_end False -> keep slices [last-n_keep+1, last]

    if keep_low_end:
        data[:, :, first + n_keep:] = 0
    else:
        data[:, :, :last - n_keep + 1] = 0

    nib.save(nib.Nifti1Image(data, img.affine, img.header), mask_path)


def _reslice_t2_to_t1(t2_path, t1_ref_path, mat_path, out_path):
    """Resample a T2/tse-space label volume onto the T1/mprage grid.

    Uses ASHS's own T2->T1 registration (ashs/flirt_t2_to_t1/flirt_t2_to_t1.mat,
    a greedy-computed RAS matrix mapping T2 physical coords -> T1 physical
    coords) when available, applied via direct voxel-to-voxel affine math.
    This is deliberately NOT done via flirt/greedy CLI resampling - their
    -applyxfm/-r conventions expect the matrix in a specific direction/format
    and are easy to mismatch with a raw greedy RAS matrix (silently produces
    a badly misplaced result rather than an error - see
    project_ashs_t2t1_registration_fix memory). Falls back to the T2/T1
    header alone (identity in RAS space) when no matrix is available for a
    subject, which was the previous behaviour for every subject.

    Confirmed via NMI(T1, resliced T2) that the matrix - correctly applied -
    is as-good-or-better than header-only for every subject checked,
    including ones with large (~10mm) corrections where header-only was
    previously assumed to be an adequate approximation.
    """
    import nibabel as nib
    import numpy as np
    from scipy.ndimage import map_coordinates

    t2img = nib.load(t2_path)
    t1img = nib.load(t1_ref_path)
    M = np.loadtxt(mat_path) if mat_path and os.path.isfile(mat_path) else np.eye(4)

    vox2vox = np.linalg.inv(t2img.affine) @ np.linalg.inv(M) @ t1img.affine
    ii, jj, kk = np.meshgrid(*[np.arange(n) for n in t1img.shape], indexing='ij')
    ijk1 = np.stack([ii, jj, kk, np.ones_like(ii)]).reshape(4, -1).astype(np.float64)
    src = (vox2vox @ ijk1)[:3]

    t2d = np.asanyarray(t2img.dataobj).astype(np.float32)
    out = map_coordinates(t2d, src, order=0, mode='constant', cval=0.0)
    out = out.reshape(t1img.shape)

    nib.save(nib.Nifti1Image(out.astype(np.int16), t1img.affine, t1img.header), out_path)


def ensure_t1_to_s0(subj_root, subj_dwi_dir, native_roi_dir, fsl_bin, config_gen):
    """Ensure the per-subject T1->DWI transform is cached; return its path or None.

    This BBR (epi_reg S0->T1, then inverted) is shared by every native ROI
    builder regardless of source (ASHS or hypothalamus_seg), since both
    segmentations share the mprage/T1 grid -- and, as of 2026-08-11, is
    cached once per SESSION (in subj_dwi_dir directly), not per network.
    It used to be cached in native_roi_dir, which is network-specific
    (native_rois/{network}/) despite this function's own docstring already
    claiming "once per subject" -- so LC-BA35-only and Fornix-native (and
    any future network) were each silently running their own independent
    epi_reg BBR fit of the exact same S0->T1 pair. Caught when a manual QA
    pass (qa_s0t1_corrected.py + diagnose_and_fix_s0t1_corrected.py) fixed
    ~70 bad LC-BA35-only registrations and Fornix-native's copies for the
    same sessions turned out to still be running the old, unreviewed fit.

    Migration: if a network-specific cache already exists from before this
    fix (most likely native_rois/LC-BA35-only, which has been through QA),
    it's promoted into the shared location instead of recomputing from
    scratch -- preserves already-reviewed/manually-fixed registrations
    rather than silently discarding them.
    """
    verbose = config_gen['verbose']

    mprage    = '{0}/ashs/mprage.nii.gz'.format(subj_root)
    s0_ref    = '{0}/dti_S0.nii.gz'.format(subj_dwi_dir)

    shared_dir = '{0}/s0_to_t1_registration'.format(subj_dwi_dir)
    mprage_brain = '{0}/mprage_brain.nii.gz'.format(shared_dir)
    s0_to_t1_pre = '{0}/s0_to_t1'.format(shared_dir)       # epi_reg prefix
    s0_to_t1     = '{0}.mat'.format(s0_to_t1_pre)
    t1_to_s0     = '{0}/t1_to_s0.mat'.format(shared_dir)

    if os.path.isfile(t1_to_s0):
        return t1_to_s0

    if not os.path.isdir(shared_dir):
        os.makedirs(shared_dir)

    # Migrate an existing network-specific cache rather than recompute --
    # prefer LC-BA35-only since that's the one that's actually been QA'd.
    native_rois_root = os.path.dirname(native_roi_dir)
    for candidate_net in ['LC-BA35-only', 'Fornix-native']:
        old_dir = os.path.join(native_rois_root, candidate_net)
        old_mat = os.path.join(old_dir, 's0_to_t1.mat')
        old_inv = os.path.join(old_dir, 't1_to_s0.mat')
        if os.path.isfile(old_inv) and not os.path.islink(old_inv) and os.path.isfile(old_mat):
            if verbose:
                print('\tMigrating existing {0} S0->T1 registration to shared cache'.format(candidate_net))
            shutil.copy2(old_mat, s0_to_t1)
            shutil.copy2(old_inv, t1_to_s0)
            if os.path.isfile(os.path.join(old_dir, 'mprage_brain.nii.gz')):
                shutil.copy2(os.path.join(old_dir, 'mprage_brain.nii.gz'), mprage_brain)
            return t1_to_s0

    for f in [mprage, s0_ref]:
        if not os.path.isfile(f):
            print('\tMissing input for T1->DWI transform: {0}'.format(f))
            return None

    if not os.path.isfile(s0_to_t1):
        if not os.path.isfile(mprage_brain):
            cmd = '{0}bet {1} {2} -R -f 0.4'.format(fsl_bin, mprage, mprage_brain)
            if verbose:
                print(cmd)
            err = run_fsl(cmd)
            if err:
                print('\tError brain-extracting mprage: {0}'.format(err))
                return None
        cmd = ('{0}epi_reg --epi={1} --t1={2} --t1brain={3} --out={4}'.format(
            fsl_bin, s0_ref, mprage, mprage_brain, s0_to_t1_pre))
        if verbose:
            print(cmd)
        err = run_fsl(cmd)
        if err or not os.path.isfile(s0_to_t1):
            print('\tError registering S0 -> T1 (BBR): {0}'.format(err))
            return None

    cmd = 'convert_xfm -omat {0} -inverse {1}'.format(t1_to_s0, s0_to_t1)
    if verbose:
        print(cmd)
    err = run_fsl(cmd)
    if err:
        print('\tError inverting s0_to_t1.mat: {0}'.format(err))
        return None

    return t1_to_s0


def make_native_roi_in_template(entry, name, subj_dwi_dir, native_roi_dir,
                                 template_ref, fsl_bin, config_gen):
    """Warp an ASHS label into Mean3G template space and return the output path.

    Chain (all per-subject transforms cached in native_roi_dir):
      ASHS seg (T2/tse space)
        --(ASHS's own flirt_t2_to_t1.mat, or header if unavailable) --> T1/mprage space [_reslice_t2_to_t1, NN]
        --(inverse of BBR s0_to_t1)       -->  DWI space        [flirt, NN]
        --(reg3G/FA_warp2Mean3G.nii.gz)   -->  Mean3G           [applywarp, NN]

    The only *searched* registration in this chain is BBR epi_reg S0->T1;
    the T2->T1 step reuses ASHS's own precomputed rigid registration rather
    than ever registering against the thin oblique T2 slab directly (the
    latter was the root cause of the flipped / mis-rotated ROI placements
    produced by the old direct mind/s0_to_t2.mat MI registration). Applying
    ASHS's matrix (rather than trusting the T2/T1 headers alone, the
    previous approach) matters most for subjects with real inter-scan head
    motion - see project_ashs_t2t1_registration_fix memory.

    Returns the output path on success, None on failure.
    """
    verbose     = config_gen['verbose']
    ashs_side   = entry['ashs_side']
    labels      = entry['ashs_labels']
    anterior_fr = entry.get('anterior_fraction', None)
    posterior_fr = entry.get('posterior_fraction', None)
    if anterior_fr is not None and posterior_fr is not None:
        raise ValueError(
            'Native ROI {0}: specify only one of anterior_fraction/'
            'posterior_fraction, not both'.format(name))

    subj_root = os.path.dirname(subj_dwi_dir)
    seg_file  = '{0}/ashs/final/ashs_{1}_lfseg_corr_usegray.nii.gz'.format(subj_root, ashs_side)
    mprage    = '{0}/ashs/mprage.nii.gz'.format(subj_root)
    fa_ref    = '{0}/dti_FA.nii.gz'.format(subj_dwi_dir)
    fa_warp   = '{0}/reg3G/FA_warp2Mean3G.nii.gz'.format(subj_dwi_dir)

    tmp_t2    = '{0}/{1}_t2.nii.gz'.format(native_roi_dir, name)
    tmp_t1    = '{0}/{1}_t1.nii.gz'.format(native_roi_dir, name)
    tmp_dwi   = '{0}/{1}_dwi.nii.gz'.format(native_roi_dir, name)
    out_path  = '{0}/{1}.nii.gz'.format(native_roi_dir, name)

    for f in [seg_file, mprage, fa_ref, fa_warp, template_ref]:
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

    # Optional: restrict to anterior or posterior fraction of the slab
    if anterior_fr is not None:
        _crop_anterior(tmp_t2, anterior_fr, keep='anterior')
    elif posterior_fr is not None:
        _crop_anterior(tmp_t2, posterior_fr, keep='posterior')

    t1_to_s0 = ensure_t1_to_s0(subj_root, subj_dwi_dir, native_roi_dir, fsl_bin, config_gen)
    if t1_to_s0 is None:
        return None

    # Step 2a: T2/tse -> T1, using ASHS's own registration when available
    t2_to_t1_mat = '{0}/ashs/flirt_t2_to_t1/flirt_t2_to_t1.mat'.format(subj_root)
    if verbose:
        print('\tReslicing {0} T2 -> T1 (mat={1})'.format(
            name, t2_to_t1_mat if os.path.isfile(t2_to_t1_mat) else 'none, using header'))
    try:
        _reslice_t2_to_t1(tmp_t2, mprage, t2_to_t1_mat, tmp_t1)
    except Exception as e:
        print('\tError resampling {0} T2 -> T1: {1}'.format(name, e))
        return None

    # Step 2b: T1 -> DWI (nearest-neighbour preserves label integrity)
    cmd = ('{0}flirt -in {1} -ref {2} -applyxfm -init {3} '
           '-interp nearestneighbour -out {4}'.format(
               fsl_bin, tmp_t1, fa_ref, t1_to_s0, tmp_dwi))
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

    for f in [tmp_t2, tmp_t1, tmp_dwi]:
        try:
            os.remove(f)
        except OSError:
            pass

    if verbose:
        print('\tNative ROI {0}: generated in Mean3G space -> {1}'.format(name, out_path))

    return out_path


def make_native_roi_from_hypothalamus(entry, name, subj_dwi_dir, native_roi_dir,
                                       template_ref, fsl_bin, config_gen):
    """Warp a hypothalamus_seg label into Mean3G template space and return the output path.

    Chain (per-subject transforms cached in native_roi_dir):
      hypothalamus_seg (T1/mprage space, run directly on the subject's T1w)
        --(inverse of BBR s0_to_t1)     -->  DWI space   [flirt, NN]
        --(reg3G/FA_warp2Mean3G.nii.gz) -->  Mean3G      [applywarp, NN]

    Unlike the ASHS source (T2 slab), hypothalamus_seg's output already shares
    the mprage grid exactly (confirmed: identical affine/shape), so there is no
    T2->T1 header-reslice step - it goes straight from T1 space to DWI space.

    Returns the output path on success, None on failure.
    """
    verbose = config_gen['verbose']
    labels  = entry['hypothalamus_labels']

    subj_root = os.path.dirname(subj_dwi_dir)
    session   = os.path.basename(subj_root)
    subj_id   = session.rsplit('_', 1)[0]
    seg_file  = '{0}/hypothalamus/{1}_hypothalamus_seg.nii.gz'.format(subj_root, subj_id)
    fa_ref    = '{0}/dti_FA.nii.gz'.format(subj_dwi_dir)
    fa_warp   = '{0}/reg3G/FA_warp2Mean3G.nii.gz'.format(subj_dwi_dir)

    tmp_t1    = '{0}/{1}_t1.nii.gz'.format(native_roi_dir, name)
    tmp_dwi   = '{0}/{1}_dwi.nii.gz'.format(native_roi_dir, name)
    out_path  = '{0}/{1}.nii.gz'.format(native_roi_dir, name)

    for f in [seg_file, fa_ref, fa_warp, template_ref]:
        if not os.path.isfile(f):
            print('\tNative ROI {0}: missing input {1}'.format(name, f))
            return None

    # Step 0: binarise each requested label, then OR them together (already T1 space)
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

    err = run_fsl(build_mask_command(fsl_bin, label_masks, tmp_t1))
    for f in label_masks:
        try:
            os.remove(f)
        except OSError:
            pass
    if err:
        print('\tError combining labels for {0}: {1}'.format(name, err))
        return None

    t1_to_s0 = ensure_t1_to_s0(subj_root, subj_dwi_dir, native_roi_dir, fsl_bin, config_gen)
    if t1_to_s0 is None:
        return None

    # Step 1: T1 -> DWI (nearest-neighbour preserves label integrity)
    cmd = ('{0}flirt -in {1} -ref {2} -applyxfm -init {3} '
           '-interp nearestneighbour -out {4}'.format(
               fsl_bin, tmp_t1, fa_ref, t1_to_s0, tmp_dwi))
    if verbose:
        print(cmd)
    err = run_fsl(cmd)
    if err:
        print('\tError warping {0} to DWI space: {1}'.format(name, err))
        return None

    # Step 2: DWI -> Mean3G (nearest-neighbour, exact template grid)
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

    for f in [tmp_t1, tmp_dwi]:
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
    Native entries generate a per-subject ASHS- or hypothalamus_seg-derived mask on the fly,
    dispatching on which label-source field is present ('ashs_labels' vs 'hypothalamus_labels').
    Returns None if the native mask cannot be built (caller should skip subject).
    """
    name = roi_name(entry)
    if roi_space(entry) == 'template':
        return '{0}/{1}.nii.gz'.format(rois_dir, name)

    out = '{0}/{1}.nii.gz'.format(native_roi_dir, name)
    if os.path.isfile(out) and not config_gen['clobber']:
        return out

    if 'hypothalamus_labels' in entry:
        return make_native_roi_from_hypothalamus(
            entry, name, subj_dwi_dir, native_roi_dir, template_ref, fsl_bin, config_gen)

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
    dwi_dirname = config_gen.get('dwi_dirname', 'dwi')
    if dwi_dirname == 'auto':
        # Corrected-eddy pipeline: exactly one of dwi_topup/dwi_fugue/dwi_bdp
        # may exist per session (see PLAN_distortion_correction_rollout.md's
        # storage architecture) -- resolve which one applies here instead of
        # assuming the old plain "dwi" layout. Canonical resolution logic
        # lives in pipeline_state.py; don't reimplement it here. Uses the
        # real-filesystem resolver, not the flag-based one -- flags/fugue.done
        # was found missing 2026-08-22 for sessions with fully complete,
        # valid fugue output, which silently leaked bdp (banned for this
        # cohort) into production tractography for 7 subjects.
        sess_rel = '{0}{1}/{2}'.format(config_gen['prefix'], subject, session)
        method = resolve_correction_method_robust(sess_rel)
        if method is None:
            print('No dwi_<method> dir found for auto-resolution [{0}]'.format(subject))
            return False
        dwi_dirname = 'dwi_{0}'.format(method)
    subj_dir = '{0}/{1}{2}/{3}{4}' \
                        .format(deriv_dir, config_gen['prefix'], subject, session, dwi_dirname)

    bedpostx_dir = '{0}/bedpostX'.format(subj_dir)
    # NOT xfms/eye.mat -- bedpostx_postproc_gpu.sh writes that unconditionally
    # even when every xfibres_gpu fit fails (found 2026-08-08, see
    # pipeline_state.py's module docstring). merged_th1samples/dyads1 only
    # exist if the fit genuinely ran.
    bedpostx_done = os.path.isfile('{0}/merged_th1samples.nii.gz'.format(bedpostx_dir)) and \
        os.path.isfile('{0}/dyads1.nii.gz'.format(bedpostx_dir))
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
