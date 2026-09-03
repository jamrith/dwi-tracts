#!/usr/bin/env python3
"""
run_profiles_figures.py <config.json> [--methods rft,nonparam,permutation]
                                      [--no-tsa]

Render the NORMAL result figures (plot_fornix_ba35_results_v4.run_network --
the same brain-overview / zoom / all-covariates / effect-size / TSA-contrast
set run_glm_generic.py produces) for each inference method a profiles run
wrote, so the methods can be compared figure-for-figure.

How it works, and why it needs no change to the plotting script:
plot_fornix_ba35_results_v4 reads results ONLY through
load_rft_trace(rft_dir, ...), which opens
`tvals_rft_{roi_a}_{roi_b}_{factor}_{thr}.poly3d`, and it derives
`rft_dir = {net_root}/polylines/rft/{glm_name}`. DwiTractsProfiles writes
byte-compatible poly3d files (same columns: tvals, tvals_thr, pvals,
clusters, logpvals, coef*) but named `tvals_{np|perm|rft}_...` under
`polylines/{output_dir}/{sfx}/{glm}/` -- deliberately namespaced so it can
never overwrite the legacy tree.

So this script stages, per method, a thin alias:

  {net_root}/polylines/rft/{glm}__{output_dir}_{sfx}/tvals_rft_*.poly3d
      -> symlinks to the method's own tvals_{sfx}_*.poly3d
  {net_root}/{output_dir}/{glm}__{output_dir}_{sfx}/summary-*
      -> symlink to the real summary dir (get_glm_n reads a resids_*.csv
         column count from it)
  {net_root}/{output_dir}/{glm}__{output_dir}_{sfx}/figures/
      -> a REAL directory, so each method's figures stay separate

then calls run_network with glm_name set to that alias. Nothing is copied;
the legacy polylines/rft/{glm} tree is never touched.

The TSA-vs-covariate contrast figure is identical across methods (it plots
the raw per-subject metric, not any inference result) and is by far the most
expensive panel, so it is rendered once and symlinked into the other
methods' figure directories.
"""
import os
import sys
import glob
import json
import argparse

import pandas as pd

sys.path.insert(0, '/gpfs01/imgshare/ConnLS/ADNI')

from dwitracts.profiles import DwiTractsProfiles
from run_profiles_generic import load_params


def stage_method(net_root, output_dir, glm_name, sfx):
    """Build the alias tree described in the module docstring.

    Returns (alias, n_traces) -- n_traces is how many poly3d files were
    linked, so a caller can skip a method that produced nothing.
    """
    alias = '{0}__{1}_{2}'.format(glm_name, output_dir, sfx)

    src_poly = os.path.join(net_root, 'polylines', output_dir, sfx, glm_name)
    dst_poly = os.path.join(net_root, 'polylines', 'rft', alias)
    os.makedirs(dst_poly, exist_ok=True)
    n = 0
    for src in sorted(glob.glob(os.path.join(src_poly, 'tvals_{0}_*.poly3d'.format(sfx)))):
        base = os.path.basename(src).replace('tvals_{0}_'.format(sfx), 'tvals_rft_', 1)
        dst = os.path.join(dst_poly, base)
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst)
        os.symlink(src, dst)
        n += 1

    real_glm_dir = os.path.join(net_root, output_dir, glm_name)
    alias_glm_dir = os.path.join(net_root, output_dir, alias)
    os.makedirs(os.path.join(alias_glm_dir, 'figures'), exist_ok=True)
    for summary in sorted(glob.glob(os.path.join(real_glm_dir, 'summary-*'))):
        dst = os.path.join(alias_glm_dir, os.path.basename(summary))
        if os.path.islink(dst):
            os.remove(dst)
        if not os.path.exists(dst):
            os.symlink(summary, dst)
    return alias, n


def build_tsa_params(params, glm_name):
    """Same assembly run_glm_generic.py does for the contrast figure."""
    try:
        params_gen = params['general']
        tracts_gen = params['tracts']['general']
        dwi_regress = params['tracts']['dwi_regressions']
        avg_dir = params['tracts']['average_directions']
        with open(params_gen['subjects_file']) as f:
            subjects = [line.strip() for line in f if line.strip()]
        covariates_df = pd.read_csv(params_gen['covariates_file'])
        metric_stem = params_gen.get('metric', 'betas')
        # initialize() has already inserted 'Intercept' at position 0
        factors = [f for f in params['glm'][glm_name]['factors'] if f != 'Intercept']
        return {
            'project_dir': tracts_gen['project_dir'],
            'deriv_dir': tracts_gen['deriv_dir'],
            'regress_dir': dwi_regress['regress_dir'],
            'beta_sm_fwhm': dwi_regress['beta_sm_fwhm'],
            'subjects': subjects,
            'covariates_df': covariates_df,
            'categorical': params['glm'][glm_name]['categorical'],
            'levels': params['glm'][glm_name]['levels'],
            'threshold': avg_dir['threshold'],
            'use_norm': avg_dir['use_normalized'],
            'metric_stem': metric_stem,
            'metric_label': 'TSA' if metric_stem == 'betas' else metric_stem,
            'factors': factors,
        }
    except Exception as e:
        print('WARNING: could not assemble tsa_params, skipping TSA contrast figure:', e)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('--methods', default=None,
                    help='comma-separated subset of the config\'s inference_method')
    ap.add_argument('--no-tsa', action='store_true',
                    help='skip the (slow) TSA-vs-covariate contrast figure')
    args = ap.parse_args()

    from plot_fornix_ba35_results_v4 import run_network

    params = load_params(args.config)
    if 'figures' not in params:
        print('Config has no "figures" block; nothing to render.')
        return 1

    prof = DwiTractsProfiles(params)
    assert prof.initialize()

    fig = params['figures']
    net_root = os.path.join(fig['tracts_dir'], fig['network_dir'])
    output_dir = prof._p('output_dir', 'profiles_out')
    glm_name = list(params['glm'].keys())[0]

    methods = prof.methods
    if args.methods:
        methods = [m.strip() for m in args.methods.split(',')]

    tsa_params = None if args.no_tsa else build_tsa_params(params, glm_name)
    contrast_src = None

    for i, method in enumerate(methods):
        sfx = DwiTractsProfiles.METHOD_SUFFIX[method]
        alias, n = stage_method(net_root, output_dir, glm_name, sfx)
        if n == 0:
            print('No {0} traces found for {1}; skipping.'.format(sfx, glm_name))
            continue
        print('\n=== figures: {0} ({1} traces) -> {2}/{3}/figures'
              .format(method, n, output_dir, alias))

        # Render the (expensive, method-independent) contrast figure once.
        this_tsa = tsa_params if contrast_src is None else None

        run_network(
            '{0} [{1}]'.format(fig['network_label'], method),
            fig['tracts_dir'], fig['network_dir'], alias,
            [tuple(p) for p in fig['pairs_2']],
            [tuple(p) for p in fig['pairs_3']],
            fig['color'],
            tsa_params=this_tsa,
            glm_output_dir=output_dir,
            # compare_label must identify the TREE as well as the method:
            # every tree shares the config's base label, so '{base}_{sfx}'
            # alone makes all of them collide onto one file in
            # method_comparison/ and only the last one rendered survives.
            compare_label=('{0}__{1}_{2}'.format(
                               fig['compare_label'],
                               output_dir[len('profiles_'):] if output_dir.startswith('profiles_')
                               else output_dir, sfx)
                           if fig.get('compare_label') else None),
            effect_scales=list(params['general'].get('effect_scales', {}).keys()),
        )

        fig_dir = os.path.join(net_root, output_dir, alias, 'figures')
        # Ship the run's parameters WITH the figures: shell geometry (K, shell
        # width, voxels/shell, empty shells, df) and the inference metadata
        # this method actually ran with (FWHM, df, cluster-forming t, n_perm,
        # p floor, per-factor cluster counts and p). Otherwise a PDF that has
        # been copied out of the tree carries no record of how it was made.
        real_summary = glob.glob(os.path.join(net_root, output_dir, glm_name, 'summary-*'))
        if real_summary:
            import shutil as _sh
            for meta in (['shell_geometry.csv', 'inference_meta-{0}.csv'.format(sfx)]):
                src = os.path.join(real_summary[0], meta)
                if os.path.isfile(src):
                    _sh.copy(src, os.path.join(fig_dir, meta))
        if contrast_src is None:
            hits = glob.glob(os.path.join(fig_dir, '*_contrasts.pdf'))
            contrast_src = hits[0] if hits else None
        elif contrast_src:
            dst = os.path.join(fig_dir, os.path.basename(contrast_src))
            if not os.path.exists(dst):
                os.symlink(contrast_src, dst)

    print('\nDONE FIGURES:', args.config)
    return 0


if __name__ == '__main__':
    sys.exit(main())
