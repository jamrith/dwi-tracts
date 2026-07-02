#!/usr/bin/env python3
"""
check_outputs.py  -  Report probtrackx output completeness per subject.

Reads a preprocess config JSON (same format used by run_probtrackx.py /
queue_probtrackx_limbic.sh) and checks whether every expected
target_paths_*.nii.gz / target_localdir_*.nii.gz file is present for
each subject.  These are the per-subject files consumed by compute_tsa.py.

Saves a PNG of pie charts summarising completion by subject status and
by individual seed-target tract.

Usage
-----
  python check_outputs.py <preprocess_config.json> [subjects_file]

If subjects_file is omitted, the path in config['general']['subjects_file'] is used.

Exit codes
----------
  0  all subjects complete
  1  one or more subjects incomplete / missing
"""

import json
import math
import os
import sys


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_subjects(path):
    subjects = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                subjects.append(line)
    return subjects


def build_expected_pairs(networks):
    """
    Return a deduplicated list of (seed, target) tuples that probtrackx
    should have produced, based on the forward + reverse runs defined in
    the network JSON.
    """
    seen = set()
    pairs = []
    for net in networks:
        seeds   = net.get('seeds', [])
        targets = net.get('targets', [])
        for seed in seeds:
            for tgt in targets:
                key = (seed, tgt)
                if key not in seen:
                    seen.add(key)
                    pairs.append(key)
        for tgt in targets:
            for seed in seeds:
                key = (tgt, seed)
                if key not in seen:
                    seen.add(key)
                    pairs.append(key)
    return pairs


def flat_path(probtrackx_dir, seed, target, fname):
    return os.path.join(probtrackx_dir, seed, fname)


def pernetwork_path(probtrackx_dir, net_name, seed, target, fname):
    return os.path.join(probtrackx_dir, net_name, seed, fname)


def detect_needs_fix(probtrackx_dir, networks):
    if not os.path.isdir(probtrackx_dir):
        return False
    net_names = {net.get('name', '') for net in networks}
    subdirs = {d for d in os.listdir(probtrackx_dir)
               if os.path.isdir(os.path.join(probtrackx_dir, d))}
    return bool(net_names & subdirs)


def check_subject(probtrackx_dir, pairs, networks):
    """
    Check a single subject's probtrackx output directory.

    Returns
    -------
    dict with keys:
      status       : 'complete' | 'partial' | 'missing_dir' | 'needs_fix'
      n_found      : int  (files present at flat path)
      n_expected   : int
      missing_flat : list[str]
      fixable      : int
      pair_found   : dict  (seed, target) -> int  (0, 1, or 2 files found)
    """
    n_expected = len(pairs) * 2
    pair_found = {p: 0 for p in pairs}
    result = {
        'n_expected':  n_expected,
        'n_found':     0,
        'missing_flat': [],
        'fixable':     0,
        'status':      'missing_dir',
        'pair_found':  pair_found,
    }

    if not os.path.isdir(probtrackx_dir):
        result['missing_flat'] = [f'{s}/target_paths_{t}.nii.gz'    for s, t in pairs] + \
                                  [f'{s}/target_localdir_{t}.nii.gz' for s, t in pairs]
        return result

    needs_fix = detect_needs_fix(probtrackx_dir, networks)
    net_map   = {net['name']: net for net in networks}

    found = 0
    missing_flat = []
    fixable = 0

    for seed, target in pairs:
        for prefix in ('target_paths', 'target_localdir'):
            fname = f'{prefix}_{target}.nii.gz'
            flat  = flat_path(probtrackx_dir, seed, target, fname)

            if os.path.isfile(flat):
                found += 1
                pair_found[(seed, target)] += 1
            else:
                missing_flat.append(f'{seed}/{fname}')
                if needs_fix:
                    for net_name, net in net_map.items():
                        seeds_n   = net.get('seeds', [])
                        targets_n = net.get('targets', [])
                        if (seed in seeds_n and target in targets_n) or \
                           (seed in targets_n and target in seeds_n):
                            old = pernetwork_path(probtrackx_dir, net_name, seed, target, fname)
                            if os.path.isfile(old):
                                fixable += 1
                                break

    result['n_found']      = found
    result['missing_flat'] = missing_flat
    result['fixable']      = fixable

    if found == n_expected:
        result['status'] = 'complete'
    elif needs_fix and fixable > 0:
        result['status'] = 'needs_fix'
    elif found > 0:
        result['status'] = 'partial'
    else:
        result['status'] = 'missing_dir'

    return result


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

# Short display labels for common ROI names
_ABBREV = {
    'Ent_L':               'Ent L',
    'Ent_R':               'Ent R',
    'bst_lh':              'BSt L',
    'bst_rh':              'BSt R',
    'ofc_lh':              'OFC L',
    'ofc_rh':              'OFC R',
    'area33_acc_lh':       'ACC L',
    'area33_acc_rh':       'ACC R',
    'area_p24ab_pacc_lh':  'pACC L',
    'area_p24ab_pacc_rh':  'pACC R',
    'thalamus_am_lh':      'Thal L',
    'thalamus_am_rh':      'Thal R',
}

def _abbrev(name):
    return _ABBREV.get(name, name)


def _draw_pie(ax, n_complete, n_partial, n_missing, n_total, title):
    """Draw a single completion pie on ax."""
    COL_COMPLETE = '#27ae60'
    COL_PARTIAL  = '#f39c12'
    COL_MISSING  = '#bdc3c7'

    sizes  = []
    colors = []
    for n, c in [(n_complete, COL_COMPLETE),
                 (n_partial,  COL_PARTIAL),
                 (n_missing,  COL_MISSING)]:
        if n > 0:
            sizes.append(n)
            colors.append(c)

    if not sizes:
        sizes  = [1]
        colors = [COL_MISSING]

    wedge_props = dict(linewidth=0.5, edgecolor='white')
    ax.pie(sizes, colors=colors, wedgeprops=wedge_props, startangle=90)

    # Compact percentage label in the centre
    pct = int(round(100 * n_complete / n_total)) if n_total else 0
    ax.text(0, 0, f'{pct}%', ha='center', va='center',
            fontsize=7, fontweight='bold', color='#2c3e50')
    ax.set_title(title, fontsize=6.5, pad=2, color='#2c3e50')


def save_plot(rows, pairs, counts, n_total, network_name,
              config_file, subjects_file, out_path):
    """Build and save the completion figure."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec

    # ------------------------------------------------------------------
    # Per-pair aggregation
    # ------------------------------------------------------------------
    pair_complete = {p: 0 for p in pairs}
    pair_partial  = {p: 0 for p in pairs}

    for _subj, res, _ptx in rows:
        for pair in pairs:
            n = res['pair_found'].get(pair, 0)
            if n == 2:
                pair_complete[pair] += 1
            elif n == 1:
                pair_partial[pair]  += 1

    # ------------------------------------------------------------------
    # Layout: summary pie (left 2 cols, middle 2 rows) + 4x5 tract grid
    # ------------------------------------------------------------------
    n_tract_cols = 5
    n_tract_rows = math.ceil(len(pairs) / n_tract_cols)   # 4 for 20 pairs

    fig = plt.figure(figsize=(20, 2 + n_tract_rows * 2.4))
    gs  = GridSpec(n_tract_rows, n_tract_cols + 2, figure=fig,
                   wspace=0.08, hspace=0.55,
                   left=0.02, right=0.98,
                   top=0.88,  bottom=0.04)

    summary_row_start = max(0, (n_tract_rows - 2) // 2)
    ax_summary = fig.add_subplot(gs[summary_row_start:summary_row_start + 2, 0:2])

    # Summary pie: subject-level status
    status_data = [
        (counts['complete'],    '#27ae60', 'Complete'),
        (counts['partial'],     '#f39c12', 'Partial'),
        (counts['missing_dir'], '#bdc3c7', 'Missing'),
        (counts['needs_fix'],   '#2980b9', 'Needs fix'),
    ]
    s_sizes  = [n for n, _, _ in status_data if n > 0]
    s_colors = [c for n, c, _ in status_data if n > 0]
    s_labels = [f'{l} ({n})' for n, _, l in status_data if n > 0]
    ax_summary.pie(s_sizes, colors=s_colors, startangle=90,
                   wedgeprops=dict(linewidth=0.8, edgecolor='white'))
    ax_summary.set_title('Subjects', fontsize=10, fontweight='bold',
                          pad=6, color='#2c3e50')
    ax_summary.legend(s_labels, loc='lower center',
                      bbox_to_anchor=(0.5, -0.18),
                      fontsize=7, frameon=False, ncol=2)

    # Tract pies
    for i, (seed, target) in enumerate(pairs):
        row = i // n_tract_cols
        col = i %  n_tract_cols + 2
        ax  = fig.add_subplot(gs[row, col])
        n_c = pair_complete[( seed, target)]
        n_p = pair_partial [(seed, target)]
        n_m = n_total - n_c - n_p
        title = f'{_abbrev(seed)}\n-> {_abbrev(target)}'
        _draw_pie(ax, n_c, n_p, n_m, n_total, title)

    # Legend for tract pies
    legend_patches = [
        mpatches.Patch(color='#27ae60', label='Complete'),
        mpatches.Patch(color='#f39c12', label='Partial'),
        mpatches.Patch(color='#bdc3c7', label='Missing'),
    ]
    fig.legend(handles=legend_patches, loc='upper right',
               bbox_to_anchor=(0.99, 0.97), fontsize=8, frameon=True,
               title='Tract files', title_fontsize=8)

    # Title
    subj_base = os.path.basename(subjects_file)
    cfg_base  = os.path.basename(config_file)
    fig.suptitle(f'Probtrackx output check  |  {network_name}  |  '
                 f'{subj_base}  ({n_total} subjects)',
                 fontsize=11, fontweight='bold', y=0.97, color='#2c3e50')

    fig.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print(f'Usage: {os.path.basename(sys.argv[0])} <preprocess_config.json> [subjects_file]')
        sys.exit(1)

    config_file = sys.argv[1]
    with open(config_file) as fh:
        config = json.load(fh)

    config_gen = config['general']
    config_ptx = config['probtrackx']

    subjects_file = sys.argv[2] if len(sys.argv) > 2 else config_gen.get('subjects_file', '')
    if not subjects_file:
        print('Error: no subjects_file in config or command line.')
        sys.exit(1)

    subjects     = load_subjects(subjects_file)
    network_name = config_ptx['network_name']
    roi_list     = config_ptx['roi_list']

    with open(roi_list) as fh:
        networks = json.load(fh)['networks']

    pairs = build_expected_pairs(networks)

    root_dir  = config_gen['root_dir']
    deriv_dir = os.path.join(root_dir, config_gen['deriv_dir'])
    prefix    = config_gen.get('prefix', '')
    session   = config_gen.get('session', '')
    if session:
        session = f'{session}/'

    # -----------------------------------------------------------------------
    # Check each subject
    # -----------------------------------------------------------------------
    counts = {'complete': 0, 'partial': 0, 'missing_dir': 0, 'needs_fix': 0}
    rows   = []

    for subject in subjects:
        subj_id  = f'{prefix}{subject}'
        subj_dir = os.path.join(deriv_dir, subj_id, f'{session}dwi')
        ptx_dir  = os.path.join(subj_dir, 'probtrackX', network_name)
        res      = check_subject(ptx_dir, pairs, networks)
        counts[res['status']] += 1
        rows.append((subject, res, ptx_dir))

    n_total = len(subjects)

    # -----------------------------------------------------------------------
    # Text summary
    # -----------------------------------------------------------------------
    print(f'\n=== Probtrackx output check: {network_name} ===\n')
    print(f'Config:           {config_file}')
    print(f'Subjects file:    {subjects_file}')
    print(f'Expected files:   {len(pairs) * 2} files/subject  '
          f'({len(pairs)} seed-target pairs x 2 file types)\n')

    pct = lambda n: f'{100*n/n_total:.1f}%' if n_total else '-'
    print(f'{"Complete":>22}  {counts["complete"]:>4}  ({pct(counts["complete"])})')
    print(f'{"Needs fix (old layout)":>22}  {counts["needs_fix"]:>4}  ({pct(counts["needs_fix"])})')
    print(f'{"Partial":>22}  {counts["partial"]:>4}  ({pct(counts["partial"])})')
    print(f'{"Missing entirely":>22}  {counts["missing_dir"]:>4}  ({pct(counts["missing_dir"])})')
    print(f'{"-"*45}')
    print(f'{"Total":>22}  {n_total:>4}')

    if counts['needs_fix'] > 0:
        print(f'\n  * {counts["needs_fix"]} subject(s) have the old per-network layout.')
        print(f'    Run preprocess/fix_probtrackx_structure.sh to migrate them.')

    # -----------------------------------------------------------------------
    # Per-subject table
    # -----------------------------------------------------------------------
    print(f'\n{"Subject":<35} {"Status":<12} {"Files":>12}')
    print('-' * 62)
    for subject, res, ptx_dir in rows:
        status = res['status'].upper().replace('_', ' ')
        files  = f'{res["n_found"]}/{res["n_expected"]}'
        extras = f'  ({res["fixable"]} fixable)' if res['status'] == 'needs_fix' else ''
        print(f'{subject:<35} {status:<12} {files:>12}{extras}')

    # Details for incomplete subjects
    incomplete = [(s, r, p) for s, r, p in rows if r['status'] != 'complete']
    if incomplete:
        print(f'\n--- Missing file details ---')
        for subject, res, ptx_dir in incomplete:
            if res['status'] == 'needs_fix':
                continue
            if res['missing_flat']:
                print(f'\n  {subject}  [{res["status"]}]')
                for mf in res['missing_flat'][:20]:
                    print(f'    {mf}')
                if len(res['missing_flat']) > 20:
                    print(f'    ... and {len(res["missing_flat"]) - 20} more')

    # -----------------------------------------------------------------------
    # Pie charts
    # -----------------------------------------------------------------------
    subj_stem = os.path.splitext(os.path.basename(subjects_file))[0]
    out_png   = f'{subj_stem}_{network_name}_check.png'
    try:
        save_plot(rows, pairs, counts, n_total, network_name,
                  config_file, subjects_file, out_png)
        print(f'\nPlot saved to {out_png}')
    except ImportError:
        print('\n(matplotlib not available - skipping plot)')

    print()
    return 0 if counts['complete'] == n_total else 1


if __name__ == '__main__':
    sys.exit(main())
