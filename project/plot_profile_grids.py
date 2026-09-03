#!/usr/bin/env python3
"""
plot_profile_grids.py -- knob-comparison grids for the along-tract profile GLM.

One figure per (factor, tract). Within a figure:

    columns = shell scheme        nested:  FLOOD-FILL | ARC-LENGTH > 1.85mm / 2.5mm
    rows    = trace construction  nested:  VOXELWISE  | METRIC-FIRST > wmean / mean / max

Each panel draws the factor's t-value along the tract, with the clusters that
survive RFT (blue) and the permutation-family test (orange) marked both on the
trace and as strips below it. The top-left panel with its RFT strip is the
legacy DwiTractsGlm pipeline.

Trees are DISCOVERED from the output_dir naming convention rather than listed,
so a new knob combination appears in the grid as soon as it is run:

    profiles_{scheme}_{construction}_{aggregation}[_core{NN}]
      scheme        floodfill | arclenfine | arclen
      construction  voxelwise | metricfirst
      aggregation   max | wmean | mean

Valid inference differs per row and the script picks it automatically:
voxelwise -> 'perm' (selection-aware permutation, the only exactly valid test
for an argmax node statistic); metric-first -> 'np' (nonparam). RFT is drawn
in every panel for comparison but is mis-specified wherever the node statistic
is a max -- see PROFILE_GLM_SPEC.md sections 12.2 and 17.

Usage:
    python project/plot_profile_grids.py                      # all factors, both tracts
    python project/plot_profile_grids.py --factors AGE,AD
    python project/plot_profile_grids.py --include-core       # add core-threshold trees
    python project/plot_profile_grids.py --network-root <dir> --glm <name>
"""
import os
import re
import glob
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# -- defaults for the LC->BA35 lateral nonbdp461 reference case --------------
DEFAULT_ROOT = ('/share/ConnLS/ADNI/tract_stats_nonbdp_thr007/'
                'LC-BA35-lateral-nonbdp-thr007')
DEFAULT_GLM = 'glm-lc-ba35-lateral-v2-LR'
DEFAULT_TRACTS = [('LC_L_BA35_L', 'Left'), ('LC_R_BA35_R', 'Right')]

# Direction wording per factor, so the subtitle states the sign convention
# instead of leaving the reader to infer it.
DIRECTION = {
    'AGE': 'negative t = lower TSA with older age',
    'FEMALE': 'negative t = lower TSA in females (0 = male, 1 = female)',
    'AD': 'negative t = lower TSA in AD',
    'MCI': 'negative t = lower TSA in MCI',
}

# Palette: text tokens for ink, categorical slots 1 and 2 for the two
# inference families (see the dataviz reference palette).
INK, INK2, MUTED, SURF = '#0b0b0b', '#52514e', '#8a8a85', '#fcfcfb'
C_RFT, C_PERM = '#2a78d6', '#eb6834'

SCHEME_ORDER = ['floodfill', 'arclenfine', 'arclen']
SCHEME_LABEL = {'floodfill': ('flood-fill', 'hop count'),
                'arclenfine': ('arc-length', '1.85 mm  (matched)'),
                'arclen': ('arc-length', '2.5 mm  (coarse)')}
ROW_ORDER = [('voxelwise', 'max'), ('metricfirst', 'wmean'),
             ('metricfirst', 'mean'), ('metricfirst', 'max')]
ROW_LABEL = {('voxelwise', 'max'): ('VOXELWISE', 'max  (legacy)'),
             ('metricfirst', 'wmean'): ('METRIC-FIRST', 'wmean'),
             ('metricfirst', 'mean'): ('METRIC-FIRST', 'mean'),
             ('metricfirst', 'max'): ('METRIC-FIRST', 'max')}

TREE_RE = re.compile(r'^profiles_(floodfill|arclenfine|arclen)_'
                     r'(voxelwise|metricfirst)_(max|wmean|mean)(_core\d+)?$')


def discover_trees(root, include_core=False):
    """Map (scheme, construction, aggregation) -> tree directory name."""
    found = {}
    for path in sorted(glob.glob(os.path.join(root, 'profiles_*'))):
        m = TREE_RE.match(os.path.basename(path))
        if not m:
            continue
        scheme, construction, agg, core = m.groups()
        if core and not include_core:
            continue
        found[(scheme, construction, agg)] = os.path.basename(path)
    return found


def load_stats(root, tree, glm, tract):
    """The along-tract stats table for one tree/tract, or None."""
    hits = glob.glob(os.path.join(root, tree, glm, 'summary-*',
                                  'stats_{0}.csv'.format(tract)))
    return pd.read_csv(hits[0]) if hits else None


def draw_panel(ax, T, factor, perm_sfx, ylim, ticks, strips, last_row, first_col):
    """One cell: t-trace, zero line, significant runs, per-method strips."""
    ax.set_facecolor(SURF)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color(MUTED)
        ax.spines[sp].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=7.5, length=3)
    ax.set_ylim(*ylim)
    ax.set_xlim(-0.03, 1.03)
    ax.set_yticks(ticks)
    if not first_col:
        ax.set_yticklabels([])
    if last_row:
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(['LC', '', 'BA35'], fontsize=8)
    else:
        ax.set_xticklabels([])

    if T is None:
        ax.text(0.5, 0.5, 'not run', ha='center', va='center',
                color=MUTED, fontsize=8.5, transform=ax.transAxes)
        return

    d = T['Distance'].values.astype(float)
    K = len(d)
    # Normalised position: flood-fill nodes are hop counts and arc-length
    # nodes are millimetres, so native units are not comparable across columns.
    x = (d - d.min()) / (d.max() - d.min()) if d.max() > d.min() else np.zeros(K)
    t = T['{0}|tval'.format(factor)].values
    ax.axhline(0, color=MUTED, lw=0.7, zorder=1)
    ax.plot(x, t, color=INK2, lw=1.5, zorder=3, solid_capstyle='round')

    notes = []
    for k, (sfx, disp) in enumerate((('rft', 'RFT'), (perm_sfx, 'Perm'))):
        col = C_RFT if sfx == 'rft' else C_PERM
        cc = '{0}|{1}_clusters'.format(factor, sfx)
        if cc not in T.columns:
            notes.append('{0} n/a'.format(disp))
            continue
        c = T[cc].values
        p = T['{0}|{1}_pval'.format(factor, sfx)].values
        n = int((c > 0).sum())
        for cid in np.unique(c[c > 0]):
            m = np.flatnonzero(c == cid)
            ax.plot([x[m.min()], x[m.max()]], [strips[k]] * 2, color=col, lw=4.2,
                    solid_capstyle='round', zorder=4)
            ax.plot(x[m], t[m], color=col, lw=2.8, zorder=5, solid_capstyle='round')
        notes.append('{0} {1}/{2} {3:.0e}'.format(disp, n, K, p[c > 0].min())
                     if n else '{0}  --'.format(disp))
    ax.text(0.975, 0.96, '\n'.join(notes), transform=ax.transAxes, ha='right',
            va='top', fontsize=7.0, color=INK2, linespacing=1.5)
    ax.text(0.02, 0.96, 'K={0}'.format(K), transform=ax.transAxes, ha='left',
            va='top', fontsize=7.0, color=MUTED)


def make_grid(root, glm, trees, factor, tract, side, out_dir, dpi=200):
    cols = [s for s in SCHEME_ORDER if any(k[0] == s for k in trees)]
    rows = [r for r in ROW_ORDER if any((k[1], k[2]) == r for k in trees)]
    if not cols or not rows:
        print('  nothing to plot for {0}/{1}'.format(factor, tract))
        return None

    tables = {}
    for j, scheme in enumerate(cols):
        for i, (constr, agg) in enumerate(rows):
            tree = trees.get((scheme, constr, agg))
            tables[(i, j)] = load_stats(root, tree, glm, tract) if tree else None

    vals = [T['{0}|tval'.format(factor)].values for T in tables.values() if T is not None]
    if not vals:
        return None
    hi = float(max(v.max() for v in vals))
    lo = float(min(v.min() for v in vals))
    span = hi - lo
    # Reserve a band below the data for the significance strips so they can
    # never collide with the trace or the per-panel annotation.
    strips = (lo - 0.20 * span, lo - 0.32 * span)
    ylim = (lo - 0.42 * span, hi + 0.32 * span)   # headroom for the 2-line annotation
    step = 2.5 if span < 22 else 5.0
    ticks = [v for v in np.arange(-30, 30.1, step)
             if lo - 0.02 * span <= v <= hi + 0.02 * span]

    fig = plt.figure(figsize=(12.6, 10.4))
    fig.patch.set_facecolor(SURF)
    L, Rt, TP, BT = 0.200, 0.985, 0.815, 0.085
    gs = fig.add_gridspec(len(rows), len(cols), left=L, right=Rt, top=TP,
                          bottom=BT, hspace=0.30, wspace=0.10)
    axes = np.empty((len(rows), len(cols)), dtype=object)
    for i, (constr, agg) in enumerate(rows):
        perm_sfx = 'perm' if constr == 'voxelwise' else 'np'
        for j in range(len(cols)):
            ax = fig.add_subplot(gs[i, j])
            axes[i, j] = ax
            draw_panel(ax, tables[(i, j)], factor, perm_sfx, ylim, ticks, strips,
                       last_row=(i == len(rows) - 1), first_col=(j == 0))

    box = lambda i, j: axes[i, j].get_position()

    # nested column labels
    sub_y, sup_y = TP + 0.022, TP + 0.062
    for j, scheme in enumerate(cols):
        b = box(0, j)
        fig.text((b.x0 + b.x1) / 2, sub_y, SCHEME_LABEL[scheme][1],
                 ha='center', va='bottom', fontsize=9.5, color=INK2)
    groups = {}
    for j, scheme in enumerate(cols):
        groups.setdefault(SCHEME_LABEL[scheme][0], []).append(j)
    for sup, js in groups.items():
        x0, x1 = box(0, min(js)).x0, box(0, max(js)).x1
        fig.text((x0 + x1) / 2, sup_y + 0.012, '  '.join(sup.upper()), ha='center',
                 va='bottom', fontsize=12, color=INK, fontweight='bold')
        fig.add_artist(Line2D([x0, x1], [sup_y + 0.005], color=INK, lw=1.4,
                              transform=fig.transFigure))

    # nested row labels
    sub_x, sup_x = L - 0.058, L - 0.135
    for i, r in enumerate(rows):
        b = box(i, 0)
        fig.text(sub_x, (b.y0 + b.y1) / 2, ROW_LABEL[r][1], ha='right',
                 va='center', fontsize=9.5, color=INK2)
    rgroups = {}
    for i, r in enumerate(rows):
        rgroups.setdefault(ROW_LABEL[r][0], []).append(i)
    for sup, iss in rgroups.items():
        y0, y1 = box(max(iss), 0).y0, box(min(iss), 0).y1
        fig.text(sup_x - 0.012, (y0 + y1) / 2, sup, ha='center', va='center',
                 fontsize=12.5, color=INK, fontweight='bold', rotation=90)
        fig.add_artist(Line2D([sup_x - 0.001] * 2, [y0, y1], color=INK, lw=1.4,
                              transform=fig.transFigure))

    pair = tract.replace('_BA35', ' → BA35')
    fig.suptitle('{0} hemisphere — {1} effect along the tract  ({2})'
                 .format(side, factor, pair), fontsize=15, color=INK, y=0.987)
    fig.text(0.5, 0.949, 'nonbdp461 · 457 subjects · covariates AGE, FEMALE, AD, MCI · '
             + DIRECTION.get(factor, ''), ha='center', fontsize=9, color=INK2)
    handles = [Line2D([], [], color=INK2, lw=1.6, label='{0} t along tract'.format(factor)),
               Line2D([], [], color=C_RFT, lw=4, solid_capstyle='round', label='RFT cluster'),
               Line2D([], [], color=C_PERM, lw=4, solid_capstyle='round',
                      label='Permutation cluster')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.937),
               ncol=3, frameon=False, fontsize=9, labelcolor=INK2)
    fig.text(L, 0.028, 'x = normalised position (LC → BA35).  Top-left panel with its RFT '
             'strip is the legacy pipeline.  Permutation p floored at 1/5001 = 2e-4.',
             fontsize=7.4, color=MUTED, ha='left')
    fig.text(L, 0.010, 'Valid inference per row: VOXELWISE → selection-aware permutation;  '
             'METRIC-FIRST → nonparam.  RFT is mis-specified on any max-based node statistic.',
             fontsize=7.4, color=MUTED, ha='left')

    stem = os.path.join(out_dir, '{0}_grid_{1}'.format(factor, side.lower()))
    for ext in ('png', 'pdf'):
        fig.savefig('{0}.{1}'.format(stem, ext), dpi=dpi, facecolor=SURF)
    plt.close(fig)
    return stem


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--network-root', default=DEFAULT_ROOT)
    ap.add_argument('--glm', default=DEFAULT_GLM)
    ap.add_argument('--factors', default='AGE,FEMALE,AD,MCI')
    ap.add_argument('--out', default=None,
                    help='output dir (default: <network-root>/profiles_comparison)')
    ap.add_argument('--include-core', action='store_true',
                    help='also include profile_core_threshold trees')
    ap.add_argument('--dpi', type=int, default=200)
    args = ap.parse_args()

    out_dir = args.out or os.path.join(args.network_root, 'profiles_comparison')
    os.makedirs(out_dir, exist_ok=True)

    trees = discover_trees(args.network_root, include_core=args.include_core)
    if not trees:
        print('No profiles_* trees found under {0}'.format(args.network_root))
        return 1
    print('Discovered {0} trees:'.format(len(trees)))
    for k in sorted(trees):
        print('   {0:<12} {1:<12} {2:<6} -> {3}'.format(k[0], k[1], k[2], trees[k]))

    for factor in [f.strip() for f in args.factors.split(',') if f.strip()]:
        for tract, side in DEFAULT_TRACTS:
            stem = make_grid(args.network_root, args.glm, trees, factor, tract,
                             side, out_dir, dpi=args.dpi)
            if stem:
                print('wrote {0}.png/.pdf'.format(stem))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
