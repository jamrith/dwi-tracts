"""
Variant of make_routes_3d_html.py with a THRESHOLD SLIDER: the tract-shell
isosurface (probtrackx path-count density) is precomputed at a range of
thresholds and a plotly slider swaps which one is shown, for both
hemispheres at once. ROIs and routes are ordinary legend-toggleable traces
and are unaffected by the slider (the slider restyles only the shell
traces, so legend choices stick).

Note the routes themselves are NOT re-extracted per threshold -- they are
the fixed thr=0.07 results; the slider is for visually judging how the
shell (and the merge zone near the seed) grows/shrinks around them.
Self-contained HTML, fetch over sftp.
"""
import os

import numpy as np
import nibabel as nib
from skimage import measure
import plotly.graph_objects as go

PROJECT_ROOT = '/gpfs01/imgshare/ConnLS/ADNI/dwi-tracts'
DATA_ROOT = '/gpfs01/imgshare/ConnLS/ADNI'
NET_DIR = os.path.join(DATA_ROOT, 'tract_stats_nonbdp_thr007', 'LC-BA35-only-nonbdp-thr007')
ROIS_DIR = os.path.join(PROJECT_ROOT, 'data/rois/mtl-fix')
OUT_DIR = os.path.join(PROJECT_ROOT, 'project', 'route_dijkstra_out')

THRESHOLDS = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.14, 0.20, 0.30]
DEFAULT_IDX = THRESHOLDS.index(0.07)


def parse_poly3d(path):
    with open(path) as f:
        lines = f.read().strip().split('\n')
    n_pts = int(lines[1].split()[0])
    return np.array([list(map(float, lines[2 + i].split())) for i in range(n_pts)])


def mesh_from_volume(V, level, affine, crop_pad=2):
    nz = np.argwhere(V > level)
    lo = np.maximum(nz.min(axis=0) - crop_pad, 0)
    hi = np.minimum(nz.max(axis=0) + crop_pad + 1, np.array(V.shape))
    sub = V[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    verts, faces, _, _ = measure.marching_cubes(sub, level)
    verts = verts + lo[None, :]
    verts = nib.affines.apply_affine(affine, verts)
    return verts, faces


def main():
    fig = go.Figure()
    shell_trace_idx = {t: [] for t in THRESHOLDS}

    # 1. Shell traces first (their indices drive the slider)
    for hemi in ('L', 'R'):
        roi_a, roi_b = 'LC_{0}'.format(hemi), 'BA35_{0}'.format(hemi)
        img = nib.load(os.path.join(NET_DIR, 'average',
                                    'avr_min_tract_counts_{0}_{1}.nii.gz'.format(roi_a, roi_b)))
        aff = img.header.get_sform()
        V_dens = np.squeeze(img.get_fdata())
        for ti, thr in enumerate(THRESHOLDS):
            v, f = mesh_from_volume(V_dens, thr, aff)
            shell_trace_idx[thr].append(len(fig.data))
            fig.add_trace(go.Mesh3d(
                x=v[:, 0], y=v[:, 1], z=v[:, 2],
                i=f[:, 0], j=f[:, 1], k=f[:, 2],
                color='gray', opacity=0.12,
                name='{0} shell >{1}'.format(hemi, thr),
                hoverinfo='name', showlegend=False,
                visible=(ti == DEFAULT_IDX),
                flatshading=True,
                lighting=dict(ambient=0.55, diffuse=0.7, specular=0.1)))
        print('{0}: shells meshed'.format(hemi))

    # 2. ROIs and routes (legend-toggleable, untouched by the slider)
    for hemi in ('L', 'R'):
        roi_a, roi_b = 'LC_{0}'.format(hemi), 'BA35_{0}'.format(hemi)
        img = nib.load(os.path.join(NET_DIR, 'average',
                                    'avr_min_tract_counts_{0}_{1}.nii.gz'.format(roi_a, roi_b)))
        aff = img.header.get_sform()
        for roi, col, op in ((roi_a, 'royalblue', 0.45), (roi_b, 'crimson', 0.35)):
            V = np.squeeze(nib.load(os.path.join(ROIS_DIR, roi + '.nii.gz')).get_fdata())
            v, f = mesh_from_volume(V, 0.5, aff)
            fig.add_trace(go.Mesh3d(
                x=v[:, 0], y=v[:, 1], z=v[:, 2],
                i=f[:, 0], j=f[:, 1], k=f[:, 2],
                color=col, opacity=op, name=roi, hoverinfo='name',
                showlegend=True, flatshading=True,
                lighting=dict(ambient=0.55, diffuse=0.7, specular=0.1)))

        baseline = parse_poly3d(os.path.join(NET_DIR, 'polylines',
                                             'maxes_{0}_{1}_sm3.poly3d'.format(roi_a, roi_b)))
        fig.add_trace(go.Scatter3d(
            x=baseline[:, 0], y=baseline[:, 1], z=baseline[:, 2],
            mode='lines', line=dict(color='black', width=4),
            name='{0} greedy baseline'.format(hemi), hoverinfo='name'))

        variants = [('nodir', 'v1', ('#a1d99b', '#fdae6b'), 4, 'legendonly'),
                    ('v2', 'v2 clearance', ('#31a354', '#e6550d'), 5, 'legendonly'),
                    ('v2rc', 'v2 recentred', ('#006d2c', '#a63603'), 6, True)]
        for tag, label, cols, width, vis in variants:
            for i in (0, 1):
                p = os.path.join(OUT_DIR, 'route_{0}_{1}_{2}.poly3d'.format(hemi, tag, i))
                if not os.path.isfile(p):
                    continue
                r = parse_poly3d(p)
                fig.add_trace(go.Scatter3d(
                    x=r[:, 0], y=r[:, 1], z=r[:, 2],
                    mode='lines', line=dict(color=cols[i], width=width),
                    name='{0} {1} route {2}'.format(hemi, label, i + 1),
                    hoverinfo='name', visible=vis))

    # 3. Slider: restyle ONLY the shell traces
    all_shell_idx = sorted(sum(shell_trace_idx.values(), []))
    steps = []
    for thr in THRESHOLDS:
        vis = [(idx in shell_trace_idx[thr]) for idx in all_shell_idx]
        steps.append(dict(method='restyle',
                          args=[{'visible': vis}, all_shell_idx],
                          label='{0:g}'.format(thr)))

    fig.update_layout(
        title='LC -> BA35 routes with adjustable path-count shell threshold (mm, MNI)',
        scene=dict(aspectmode='data',
                   xaxis_title='x (mm)', yaxis_title='y (mm)', zaxis_title='z (mm)'),
        legend=dict(itemsizing='constant', font=dict(size=10)),
        sliders=[dict(active=DEFAULT_IDX, steps=steps,
                      currentvalue=dict(prefix='shell threshold: '),
                      pad=dict(t=10))],
        margin=dict(l=0, r=0, t=40, b=0))

    out_f = os.path.join(OUT_DIR, 'routes_3d_thresh.html')
    fig.write_html(out_f, include_plotlyjs=True, full_html=True)
    print('wrote {0} ({1:.1f} MB)'.format(out_f, os.path.getsize(out_f) / 1e6))


if __name__ == '__main__':
    main()
