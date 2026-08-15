"""
Interactive plotly 3D view of the Dijkstra route-extraction results:
translucent tract-shell isosurface (density > 0.07), LC/BA35 ROI meshes,
pipeline greedy baseline polyline, and the v1/v2/v2-recentred routes for
both hemispheres. Everything in world mm. Fully self-contained HTML
(plotly.js embedded) -- fetch over sftp and open locally.
"""
import os
import sys

import numpy as np
import nibabel as nib
from skimage import measure
import plotly.graph_objects as go

PROJECT_ROOT = '/gpfs01/imgshare/ConnLS/ADNI/dwi-tracts'
DATA_ROOT = '/gpfs01/imgshare/ConnLS/ADNI'
NET_DIR = os.path.join(DATA_ROOT, 'tract_stats_nonbdp_thr007', 'LC-BA35-only-nonbdp-thr007')
ROIS_DIR = os.path.join(PROJECT_ROOT, 'data/rois/mtl-fix')
OUT_DIR = os.path.join(PROJECT_ROOT, 'project', 'route_dijkstra_out')
THRESHOLD = 0.07


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


def add_mesh(fig, verts, faces, color, opacity, name, group, visible=True):
    fig.add_trace(go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color=color, opacity=opacity, name=name,
        legendgroup=group, showlegend=True, hoverinfo='name',
        visible=True if visible else 'legendonly',
        flatshading=True,
        lighting=dict(ambient=0.55, diffuse=0.7, specular=0.1)))


def add_line(fig, curve, color, width, name, group, visible=True):
    fig.add_trace(go.Scatter3d(
        x=curve[:, 0], y=curve[:, 1], z=curve[:, 2],
        mode='lines', line=dict(color=color, width=width),
        name=name, legendgroup=group, hoverinfo='name',
        visible=True if visible else 'legendonly'))


def main():
    fig = go.Figure()
    for hemi in ('L', 'R'):
        roi_a, roi_b = 'LC_{0}'.format(hemi), 'BA35_{0}'.format(hemi)
        img = nib.load(os.path.join(NET_DIR, 'average',
                                    'avr_min_tract_counts_{0}_{1}.nii.gz'.format(roi_a, roi_b)))
        aff = img.header.get_sform()
        V_dens = np.squeeze(img.get_fdata())
        V_seed = np.squeeze(nib.load(os.path.join(ROIS_DIR, roi_a + '.nii.gz')).get_fdata())
        V_target = np.squeeze(nib.load(os.path.join(ROIS_DIR, roi_b + '.nii.gz')).get_fdata())

        v, f = mesh_from_volume(V_dens, THRESHOLD, aff)
        add_mesh(fig, v, f, 'gray', 0.12, '{0} tract shell (>0.07)'.format(hemi), 'shell' + hemi)
        v, f = mesh_from_volume(V_seed, 0.5, aff)
        add_mesh(fig, v, f, 'royalblue', 0.45, '{0} LC (seed)'.format(hemi), 'seed' + hemi)
        v, f = mesh_from_volume(V_target, 0.5, aff)
        add_mesh(fig, v, f, 'crimson', 0.35, '{0} BA35 (target)'.format(hemi), 'targ' + hemi)

        baseline = parse_poly3d(os.path.join(NET_DIR, 'polylines',
                                             'maxes_{0}_{1}_sm3.poly3d'.format(roi_a, roi_b)))
        add_line(fig, baseline, 'black', 4, '{0} greedy baseline'.format(hemi), 'base' + hemi)

        variants = [('nodir', 'v1', ('#a1d99b', '#fdae6b'), 4, False),
                    ('v2', 'v2 clearance', ('#31a354', '#e6550d'), 5, False),
                    ('v2rc', 'v2 recentred', ('#006d2c', '#a63603'), 6, True)]
        for tag, label, cols, width, vis in variants:
            for i in (0, 1):
                p = os.path.join(OUT_DIR, 'route_{0}_{1}_{2}.poly3d'.format(hemi, tag, i))
                if not os.path.isfile(p):
                    continue
                add_line(fig, parse_poly3d(p), cols[i], width,
                         '{0} {1} route {2}'.format(hemi, label, i + 1),
                         '{0}{1}'.format(tag, hemi), visible=vis)

    fig.update_layout(
        title='LC -> BA35 route extraction (thr=0.07): shells, ROIs, and routes (mm, MNI)',
        scene=dict(aspectmode='data',
                   xaxis_title='x (mm)', yaxis_title='y (mm)', zaxis_title='z (mm)'),
        legend=dict(itemsizing='constant', font=dict(size=10)),
        margin=dict(l=0, r=0, t=40, b=0))

    out_f = os.path.join(OUT_DIR, 'routes_3d.html')
    fig.write_html(out_f, include_plotlyjs=True, full_html=True)
    print('wrote {0} ({1:.1f} MB)'.format(out_f, os.path.getsize(out_f) / 1e6))


if __name__ == '__main__':
    main()
