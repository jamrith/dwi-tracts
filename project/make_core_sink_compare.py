"""
Prototype of core-sink routing: instead of terminating at the first BA35
voxel touched (current behaviour -- stops at the anterior-superior lip near
amygdala/hippocampal head), the Dijkstra sinks are restricted to the BA35
CORE (voxels within CORE_RADIUS_MM of the mask medoid), so the route
continues through the ROI interior into perirhinal/TEC territory. No new
tractography: inside the ROI the density is floored at threshold and the
clearance term steers along the ROI's medial axis.

Also prototypes the endpoint-restoration fix: after pipeline smoothing +
recentring, the raw graph path's true endpoints (inside seed/sink by
construction) are re-appended if processing drifted them away.

Output: routes_core_sink_compare.html -- plotly 3D, both hemispheres,
current vs core-sink routes, with the sink region shown. 461 geometry.
"""
import os
import sys

import numpy as np
import nibabel as nib
from skimage import measure
import plotly.graph_objects as go

sys.path.insert(0, '/gpfs01/imgshare/ConnLS/ADNI/dwi-tracts')
sys.path.insert(0, '/gpfs01/imgshare/ConnLS/ADNI/dwi-tracts/project')
from dwitracts import utils as dwiutils
from route_dijkstra import extract_routes, recenter_polyline

DATA_ROOT = '/gpfs01/imgshare/ConnLS/ADNI'
NET_DIR = os.path.join(DATA_ROOT, 'tract_stats_nonbdp_thr007', 'LC-BA35-only-nonbdp-thr007')
ROIS_DIR = os.path.join(DATA_ROOT, 'dwi-tracts', 'data/rois/mtl-fix')
OUT_DIR = os.path.join(DATA_ROOT, 'dwi-tracts', 'project', 'route_dijkstra_out')
THR = 0.07
CORE_RADIUS_MM = 4.0

COLORS = {('lateral', 'entry'): '#a1d99b', ('medial', 'entry'): '#fdae6b',
          ('lateral', 'core'): '#006d2c', ('medial', 'core'): '#a63603'}


def roi_core(V_roi, affine, radius_mm):
    """Voxels of V_roi within radius_mm of the mask medoid (mask voxel
    nearest the centroid -- robust for curved/elongated ROIs whose centroid
    can fall outside the mask)."""
    idx = np.argwhere(V_roi > 0)
    mm = nib.affines.apply_affine(affine, idx)
    medoid = mm[np.argmin(np.linalg.norm(mm - mm.mean(axis=0), axis=1))]
    keep = np.linalg.norm(mm - medoid[None, :], axis=1) <= radius_mm
    M = np.zeros(V_roi.shape, dtype=float)
    M[tuple(idx[keep].T)] = 1.0
    return M, medoid


def smooth_like_pipeline(route_mm, header):
    T = np.linalg.inv(header.get_sform())
    route_vox = nib.affines.apply_affine(T, route_mm)
    route_vox = np.round(dwiutils.smooth_polyline_ma(np.round(route_vox), 3))
    return dwiutils.smooth_polyline_ma(dwiutils.voxel_to_world(route_vox, header), 7)


def restore_endpoints(proc_mm, raw_mm, tol=0.1):
    """Re-anchor the processed polyline on the raw graph path's endpoints,
    which lie inside the seed / sink by construction."""
    out = proc_mm
    if np.linalg.norm(out[0] - raw_mm[0]) > tol:
        out = np.vstack([raw_mm[0][None, :], out])
    if np.linalg.norm(out[-1] - raw_mm[-1]) > tol:
        out = np.vstack([out, raw_mm[-1][None, :]])
    return out


def finish(routes_raw, V_dens, header):
    named = {}
    for name, raw in routes_raw.items():
        sm = smooth_like_pipeline(raw, header)
        rc = recenter_polyline(sm, V_dens, header, THR)
        named[name] = restore_endpoints(rc, raw)
    return named


def mesh_from_volume(V, level, affine, crop_pad=2):
    nz = np.argwhere(V > level)
    lo = np.maximum(nz.min(axis=0) - crop_pad, 0)
    hi = np.minimum(nz.max(axis=0) + crop_pad + 1, np.array(V.shape))
    sub = V[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    verts, faces, _, _ = measure.marching_cubes(sub, level)
    return nib.affines.apply_affine(affine, verts + lo[None, :]), faces


def name_routes(routes, baseline):
    from scipy.spatial import cKDTree
    def mcd(a, b):
        return 0.5 * (cKDTree(b).query(a)[0].mean() + cKDTree(a).query(b)[0].mean())
    fits = [mcd(r, baseline) for r in routes]
    order = np.argsort(fits)
    return {'lateral': routes[int(order[0])], 'medial': routes[int(order[1])]}


def parse_poly3d(path):
    with open(path) as f:
        lines = f.read().strip().split('\n')
    n_pts = int(lines[1].split()[0])
    return np.array([list(map(float, lines[2 + i].split())) for i in range(n_pts)])


def main():
    fig = go.Figure()
    for hemi in ('L', 'R'):
        roi_a, roi_b = 'LC_{0}'.format(hemi), 'BA35_{0}'.format(hemi)
        img = nib.load(os.path.join(NET_DIR, 'average',
                                    'avr_min_tract_counts_{0}_{1}.nii.gz'.format(roi_a, roi_b)))
        aff = img.header.get_sform()
        V_dens = np.squeeze(img.get_fdata())
        V_seed = np.squeeze(nib.load(os.path.join(ROIS_DIR, roi_a + '.nii.gz')).get_fdata())
        V_targ = np.squeeze(nib.load(os.path.join(ROIS_DIR, roi_b + '.nii.gz')).get_fdata())
        baseline = parse_poly3d(os.path.join(NET_DIR, 'polylines',
                                             'maxes_{0}_{1}_sm3.poly3d'.format(roi_a, roi_b)))
        V_core, medoid = roi_core(V_targ, aff, CORE_RADIUS_MM)
        print(hemi, 'BA35 medoid at', medoid.round(1), '| core voxels:', int(V_core.sum()))

        variants = {}
        for tag, sink in (('entry', None), ('core', V_core)):
            routes, diags = extract_routes(V_dens, V_seed, V_targ, img.header, THR,
                                           V_sink=sink, n_routes=2)
            named = name_routes(routes, baseline)
            variants[tag] = finish(named, V_dens, img.header)
            for nm in ('lateral', 'medial'):
                print(' {0} {1} {2}: {3} pts, end {4}'.format(
                    hemi, tag, nm, len(variants[tag][nm]),
                    variants[tag][nm][-1].round(1)))

        # meshes
        v, f = mesh_from_volume(V_dens, THR, aff)
        fig.add_trace(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2],
                                i=f[:, 0], j=f[:, 1], k=f[:, 2],
                                color='gray', opacity=0.10, flatshading=True,
                                name='{0} shell >0.07'.format(hemi), hoverinfo='name'))
        for M, col, op, nm in ((V_seed, 'royalblue', 0.45, '{0} LC (seed)'.format(hemi)),
                               (V_targ, 'crimson', 0.25, '{0} BA35'.format(hemi)),
                               (V_core, 'gold', 0.85, '{0} BA35 core sink (r={1:.0f}mm)'
                                .format(hemi, CORE_RADIUS_MM))):
            v, f = mesh_from_volume(M, 0.5, aff)
            fig.add_trace(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2],
                                    i=f[:, 0], j=f[:, 1], k=f[:, 2],
                                    color=col, opacity=op, flatshading=True,
                                    name=nm, hoverinfo='name'))

        for tag, label, width, vis in (('entry', 'current (stop at entry)', 4, 'legendonly'),
                                       ('core', 'core sink (to TEC)', 6, True)):
            for nm in ('lateral', 'medial'):
                r = variants[tag][nm]
                fig.add_trace(go.Scatter3d(
                    x=r[:, 0], y=r[:, 1], z=r[:, 2], mode='lines',
                    line=dict(color=COLORS[(nm, tag)], width=width),
                    name='{0} {1} - {2}'.format(hemi, nm, label),
                    hoverinfo='name', visible=vis))

    fig.update_layout(
        title=('LC -> BA35 routes: stop-at-entry (pale, toggle on) vs BA35 core sink '
               '(dark) -- gold = sink region at the ROI medoid'),
        scene=dict(aspectmode='data',
                   xaxis_title='x (mm)', yaxis_title='y (mm)', zaxis_title='z (mm)'),
        legend=dict(itemsizing='constant', font=dict(size=10)),
        margin=dict(l=0, r=0, t=40, b=0))
    out_f = os.path.join(OUT_DIR, 'routes_core_sink_compare.html')
    fig.write_html(out_f, include_plotlyjs=True, full_html=True)
    print('wrote {0} ({1:.1f} MB)'.format(out_f, os.path.getsize(out_f) / 1e6))


if __name__ == '__main__':
    main()
