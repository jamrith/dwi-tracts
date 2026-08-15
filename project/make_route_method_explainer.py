"""
Builds route_method_explainer.html: a self-contained explainer of the
Dijkstra route-extraction method (route_dijkstra.py), illustrated with the
ACTUAL intermediate arrays from the right-hemisphere LC->BA35 run
(density data, EDT/clearance cost, Dijkstra cost-to-reach wavefront,
suppression corridor, recentring cross-sections), plus interactive plotly
3D/line figures. Matplotlib PNGs are embedded as base64; plotly.js is
inlined -- fetch the single file over sftp and open locally.
"""
import os
import io
import sys
import base64

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from scipy import ndimage
from skimage import measure
import plotly.graph_objects as go

sys.path.insert(0, '/gpfs01/imgshare/ConnLS/ADNI/dwi-tracts')
sys.path.insert(0, '/gpfs01/imgshare/ConnLS/ADNI/dwi-tracts/project')
from dwitracts import utils as dwiutils
from route_dijkstra import extract_routes, recenter_polyline

PROJECT_ROOT = '/gpfs01/imgshare/ConnLS/ADNI/dwi-tracts'
DATA_ROOT = '/gpfs01/imgshare/ConnLS/ADNI'
NET_DIR = os.path.join(DATA_ROOT, 'tract_stats_nonbdp_thr007', 'LC-BA35-only-nonbdp-thr007')
ROIS_DIR = os.path.join(PROJECT_ROOT, 'data/rois/mtl-fix')
OUT_DIR = os.path.join(PROJECT_ROOT, 'project', 'route_dijkstra_out')
FIG_DIR = os.path.join(OUT_DIR, 'explainer_figs')
THR = 0.07
HEMI = 'R'

os.makedirs(FIG_DIR, exist_ok=True)

GREEN, ORANGE = '#31a354', '#e6550d'
DGREEN, DORANGE = '#006d2c', '#a63603'


def fig_to_b64(fig, name):
    fig.savefig(os.path.join(FIG_DIR, name), dpi=170, bbox_inches='tight')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=170, bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('ascii')


def world_to_vox(curve_mm, header):
    T = np.linalg.inv(header.get_sform())
    return nib.affines.apply_affine(T, curve_mm)


def vol_from_nodes(values, coords, shape, fill=np.nan):
    W = np.full(shape, fill, dtype=float)
    W[tuple(coords.T)] = values
    return W


def crop_lims(mask2d, pad=6):
    nz = np.argwhere(mask2d)
    return (nz[:, 1].min() - pad, nz[:, 1].max() + pad,
            nz[:, 0].min() - pad, nz[:, 0].max() + pad)


def style_ax(ax, lims):
    ax.set_xlim(lims[0], lims[1]); ax.set_ylim(lims[2], lims[3])
    ax.set_xticks([]); ax.set_yticks([])


# ---------------- load data + run the method with capture ----------------
roi_a, roi_b = 'LC_{0}'.format(HEMI), 'BA35_{0}'.format(HEMI)
img = nib.load(os.path.join(NET_DIR, 'average',
                            'avr_min_tract_counts_{0}_{1}.nii.gz'.format(roi_a, roi_b)))
V_dens = np.squeeze(img.get_fdata())
V_seed = np.squeeze(nib.load(os.path.join(ROIS_DIR, roi_a + '.nii.gz')).get_fdata())
V_target = np.squeeze(nib.load(os.path.join(ROIS_DIR, roi_b + '.nii.gz')).get_fdata())
seed_m, targ_m = V_seed > 0, V_target > 0

cap = {}
routes, diags = extract_routes(V_dens, V_seed, V_target, img.header, THR,
                               n_routes=2, capture=cap, verbose=True)
coords = cap['coords']
shape = V_dens.shape


def smooth(r_mm):
    r_vox = np.round(world_to_vox(r_mm, img.header))
    r_vox = np.round(dwiutils.smooth_polyline_ma(r_vox, 3))
    return dwiutils.smooth_polyline_ma(dwiutils.voxel_to_world(r_vox, img.header), 7)


sm = [smooth(r) for r in routes]
rc = [recenter_polyline(r, V_dens, img.header, THR) for r in sm]
raw_vox = [world_to_vox(r, img.header) for r in routes]
sm_vox = [world_to_vox(r, img.header) for r in sm]
rc_vox = [world_to_vox(r, img.header) for r in rc]

# ---------------- F1: the raw data ----------------
fig, axes = plt.subplots(1, 3, figsize=(8.6, 3.0))
vmax = np.percentile(V_dens[V_dens > 0], 99.5)
for ax, axis, name in zip(axes, (0, 1, 2), ('sagittal', 'coronal', 'axial')):
    mip = V_dens.max(axis=axis).T
    ax.imshow(mip, origin='lower', cmap='gray_r',
              norm=mcolors.PowerNorm(0.5, vmin=0, vmax=vmax))
    ax.contour(seed_m.max(axis=axis).T, levels=[0.5], colors='tab:blue', linewidths=0.8)
    ax.contour(targ_m.max(axis=axis).T, levels=[0.5], colors='tab:red', linewidths=0.8)
    style_ax(ax, crop_lims(mip > 0.005))
    ax.set_title('{0} MIP'.format(name), fontsize=8)
fig.suptitle('probtrackx path-count density (avr_min_tract_counts), {0} hemisphere'.format(HEMI),
             fontsize=9)
F1 = fig_to_b64(fig, 'f1_data.png')

# ---------------- F2: cost ingredients on a sagittal slice ----------------
x0 = int(np.median(raw_vox[0][:, 0]))
dens_sl = V_dens[x0].T
dom_sl = cap['domain'][x0].T
V_edt = vol_from_nodes(cap['edt_mm'], coords, shape)
V_clr = vol_from_nodes(1.0 / (cap['edt_mm'] + 0.5), coords, shape)
V_rdg = vol_from_nodes(np.power(1.0 - cap['dens_n'], 2.0), coords, shape)
V_nc = vol_from_nodes(cap['node_cost_base'], coords, shape)
lims = crop_lims(dom_sl, pad=5)

fig, axes = plt.subplots(1, 4, figsize=(10.5, 2.9))
panels = [(dens_sl, 'gray_r', 'density (slice x={0})'.format(x0), None),
          (V_edt[x0].T, 'viridis', 'clearance EDT (mm)', None),
          (V_clr[x0].T, 'magma_r', 'clearance cost 1/(EDT+0.5)', 1.6),
          (V_nc[x0].T, 'magma_r', 'total node cost / mm', 2.2)]
for ax, (S, cm, title, vmx) in zip(axes, panels):
    im = ax.imshow(S, origin='lower', cmap=cm, vmax=vmx)
    ax.contour(dom_sl, levels=[0.5], colors='k', linewidths=0.5)
    on = np.abs(raw_vox[0][:, 0] - x0) <= 1
    ax.plot(raw_vox[0][on, 1], raw_vox[0][on, 2], '.', color=GREEN, ms=3)
    style_ax(ax, lims)
    ax.set_title(title, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=6)
F2 = fig_to_b64(fig, 'f2_cost.png')

# ---------------- F3: turn penalty ----------------
ang = np.linspace(0, 90, 200)
cost = 1.0 * (1.0 - np.cos(np.radians(ang)))
fig, ax = plt.subplots(figsize=(3.4, 2.4))
ok = ang <= 60
ax.plot(ang[ok], cost[ok], color='k')
ax.axvspan(60, 90, color='crimson', alpha=0.15)
ax.text(75, 0.25, 'forbidden\n(> 60 deg)', ha='center', fontsize=7, color='crimson')
ax.set_xlabel('turn angle per step (deg)', fontsize=8)
ax.set_ylabel('added cost  w_turn (1-cos)', fontsize=8)
ax.tick_params(labelsize=7)
F3 = fig_to_b64(fig, 'f3_turn.png')

# ---------------- F4: Dijkstra cost-to-reach wavefront ----------------
V_ctr = vol_from_nodes(cap['routes'][0]['cost_to_reach'], coords, shape)
fig, axes = plt.subplots(1, 3, figsize=(8.6, 3.0))
for ax, axis, name in zip(axes, (0, 1, 2), ('sagittal', 'coronal', 'axial')):
    proj = np.nanmin(np.where(np.isfinite(V_ctr), V_ctr, np.nan), axis=axis).T
    im = ax.imshow(proj, origin='lower', cmap='viridis', vmax=np.nanpercentile(proj, 98))
    keep = [i for i in range(3) if i != axis]
    ax.plot(raw_vox[0][:, keep[0]], raw_vox[0][:, keep[1]], '-', color='w', lw=1.6)
    ax.plot(raw_vox[0][:, keep[0]], raw_vox[0][:, keep[1]], '-', color=GREEN, lw=1.0)
    ax.contour(seed_m.max(axis=axis).T, levels=[0.5], colors='tab:blue', linewidths=0.8)
    ax.contour(targ_m.max(axis=axis).T, levels=[0.5], colors='tab:red', linewidths=0.8)
    style_ax(ax, crop_lims(np.isfinite(proj)))
    ax.set_title('{0} (min-projection)'.format(name), fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=6)
fig.suptitle('accumulated cost-to-reach from LC (Dijkstra wavefront) + optimal lateral route', fontsize=9)
F4 = fig_to_b64(fig, 'f4_wavefront.png')

# ---------------- F5: Gaussian suppression field -> medial route ----------------
pen = cap['routes'][1]['node_cost'] - cap['node_cost_base']
V_pen = vol_from_nodes(pen, coords, shape, fill=0)
V_exempt = vol_from_nodes(cap['exempt'].astype(float), coords, shape, fill=0)
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4))
for ax, axis, name in zip(axes, (1, 2), ('coronal', 'axial')):
    mip = V_dens.max(axis=axis).T
    ax.imshow(mip, origin='lower', cmap='gray_r',
              norm=mcolors.PowerNorm(0.5, vmin=0, vmax=vmax))
    P = V_pen.max(axis=axis).T
    im = ax.imshow(np.ma.masked_less(P, 0.05), origin='lower', cmap='Reds',
                   alpha=0.65, vmin=0, vmax=10)
    ax.contour(V_exempt.max(axis=axis).T, levels=[0.5], colors='tab:blue',
               linewidths=0.8, linestyles='dotted')
    keep = [i for i in range(3) if i != axis]
    ax.plot(raw_vox[0][:, keep[0]], raw_vox[0][:, keep[1]], '-', color=GREEN, lw=1.5,
            label='lateral (route 1)')
    ax.plot(raw_vox[1][:, keep[0]], raw_vox[1][:, keep[1]], '-', color=ORANGE, lw=1.5,
            label='medial (route 2)')
    style_ax(ax, crop_lims(mip > 0.005))
    ax.set_title(name, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=6)
axes[0].legend(fontsize=7, loc='upper left')
fig.suptitle('Gaussian suppression field around the lateral route (peak +10/mm, sigma 4 mm) '
             'with ROI-end exemption (blue dotted)', fontsize=9)
F5 = fig_to_b64(fig, 'f5_suppression.png')
supp = pen > 0.5  # for the 3D corridor cloud


# ---------------- cross-section sampler ----------------
def cross_section(p_mm, t_mm, half_mm=10.0, step=0.5):
    t = t_mm / np.linalg.norm(t_mm)
    a = np.array([0.0, 0.0, 1.0]) if abs(t[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(t, a); u /= np.linalg.norm(u)
    v = np.cross(t, u)
    s = np.arange(-half_mm, half_mm + 1e-6, step)
    U, W = np.meshgrid(s, s)
    pts = p_mm[None, None, :] + U[..., None] * u[None, None, :] + W[..., None] * v[None, None, :]
    vox = world_to_vox(pts.reshape(-1, 3), img.header).T
    D = ndimage.map_coordinates(V_dens, vox, order=1).reshape(U.shape)
    return D, u, v, s


def tangent(curve, i):
    j0, j1 = max(0, i - 1), min(len(curve) - 1, i + 1)
    return curve[j1] - curve[j0]


# ---------------- F6: recentring ----------------
fig, axes = plt.subplots(1, 3, figsize=(8.6, 3.0))
for ax, frac in zip(axes, (0.3, 0.5, 0.7)):
    i = int(frac * (len(sm[0]) - 1))
    p_old = sm[0][i]
    t = tangent(sm[0], i)
    D, u, v, s = cross_section(p_old, t)
    ax.imshow(D, origin='lower', cmap='gray_r',
              extent=[s[0], s[-1], s[0], s[-1]],
              norm=mcolors.PowerNorm(0.5, vmin=0, vmax=vmax))
    ax.contour(D, levels=[THR], colors='k', linewidths=0.5,
               extent=[s[0], s[-1], s[0], s[-1]])
    j = int(np.argmin(np.linalg.norm(rc[0] - p_old[None, :], axis=1)))
    d_new = rc[0][j] - p_old
    ax.plot(0, 0, 'o', color='gray', ms=5, label='before')
    ax.plot(d_new @ u, d_new @ v, 'o', color=DGREEN, ms=5, label='after recentring')
    ax.set_title('cross-section at {0:.0f}% arclength'.format(100 * frac), fontsize=8)
    ax.tick_params(labelsize=6)
axes[0].legend(fontsize=7, loc='upper left')
axes[0].set_ylabel('mm (in-plane)', fontsize=7)
fig.suptitle('recentring: vertex pulled to density-weighted centroid of its cross-section '
             '(black contour = 0.07)', fontsize=9)
F6 = fig_to_b64(fig, 'f6_recenter.png')

# ---------------- F7: why the routes merge near the target ----------------
targ_mm = nib.affines.apply_affine(img.header.get_sform(), np.argwhere(targ_m))
targ_c = targ_mm.mean(axis=0)
d_to_t = np.linalg.norm(rc[0] - targ_c[None, :], axis=1)
wants = np.linspace(25.0, d_to_t.min(), 4)
fig, axes = plt.subplots(1, 4, figsize=(10.5, 2.9))
for ax, want in zip(axes, wants):
    i = int(np.argmin(np.abs(d_to_t - want)))
    p = rc[0][i]
    D, u, v, s = cross_section(p, tangent(rc[0], i))
    ax.imshow(D, origin='lower', cmap='gray_r',
              extent=[s[0], s[-1], s[0], s[-1]],
              norm=mcolors.PowerNorm(0.5, vmin=0, vmax=vmax))
    ax.contour(D, levels=[THR], colors='k', linewidths=0.5,
               extent=[s[0], s[-1], s[0], s[-1]])
    ax.plot(0, 0, 'o', color=DGREEN, ms=5)
    j = int(np.argmin(np.linalg.norm(rc[1] - p[None, :], axis=1)))
    d2 = rc[1][j] - p
    ax.plot(d2 @ u, d2 @ v, 'o', color=DORANGE, ms=5)
    ax.set_title('{0:.0f} mm from BA35 centre'.format(d_to_t[i]), fontsize=8)
    ax.tick_params(labelsize=6)
fig.suptitle('cross-sections approaching BA35: the two shells (green=lateral, orange=medial) '
             'collapse into one blob', fontsize=9)
F7 = fig_to_b64(fig, 'f7_merge.png')

# ---------------- P1: plotly 3D scene ----------------
def mesh_from_volume(V, level, affine, crop_pad=2):
    nz = np.argwhere(V > level)
    lo = np.maximum(nz.min(axis=0) - crop_pad, 0)
    hi = np.minimum(nz.max(axis=0) + crop_pad + 1, np.array(V.shape))
    sub = V[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    verts, faces, _, _ = measure.marching_cubes(sub, level)
    return nib.affines.apply_affine(affine, verts + lo[None, :]), faces


aff = img.header.get_sform()
p1 = go.Figure()
v, f = mesh_from_volume(V_dens, THR, aff)
p1.add_trace(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2],
                       color='gray', opacity=0.12, name='shell >0.07', hoverinfo='name',
                       flatshading=True))
for M, col, nm in ((V_seed, 'royalblue', 'LC (seed)'), (V_target, 'crimson', 'BA35 (target)')):
    v, f = mesh_from_volume(M, 0.5, aff)
    p1.add_trace(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2],
                           color=col, opacity=0.4, name=nm, hoverinfo='name', flatshading=True))
corr_mm = nib.affines.apply_affine(aff, coords[supp])
sub = corr_mm[::3]
p1.add_trace(go.Scatter3d(x=sub[:, 0], y=sub[:, 1], z=sub[:, 2], mode='markers',
                          marker=dict(size=2, color='crimson', opacity=0.25),
                          name='suppression corridor', hoverinfo='name'))
for r, col, nm in ((rc[0], DGREEN, 'lateral (recentred)'), (rc[1], DORANGE, 'medial (recentred)')):
    p1.add_trace(go.Scatter3d(x=r[:, 0], y=r[:, 1], z=r[:, 2], mode='lines',
                              line=dict(color=col, width=6), name=nm, hoverinfo='name'))
p1.update_layout(scene=dict(aspectmode='data'), height=620,
                 legend=dict(itemsizing='constant', font=dict(size=10)),
                 margin=dict(l=0, r=0, t=10, b=0))
P1 = p1.to_html(full_html=False, include_plotlyjs='inline')

# ---------------- P2: along-route profiles ----------------
def route_profile(r_mm):
    vox = world_to_vox(r_mm, img.header).T
    dens = ndimage.map_coordinates(V_dens, vox, order=1)
    V_edt_f = vol_from_nodes(cap['edt_mm'], coords, shape, fill=0)
    edt = ndimage.map_coordinates(V_edt_f, vox, order=1)
    arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(r_mm, axis=0), axis=1))])
    return arc, dens, edt


p2 = go.Figure()
for r, col, nm in ((rc[0], DGREEN, 'lateral'), (rc[1], DORANGE, 'medial')):
    arc, dens, edt = route_profile(r)
    p2.add_trace(go.Scatter(x=arc, y=dens, name='{0}: density'.format(nm),
                            line=dict(color=col)))
    p2.add_trace(go.Scatter(x=arc, y=edt, name='{0}: clearance (mm)'.format(nm),
                            line=dict(color=col, dash='dot'), yaxis='y2'))
p2.add_hline(y=THR, line=dict(color='gray', dash='dash'))
p2.update_layout(height=380, xaxis_title='arclength from LC (mm)',
                 yaxis=dict(title='path-count density'),
                 yaxis2=dict(title='clearance EDT (mm)', overlaying='y', side='right'),
                 legend=dict(font=dict(size=10)), margin=dict(l=10, r=10, t=10, b=10))
P2 = p2.to_html(full_html=False, include_plotlyjs=False)

# ---------------- assemble HTML ----------------
def sec(title, body):
    return '<h2>{0}</h2>\n{1}\n'.format(title, body)


def im(b64, cap_txt):
    return ('<figure><img src="data:image/png;base64,{0}" style="max-width:100%">'
            '<figcaption>{1}</figcaption></figure>'.format(b64, cap_txt))


html = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>LC-BA35 route extraction: how it works</title>
<style>
body {{ font-family: 'Nimbus Sans', Helvetica, Arial, sans-serif; max-width: 980px;
       margin: 2em auto; padding: 0 1em; color: #222; line-height: 1.45; }}
h1 {{ font-size: 1.5em; }} h2 {{ font-size: 1.15em; margin-top: 1.8em;
       border-bottom: 1px solid #ddd; padding-bottom: 0.2em; }}
figure {{ margin: 1em 0; }} figcaption {{ font-size: 0.82em; color: #555; }}
code {{ background: #f4f4f4; padding: 0 0.25em; }}
table {{ border-collapse: collapse; font-size: 0.85em; }}
td, th {{ border: 1px solid #ccc; padding: 0.3em 0.6em; text-align: left; }}
.note {{ background: #f6f8e8; border-left: 3px solid #a3b86c; padding: 0.6em 0.9em;
         font-size: 0.9em; }}
</style></head><body>
<h1>Globally optimal route extraction for LC &rarr; BA35 tracts</h1>
<p>This documents how <code>route_dijkstra.py</code> turns the probtrackx
path-count distribution into anatomically plausible centre-line polylines,
using the actual right-hemisphere thr=0.07 data at every step. It replaces the
greedy per-shell argmax (<code>main.py</code>) and the shell-component tracer
(<code>method_shell_components</code> + truncation + snap-fix).
Route naming: the route nearer the pipeline's greedy baseline polyline is the
<b>lateral</b> route (it re-finds the native corridor; green below); the
second, distinct corridor is the <b>medial</b> route (orange). The number of
routes per seed/target pair and their names are set in the tracts config's
<code>"polylines"</code> block.</p>

{s1}{s2}{s3}{s4}{s5}{s6}{s7}{s8}{s9}
<h2>Parameters used here</h2>
<table>
<tr><th>parameter</th><th>value</th><th>role</th></tr>
<tr><td>threshold</td><td>0.07</td><td>density mask for the graph domain</td></tr>
<tr><td>w_density, gamma</td><td>1.0, 2</td><td>ridge term (1 - density)^2</td></tr>
<tr><td>w_center, edt_eps</td><td>1.0, 0.5</td><td>clearance term 1/(EDT + 0.5)</td></tr>
<tr><td>w_turn, max_turn</td><td>1.0, 60&deg;</td><td>smoothness; hard cap per step</td></tr>
<tr><td>w_len</td><td>0.05</td><td>tie-breaker only</td></tr>
<tr><td>suppress sigma / peak</td><td>4 mm / +10 per mm</td><td>Gaussian route-2 repulsion</td></tr>
<tr><td>recentring</td><td>r=3 mm slab=1 mm, 10 iter</td><td>cross-section centroid pull</td></tr>
</table>
<p style="font-size:0.8em;color:#777">Generated by
<code>dwi-tracts/project/make_route_method_explainer.py</code>; routes and
intermediates recomputed live from
<code>tract_stats_nonbdp_thr007/LC-BA35-only-nonbdp-thr007</code>.</p>
</body></html>
"""

s1 = sec('1. The input: a streamline histogram, not a tract', """
<p>The only volumetric input is <code>avr_min_tract_counts</code> -- the
group-average probtrackx visitation histogram. It is bright where many
streamlines passed, blurry at its edges, and near LC the lateral and medial
corridors overlap into a single trunk. Any method that works shell-by-shell
on this volume inherits that blur; the method below never segments it -- it
only ever asks "what does a whole path cost".</p>
""" + im(F1, 'Maximum-intensity projections of the path-count density '
              '(PowerNorm 0.5 to make the faint shell visible). Blue: LC seed. Red: BA35 target.'))

s2 = sec('2. The graph: voxels &times; incoming directions', """
<p>Every suprathreshold voxel (plus all voxels of both real ROIs) becomes 26
graph states -- one per incoming step direction. A step to a neighbour is an
edge, and it is simply <b>deleted</b> if it would turn more than 60&deg;
relative to the incoming direction. Because the state remembers where you
came from, curvature is part of the optimisation itself, not a post-hoc
filter. With ~10k voxels this is ~260k states / ~2.5M edges -- Dijkstra
solves it exactly in about 2 s.</p>
""")

s3 = sec('3. What a step costs', """
<p>Each mm of travel is charged for <i>where</i> it lands and <i>how it
bends</i>:</p>
<ul>
<li><b>ridge term</b> (1 - density)&sup2;: cheap in the bright core, expensive
in the faint fringe;</li>
<li><b>clearance term</b> 1/(EDT + 0.5): EDT is the distance to the mask
boundary, so this is hyperbolically expensive against the shell wall and
cheapest on the medial axis -- the path behaves exactly "afraid of low- and
no-density areas" and will not cut corners;</li>
<li><b>turn term</b> w(1 - cos&theta;) up to the hard 60&deg; cap;</li>
<li><b>length tie-breaker</b> 0.05/mm, too small to fight the other terms.</li>
</ul>
""" + im(F2, 'Cost ingredients on the sagittal slice through the route midpoint. '
             'Green dots: route-1 vertices near this slice. The total node cost (right) is what '
             'each mm of travel is charged.')
     + im(F3, 'The turning charge per step. Anything beyond 60 degrees is not an edge at all.'))

s4 = sec('4. The search: one exact solve, both ends anchored', """
<p>All LC voxels are sources and all <b>real</b> BA35 voxels are sinks, so
the optimal route starts in LC and stops inside actual BA35 by construction --
the dilated-mask truncation and snap-to-ROI repairs of the old approach have
nothing left to fix. The figure shows the accumulated cost radiating from LC
(the Dijkstra "wavefront"); the route is read out by backtracking from the
cheapest BA35 state.</p>
""" + im(F4, 'Accumulated cost-to-reach from LC (min over incoming directions, '
             'min-projections). The optimal (lateral) route follows the valley of this field.')
     + """
<div class="note"><b>Termination is "first entry", not "into the ROI
interior".</b> Every real-ROI voxel is a sink, so the route stops at the
<i>first</i> BA35 voxel the search reaches -- and because the density only
overlaps a fraction of BA35 (~15-20% of its voxels here), that first contact
can sit at the ROI's edge (for LC-BA35 this lands near the amygdala/
hippocampal-head border) well short of the ROI's own centroid. A
"core sink" variant was prototyped (restricting termination to voxels near
the ROI medoid, forcing the route to traverse the ROI interior --
see <code>route_dijkstra_out/routes_core_sink_compare.html</code>) but was
NOT adopted: the decision was to keep endpoints anchored strictly to where
the streamline data itself enters the ROI, rather than extending the route
through territory the tractography has no support for. Separately, pipeline
smoothing and recentring can drift the terminal vertex a couple mm off the
ROI surface even in the entry-point behaviour above; the injection driver
now re-anchors each processed route on its raw graph path's true endpoints
after smoothing (<code>restore_endpoints</code> in
<code>run_route_injection_v2.py</code>).</div>
""")

s5 = sec('5. A second route: penalise the first corridor, solve again', """
<p>For the second (medial) route the same graph is re-solved after adding a
<b>Gaussian</b> repulsion around the lateral route: +10/mm on the route itself, falling off as
exp(-d&sup2;/2&sigma;&sup2;) with &sigma; = 4 mm -- <i>except</i> near the
two ROIs (blue dotted zone), where all routes must legitimately converge.
The smooth falloff means the medial route is pushed off the lateral one by a graded force
rather than a hard cost cliff, so it settles at the natural centre of the
remaining shell instead of hugging the corridor boundary. No
connected-component detection on blurred shells, no Hungarian matching: the
second-cheapest topologically distinct corridor simply wins.</p>
""" + im(F5, 'The Gaussian suppression field (red, colorbar = added cost per mm) repels '
             'the medial route (orange) onto the other side of the shell; the exemption zone (blue '
             'dotted) lets both routes share the merged trunk at the ROI ends.'))

s6 = sec('6. Recentring: a final middle-of-the-shell guarantee', """
<p>After pipeline-style smoothing, each vertex is pulled (10 relaxed
iterations) toward the density-weighted centroid of a thin slab perpendicular
to the route -- its local cross-section (radius 3 mm, so it can never grab
the sibling route ~9 mm away). Endpoints stay anchored in the ROIs.</p>
""" + im(F6, 'Actual cross-sections of the density at 30/50/70% arclength of the lateral route. '
             'Grey dot: vertex before; green: after recentring.'))

s7 = sec('7. Why the routes join near BA35 -- and why that is correct', """
<p>The cross-sections below walk the lateral route into the target. Around 12 mm out
the two shells are still separable; by ~6 mm the histogram is a single blob,
and inside it there is no data to support two distinct centre-lines. The
merge you see in 3D is the data itself merging: keeping the routes apart
there would mean inventing geometry the path distributions do not contain.
This is also why the suppression corridor is exempted near the ROIs.</p>
<div class="note">If separated endpoints were ever needed (e.g. medial vs
lateral BA35 entry points), the honest fix is upstream: split the target ROI
and re-run probtrackx per sub-target, not force the polylines apart.</div>
""" + im(F7, 'Cross-sections perpendicular to the lateral route approaching BA35. Green: lateral; '
             'orange: nearest point of the medial route projected into the same plane.'))

s8 = sec('8. Interactive: the scene and the evidence along each route', """
<p>Left-drag to rotate; click legend entries to toggle. The corridor cloud is
the lateral route's suppression zone. Below: density and clearance sampled along each
final route -- both stay above threshold (dashed) end to end, and clearance
dips exactly where the shells pinch.</p>
""" + P1 + P2)

s9 = sec('9. How many routes to fit: a config setting, not a script argument', """
<p>Route count, naming, and extraction parameters are declared per
seed/target pair in a <code>"polylines"</code> block on the dwi-tracts
tracts config JSON, read by
<code>project/run_route_injection_v2.py</code>:</p>
<pre style="background:#f4f4f4;padding:0.8em;font-size:0.82em;overflow-x:auto">
"polylines": {{
   "threshold": 0.07,
   "route_names": ["lateral", "medial"],
   "n_routes": {{
      "LC_L,BA35_L": 2,
      "LC_R,BA35_R": 2
   }},
   "dijkstra": {{}},
   "recenter": {{}}
}}
</pre>
<p><code>route_names</code> maps names to routes by ascending distance from
the pipeline's own greedy baseline polyline (nearest first), so name order
matches which corridor is "native-like" vs "distinct" regardless of which
gets extracted first. Extra routes beyond the named list auto-name
<code>route3</code>, <code>route4</code>, ... A pair absent from
<code>n_routes</code> (or set to 0) is skipped entirely -- the pipeline's
own native tract stands for it. <code>dijkstra</code> and
<code>recenter</code> are kwarg-override dicts passed straight through to
<code>extract_routes</code> / <code>recenter_polyline</code>, so tuning the
cost weights or turn cap for a specific pair needs no code change. One run
of the driver (<code>--cohort ... --route lateral</code>) injects that
route for <i>every</i> configured pair in a single combined-LR pass -- see
<code>project/ROUTES_V2.md</code> for the full architecture and current
script inventory.</p>
""")

html = html.format(s1=s1, s2=s2, s3=s3, s4=s4, s5=s5, s6=s6, s7=s7, s8=s8, s9=s9)
out_f = os.path.join(OUT_DIR, 'route_method_explainer.html')
with open(out_f, 'w') as f:
    f.write(html)
print('wrote {0} ({1:.1f} MB)'.format(out_f, os.path.getsize(out_f) / 1e6))
