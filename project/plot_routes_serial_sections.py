"""
Serial coronal sections through the MNI152 T1 template with the v2 lateral
(green) and medial (orange) LC->BA35 routes rendered as thick tubes (2 mm
radius, voxelized from the injected polylines), both hemispheres in every
panel. Output: route_dijkstra_out/routes_serial_sections.png + caption .md.
"""
import os

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from scipy.spatial import cKDTree

CFG_DIR = '/gpfs01/imgshare/ConnLS/ADNI/dwi-tracts/project/qa_diag/route_v2_configs/nonbdp461'
OUT_DIR = '/gpfs01/imgshare/ConnLS/ADNI/dwi-tracts/project/route_dijkstra_out'
MNI = '/gpfs01/imgshare/ConnLS/ADNI/fsl_mod/data/standard/MNI152_T1_1mm.nii.gz'
TUBE_RADIUS_MM = 2.0
N_SLICES = 10

COLORS = {'lateral': '#31a354', 'medial': '#e6550d'}


def parse_poly3d(path):
    with open(path) as f:
        lines = f.read().strip().split('\n')
    n_pts = int(lines[1].split()[0])
    return np.array([list(map(float, lines[2 + i].split())) for i in range(n_pts)])


def densify(curve_mm, step=0.25):
    seg = np.linalg.norm(np.diff(curve_mm, axis=0), axis=1)
    arc = np.concatenate([[0], np.cumsum(seg)])
    s = np.arange(0, arc[-1], step)
    return np.column_stack([np.interp(s, arc, curve_mm[:, i]) for i in range(3)])


def tube_mask(curves_mm, img, radius):
    """Boolean volume: voxels within `radius` mm of any of the curves."""
    pts = np.vstack([densify(c) for c in curves_mm])
    tree = cKDTree(pts)
    inv = np.linalg.inv(img.affine)
    vox = nib.affines.apply_affine(inv, pts)
    lo = np.maximum(np.floor(vox.min(axis=0)).astype(int) - 4, 0)
    hi = np.minimum(np.ceil(vox.max(axis=0)).astype(int) + 5, np.array(img.shape[:3]))
    gx, gy, gz = np.meshgrid(*[np.arange(lo[i], hi[i]) for i in range(3)], indexing='ij')
    sub_vox = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    sub_mm = nib.affines.apply_affine(img.affine, sub_vox)
    d, _ = tree.query(sub_mm)
    M = np.zeros(img.shape[:3], dtype=bool)
    M[gx.ravel()[d <= radius], gy.ravel()[d <= radius], gz.ravel()[d <= radius]] = True
    return M


def main():
    img = nib.load(MNI)
    T1 = img.get_fdata()

    masks = {}
    for route in ('lateral', 'medial'):
        curves = [parse_poly3d(os.path.join(CFG_DIR, '{0}_v2_route_{1}.poly3d'.format(route, h)))
                  for h in ('L', 'R')]
        masks[route] = tube_mask(curves, img, TUBE_RADIUS_MM)

    union = masks['lateral'] | masks['medial']
    nz = np.argwhere(union)
    y0, y1 = nz[:, 1].min(), nz[:, 1].max()
    x0, x1 = nz[:, 0].min() - 12, nz[:, 0].max() + 12
    z0, z1 = nz[:, 2].min() - 10, nz[:, 2].max() + 10
    ys = np.unique(np.round(np.linspace(y0 + 1, y1 - 1, N_SLICES)).astype(int))

    vmax = np.percentile(T1, 99)
    fig, axes = plt.subplots(2, 5, figsize=(10.0, 5.4))
    for ax, y in zip(axes.ravel(), ys):
        ax.imshow(T1[x0:x1, y, z0:z1].T, origin='lower', cmap='gray',
                  vmax=vmax, interpolation='bilinear')
        for route in ('lateral', 'medial'):
            S = masks[route][x0:x1, y, z0:z1].T
            rgba = mcolors.to_rgba(COLORS[route])
            overlay = np.zeros(S.shape + (4,))
            overlay[S] = rgba
            overlay[S, 3] = 0.85
            ax.imshow(overlay, origin='lower', interpolation='nearest')
        y_mm = nib.affines.apply_affine(img.affine, [[0, y, 0]])[0][1]
        ax.set_title('y = {0:+.0f} mm'.format(y_mm), fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    handles = [plt.Line2D([0], [0], color=COLORS[r], lw=4, label=r)
               for r in ('lateral', 'medial')]
    axes[0, 0].legend(handles=handles, fontsize=7, loc='lower left', framealpha=0.8)
    fig.suptitle('LC -> BA35 v2 routes over MNI152, serial coronal sections '
                 '(posterior to anterior)', fontsize=10)
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, 'routes_serial_sections.png')
    fig.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close(fig)

    with open(os.path.join(OUT_DIR, 'routes_serial_sections.md'), 'w') as f:
        f.write(
            '# Serial coronal sections: v2 LC$\\rightarrow$BA35 routes\n\n'
            'Lateral (green) and medial (orange) routes from the v2 Dijkstra\n'
            'extraction (461-cohort thr $=0.07$ geometry, both hemispheres),\n'
            'voxelized as tubes of radius $r = {0:.0f}$ mm around the injected\n'
            'polylines and overlaid on the MNI152 1 mm T1 template. Sections run\n'
            'posterior $\\rightarrow$ anterior at equal spacing across the tract\n'
            'extent; y coordinates are MNI mm. The two routes share the trunk\n'
            'near LC (posterior sections) and converge again at BA35 (anterior\n'
            'sections), as dictated by the path-count distributions.\n'.format(TUBE_RADIUS_MM))
    print('wrote', out_png)


if __name__ == '__main__':
    main()
