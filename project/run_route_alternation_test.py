"""
Does the sequential (first-mover) route extraction actually bias the result?

Test on the 461-cohort thr=0.07 geometry, both hemispheres:
1. sequential:  r1 = argmin cost;  r2 = argmin cost + gauss_penalty(r1)   [status quo]
2. alternating: iterate  r1 <- argmin cost + pen(r2);  r2 <- argmin cost + pen(r1)
   until neither route moves by more than 0.3 mm (block coordinate descent
   on the symmetric joint objective).
3. reversed start: begin the alternation from the OPPOSITE assignment
   (solve the penalised route first), alternate to a fixed point, and check
   whether it lands on the same pair as (2).

If (1) ~= (2) ~= (3), the first-mover advantage is immaterial on this data
and the sequential results (already submitted for TSA/GLM) stand.
"""
import os
import sys
import json

import numpy as np
import nibabel as nib
from scipy.spatial import cKDTree

sys.path.insert(0, '/gpfs01/imgshare/ConnLS/ADNI/dwi-tracts')
sys.path.insert(0, '/gpfs01/imgshare/ConnLS/ADNI/dwi-tracts/project')
from route_dijkstra import extract_routes

DATA_ROOT = '/gpfs01/imgshare/ConnLS/ADNI'
NET_DIR = os.path.join(DATA_ROOT, 'tract_stats_nonbdp_thr007', 'LC-BA35-only-nonbdp-thr007')
ROIS_DIR = os.path.join(DATA_ROOT, 'dwi-tracts', 'data/rois/mtl-fix')
OUT_DIR = os.path.join(DATA_ROOT, 'dwi-tracts', 'project', 'route_dijkstra_out')
THR = 0.07
TOL_MM = 0.3
MAX_IT = 6


def mean_curve_dist(a, b):
    return 0.5 * (cKDTree(b).query(a)[0].mean() + cKDTree(a).query(b)[0].mean())


def main():
    results = {}
    for hemi in ('L', 'R'):
        roi_a, roi_b = 'LC_{0}'.format(hemi), 'BA35_{0}'.format(hemi)
        img = nib.load(os.path.join(NET_DIR, 'average',
                                    'avr_min_tract_counts_{0}_{1}.nii.gz'.format(roi_a, roi_b)))
        V_dens = np.squeeze(img.get_fdata())
        V_seed = np.squeeze(nib.load(os.path.join(ROIS_DIR, roi_a + '.nii.gz')).get_fdata())
        V_targ = np.squeeze(nib.load(os.path.join(ROIS_DIR, roi_b + '.nii.gz')).get_fdata())

        def solve(repel=None):
            r, d = extract_routes(V_dens, V_seed, V_targ, img.header, THR,
                                  n_routes=1, repel_routes_mm=repel)
            return r[0], d[0]['cost']

        print('== {0} =='.format(hemi))
        res = {}

        # 1. sequential (status quo)
        r1, c1 = solve()
        r2, c2 = solve(repel=[r1])
        print(' sequential: cost r1={0:.1f} r2(pen)={1:.1f} sep={2:.2f}mm'
              .format(c1, c2, mean_curve_dist(r1, r2)))
        res['sequential_sep_mm'] = mean_curve_dist(r1, r2)

        # 2. alternate from sequential
        a1, a2 = r1, r2
        moves = []
        for it in range(MAX_IT):
            a1_new, _ = solve(repel=[a2])
            a2_new, _ = solve(repel=[a1_new])
            m1, m2 = mean_curve_dist(a1_new, a1), mean_curve_dist(a2_new, a2)
            moves.append((round(float(m1), 3), round(float(m2), 3)))
            a1, a2 = a1_new, a2_new
            print(' alt iter {0}: r1 moved {1:.3f}mm, r2 moved {2:.3f}mm'.format(it + 1, m1, m2))
            if m1 < TOL_MM and m2 < TOL_MM:
                break
        res['alt_moves_mm'] = moves
        res['alt_vs_seq_r1_mm'] = float(mean_curve_dist(a1, r1))
        res['alt_vs_seq_r2_mm'] = float(mean_curve_dist(a2, r2))
        print(' fixed point vs sequential: r1 {0:.3f}mm, r2 {1:.3f}mm'
              .format(res['alt_vs_seq_r1_mm'], res['alt_vs_seq_r2_mm']))

        # 3. reversed start: give the OTHER corridor first-mover status
        b2, _ = solve()                # unconstrained solve (lands on corridor 1)
        b1, _ = solve(repel=[b2])      # forced to corridor 2
        # now alternate with roles swapped: b1 is "route 1"
        for it in range(MAX_IT):
            b1_new, _ = solve(repel=[b2])
            b2_new, _ = solve(repel=[b1_new])
            m1, m2 = mean_curve_dist(b1_new, b1), mean_curve_dist(b2_new, b2)
            b1, b2 = b1_new, b2_new
            if m1 < TOL_MM and m2 < TOL_MM:
                break
        # match reversed pair to forward pair (order-free comparison)
        d_same = res_pairs = min(mean_curve_dist(b1, a1) + mean_curve_dist(b2, a2),
                                 mean_curve_dist(b1, a2) + mean_curve_dist(b2, a1))
        res['reversed_vs_alt_total_mm'] = float(d_same)
        print(' reversed-start fixed point vs forward fixed point: {0:.3f}mm total'.format(d_same))
        results[hemi] = res

    with open(os.path.join(OUT_DIR, 'alternation_test.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
