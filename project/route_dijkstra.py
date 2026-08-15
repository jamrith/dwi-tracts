"""
route_dijkstra.py

Globally optimal tract-route extraction, replacing the greedy per-shell
argmax (main.py estimate_unidirectional_tracts) and the shell-component
tracer (run_dorsal_injection_*: method_shell_components +
truncate_routes_at_target + snap_to_target_multi).

Formulation
-----------
Shortest path on a direction-augmented voxel graph:

  node  = (voxel v in domain, incoming step direction i in 26-neighbourhood)
  edge  = (v, i) -> (v + off_j, j), allowed only if the turn angle between
          off_i and off_j is <= max_turn_deg (hard curvature cap per ~1-1.7mm
          step)
  cost  = step_len_mm * [ w_density * (1 - density_norm)^gamma      ridge term
                        + w_center  / (edt_mm + edt_eps)            clearance term
                        + w_dir     * (1 - |cos(step, dirfield)|)   optional probtrackx
                        + w_len ]                                   direct-path term
        + w_turn * (1 - cos(turn angle))                            smoothness term

The clearance term is hyperbolic in the distance-to-boundary (maximum-
clearance path formulation from motion planning): walking near the wall of
the tract shell costs ~1/edt_eps per mm while the medial axis costs a small
fraction of that, so the optimal path hugs the middle of the shell instead
of cutting corners along the low-density boundary. w_len is kept tiny --
only a tie-breaker, not a real preference for short paths.

  sources = every state at a seed-ROI voxel (all incoming directions, so the
            first real step pays no turn penalty)
  sinks   = every state at a REAL target-ROI voxel

Because sinks are the real (non-dilated) target ROI and the domain explicitly
includes the ROI voxels, the returned route terminates inside the real ROI by
construction -- no dilated-mask hit-test, no truncation, no snap fix.

Multiple routes: after route k, every domain voxel gets an additive Gaussian
penalty suppress_cost * exp(-d^2 / (2 * suppress_radius_mm^2)) of its
distance d to the routes found so far (zeroed near the seed/target ROIs,
where all routes must converge), and the search is re-run. Routes come out
ordered by cost; topological distinctness falls out of the suppression.

Domain: (density > threshold) | seed ROI | target ROI. Inside the ROIs the
density is floored at the threshold so entering the ROI is not penalized as
near-zero density.
"""
import math

import numpy as np
import nibabel as nib
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree


def _offsets_26(zooms):
    offs = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                offs.append((dx, dy, dz))
    offs = np.array(offs, dtype=int)
    lens = np.linalg.norm(offs * np.asarray(zooms)[None, :], axis=1)
    units = (offs * np.asarray(zooms)[None, :]) / lens[:, None]
    return offs, lens, units


def extract_routes(V_dens, V_seed, V_target, header,
                   threshold,
                   V_sink=None,
                   n_routes=2,
                   w_density=1.0, gamma=2.0,
                   w_center=1.0, edt_eps=0.5,
                   w_turn=1.0, max_turn_deg=60.0,
                   w_len=0.05,
                   V_dirfield=None, w_dir=0.0, dir_missing_penalty=0.5,
                   suppress_radius_mm=4.0, suppress_cost=10.0,
                   endpoint_exempt_dilate=3,
                   repel_routes_mm=None,
                   capture=None,
                   verbose=False):
    """Extract up to n_routes globally-optimal seed->target routes.

    V_dens:      3D density volume (e.g. avr_min_tract_counts), squeezed
    V_seed:      3D seed ROI mask (>0 inside), squeezed
    V_target:    3D REAL target ROI mask (>0 inside), squeezed
    header:      nibabel header of the density image (for zooms/sform)
    V_dirfield:  optional (X,Y,Z,3) group streamline-orientation field
                 (sign-ambiguous; |cos| is used). Only consulted if w_dir > 0.
    V_sink:      optional 3D mask restricting WHERE routes may terminate
                 (must lie within V_target). The full target ROI still joins
                 the graph domain, so routes traverse the ROI interior to
                 reach the sink (e.g. its core) instead of stopping at the
                 first ROI voxel touched. Default: all of V_target.
    repel_routes_mm: optional list of (N,3) world-mm polylines whose Gaussian
                 suppression fields are applied from the FIRST solve onwards
                 (used for alternating joint re-optimization of multiple
                 routes, where each route is re-solved against the others).

    Returns (routes_mm, diags): routes_mm is a list of (N,3) world-mm
    polylines (unsmoothed voxel-centre paths), diags a list of per-route
    dicts. Fewer than n_routes are returned if no path exists.

    capture: optional dict; if given, it is filled with the method's
    intermediate arrays for QA/explanation: 'coords' (N,3 voxel), 'domain',
    'dens_n', 'edt_mm', 'node_cost_base', 'exempt' (all per domain voxel),
    and per-route entries under 'routes': {'node_cost', 'cost_to_reach'
    (min over incoming directions, per domain voxel), 'path_ids'}.
    """
    zooms = header.get_zooms()[:3]
    seed = V_seed > 0
    target = V_target > 0

    domain = (V_dens > threshold) | seed | target
    coords = np.argwhere(domain)
    N = coords.shape[0]
    vol_id = -np.ones(V_dens.shape, dtype=np.int64)
    vol_id[tuple(coords.T)] = np.arange(N)

    # Density term: floor at threshold inside the ROIs
    dens = V_dens.copy()
    roi_any = seed | target
    dens[roi_any] = np.maximum(dens[roi_any], threshold)
    dens_n = dens[tuple(coords.T)]
    dens_n = dens_n / dens_n.max()

    # Clearance term: hyperbolic in the mm distance to the domain boundary,
    # so boundary-hugging is punished much harder than center-line travel
    # and corner-cutting through thin/low-density margins never pays off
    edt = ndimage.distance_transform_edt(domain, sampling=zooms)
    edt_mm = edt[tuple(coords.T)]

    node_cost_base = (w_density * np.power(1.0 - dens_n, gamma)
                      + w_center / (edt_mm + edt_eps)
                      + w_len)

    offs, lens, units = _offsets_26(zooms)
    n_off = offs.shape[0]
    cos_turn = units @ units.T
    min_cos = math.cos(math.radians(max_turn_deg))

    # Per-offset neighbour lookup: nb_id[j][n] = domain id of coords[n]+offs[j], or -1
    shape = np.array(V_dens.shape)
    nb_id = np.full((n_off, N), -1, dtype=np.int64)
    for j in range(n_off):
        w = coords + offs[j][None, :]
        ok = np.all((w >= 0) & (w < shape[None, :]), axis=1)
        nb_id[j, ok] = vol_id[tuple(w[ok].T)]

    # Optional probtrackx-direction term, per (voxel, outgoing offset)
    dir_pen = None
    if V_dirfield is not None and w_dir > 0:
        F = V_dirfield[tuple(coords.T)]          # (N,3)
        Fn = np.linalg.norm(F, axis=1)
        has_dir = Fn > 1e-9
        Fu = np.zeros_like(F)
        Fu[has_dir] = F[has_dir] / Fn[has_dir, None]
        # |cos| between each unit offset and the local orientation
        abscos = np.abs(Fu @ units.T)            # (N, n_off)
        dir_pen = np.where(has_dir[:, None],
                           w_dir * (1.0 - abscos),
                           w_dir * dir_missing_penalty)

    seed_ids = vol_id[seed & domain]
    seed_ids = seed_ids[seed_ids >= 0]
    sink = target if V_sink is None else ((V_sink > 0) & target)
    target_ids = vol_id[sink & domain]
    target_ids = target_ids[target_ids >= 0]
    if seed_ids.size == 0 or target_ids.size == 0:
        raise ValueError('Seed or sink has no voxels in the domain')

    # Exemption zone for the route-suppression penalty (routes must share ends)
    exempt = ndimage.binary_dilation(roi_any, np.ones((3, 3, 3)),
                                     iterations=endpoint_exempt_dilate)
    exempt_n = exempt[tuple(coords.T)]

    sform = header.get_sform()

    if capture is not None:
        capture.update({'coords': coords, 'domain': domain,
                        'dens_n': dens_n, 'edt_mm': edt_mm,
                        'node_cost_base': node_cost_base,
                        'exempt': exempt_n, 'routes': []})

    def build_and_run(node_cost):
        rows, cols, data = [], [], []
        for i in range(n_off):
            for j in range(n_off):
                c = cos_turn[i, j]
                if c < min_cos:
                    continue
                valid = nb_id[j] >= 0
                u = np.flatnonzero(valid)
                v = nb_id[j, u]
                step = lens[j] * node_cost[v] + w_turn * (1.0 - c)
                if dir_pen is not None:
                    step = step + lens[j] * dir_pen[v, j]
                rows.append(u * n_off + i)
                cols.append(v * n_off + j)
                data.append(step)
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        data = np.concatenate(data)
        G = coo_matrix((data, (rows, cols)), shape=(N * n_off, N * n_off)).tocsr()

        src = (seed_ids[:, None] * n_off + np.arange(n_off)[None, :]).ravel()
        dist, pred = dijkstra(G, directed=True, indices=src,
                              min_only=True, return_predecessors=True)[:2]

        if capture is not None:
            capture['routes'].append({
                'node_cost': node_cost.copy(),
                'cost_to_reach': dist.reshape(N, n_off).min(axis=1)})

        tgt_states = (target_ids[:, None] * n_off + np.arange(n_off)[None, :]).ravel()
        tgt_dist = dist[tgt_states]
        if not np.any(np.isfinite(tgt_dist)):
            return None, None
        best = tgt_states[int(np.argmin(tgt_dist))]

        state_path = [best]
        while pred[state_path[-1]] >= 0:
            state_path.append(pred[state_path[-1]])
        state_path.reverse()
        vox_ids = [s // n_off for s in state_path]
        # collapse consecutive duplicates (start states share the voxel)
        path = [vox_ids[0]]
        for vid in vox_ids[1:]:
            if vid != path[-1]:
                path.append(vid)
        return np.array(path, dtype=np.int64), float(np.min(tgt_dist))

    def route_penalty(route_list):
        tree = cKDTree(np.vstack(route_list))
        d, _ = tree.query(nib.affines.apply_affine(sform, coords))
        pen = suppress_cost * np.exp(-0.5 * np.square(d / suppress_radius_mm))
        pen[exempt_n] = 0.0
        return pen

    repel = list(repel_routes_mm) if repel_routes_mm else []
    routes_mm, diags = [], []
    node_cost = node_cost_base + route_penalty(repel) if repel else node_cost_base.copy()
    for k in range(n_routes):
        path_ids, cost = build_and_run(node_cost)
        if path_ids is None:
            if verbose:
                print('  route {0}: no path found, stopping'.format(k))
            break
        path_vox = coords[path_ids]
        if capture is not None:
            capture['routes'][-1]['path_ids'] = path_ids
        path_mm = nib.affines.apply_affine(sform, path_vox)
        seg = np.linalg.norm(np.diff(path_mm, axis=0), axis=1)
        diag = {'cost': cost,
                'n_points': int(path_mm.shape[0]),
                'length_mm': float(seg.sum())}
        if routes_mm:
            tree = cKDTree(np.vstack(routes_mm))
            d, _ = tree.query(path_mm)
            diag['min_sep_to_prev_mm'] = float(d.min())
            diag['mean_sep_to_prev_mm'] = float(d.mean())
            diag['max_sep_to_prev_mm'] = float(d.max())
        routes_mm.append(path_mm)
        diags.append(diag)
        if verbose:
            print('  route {0}: {1}'.format(k, diag))

        if k + 1 < n_routes:
            # suppress a corridor around all routes so far (except near ROIs):
            # Gaussian falloff with sigma = suppress_radius_mm, so the next
            # route is repelled smoothly rather than by a hard cost cliff
            node_cost = node_cost_base + route_penalty(routes_mm + repel)

    return routes_mm, diags


def recenter_polyline(route_mm, V_dens, header, threshold,
                      radius_mm=3.0, slab_mm=1.0,
                      n_iter=10, relax=0.5, smooth_window=3,
                      fix_ends=True):
    """Iteratively pull each vertex to the density-weighted centroid of the
    tract cross-section it sits in, so the final polyline runs down the
    middle of the tract shell regardless of how it was initialized.

    At each iteration, for every interior vertex: take the local tangent
    (central difference), collect suprathreshold density voxels within
    radius_mm whose axial offset along the tangent is <= slab_mm (a thin
    perpendicular slab = the local cross-section), move the vertex a
    fraction `relax` toward their density-weighted centroid, then lightly
    smooth the whole polyline. Endpoints stay fixed (they are anchored
    inside the ROIs by the graph search).

    radius_mm should be smaller than the separation between distinct
    routes after divergence (so the slab never captures the sibling
    route's shell) but larger than half the local tract caliber.
    """
    sel = V_dens > threshold
    pts_vox = np.argwhere(sel)
    pts_mm = nib.affines.apply_affine(header.get_sform(), pts_vox)
    w = V_dens[tuple(pts_vox.T)]
    tree = cKDTree(pts_mm)

    route = route_mm.copy().astype(float)
    n = route.shape[0]
    if n < 3:
        return route

    for _ in range(n_iter):
        tang = np.zeros_like(route)
        tang[1:-1] = route[2:] - route[:-2]
        tang[0] = route[1] - route[0]
        tang[-1] = route[-1] - route[-2]
        tl = np.linalg.norm(tang, axis=1)
        tl[tl == 0] = 1.0
        tang = tang / tl[:, None]

        new = route.copy()
        idx_lists = tree.query_ball_point(route, radius_mm)
        rng = range(1, n - 1) if fix_ends else range(n)
        for vi in rng:
            ids = idx_lists[vi]
            if not ids:
                continue
            offs = pts_mm[ids] - route[vi][None, :]
            axial = offs @ tang[vi]
            keep = np.abs(axial) <= slab_mm
            if not np.any(keep):
                continue
            ww = w[ids][keep]
            centroid = np.average(pts_mm[ids][keep], axis=0, weights=ww)
            new[vi] = route[vi] + relax * (centroid - route[vi])

        # light moving-average smoothing, endpoints restored
        p0, p1 = new[0].copy(), new[-1].copy()
        sm = new.copy()
        hw = smooth_window // 2
        for vi in range(n):
            a, b = max(0, vi - hw), min(n, vi + hw + 1)
            sm[vi] = new[a:b].mean(axis=0)
        if fix_ends:
            sm[0], sm[-1] = p0, p1
        route = sm

    return route
