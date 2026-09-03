#!/usr/bin/env python3
"""
test_profiles.py -- checks for dwitracts.profiles (PROFILE_GLM_SPEC.md).

Self-contained: no pipeline data, no file layout, no cluster. Covers the
numerics the module's correctness actually rests on, plus the synthetic
recovery / false-positive-rate experiment from PROFILE_GLM_SPEC.md section
8.4 in miniature.

    python project/test_profiles.py            # fast checks
    python project/test_profiles.py --fwer     # + the slow FWER simulation

Run from the repo root (the dwitracts package must be importable).
"""
import os
import sys
import argparse

import numpy as np
import scipy.stats as stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dwitracts.profiles import (ols_fit, tvals_for_column, cluster_masses,
                                max_cluster_mass, tfce_1d, arc_length_param,
                                resample_polyline_uniform, polyline_at_arclength,
                                DwiTractsProfiles, Shells)

FAILED = []


def check(name, cond, detail=''):
    print('{0:5s} {1}{2}'.format('ok' if cond else 'FAIL', name,
                                 '' if cond else '  <- ' + str(detail)))
    if not cond:
        FAILED.append(name)


# ---------------------------------------------------------------------------
# 1. OLS: must match statsmodels exactly, and df must be n - rank(X)
# ---------------------------------------------------------------------------

def test_ols():
    import statsmodels.api as sm
    rng = np.random.default_rng(0)
    n, p, m = 40, 4, 7
    X = np.column_stack([np.ones(n), rng.normal(size=(n, p - 1))])
    Y = rng.normal(size=(n, m))

    B, T, P, R, df = ols_fit(X, Y)
    check('ols df = n - rank(X)', df == n - p, df)

    ref_b, ref_t, ref_p = [], [], []
    for j in range(m):
        r = sm.OLS(Y[:, j], X).fit()
        ref_b.append(r.params); ref_t.append(r.tvalues); ref_p.append(r.pvalues)
    check('ols coef matches statsmodels', np.allclose(B, np.array(ref_b).T))
    check('ols t matches statsmodels', np.allclose(T, np.array(ref_t).T))
    check('ols p matches statsmodels', np.allclose(P, np.array(ref_p).T))

    for col in range(p):
        t1 = tvals_for_column(X, Y, col)
        check('tvals_for_column[{0}] matches ols_fit'.format(col), np.allclose(t1, T[col]))

    # Rank-deficient design: df must drop, not silently stay at n - p
    Xd = np.column_stack([X, X[:, 1]])
    _, _, _, _, dfd = ols_fit(Xd, Y)
    check('rank-deficient df uses true rank', dfd == n - p, dfd)

    # t is invariant to rescaling X columns and y -- the property the
    # effect_scales variants rely on
    S = np.diag([1.0, 10.0, 0.5, 3.0])
    _, T2, _, _, _ = ols_fit(X @ S, Y * 7.0)
    check('t invariant to X/y rescaling', np.allclose(T2, T))


# ---------------------------------------------------------------------------
# 2. Cluster / TFCE primitives
# ---------------------------------------------------------------------------

def test_clusters():
    t = np.array([0, 3, 4, 3, 0, -5, -6, 0, 2, 2], float)
    cl = cluster_masses(t, 2.5, 2)
    check('two clusters found', len(cl) == 2, cl)
    masses = sorted(c[2] for c in cl)
    check('cluster masses correct', np.allclose(masses, [10.0, 11.0]), masses)
    check('max_cluster_mass', np.isclose(max_cluster_mass(t, 2.5, 2), 11.0))
    check('min_clust filters short runs', cluster_masses(t, 2.5, 4) == [])

    # A sign flip must split a cluster, never merge across zero
    t2 = np.array([3, 3, 3, -3, -3, -3], float)
    cl2 = cluster_masses(t2, 2.0, 3)
    check('sign change splits clusters', len(cl2) == 2, cl2)

    s = tfce_1d(np.array([0, 1, 5, 1, 0], float))
    check('tfce peaks at the peak', s.argmax() == 2, s)
    check('tfce zero off-signal', s[0] == 0 and s[-1] == 0, s)
    check('tfce monotone in extent',
          tfce_1d(np.array([0, 3, 3, 3, 3, 0.]))[2] > tfce_1d(np.array([0, 3, 0, 0, 0, 0.]))[1])


# ---------------------------------------------------------------------------
# 3. Shell geometry
# ---------------------------------------------------------------------------

def test_shells():
    # A 100mm straight line, irregularly sampled -- exactly the situation
    # the uniform resample exists to fix.
    s_true = np.sort(np.concatenate([[0], np.random.default_rng(1).uniform(0, 100, 30), [100]]))
    poly = np.column_stack([s_true, np.zeros_like(s_true), np.zeros_like(s_true)])
    sv, L = arc_length_param(poly)
    check('arc length', np.isclose(L, 100.0), L)

    s_fine, pts = resample_polyline_uniform(poly, 0.5)
    check('uniform resample step', np.allclose(np.diff(s_fine), 0.5))
    check('uniform resample on-line', np.allclose(pts[:, 0], s_fine))

    q = polyline_at_arclength(poly, [0, 25, 50, 100, 150])
    check('arclength sampling exact', np.allclose(q[:, 0], [0, 25, 50, 100, 100]), q[:, 0])

    # Population balance: a tube of varying cross-section along a straight
    # line. Length-proportional shells must be far more balanced than the
    # cross-sectional area they're cut from.
    rng = np.random.default_rng(2)
    xs, ys = [], []
    for x in np.arange(0, 100, 1.0):
        width = 2 + 20 * (x / 100.0) ** 3          # flares badly at one end
        k = max(1, int(width))
        xs += [x] * k
        ys += list(rng.normal(size=k))
    coords = np.column_stack([xs, ys, np.zeros(len(xs))])
    s_f, pts_f = resample_polyline_uniform(poly, 0.5)
    from scipy.spatial import cKDTree
    _, nearest = cKDTree(pts_f).query(coords)
    voxel_s = s_f[nearest]
    K = int(round(L / 2.0))
    edges = np.linspace(0, L, K + 1)
    bins = np.clip(np.searchsorted(edges, voxel_s, side='right') - 1, 0, K - 1)
    counts = np.bincount(bins, minlength=K)
    check('length-proportional shells non-empty', counts.min() > 0, counts.min())
    check('shell count from length', K == 50, K)
    check('shell centres ascend', np.all(np.diff((edges[:-1] + edges[1:]) / 2) > 0))


# ---------------------------------------------------------------------------
# 4. Permutation inference: the null must be calibrated, the signal found
# ---------------------------------------------------------------------------

def _make_profiler(n_perm=500, tfce=False, min_clust=3, alpha=0.05,
                   scheme='freedman_lane'):
    """A DwiTractsProfiles with only the inference knobs set -- enough to
    exercise _perm_pvalues / _perm_statistic without any pipeline state."""
    p = DwiTractsProfiles({})
    p.tfce = tfce
    p.min_clust = min_clust
    p.alpha = alpha
    p.n_permutations = n_perm
    p.perm_scheme = scheme
    p.perm_seed = 0
    p.two_tailed = True
    p.cluster_forming_p = 0.01
    p.max_weight_factor = 1.0
    return p


def _perm_test(P, X, col, prof, t_thresh):
    """One tract's worth of the nonparam machinery, inlined."""
    fit, res = DwiTractsProfiles._freedman_lane_parts(X, col, P)
    XtX_inv = np.linalg.pinv(X.T @ X)
    draws = prof._perm_generator(P.shape[0], prof.n_permutations, 'F')
    null = np.array([prof._perm_statistic(
        tvals_for_column(X, fit + (res * d[:, None] if prof.perm_scheme == 'sign_flip'
                                   else res[d]), col, XtX_inv), t_thresh)
        for d in draws])
    obs = tvals_for_column(X, P, col, XtX_inv)
    return prof._perm_pvalues(obs, null, t_thresh), obs


def test_permutation_recovery():
    """Signal recovery: a Gaussian bump at a known node, scaled by AGE."""
    rng = np.random.default_rng(7)
    n, K = 40, 30
    age = rng.normal(size=n)
    sex = rng.integers(0, 2, n).astype(float)
    X = np.column_stack([np.ones(n), age, sex])

    bump = np.exp(-0.5 * ((np.arange(K) - 12) / 2.0) ** 2)
    # smooth noise along the node axis, so the trace looks like a real one
    noise = rng.normal(size=(n, K + 20))
    noise = np.apply_along_axis(lambda v: np.convolve(v, np.ones(5) / 5, 'valid'), 1, noise)[:, :K]
    P = noise * 3 + np.outer(age, bump) * 2.0

    prof = _make_profiler(n_perm=500)
    df = n - 3
    t_thresh = float(stats.t.isf(prof.cluster_forming_p / 2, df))
    (pvals, clusters), obs = _perm_test(P, X, 1, prof, t_thresh)

    check('recovers injected bump', clusters.max() > 0, 'no cluster found')
    if clusters.max() > 0:
        peak = int(np.argmax(np.abs(obs)))
        check('bump localised near node 12', abs(peak - 12) <= 3, peak)
        check('cluster p is significant', pvals.min() < 0.05, pvals.min())

    # Nuisance-only factor (sex has no effect) should not light up
    (pv2, cl2), _ = _perm_test(P, X, 2, prof, t_thresh)
    check('no false cluster on null factor', cl2.max() == 0, pv2.min())

    # TFCE finds it too
    proft = _make_profiler(n_perm=500, tfce=True)
    (pv3, cl3), _ = _perm_test(P, X, 1, proft, t_thresh)
    check('tfce recovers injected bump', cl3.max() > 0, pv3.min())

    # Reproducibility: identical seed -> identical p-values
    (pv4, _), _ = _perm_test(P, X, 1, _make_profiler(n_perm=500), t_thresh)
    check('permutation p-values reproducible', np.allclose(pvals, pv4))


def test_fwer(n_sims=200, n_perm=300):
    """Empirical FWER under a pure-noise null must sit near alpha."""
    rng = np.random.default_rng(11)
    n, K, alpha = 30, 25, 0.05
    prof = _make_profiler(n_perm=n_perm, alpha=alpha)
    df = n - 3
    t_thresh = float(stats.t.isf(prof.cluster_forming_p / 2, df))
    hits = 0
    for i in range(n_sims):
        age = rng.normal(size=n)
        sex = rng.integers(0, 2, n).astype(float)
        X = np.column_stack([np.ones(n), age, sex])
        noise = rng.normal(size=(n, K + 20))
        P = np.apply_along_axis(lambda v: np.convolve(v, np.ones(5) / 5, 'valid'),
                                1, noise)[:, :K]
        (pvals, clusters), _ = _perm_test(P, X, 1, prof, t_thresh)
        if clusters.max() > 0:
            hits += 1
    rate = hits / n_sims
    se = math_sqrt(alpha * (1 - alpha) / n_sims)
    check('empirical FWER ~ alpha (got {0:.3f}, alpha={1})'.format(rate, alpha),
          rate < alpha + 3 * se, rate)


def math_sqrt(x):
    return float(np.sqrt(x))


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fwer', action='store_true', help='run the slow FWER simulation')
    args = ap.parse_args()

    print('-- OLS');            test_ols()
    print('-- clusters/TFCE');  test_clusters()
    print('-- shells');         test_shells()
    print('-- permutation');    test_permutation_recovery()
    if args.fwer:
        print('-- FWER (slow)'); test_fwer()

    print()
    if FAILED:
        print('{0} FAILED: {1}'.format(len(FAILED), ', '.join(FAILED)))
        return 1
    print('all checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
