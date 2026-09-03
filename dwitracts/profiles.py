"""
Along-tract profile GLMs -- a parallel implementation to dwitracts.glm's
DwiTractsGlm, per PROFILE_GLM_SPEC.md.

This module deliberately does NOT touch DwiTractsGlm.fit_glms /
extract_distance_traces / extract_distance_traces_rft1d /
extract_distance_trace_clusters. The legacy path stays runnable, byte for
byte, as the comparison baseline; everything here writes under its own
`profiles.output_dir` tree so the two can never collide.

What it fixes relative to the legacy path (see PROFILE_GLM_SPEC.md sections
1 and 3-5 for the full argument):

  * shell_scheme='length_proportional' -- shells are equal-physical-width
    slabs of arc length along the tract's own polyline, so shell population
    is balanced by construction (legacy flood-fill shells measured
    min=7/max=2002 voxels on LC_L->BA35_L) and a shell means the same
    physical distance in every tract. Shell count K = round(L / width)
    falls out of the tract's length.
  * trace_construction='metric_first' -- the per-subject metric is
    aggregated within each shell FIRST, then one honest OLS is fit per
    shell across subjects. The node statistic is then a genuine t, unlike
    the legacy max|t|-over-a-shell-of-correlated-voxels extreme-value
    statistic, which is not t-distributed and makes rft1d's STAT='T'
    assumptions false.
  * df = N_sub - rank(X) (per node, after outlier trimming), not the
    legacy N_sub - 1.
  * inference_method may request any subset of:
      'rft'         - the legacy analytic 1D-RFT path (utils.
                      get_tvalue_rft1d_clusters), unchanged, valid under
                      metric_first construction.
      'nonparam'    - Freedman-Lane / sign-flip permutation on the
                      per-subject K x N_sub node profiles, max-cluster-mass
                      (or max-TFCE) FWER. No assumption on the node
                      statistic's distribution, no FWHM estimate, no df
                      dependence.
      'permutation' - selection-aware permutation: each permutation refits
                      the voxelwise GLM and REBUILDS the trace (re-picking
                      the argmax voxel per shell), so the peak-selection
                      step is inside the null. The only option that makes
                      inference on a voxelwise+max trace exactly valid.

NOTE on spm1d: PROFILE_GLM_SPEC.md 5b proposed wrapping
spm1d.stats.nonparam. spm1d is not installed in this project's environment,
and the multi-covariate case ("all factors simultaneously") would have
needed ter-Braak residualisation bolted on top of it anyway. The
'nonparam' method here is therefore implemented natively: same null
(sign-flip / permutation), same max-statistic FWER logic, Freedman-Lane
nuisance handling (exact for the tested factor's own partial model up to
the usual exchangeability caveat), and a native 1D TFCE. No new dependency.

Outputs mirror the legacy contract exactly, so the downstream tooling
(plot_fornix_ba35_results_v4.py, create_pajek_graphs, aggregate_stats)
works against this tree with at most a directory-name change:

  {output_dir}/{glm}/summary-{agg}_thr{NN}/stats_{tract}.csv
  {output_dir}/{glm}/summary-{agg}_thr{NN}/resids_{tract}.csv     (K x N_sub)
  {output_dir}/{glm}/summary-{agg}_thr{NN}/tcounts-{suffix}.csv
  {output_dir}/{glm}/pval-{suffix}_thr{NN}/{tract}_{factor}.nii.gz
  {output_dir}/{glm}/tval-{suffix}_thr{NN}/{tract}_{factor}.nii.gz
  {tracts_dir}/polylines/{output_dir}/{suffix}/{glm}/tvals_{suffix}_{tract}_{factor}_{NN}.poly3d

where {suffix} is 'rft', 'np' or 'perm'. The RFT variant writes exactly the
legacy column names ({factor}|rft_pval, |rft_clusters, |rft_logpval), so
the plotting side needs no change at all for it.
"""

import os
import gc
import copy
import shutil
import warnings
import zlib

import numpy as np
import pandas as pd
import nibabel as nib
import scipy.stats as stats
import statsmodels.stats.multitest as smm
from scipy.spatial import cKDTree
from tqdm import tqdm

import rft1d

from . import utils
from dwitracts.glm import DwiTractsGlm


# ---------------------------------------------------------------------------
# Small numerical helpers (kept module-level and dependency-free so they can
# be unit-tested without any of the pipeline's file layout)
# ---------------------------------------------------------------------------

def ols_fit(X, Y):
    """Vectorised multi-outcome OLS.

    X: (n, p) design. Y: (n, m) outcomes (one column per node/voxel).

    Returns (B, T, P, resid, df) with B/T/P shaped (p, m), resid (n, m) and
    df the scalar residual degrees of freedom n - rank(X). df uses the true
    rank (not p), so a rank-deficient design degrades gracefully rather than
    silently inflating t.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    XtX_inv = np.linalg.pinv(X.T @ X)
    B = XtX_inv @ (X.T @ Y)
    resid = Y - X @ B
    rank = np.linalg.matrix_rank(X)
    df = X.shape[0] - rank
    if df <= 0:
        raise ValueError('Design has no residual degrees of freedom '
                         '(n={0}, rank={1}).'.format(X.shape[0], rank))
    sigma2 = np.sum(resid ** 2, axis=0) / df
    var_b = np.outer(np.diag(XtX_inv), sigma2)
    with np.errstate(divide='ignore', invalid='ignore'):
        T = np.where(var_b > 0, B / np.sqrt(var_b), 0.0)
    P = 2.0 * stats.t.sf(np.abs(T), df)
    return B, T, P, resid, df


def tvals_for_column(X, Y, col, XtX_inv=None):
    """t-statistics for one design column only, for every outcome column.

    The hot inner loop of both permutation methods: everything that doesn't
    depend on Y (pinv, the diagonal element of (X'X)^-1) is hoisted by the
    caller via XtX_inv. Returns (m,).
    """
    if XtX_inv is None:
        XtX_inv = np.linalg.pinv(X.T @ X)
    B = XtX_inv @ (X.T @ Y)
    resid = Y - X @ B
    df = X.shape[0] - np.linalg.matrix_rank(X)
    sigma2 = np.sum(resid ** 2, axis=0) / df
    var_b = XtX_inv[col, col] * sigma2
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(var_b > 0, B[col] / np.sqrt(var_b), 0.0)


def cluster_masses(tvals, t_thresh, min_clust):
    """Contiguous supra-threshold runs of a 1-D statistic trace.

    Positive (t >= +t_thresh) and negative (t <= -t_thresh) runs are found
    separately, so a sign change always splits a cluster -- matching
    utils.get_tvalue_rft1d_clusters' separate positive/negative passes.

    Returns a list of (start, stop, mass, sign), mass = sum|t| over the run.
    """
    out = []
    n = tvals.size
    for sign in (1, -1):
        supra = (sign * tvals) >= t_thresh
        i = 0
        while i < n:
            if not supra[i]:
                i += 1
                continue
            j = i
            while j < n and supra[j]:
                j += 1
            if (j - i) >= min_clust:
                out.append((i, j, float(np.sum(np.abs(tvals[i:j]))), sign))
            i = j
    return out


def max_cluster_mass(tvals, t_thresh, min_clust):
    """Largest cluster mass in a trace, or 0.0 if no cluster survives."""
    cl = cluster_masses(tvals, t_thresh, min_clust)
    return max((c[2] for c in cl), default=0.0)


def tfce_1d(tvals, dh=0.1, E=0.5, H=2.0, min_h=0.0):
    """Threshold-free cluster enhancement for a 1-D trace.

    Signed: each node's score is accumulated over ascending thresholds of
    |t| within same-signed contiguous runs, then given back the sign of the
    node. Returns (n,) of TFCE scores (always >= 0; use the returned scores
    with a max-statistic null exactly like cluster mass).
    """
    a = np.abs(tvals)
    top = a.max() if a.size else 0.0
    scores = np.zeros_like(a, dtype=float)
    if top <= min_h:
        return scores
    hs = np.arange(min_h + dh, top + dh, dh)
    for h in hs:
        for sign in (1, -1):
            supra = (sign * tvals) >= h
            n = supra.size
            i = 0
            while i < n:
                if not supra[i]:
                    i += 1
                    continue
                j = i
                while j < n and supra[j]:
                    j += 1
                extent = float(j - i)
                scores[i:j] += (extent ** E) * (h ** H) * dh
                i = j
    return scores


def arc_length_param(polyline_mm):
    """Cumulative arc length at each polyline vertex. Returns (s_vertex, L)."""
    seg = np.linalg.norm(np.diff(polyline_mm, axis=0), axis=1)
    s_vertex = np.concatenate([[0.0], np.cumsum(seg)])
    return s_vertex, float(s_vertex[-1])


def resample_polyline_uniform(polyline_mm, step_mm=0.5):
    """Fine, UNIFORM arc-length resampling -- decoupled from the original
    (irregularly spaced) polyline vertices. Returns (s_fine, pts_fine)."""
    s_vertex, L = arc_length_param(polyline_mm)
    s_fine = np.arange(0.0, L, step_mm)
    if s_fine.size == 0:
        s_fine = np.array([0.0])
    pts = np.stack([np.interp(s_fine, s_vertex, polyline_mm[:, d]) for d in range(3)], axis=1)
    return s_fine, pts


def polyline_at_arclength(polyline_mm, s_query):
    """Exact positions at given arc lengths along a polyline (mm).

    Used for length_proportional shells, whose centres are already in
    arc-length mm -- placement is exact rather than the proportional
    rescaling utils.resample_polyline_by_distance has to do for flood-fill
    shell indices.
    """
    s_vertex, L = arc_length_param(polyline_mm)
    s_query = np.clip(np.asarray(s_query, dtype=float), 0.0, L)
    return np.stack([np.interp(s_query, s_vertex, polyline_mm[:, d]) for d in range(3)], axis=1)


# ---------------------------------------------------------------------------

class Shells:
    """One tract's discretisation of 'distance along the tract'.

    idx      -- flat indices of the tract-mask voxels, in a fixed order
    bin_id   -- (n_vox,) shell index in [0, K) for each of those voxels
    centers  -- (K,) the value written to the CSV 'Distance' column: mm of
                arc length for length_proportional, the flood-fill distance
                value for flood_fill
    weights  -- (n_vox,) tract probability at each voxel
    """

    def __init__(self, scheme, idx, bin_id, centers, weights, shape, arc_length=None):
        self.scheme = scheme
        self.idx = idx
        self.bin_id = bin_id
        self.centers = centers
        self.weights = weights
        self.shape = shape
        self.arc_length = arc_length
        self.K = centers.size
        self.members = [np.flatnonzero(bin_id == k) for k in range(self.K)]

    @property
    def counts(self):
        return np.array([m.size for m in self.members])


class DwiTractsProfiles:
    """Along-tract profile GLM + cluster inference (see module docstring).

    Construct with the same params dict a DwiTractsGlm run uses (general /
    glm / traces / tracts / preproc blocks), plus a 'profiles' block:

      "profiles": {
        "output_dir": "profiles_arclen_metricfirst_wmean",  // name it for the knobs
        "shell_scheme": "length_proportional",   // | "flood_fill"
        "shell_width_mm": 2.0,
        "resample_step_mm": 0.5,
        "trace_construction": "metric_first",    // | "voxelwise"
        "profile_aggregation": "wmean",          // mean|wmean|median|trimmed|max
        "profile_core_threshold": null,
        "max_weight_factor": 1.0,
        "inference_method": ["rft", "nonparam"], // subset of rft|nonparam|permutation
        "n_permutations": 5000,
        "cluster_forming_p": 0.01,
        "two_tailed": true,
        "tfce": false,
        "permutation_scheme": "freedman_lane",   // | "sign_flip"
        "permutation_family": "tract",           // | "network"
        "permutation_seed": 0,
        "write_voxel_maps": false,
        "min_clust": 3,
        "pval_alpha": 0.05,
        "fdr_method": "fdr_tsbky",
        "fdr_alpha": 0.05
      }

    Anything omitted falls back to the value in 'general'/'traces' where one
    exists (min_clust, fdr_method, fdr_alpha, max_weight_factor), else to
    the default shown above.

    NB: initialize() delegates to DwiTractsGlm.initialize() so the design
    matrices, subject ordering, effect_scales variants and tract list are
    bit-identical to a legacy run against the same config -- which is the
    whole point of the comparison plan in PROFILE_GLM_SPEC.md section 8.
    That call inserts 'Intercept' into params['glm'][g]['factors'] in place,
    exactly as a legacy run does, so do not initialize both a DwiTractsGlm
    and a DwiTractsProfiles against the same params object in one process.
    """

    METHOD_SUFFIX = {'rft': 'rft', 'nonparam': 'np', 'permutation': 'perm'}

    def __init__(self, params):
        self.params = params
        self.is_init = False
        # {tract: (Y, missing)} for the selection-aware permutation, which
        # would otherwise re-read every subject's volume once per FACTOR
        # per tract (457 subjects x 4 factors x 2 tracts = 3656 NIfTI
        # reads, which dominated the run). Cleared by run_inference().
        self._vol_cache = {}

    # -- config access -----------------------------------------------------

    def _p(self, key, default=None):
        return self.params.get('profiles', {}).get(key, default)

    def initialize(self):
        params_prof = self.params.get('profiles', None)
        if params_prof is None:
            print('No "profiles" block in config.')
            return False

        base = DwiTractsGlm(self.params)
        if not base.initialize():
            return False

        # Everything the legacy path derives, reused verbatim.
        for attr in ('source_dir', 'project_dir', 'rois_dir', 'tracts_dir', 'final_dir',
                     'rois', 'roi_suffix', 'target_rois', 'tract_names', 'subjects',
                     'subject_data', 'Xs', 'Xs_scaled', 'tract_threshold'):
            setattr(self, attr, getattr(base, attr))

        params_gen = self.params['general']
        params_trace = self.params['traces']

        self.scheme = self._p('shell_scheme', 'length_proportional')
        if self.scheme not in ('flood_fill', 'length_proportional'):
            print('Unknown shell_scheme: {0}'.format(self.scheme))
            return False

        self.construction = self._p('trace_construction', 'metric_first')
        if self.construction not in ('voxelwise', 'metric_first'):
            print('Unknown trace_construction: {0}'.format(self.construction))
            return False

        self.aggregation = self._p('profile_aggregation', 'wmean')
        if self.aggregation not in ('mean', 'wmean', 'median', 'trimmed', 'max'):
            print('Unknown profile_aggregation: {0}'.format(self.aggregation))
            return False
        if self.construction == 'metric_first' and self.aggregation == 'max':
            warnings.warn('profile_aggregation="max" takes each SUBJECT\'s own '
                          'peak voxel in the shell. That selection is '
                          'label-independent, so it is NOT circular the way '
                          'voxelwise+max is -- but a per-subject max is an '
                          'order statistic: upward-biased and skewed, so the '
                          'node errors are not normal and "rft" on it is '
                          'mis-specified. "nonparam" only needs '
                          'exchangeability and stays valid; prefer it here.')

        self.shell_width_mm = float(self._p('shell_width_mm', 2.0))
        self.resample_step_mm = float(self._p('resample_step_mm', 0.5))
        self.core_threshold = self._p('profile_core_threshold', None)
        self.max_weight_factor = float(self._p('max_weight_factor',
                                               params_trace.get('max_weight_factor', 1.0)))

        methods = self._p('inference_method', ['rft', 'nonparam'])
        if isinstance(methods, str):
            methods = [methods]
        for m in methods:
            if m not in self.METHOD_SUFFIX:
                print('Unknown inference_method: {0}'.format(m))
                return False
        if self.construction == 'metric_first' and 'permutation' in methods:
            # Selection-aware permutation only buys something when the trace
            # construction reads the GROUP statistic -- voxelwise+max re-picks
            # its argmax voxel from the permuted t-map, so the selection has
            # to live inside the null. Metric-first aggregation (including
            # profile_aggregation="max", which picks each SUBJECT's own peak
            # voxel) never touches the covariates, so re-aggregating under a
            # permutation returns a bit-identical profile matrix and the test
            # degenerates to exactly "nonparam" -- at many times the cost.
            # Rejected rather than silently aliased, because
            # _selection_aware_traces would otherwise rebuild a VOXELWISE
            # argmax|t| trace and test something different from the trace in
            # this run's own summary CSV.
            print('inference_method "permutation" is redundant under '
                  'trace_construction="metric_first": the aggregation is '
                  'label-independent, so its null is identical to "nonparam". '
                  'Use "nonparam".')
            return False
        if self.construction == 'metric_first' and 'permutation' in methods:
            # Selection-aware permutation only buys something when the trace
            # construction reads the GROUP statistic -- voxelwise+max re-picks
            # its argmax voxel from the permuted t-map, so the selection has
            # to live inside the null. Metric-first aggregation (including
            # profile_aggregation="max", which picks each SUBJECT's own peak
            # voxel on that subject's own metric) never touches the
            # covariates, so re-aggregating under a permutation returns a
            # bit-identical profile matrix and the test degenerates to
            # exactly "nonparam" -- at many times the cost. Rejected rather
            # than silently aliased, because _selection_aware_traces would
            # otherwise rebuild a VOXELWISE argmax|t| trace and test
            # something different from the trace in this run's own summary
            # CSV.
            print('inference_method "permutation" is redundant under '
                  'trace_construction="metric_first": the aggregation is '
                  'label-independent, so its null is identical to "nonparam". '
                  'Use "nonparam".')
            return False
        if self.construction == 'voxelwise' and 'nonparam' in methods:
            print('inference_method "nonparam" needs per-subject node profiles, which '
                  'only trace_construction="metric_first" produces. Use "permutation" '
                  '(selection-aware) for a voxelwise trace.')
            return False
        self.methods = methods

        self.n_permutations = int(self._p('n_permutations', 5000))
        self.cluster_forming_p = float(self._p('cluster_forming_p', 0.01))
        self.two_tailed = bool(self._p('two_tailed', True))
        self.tfce = bool(self._p('tfce', False))
        self.perm_scheme = self._p('permutation_scheme', 'freedman_lane')
        if self.perm_scheme not in ('freedman_lane', 'sign_flip'):
            print('Unknown permutation_scheme: {0}'.format(self.perm_scheme))
            return False
        self.perm_family = self._p('permutation_family', 'tract')
        if self.perm_family not in ('tract', 'network'):
            print('Unknown permutation_family: {0}'.format(self.perm_family))
            return False
        self.perm_seed = int(self._p('permutation_seed', 0))
        self.write_voxel_maps = bool(self._p('write_voxel_maps', False))

        self.min_clust = int(self._p('min_clust', params_gen.get('min_clust', 3)))
        self.alpha = float(self._p('pval_alpha', params_trace.get('pval_alpha', 0.05)))
        self.fdr_method = self._p('fdr_method', params_gen.get('fdr_method', 'none'))
        self.fdr_alpha = float(self._p('fdr_alpha', params_gen.get('fdr_alpha', 0.05)))
        self.nan_value = float(params_gen.get('nan_value', 0))

        # Trace threshold, as the legacy path defines it
        if params_trace['tract_threshold'] < self.tract_threshold:
            print('Trace threshold ({0:1.3f}) cannot be less than GLM threshold ({1:1.3f}).'
                  .format(params_trace['tract_threshold'], self.tract_threshold))
            return False
        self.trace_threshold = params_trace['tract_threshold']
        self.thresh_str = '{0:02d}'.format(round(self.trace_threshold * 100))

        # summary dir is named for whatever actually collapses the shell --
        # the aggregation for metric_first, the legacy summary_metric for
        # voxelwise -- so the two constructions never share a directory.
        self.summary_metric = (self.aggregation if self.construction == 'metric_first'
                               else params_gen['summary_metric'])

        self.profiles_dir = os.path.join(self.tracts_dir, self._p('output_dir', 'profiles_out'))
        self.use_norm = self.params['tracts']['average_directions']['use_normalized']
        self.metric_stem = self.params['tracts']['dwi_regressions'].get('metric_stem', 'betas')

        self.is_init = True
        return True

    # -- directory layout --------------------------------------------------

    def glm_dir(self, glm):
        return os.path.join(self.profiles_dir, glm)

    def summary_dir(self, glm):
        return '{0}/summary-{1}_thr{2}'.format(self.glm_dir(glm), self.summary_metric,
                                               self.thresh_str)

    def cache_dir(self, glm):
        return os.path.join(self.glm_dir(glm), 'cache')

    # -- tract name <-> roi pair ------------------------------------------

    def _split_tract(self, tract_name):
        roi_a = next(r for r in self.rois if tract_name.startswith(r + '_')
                     and tract_name[len(r) + 1:] in self.target_rois[r])
        return roi_a, tract_name[len(roi_a) + 1:]

    def _tract_file(self, tract_name):
        stem = 'tract_final_norm_bidir' if self.use_norm else 'tract_final_bidir'
        return '{0}/{1}_{2}.nii.gz'.format(self.final_dir, stem, tract_name)

    def _polyline_file(self, tract_name):
        roi_a, roi_b = self._split_tract(tract_name)
        return '{0}/polylines/maxes_{1}_{2}_sm3.poly3d'.format(self.tracts_dir, roi_a, roi_b)

    # -- shells ------------------------------------------------------------

    def build_shells(self, tract_name, verbose=False):
        """Discretise 'distance along the tract' for one tract.

        Returns a Shells object, or None if the tract has too little data.
        """
        tract_file = self._tract_file(tract_name)
        if not os.path.exists(tract_file):
            return None
        V_img = nib.load(tract_file)
        V_tract = V_img.get_fdata()
        V_tract[V_tract < self.trace_threshold] = 0
        shape = V_tract.shape
        flat_tract = V_tract.ravel()

        if self.scheme == 'flood_fill':
            dist_file = '{0}/dist/dist_bidir_{1}.nii.gz'.format(self.tracts_dir, tract_name)
            if not os.path.exists(dist_file):
                return None
            V_dist = nib.load(dist_file).get_fdata()
            V_dist[V_tract < self.trace_threshold] = 0
            flat_dist = V_dist.ravel()
            idx = np.flatnonzero(flat_dist > 0)
            if idx.size == 0:
                return None
            centers = np.unique(flat_dist[idx])
            bin_id = np.searchsorted(centers, flat_dist[idx])
            arc = None
        else:
            poly_file = self._polyline_file(tract_name)
            if not os.path.isfile(poly_file):
                if verbose:
                    print('   No polyline for {0} ({1}).'.format(tract_name, poly_file))
                return None
            poly = utils.read_polyline_mgui(poly_file)
            idx = np.flatnonzero(flat_tract > 0)
            if idx.size == 0:
                return None
            vox = np.array(np.unravel_index(idx, shape)).T
            mm = nib.affines.apply_affine(V_img.header.get_sform(), vox)

            s_vertex, L = arc_length_param(poly)
            s_fine, pts_fine = resample_polyline_uniform(poly, self.resample_step_mm)
            # k=1 nearest fine-anchor query -- same pattern route_dijkstra's
            # route_penalty uses to project voxels onto a path.
            _, nearest = cKDTree(pts_fine).query(mm)
            voxel_s = s_fine[nearest]

            K = max(1, int(round(L / self.shell_width_mm)))
            edges = np.linspace(0.0, L, K + 1)
            bin_id = np.clip(np.searchsorted(edges, voxel_s, side='right') - 1, 0, K - 1)
            centers = (edges[:-1] + edges[1:]) / 2.0
            arc = L

        shells = Shells(self.scheme, idx, bin_id, centers, flat_tract[idx], shape, arc)

        # An empty shell is KEPT, deliberately. It means no voxel survives
        # tract_threshold over that stretch of arc length -- a genuine gap in
        # the tract, not a binning artifact (measured on LC_L->BA35_L at
        # tract_threshold 0.5: one empty slab at every width from 2.0 to 3.0
        # mm). Its node carries t = 0, p = 1 and an all-zero residual row, so
        # it falls below any cluster-forming threshold and BREAKS a cluster
        # that would otherwise span the gap. That break is the correct
        # inference: there is no evidence there to join the two sides with.
        # (The all-zero residual row is excluded from the FWHM estimate by
        # the 'nz' filter in _rft_traces, so it doesn't bias smoothness.)
        empty = int(np.sum(shells.counts == 0))
        if empty and verbose:
            print('   NOTE: {0} of {1} shells are empty for {2} (no voxel above '
                  'tract_threshold); those nodes break clusters, as intended.'
                  .format(empty, shells.K, tract_name))
        if verbose:
            c = shells.counts
            print('   {0}: K={1} shells, voxels/shell min={2} max={3} mean={4:.1f}'
                  .format(tract_name, shells.K, c.min(), c.max(), c.mean()))
        return shells

    # -- per-subject data --------------------------------------------------

    def _subject_volume_file(self, subject, tract_name):
        params_tracts = self.params['tracts']
        params_regress = params_tracts['dwi_regressions']
        prefix_sub = '{0}{1}'.format(params_tracts['general']['prefix'], subject)
        subject_dir = os.path.join(self.project_dir, params_tracts['general']['deriv_dir'],
                                   prefix_sub, params_tracts['general']['sub_dirs'])
        subj_output_dir = '{0}/dwi/{1}/{2}'.format(subject_dir, params_regress['regress_dir'],
                                                   params_tracts['general']['network_name'])
        return '{0}/{1}_mni_sm_{2}um_{3}.nii.gz'.format(
            subj_output_dir, self.metric_stem,
            int(1000.0 * params_regress['beta_sm_fwhm']), tract_name)

    def load_subject_data(self, tract_name, idx, verbose=False):
        """Per-subject metric values at the tract-mask voxels.

        Returns (Y, missing) with Y (n_present, n_vox) and missing (N_sub,)
        bool -- exactly the same read/skip logic fit_glms uses.
        """
        N_sub = len(self.subjects)
        Y = np.zeros((N_sub, idx.size))
        missing = np.zeros(N_sub, dtype=bool)
        for i, subject in enumerate(self.subjects):
            f = self._subject_volume_file(subject, tract_name)
            try:
                Y[i, :] = nib.load(f).get_fdata().ravel()[idx]
            except FileNotFoundError:
                missing[i] = True
        if verbose and missing.any():
            print('   Missing tract data for {0} subjects.'.format(int(missing.sum())))
        return Y[~missing], missing

    # -- shell aggregation (metric-first) ----------------------------------

    def aggregate_profiles(self, Y, shells):
        """Collapse per-voxel per-subject values to per-shell per-subject.

        Y: (n_sub, n_vox). Returns (n_sub, K) -- one profile per subject.
        This happens in METRIC space, before any inference touches the data,
        which is what removes the legacy path's circularity.
        """
        n_sub = Y.shape[0]
        P = np.zeros((n_sub, shells.K))
        w_all = shells.weights
        for k, members in enumerate(shells.members):
            if members.size == 0:
                continue
            sel = members
            if self.core_threshold is not None:
                core = sel[w_all[sel] >= float(self.core_threshold)]
                if core.size > 0:
                    sel = core
            block = Y[:, sel]
            if self.aggregation == 'mean':
                P[:, k] = block.mean(axis=1)
            elif self.aggregation == 'wmean':
                w = np.power(w_all[sel], self.max_weight_factor)
                sw = w.sum()
                P[:, k] = (block @ w) / sw if sw > 0 else block.mean(axis=1)
            elif self.aggregation == 'median':
                P[:, k] = np.median(block, axis=1)
            elif self.aggregation == 'trimmed':
                P[:, k] = stats.trim_mean(block, 0.1, axis=1)
            elif self.aggregation == 'max':
                w = np.power(w_all[sel], self.max_weight_factor)
                pick = np.argmax(np.abs(block * w), axis=1)
                P[:, k] = block[np.arange(n_sub), pick]
        return P

    # -- fitting -----------------------------------------------------------

    def _variants(self, glm):
        """('' , X_raw, standardize_y=False) plus one per general.effect_scales
        scheme, matching fit_glms(). t/p are identical across variants (OLS is
        invariant to positive rescaling of X and/or y) so only the raw variant
        contributes t/p/residuals; the rest contribute coefficients only."""
        out = [('', self.Xs[glm], False)]
        out += [('_{0}'.format(name), Xv, True)
                for name, Xv in self.Xs_scaled.get(glm, {}).items()]
        return out

    def fit_nodes(self, glm, Y, missing, outlier_z):
        """OLS across subjects at every column of Y independently.

        Y: (n_present, m) -- m is K for metric_first, n_vox for voxelwise.

        Returns dict with 'tvals'/'pvals' (p, m), 'coefs' {suffix: (p, m)},
        'resid' (n_present, m) and 'df' (m,).

        df is n_used - rank(X) per column (PROFILE_GLM_SPEC.md's unconditional
        df fix), which differs from the legacy N_sub - 1 both by the design's
        rank and, where outlier_z > 0, by however many rows that column
        actually dropped.
        """
        variants = self._variants(glm)
        keep = ~missing
        n, m = Y.shape
        p = variants[0][1].shape[1]

        tvals = np.zeros((p, m))
        pvals = np.ones((p, m))
        coefs = {suf: np.zeros((p, m)) for suf, _, _ in variants}
        resid = np.zeros((n, m))
        df_col = np.zeros(m, dtype=int)

        if outlier_z is None or outlier_z <= 0:
            # No trimming -- every column shares one design, so the whole
            # thing is a single matrix solve.
            for suf, Xv, do_std in variants:
                Xk = Xv[keep]
                Yv = Y
                if do_std:
                    sd = Y.std(axis=0)
                    Yv = np.where(sd > 0, (Y - Y.mean(axis=0)) / np.where(sd > 0, sd, 1.0), 0.0)
                B, T, P, R, df = ols_fit(Xk, Yv)
                coefs[suf] = B
                if suf == '':
                    tvals, pvals, resid = T, P, R
                    df_col[:] = df
            return {'tvals': tvals, 'pvals': pvals, 'coefs': coefs,
                    'resid': resid, 'df': df_col}

        # Per-column outlier trimming: the kept-row set differs per column,
        # so each column is its own small solve (same rule as fit_glms).
        for j in range(m):
            y = Y[:, j]
            lo = y.mean() - outlier_z * y.std()
            hi = y.mean() + outlier_z * y.std()
            ok = np.flatnonzero((y >= lo) & (y <= hi))
            if ok.size < p + 1:
                ok = np.arange(n)
            for suf, Xv, do_std in variants:
                Xk = Xv[keep][ok]
                yv = y[ok]
                if do_std:
                    sd = yv.std()
                    yv = (yv - yv.mean()) / sd if sd > 0 else np.zeros_like(yv)
                B, T, P, R, df = ols_fit(Xk, yv[:, None])
                coefs[suf][:, j] = B[:, 0]
                if suf == '':
                    tvals[:, j] = T[:, 0]
                    pvals[:, j] = P[:, 0]
                    resid[ok, j] = R[:, 0]
                    df_col[j] = df
        return {'tvals': tvals, 'pvals': pvals, 'coefs': coefs,
                'resid': resid, 'df': df_col}

    # -- voxelwise collapse ------------------------------------------------

    def _collapse_voxelwise(self, shells, fit, factor_ix):
        """Legacy-style trace: per shell, the P(tract)-weighted argmax|t| voxel.

        Returns (sel, ) with sel (K,) the chosen voxel index per shell (-1 if
        the shell is empty). Everything else -- t, p, coef, residuals -- is
        then read off that voxel, exactly as extract_distance_traces does.
        """
        tvals = fit['tvals'][factor_ix]
        sel = np.full(shells.K, -1, dtype=int)
        for k, members in enumerate(shells.members):
            if members.size == 0:
                continue
            w = np.power(shells.weights[members], self.max_weight_factor)
            sel[k] = members[np.argmax(np.abs(w * tvals[members]))]
        return sel

    # -- main pass ---------------------------------------------------------

    def compute_profiles(self, clobber=False, verbose=True):
        """Build the along-tract traces and write the summary contract.

        Writes, per glm:
          summary-{agg}_thr{NN}/stats_{tract}.csv    Distance + per-factor
                                                     tval/pval/fdr_pval/coef*
          summary-{agg}_thr{NN}/resids_{tract}.csv   K x N_sub
          cache/{tract}.npz                          shells + per-subject
                                                     profiles, for the
                                                     inference stage
        """
        if not self.is_init:
            print('Not initialized! Call initialize() first.')
            return False

        params_glm = self.params['glm']

        for glm in params_glm:
            sdir = self.summary_dir(glm)
            if os.path.exists(sdir):
                if not clobber:
                    print('Output directory exists! (clobber=True to overwrite): {0}'.format(sdir))
                    return False
                shutil.rmtree(sdir)
            os.makedirs(sdir)
            cdir = self.cache_dir(glm)
            if os.path.exists(cdir):
                shutil.rmtree(cdir)
            os.makedirs(cdir)
            if self.write_voxel_maps:
                for sub in ('tval', 'coef', 'pval'):
                    os.makedirs(os.path.join(self.glm_dir(glm), sub), exist_ok=True)

        if verbose:
            print('Computing profiles for {0} tracts [{1} shells, {2}].'
                  .format(len(self.tract_names), self.scheme, self.construction))

        for tract_name in tqdm(self.tract_names, desc='Tracts'):
            shells = self.build_shells(tract_name, verbose=verbose)
            if shells is None or shells.K < 3:
                if verbose:
                    print('   Skipping {0} (no usable tract/shells).'.format(tract_name))
                continue

            Y, missing = self.load_subject_data(tract_name, shells.idx, verbose=verbose)
            if Y.shape[0] < 5:
                print('   Skipping {0}: only {1} subjects with data.'
                      .format(tract_name, Y.shape[0]))
                continue

            V_img = nib.load(self._tract_file(tract_name))

            for glm in params_glm:
                factors = self.params['glm'][glm]['factors']
                outlier_z = self.params['glm'][glm].get('outlier_z', 0)

                if self.construction == 'metric_first':
                    P = self.aggregate_profiles(Y, shells)             # (n, K)
                    fit = self.fit_nodes(glm, P, missing, outlier_z)
                    node_resid = fit['resid']                           # (n, K)
                    tvals = fit['tvals']
                    pvals = fit['pvals']
                    coefs = fit['coefs']
                    df_col = fit['df']
                    subj_profiles = P
                else:
                    fit = self.fit_nodes(glm, Y, missing, outlier_z)    # per voxel
                    tvals = np.zeros((len(factors), shells.K))
                    pvals = np.ones((len(factors), shells.K))
                    coefs = {suf: np.zeros((len(factors), shells.K)) for suf in fit['coefs']}
                    node_resid = np.zeros((Y.shape[0], shells.K))
                    df_col = np.zeros(shells.K, dtype=int)
                    # The chosen voxel is factor-specific (argmax|t| of THAT
                    # factor), exactly as extract_distance_traces does it.
                    for fi in range(len(factors)):
                        sel = self._collapse_voxelwise(shells, fit, fi)
                        ok = sel >= 0
                        tvals[fi, ok] = fit['tvals'][fi, sel[ok]]
                        pvals[fi, ok] = fit['pvals'][fi, sel[ok]]
                        for suf in coefs:
                            coefs[suf][fi, ok] = fit['coefs'][suf][fi, sel[ok]]
                        if fi == 1 or len(factors) == 1:
                            # residual trace / df follow the first non-intercept
                            # factor's selection, matching the legacy single
                            # resids_{tract}.csv (which is likewise overwritten
                            # per factor and ends up being the last one's).
                            node_resid[:, ok] = fit['resid'][:, sel[ok]]
                            df_col[ok] = fit['df'][sel[ok]]
                    if df_col.max() == 0:
                        df_col[:] = fit['df'].max()
                    subj_profiles = None

                self._write_summary(glm, tract_name, shells, factors, tvals, pvals,
                                    coefs, node_resid)

                np.savez_compressed(
                    os.path.join(self.cache_dir(glm), '{0}.npz'.format(tract_name)),
                    idx=shells.idx, bin_id=shells.bin_id, centers=shells.centers,
                    weights=shells.weights, shape=np.array(shells.shape),
                    missing=missing, df=df_col,
                    profiles=(subj_profiles if subj_profiles is not None
                              else np.zeros((0, 0))))

                if self.write_voxel_maps and self.construction == 'voxelwise':
                    self._write_voxel_maps(glm, tract_name, shells, factors, fit, V_img)

            del Y
            gc.collect()

        if verbose:
            print('Done.')
        return True

    def _write_summary(self, glm, tract_name, shells, factors, tvals, pvals, coefs, node_resid):
        sdir = self.summary_dir(glm)
        cols = {'Distance': shells.centers}
        for fi, factor in enumerate(factors):
            if fi == 0 and factor == 'Intercept':
                continue
            fstr = factor.replace('*', 'X')
            t = tvals[fi].copy()
            t[np.isnan(t)] = self.nan_value
            cols['{0}|tval'.format(fstr)] = t
            cols['{0}|pval'.format(fstr)] = pvals[fi]
            try:
                R = smm.multipletests(pvals[fi], self.fdr_alpha,
                                      self.fdr_method if self.fdr_method != 'none' else 'fdr_bh')
                cols['{0}|fdr_pval'.format(fstr)] = R[1]
            except (ZeroDivisionError, ValueError):
                cols['{0}|fdr_pval'.format(fstr)] = pvals[fi]
            for suf in sorted(coefs):
                cols['{0}|coef{1}'.format(fstr, suf)] = coefs[suf][fi]
        pd.DataFrame(cols).to_csv('{0}/stats_{1}.csv'.format(sdir, tract_name), index=False)
        # (K x N_sub), same orientation the RFT stage expects
        np.savetxt('{0}/resids_{1}.csv'.format(sdir, tract_name), node_resid.T,
                   delimiter=',', header='', comments='', fmt='%1.8f')

    def _write_voxel_maps(self, glm, tract_name, shells, factors, fit, V_img):
        """Optional per-voxel stat maps (voxelwise construction only).

        Off by default: PROFILE_GLM_SPEC.md's decision is that the profile
        path emits 1-D traces, not voxel maps. Enable via
        profiles.write_voxel_maps for QA against the legacy tree.
        """
        gdir = self.glm_dir(glm)
        V = np.zeros(shells.shape)
        for fi, factor in enumerate(factors):
            if fi == 0 and factor == 'Intercept':
                continue
            fstr = factor.replace('*', 'X')
            for sub, arr in (('tval', fit['tvals'][fi]), ('pval', fit['pvals'][fi]),
                             ('coef', fit['coefs'][''][fi])):
                V.fill(0)
                V.ravel()[shells.idx] = arr
                nib.save(nib.Nifti1Image(V, V_img.affine, V_img.header),
                         '{0}/{1}/{2}_{3}.nii.gz'.format(gdir, sub, tract_name, fstr))

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _perm_generator(self, n, n_perm, factor_str):
        """Deterministic permutation / sign-flip draws for one factor.

        Seeded from (permutation_seed, factor) so a rerun reproduces the same
        null exactly, and so two factors never share a null by accident.
        """
        # zlib.crc32, not hash(): Python's str hash is salted per process
        # (PYTHONHASHSEED), which would silently make 'permutation_seed'
        # non-reproducible across runs.
        seed = (self.perm_seed, zlib.crc32(factor_str.encode('utf-8')))
        rng = np.random.default_rng(seed)
        if self.perm_scheme == 'sign_flip':
            return [rng.choice([-1.0, 1.0], size=n) for _ in range(n_perm)]
        return [rng.permutation(n) for _ in range(n_perm)]

    @staticmethod
    def _freedman_lane_parts(X, col, Y):
        """Split Y into its nuisance-fit and nuisance-residual parts.

        Freedman-Lane: permute only the part of Y orthogonal to the nuisance
        columns Z (= X without the tested column), then re-add the nuisance
        fit and refit the FULL model. This keeps the nuisance structure in
        place under the null instead of destroying it, which is what makes
        the test approximately exact with covariates present. It remains an
        approximation for "all factors simultaneously" -- each factor is
        tested against a null in which the OTHER factors' effects are held
        fixed at their estimated values; anything more exact is PALM
        territory (PROFILE_GLM_SPEC.md 5b).
        """
        Z = np.delete(X, col, axis=1)
        if Z.shape[1] == 0:
            return np.zeros_like(Y), Y
        Hz = Z @ np.linalg.pinv(Z)
        fit = Hz @ Y
        return fit, Y - fit

    def _perm_statistic(self, trace, t_thresh):
        """Max-statistic used to build the FWER null for a single trace."""
        if self.tfce:
            return float(tfce_1d(trace).max()) if trace.size else 0.0
        return max_cluster_mass(trace, t_thresh, self.min_clust)

    def _perm_pvalues(self, trace, null, t_thresh):
        """Turn an observed trace + a max-statistic null into node p-values.

        Cluster mode: every node of a surviving cluster gets that cluster's
        FWER p. TFCE mode: every node gets its own p from the max-TFCE null.
        Returns (pvals, clusters) with clusters 0 outside a significant run.
        """
        n_perm = null.size
        pvals = np.ones(trace.size)
        clusters = np.zeros(trace.size, dtype=int)
        if self.tfce:
            scores = tfce_1d(trace)
            for i, s in enumerate(scores):
                pvals[i] = (1.0 + np.count_nonzero(null >= s)) / (n_perm + 1.0)
            label = 1
            i = 0
            while i < trace.size:
                if pvals[i] >= self.alpha:
                    i += 1
                    continue
                j = i
                while j < trace.size and pvals[j] < self.alpha and \
                        np.sign(trace[j]) == np.sign(trace[i]):
                    j += 1
                if (j - i) >= self.min_clust:
                    clusters[i:j] = label
                    label += 1
                i = j
            return pvals, clusters
        label = 1
        for (i, j, mass, _sign) in cluster_masses(trace, t_thresh, self.min_clust):
            p = (1.0 + np.count_nonzero(null >= mass)) / (n_perm + 1.0)
            pvals[i:j] = p
            if p < self.alpha:
                clusters[i:j] = label
                label += 1
        return pvals, clusters

    # -- nonparam (5b): permutation on the per-subject node profiles -------

    def _nonparam_traces(self, glm, factor_ix, factor_str, verbose):
        """Observed traces + max-statistic null, from cached node profiles.

        Note the observed t here is refit WITHOUT per-node outlier trimming,
        because the null cannot honour a trimming rule that depends on the
        (permuted) data. The trimmed t written by compute_profiles stays in
        the stats CSV as '{factor}|tval'; the untrimmed trace this method
        actually tests is written alongside as '{factor}|np_tval', so the
        two are never silently conflated.
        """
        cdir = self.cache_dir(glm)
        X_full = self.Xs[glm]
        obs = {}
        parts = {}
        for tract_name in self.tract_names:
            f = os.path.join(cdir, '{0}.npz'.format(tract_name))
            if not os.path.isfile(f):
                continue
            z = np.load(f)
            P = z['profiles']
            if P.size == 0:
                continue
            keep = ~z['missing']
            X = X_full[keep]
            t = tvals_for_column(X, P, factor_ix)
            obs[tract_name] = t
            parts[tract_name] = (X, P)

        if not obs:
            return {}, {}, 0.0, 0

        n_set = {P.shape[0] for _, P in parts.values()}
        family = self.perm_family
        if family == 'network' and len(n_set) > 1:
            warnings.warn('permutation_family="network" needs the same subject set in '
                          'every tract (found {0} distinct N). Falling back to '
                          '"tract".'.format(len(n_set)))
            family = 'tract'

        df = min(P.shape[0] - np.linalg.matrix_rank(X) for X, P in parts.values())
        pt = self.cluster_forming_p / 2.0 if self.two_tailed else self.cluster_forming_p
        t_thresh = float(stats.t.isf(pt, df))

        nulls = {}
        for tract_name, (X, P) in parts.items():
            fit, res = self._freedman_lane_parts(X, factor_ix, P)
            XtX_inv = np.linalg.pinv(X.T @ X)
            draws = self._perm_generator(P.shape[0], self.n_permutations, factor_str)
            null = np.zeros(self.n_permutations)
            for b, d in enumerate(tqdm(draws, desc='  perm {0} [{1}]'.format(factor_str, tract_name),
                                       leave=False, disable=not verbose)):
                Yp = fit + (res * d[:, None] if self.perm_scheme == 'sign_flip' else res[d])
                null[b] = self._perm_statistic(tvals_for_column(X, Yp, factor_ix, XtX_inv), t_thresh)
            nulls[tract_name] = null

        if family == 'network':
            joint = np.max(np.stack(list(nulls.values())), axis=0)
            nulls = {t: joint for t in nulls}
        return obs, nulls, t_thresh, df

    # -- permutation (5c): selection-aware, rebuilds the trace each draw ---

    def _selection_aware_traces(self, glm, factor_ix, factor_str, verbose):
        """Freedman-Lane at the VOXEL level, re-picking the argmax voxel per
        shell inside every permutation.

        This is the only route that makes inference on a voxelwise + max
        trace exactly valid: the peak-selection step that inflates the node
        statistic is inside the null, so the null carries exactly the same
        inflation the observed statistic does.
        """
        cdir = self.cache_dir(glm)
        X_full = self.Xs[glm]
        obs, nulls = {}, {}
        dfs = []

        tract_list = [t for t in self.tract_names
                      if os.path.isfile(os.path.join(cdir, '{0}.npz'.format(t)))]

        # First pass: observed traces + df, so the cluster-forming threshold
        # is fixed before any permutation is drawn.
        cached = {}
        for tract_name in tract_list:
            z = np.load(os.path.join(cdir, '{0}.npz'.format(tract_name)))
            keep = ~z['missing']
            X = X_full[keep]
            if tract_name in self._vol_cache:
                Y, missing = self._vol_cache[tract_name]
            else:
                Y, missing = self.load_subject_data(tract_name, z['idx'])
                self._vol_cache[tract_name] = (Y, missing)
            if Y.shape[0] != X.shape[0]:
                warnings.warn('Subject availability changed since compute_profiles for '
                              '{0}; skipping.'.format(tract_name))
                continue
            shells = Shells(self.scheme, z['idx'], z['bin_id'], z['centers'],
                            z['weights'], tuple(z['shape']))
            XtX_inv = np.linalg.pinv(X.T @ X)
            t_vox = tvals_for_column(X, Y, factor_ix, XtX_inv)
            obs[tract_name] = self._collapse_trace(shells, t_vox)
            dfs.append(Y.shape[0] - np.linalg.matrix_rank(X))
            cached[tract_name] = (X, XtX_inv, Y, shells)

        if not obs:
            return {}, {}, 0.0, 0

        df = min(dfs)
        pt = self.cluster_forming_p / 2.0 if self.two_tailed else self.cluster_forming_p
        t_thresh = float(stats.t.isf(pt, df))

        family = self.perm_family
        if family == 'network' and len({Y.shape[0] for _, _, Y, _ in cached.values()}) > 1:
            warnings.warn('permutation_family="network" needs the same subject set in '
                          'every tract. Falling back to "tract".')
            family = 'tract'

        for tract_name, (X, XtX_inv, Y, shells) in cached.items():
            fit, res = self._freedman_lane_parts(X, factor_ix, Y)
            draws = self._perm_generator(Y.shape[0], self.n_permutations, factor_str)
            null = np.zeros(self.n_permutations)
            for b, d in enumerate(tqdm(draws, desc='  perm {0} [{1}]'.format(factor_str, tract_name),
                                       leave=False, disable=not verbose)):
                Yp = fit + (res * d[:, None] if self.perm_scheme == 'sign_flip' else res[d])
                t_vox = tvals_for_column(X, Yp, factor_ix, XtX_inv)
                null[b] = self._perm_statistic(self._collapse_trace(shells, t_vox), t_thresh)
            nulls[tract_name] = null
            del fit, res          # Y is cached across factors; don't free it
            gc.collect()

        if family == 'network':
            joint = np.max(np.stack(list(nulls.values())), axis=0)
            nulls = {t: joint for t in nulls}
        return obs, nulls, t_thresh, df

    def _collapse_trace(self, shells, t_vox):
        """P(tract)-weighted argmax|t| per shell -- the legacy trace rule."""
        out = np.zeros(shells.K)
        for k, members in enumerate(shells.members):
            if members.size == 0:
                continue
            w = np.power(shells.weights[members], self.max_weight_factor)
            out[k] = t_vox[members[np.argmax(np.abs(w * t_vox[members]))]]
        return out

    # -- RFT (5a) ----------------------------------------------------------

    def _rft_traces(self, glm, factor_str, verbose):
        """Legacy analytic 1D-RFT, with the df fix.

        FWHM is estimated from the node residuals exactly as
        extract_distance_traces_rft1d does (mean over tracts), so RFT numbers
        from this module are directly comparable to the legacy baseline apart
        from (a) the shell scheme, (b) the node statistic, and (c) df now
        being N_sub - rank(X) instead of N_sub - 1.
        """
        sdir = self.summary_dir(glm)
        cdir = self.cache_dir(glm)
        mean_fwhm, denom = 0.0, 0
        dfs = []
        last_err = None
        for tract_name in self.tract_names:
            rf = '{0}/resids_{1}.csv'.format(sdir, tract_name)
            if not os.path.isfile(rf):
                continue
            resids = np.loadtxt(rf, delimiter=',')
            nz = np.sum(resids, 1) != 0
            resids = resids[nz, :]
            try:
                fwhm = rft1d.geom.estimate_fwhm(resids.T)
                if not np.isinf(fwhm):
                    mean_fwhm += fwhm
                    denom += 1
            except Exception as e:
                # Per-tract failure (too few usable nodes) is expected and
                # skipped -- but if EVERY tract fails we must say why rather
                # than quietly emitting an empty RFT tree.
                last_err = e
            cf = os.path.join(cdir, '{0}.npz'.format(tract_name))
            if os.path.isfile(cf):
                d = np.load(cf)['df']
                d = d[d > 0]
                if d.size:
                    dfs.append(int(round(float(np.median(d)))))
        if denom == 0 or not dfs:
            print('    No usable FWHM estimate for any tract; skipping RFT.'
                  '{0}'.format(' Last error: {0!r}'.format(last_err) if last_err else ''))
            return {}, 0.0, 0
        mean_fwhm /= denom
        df = int(min(dfs))
        if verbose:
            print('    Mean FWHM {0:1.5f}, df {1}'.format(mean_fwhm, df))

        out = {}
        for tract_name in self.tract_names:
            sf = '{0}/stats_{1}.csv'.format(sdir, tract_name)
            if not os.path.isfile(sf):
                continue
            tvals = pd.read_csv(sf)['{0}|tval'.format(factor_str)].values
            pvals, clusters, logpvals = utils.get_tvalue_rft1d_clusters(
                tvals, self.alpha, df, mean_fwhm, self.min_clust)
            out[tract_name] = (tvals, pvals, clusters, logpvals)
        return out, mean_fwhm, df

    # -- orchestration -----------------------------------------------------

    def run_inference(self, clobber=False, verbose=True):
        """Run every method in profiles.inference_method over the traces
        compute_profiles() wrote. Each method writes its own suffixed
        outputs; none of them touch another's."""
        if not self.is_init:
            print('Not initialized! Call initialize() first.')
            return False

        params_glm = self.params['glm']

        for glm in params_glm:
            self._vol_cache = {}
            sdir = self.summary_dir(glm)
            if not os.path.isdir(sdir):
                print('No profiles found at {0}; run compute_profiles() first.'.format(sdir))
                return False
            factors = params_glm[glm]['factors']

            for method in self.methods:
                sfx = self.METHOD_SUFFIX[method]
                if verbose:
                    print('{0}: inference "{1}"'.format(glm, method))

                pdir = '{0}/pval-{1}_thr{2}'.format(self.glm_dir(glm), sfx, self.thresh_str)
                tdir = '{0}/tval-{1}_thr{2}'.format(self.glm_dir(glm), sfx, self.thresh_str)
                # Namespaced by the profiles output_dir: the legacy RFT path
                # writes polylines/rft/{glm} with the exact same filenames, so
                # a bare polylines/{sfx}/{glm} here would overwrite the legacy
                # baseline's own poly3d files. Point the plotting side's
                # rft_dir at this path to render profile results.
                poly_dir = '{0}/polylines/{1}/{2}/{3}'.format(
                    self.tracts_dir, self._p('output_dir', 'profiles_out'), sfx, glm)
                for d in (pdir, tdir):
                    if os.path.isdir(d):
                        if not clobber:
                            print(' Output exists! (clobber=True to overwrite): {0}'.format(d))
                            return False
                        shutil.rmtree(d)
                    os.makedirs(d)
                os.makedirs(poly_dir, exist_ok=True)

                rows = []
                for fi, factor in enumerate(factors):
                    if fi == 0 and factor == 'Intercept':
                        continue
                    factor_str = factor.replace('*', 'X')
                    if verbose:
                        print(' {0}'.format(factor_str))

                    results = self._infer_factor(glm, method, fi, factor_str, verbose)
                    if not results:
                        continue
                    results = self._apply_cluster_fdr(results, verbose)
                    rows += self._write_factor_results(glm, sfx, factor_str, results,
                                                       pdir, tdir, poly_dir)

                out = pd.DataFrame(rows, columns=['Factor', 'From', 'To', 'Direction',
                                                  'T_count', 'T_sum', 'T_max', 'T_mean_all'])
                out.to_csv('{0}/tcounts-{1}.csv'.format(sdir, sfx), index=False)
                if verbose:
                    print(' Wrote {0}/tcounts-{1}.csv'.format(sdir, sfx))
        self.export_metadata(verbose=verbose)
        return True

    def _infer_factor(self, glm, method, factor_ix, factor_str, verbose):
        """One factor, one method -> {tract: (tvals_nt, pvals, clusters, logpvals)}."""
        if method == 'rft':
            out, _fwhm, _df = self._rft_traces(glm, factor_str, verbose)
            return out

        if method == 'nonparam':
            obs, nulls, t_thresh, df = self._nonparam_traces(glm, factor_ix, factor_str, verbose)
        else:
            obs, nulls, t_thresh, df = self._selection_aware_traces(glm, factor_ix,
                                                                    factor_str, verbose)
        if not obs:
            return {}
        if verbose:
            print('    {0} permutations, cluster-forming |t| >= {1:.3f} (df {2}){3}'
                  .format(self.n_permutations, t_thresh, df,
                          ', TFCE' if self.tfce else ''))
        out = {}
        for tract_name, trace in obs.items():
            pvals, clusters = self._perm_pvalues(trace, nulls[tract_name], t_thresh)
            # log p of a permutation test is exactly log of the discrete
            # p; it can never beat log(1/(n_perm+1)) -- that floor is the
            # honest resolution limit of this null, not an underflow.
            logpvals = np.where(clusters > 0, np.log(np.maximum(pvals, 1e-300)), 0.0)
            out[tract_name] = (trace, pvals, clusters, logpvals)
        return out

    def _apply_cluster_fdr(self, results, verbose):
        """FDR across every cluster of every tract, as the legacy RFT path
        does when fdr_method != 'none'. Rejected clusters lose both their
        t-values and their cluster label, so 'clusters' stays the single
        source of truth downstream."""
        if self.fdr_method == 'none':
            return results
        pvals, keys = [], []
        for tract_name, (t, p, c, lp) in results.items():
            for cid in np.unique(c[c > 0]):
                pvals.append(float(np.max(p[c == cid])))
                keys.append((tract_name, cid))
        if not pvals:
            return results
        try:
            R = smm.multipletests(pvals, self.fdr_alpha, self.fdr_method)
        except (ZeroDivisionError, ValueError):
            if verbose:
                print('    FDR correction failed; leaving cluster p-values uncorrected.')
            return results
        for (tract_name, cid), p_fdr, sig in zip(keys, R[1], R[0]):
            t, p, c, lp = results[tract_name]
            mask = c == cid
            p[mask] = p_fdr
            if not sig:
                c[mask] = 0
                lp[mask] = 0.0
        return results

    def _write_factor_results(self, glm, sfx, factor_str, results, pdir, tdir, poly_dir):
        """Stats columns, thresholded volumes, poly3d trace and tcounts rows."""
        sdir = self.summary_dir(glm)
        cdir = self.cache_dir(glm)
        rows = []
        for tract_name, (tvals_nt, pvals, clusters, logpvals) in results.items():
            tvals_thr = tvals_nt.copy()
            tvals_thr[clusters == 0] = 0

            sf = '{0}/stats_{1}.csv'.format(sdir, tract_name)
            T = pd.read_csv(sf)
            if sfx != 'rft':
                # The permutation methods test an untrimmed refit of the
                # trace (see _nonparam_traces); keep it visible next to the
                # trimmed '{factor}|tval' rather than overwriting it.
                T['{0}|{1}_tval'.format(factor_str, sfx)] = tvals_nt
            T['{0}|{1}_pval'.format(factor_str, sfx)] = pvals
            T['{0}|{1}_clusters'.format(factor_str, sfx)] = clusters
            T['{0}|{1}_logpval'.format(factor_str, sfx)] = logpvals
            T.to_csv(sf, index=False)

            z = np.load(os.path.join(cdir, '{0}.npz'.format(tract_name)))
            shape = tuple(z['shape'])
            idx, bin_id = z['idx'], z['bin_id']
            V_img = nib.load(self._tract_file(tract_name))
            V_p = np.zeros(shape)
            V_t = np.zeros(shape)
            V_p.ravel()[idx] = pvals[bin_id]
            V_t.ravel()[idx] = tvals_thr[bin_id]
            hdr = copy.copy(V_img.header)
            hdr['datatype'] = 16
            nib.save(nib.Nifti1Image(V_p, V_img.affine, hdr),
                     '{0}/{1}_{2}.nii.gz'.format(pdir, tract_name, factor_str))
            nib.save(nib.Nifti1Image(V_t, V_img.affine, hdr),
                     '{0}/{1}_{2}.nii.gz'.format(tdir, tract_name, factor_str))

            # Polyline placement. For length_proportional the shell centres
            # ARE arc lengths, so this is exact sampling rather than the
            # proportional rescaling flood-fill indices need.
            dists = T['Distance'].values
            poly_file = self._polyline_file(tract_name)
            if os.path.isfile(poly_file):
                poly = utils.read_polyline_mgui(poly_file)
                if self.scheme == 'length_proportional':
                    line = polyline_at_arclength(poly, dists)
                else:
                    line = utils.resample_polyline_by_distance(poly, dists)
                coef_prefix = '{0}|coef'.format(factor_str)
                coef_cols = sorted(c for c in T.columns
                                   if c == coef_prefix or c.startswith(coef_prefix + '_'))
                data = np.transpose(np.stack(
                    [tvals_nt, tvals_thr, pvals, clusters.astype(float), logpvals] +
                    [T[c].values for c in coef_cols]))
                names = ['tvals', 'tvals_thr', 'pvals', 'clusters', 'logpvals'] + \
                        ['coef{0}'.format(c[len(coef_prefix):]) for c in coef_cols]
                utils.write_polyline_mgui(
                    line, '{0}/tvals_{1}_{2}_{3}_{4}.poly3d'.format(
                        poly_dir, sfx, tract_name, factor_str, self.thresh_str),
                    factor_str, data, names, data_formats={'pvals': '{0:.6e}'})

            if np.any(tvals_thr != 0):
                roi_a, roi_b = self._split_tract(tract_name)
                pos, neg = tvals_thr > 0, tvals_thr < 0
                nt_pos, nt_neg = tvals_nt > 0, tvals_nt < 0
                rows.append([factor_str, roi_a, roi_b, 'pos',
                             int(np.count_nonzero(pos)), float(np.sum(tvals_thr[pos])),
                             float(np.max(tvals_thr[pos])) if pos.any() else 0.0,
                             float(np.mean(tvals_nt[nt_pos])) if nt_pos.any() else 0.0])
                rows.append([factor_str, roi_a, roi_b, 'neg',
                             int(np.count_nonzero(neg)), float(-np.sum(tvals_thr[neg])),
                             float(np.max(-tvals_thr[neg])) if neg.any() else 0.0,
                             float(np.mean(-tvals_nt[nt_neg])) if nt_neg.any() else 0.0])
        return rows

    # -- metadata export ---------------------------------------------------

    def export_metadata(self, verbose=True):
        """Write the run's geometry and inference parameters next to the results.

        Everything here is recomputed from artefacts that already exist
        (cache/{tract}.npz, resids_*.csv, stats_*.csv), NOT from state held
        over from a fit -- so this can be called on a tree produced by an
        earlier run without repeating any permutations. FWHM is estimated
        exactly as _rft_traces does, so the exported value is the one the RFT
        inference actually used.

        Writes, per glm, into the summary dir:
          shell_geometry.csv        one row per tract
          inference_meta-{sfx}.csv  one row per (factor, tract) per method
        """
        if not self.is_init:
            print('Not initialized! Call initialize() first.')
            return False

        for glm in self.params['glm']:
            sdir = self.summary_dir(glm)
            cdir = self.cache_dir(glm)
            if not os.path.isdir(sdir):
                continue
            factors = self.params['glm'][glm]['factors']

            geom, fwhm_by_tract = [], {}
            for tract_name in self.tract_names:
                cf = os.path.join(cdir, '{0}.npz'.format(tract_name))
                if not os.path.isfile(cf):
                    continue
                z = np.load(cf)
                bin_id, centers = z['bin_id'], z['centers']
                counts = np.bincount(bin_id, minlength=centers.size)
                df = z['df']; df = df[df > 0]
                n_sub = int((~z['missing']).sum())

                fwhm = np.nan
                rf = '{0}/resids_{1}.csv'.format(sdir, tract_name)
                if os.path.isfile(rf):
                    resids = np.loadtxt(rf, delimiter=',')
                    nz = np.sum(resids, 1) != 0
                    try:
                        f = rft1d.geom.estimate_fwhm(resids[nz, :].T)
                        if not np.isinf(f):
                            fwhm = float(f)
                    except Exception:
                        pass
                fwhm_by_tract[tract_name] = fwhm

                # profile_core_threshold restricts aggregation to the tract
                # core, but a shell with NO core voxel silently falls back to
                # its full membership (see aggregate_profiles) -- so some nodes
                # are core-restricted and others are not. Count them: a trace
                # with many fallbacks is not the core trace it claims to be.
                n_core_fallback = 0
                core_frac = np.nan
                if self.core_threshold is not None:
                    w = z['weights']
                    keep = w >= float(self.core_threshold)
                    core_frac = float(keep.mean())
                    for k in range(centers.size):
                        m = bin_id == k
                        if m.any() and not (m & keep).any():
                            n_core_fallback += 1

                geom.append({
                    'tract': tract_name,
                    'shell_scheme': self.scheme,
                    'trace_construction': self.construction,
                    'aggregation': self.summary_metric,
                    'K': int(centers.size),
                    'shell_width_mm': (self.shell_width_mm
                                       if self.scheme == 'length_proportional' else np.nan),
                    'resample_step_mm': (self.resample_step_mm
                                         if self.scheme == 'length_proportional' else np.nan),
                    'arc_length_mm': (float(centers[-1] - centers[0])
                                      if self.scheme == 'length_proportional' else np.nan),
                    'n_voxels': int(bin_id.size),
                    'vox_per_shell_min': int(counts.min()),
                    'vox_per_shell_max': int(counts.max()),
                    'vox_per_shell_mean': float(counts.mean()),
                    'vox_per_shell_median': float(np.median(counts)),
                    'n_empty_shells': int((counts == 0).sum()),
                    'tract_threshold': self.trace_threshold,
                    'profile_core_threshold': (self.core_threshold
                                               if self.core_threshold is not None else np.nan),
                    'core_voxel_fraction': core_frac,
                    'n_shells_core_fallback': n_core_fallback,
                    'n_subjects': n_sub,
                    'df_median': int(np.median(df)) if df.size else 0,
                    'df_min': int(df.min()) if df.size else 0,
                    'df_max': int(df.max()) if df.size else 0,
                    'fwhm_nodes': fwhm,
                })
            if not geom:
                continue
            pd.DataFrame(geom).to_csv('{0}/shell_geometry.csv'.format(sdir), index=False)

            # The RFT stage uses ONE pooled FWHM and one df for the whole
            # network -- recompute both the same way so the exported numbers
            # are the ones inference actually ran with.
            vals = [g['fwhm_nodes'] for g in geom if not np.isnan(g['fwhm_nodes'])]
            fwhm_pooled = float(np.mean(vals)) if vals else np.nan
            df_pooled = int(min(g['df_median'] for g in geom))
            pt = self.cluster_forming_p / 2.0 if self.two_tailed else self.cluster_forming_p
            t_thresh = float(stats.t.isf(pt, df_pooled)) if df_pooled > 0 else np.nan

            for method in self.methods:
                sfx = self.METHOD_SUFFIX[method]
                rows = []
                for tract_name in self.tract_names:
                    sf = '{0}/stats_{1}.csv'.format(sdir, tract_name)
                    if not os.path.isfile(sf):
                        continue
                    T = pd.read_csv(sf)
                    for fi, factor in enumerate(factors):
                        if fi == 0 and factor == 'Intercept':
                            continue
                        fstr = factor.replace('*', 'X')
                        cc = '{0}|{1}_clusters'.format(fstr, sfx)
                        if cc not in T.columns:
                            continue
                        c = T[cc].values
                        pv = T['{0}|{1}_pval'.format(fstr, sfx)].values
                        lp = T['{0}|{1}_logpval'.format(fstr, sfx)].values
                        sig = c > 0
                        rows.append({
                            'method': method, 'suffix': sfx,
                            'factor': fstr, 'tract': tract_name,
                            'K': int(len(T)),
                            'n_sig_nodes': int(sig.sum()),
                            'n_clusters': int(np.unique(c[sig]).size),
                            'best_pval': float(pv[sig].min()) if sig.any() else 1.0,
                            'best_logpval': float(lp[sig].min()) if sig.any() else 0.0,
                            'fwhm_nodes_tract': fwhm_by_tract.get(tract_name, np.nan),
                            'fwhm_nodes_pooled': fwhm_pooled,
                            'df': df_pooled,
                            'cluster_forming_t': (t_thresh if sfx != 'rft' else np.nan),
                            'cluster_forming_p': (self.cluster_forming_p if sfx != 'rft' else np.nan),
                            'alpha': self.alpha,
                            'min_clust': self.min_clust,
                            'n_permutations': (self.n_permutations if sfx != 'rft' else np.nan),
                            'p_floor': (1.0 / (self.n_permutations + 1.0)
                                        if sfx != 'rft' else np.nan),
                            'permutation_scheme': (self.perm_scheme if sfx != 'rft' else ''),
                            'permutation_family': (self.perm_family if sfx != 'rft' else ''),
                            'tfce': (self.tfce if sfx != 'rft' else ''),
                            'two_tailed': self.two_tailed,
                            'fdr_method': self.fdr_method,
                            'fdr_alpha': self.fdr_alpha,
                        })
                if rows:
                    pd.DataFrame(rows).to_csv(
                        '{0}/inference_meta-{1}.csv'.format(sdir, sfx), index=False)
            if verbose:
                print(' Wrote shell_geometry.csv + inference_meta-*.csv to {0}'.format(sdir))
        return True

    # -- convenience -------------------------------------------------------

    def run(self, clobber=False, verbose=True):
        """compute_profiles() then run_inference()."""
        return (self.compute_profiles(clobber=clobber, verbose=verbose) and
                self.run_inference(clobber=clobber, verbose=verbose))
