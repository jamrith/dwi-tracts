# Along-tract GLM: trace construction, shell schemes, and cluster inference

**Status:** draft for decision · **Date:** 2026-09-03

**Scope:** a *new, parallel* `dwitracts` module that does **not** modify `fit_glms`,
`extract_distance_traces`, `extract_distance_traces_rft1d`, or
`extract_distance_trace_clusters`. The legacy path stays runnable and is the
comparison baseline. This is a decision doc — it lays out the choices, what each
assumes / buys / costs, and a recommendation; it is deliberately light on final
function signatures.

Related: `project/length_proportional_shells.py` (prototype for one of the knobs
below), `project/route_dijkstra_out/dijkstra_explainer.md` §9, `PIPELINE_EXPLAINED.md`.

---

## 1. Why change anything

**Current path (`DwiTractsGlm`):** per-voxel OLS across subjects (`fit_glms`) →
collapse the voxel *t*-map to a 1-D trace by taking, per flood-fill distance
shell, the P(tract)-weighted `argmax|t|` voxel (`extract_distance_traces`,
`summary_metric="max"`) → 1-D RFT cluster inference with `STAT="T"`
(`extract_distance_traces_rft1d`).

Three weak points:

1. **The node statistic is not a *t*.** `max|t|` over a shell of spatially
   correlated voxels is an extreme-value statistic; its null is stochastically
   larger than `t_df` and not *t*-shaped. `rft1d STAT="T"` (critical height,
   resel counts) assumes each node is a central *t*. The FWHM is also estimated
   from the *max-selected* residuals, and `df = N_sub - 1` is too high — it
   should be `N_sub - rank(X)` (e.g. `N_sub - 3` for `Intercept + AGE + FEMALE`,
   less after per-node outlier trimming). Both push mildly anticonservative.

2. **Flood-fill shells have wildly unequal populations.** Measured on
   LC_L→BA35_L (nonbdp461): 37 shells, voxels/shell min 7, max 2002, mean 518
   (CV 0.56). A 2000-voxel shell yields an extreme `max` by chance far more
   readily than a 7-voxel one, independent of any real effect. The legacy AGE
   trace shows a spurious `t ≈ +2.6` sign flip exactly at the oversized BA35
   end that no length-proportional binning reproduces.

3. **The permutation path that once existed was invalid.**
   `extract_distance_trace_clusters` shuffled voxel→shell labels with the
   *t*-map frozen — a spatial-arrangement null that ignores field smoothness and
   is anticonservative — plus a hard-coded `N_perm = 500` and a stale-variable
   FDR bug. It is **not** a template for the permutation option below.

---

## 2. The parallel module

A new in-package module (working name `dwitracts/profiles.py`, class
`DwiTractsProfiles`), mirroring `DwiTractsGlm`'s config / `initialize()`
conventions, driven by a new `run_profiles_*.py`. It:

- reads the **same inputs** `DwiTractsGlm` does — tracts config, per-subject
  metric volumes `{metric_stem}_mni_sm_*_{tract}.nii.gz` (TSA betas or
  FA/MD/RD/AD), covariates, `dist_bidir_*`, `tract_final_(norm_)bidir_*`,
  `maxes_*_sm3.poly3d`;
- writes under its **own** `output_dir` value (e.g.
  `profiles_arclen_metricfirst_wmean`, named for the knobs -- see 13.1) so it can
  never overwrite a legacy `glms_*` tree;
- emits the **existing downstream contract unchanged** —
  `summary-{metric}_thr{NN}/stats_{tract}.csv`
  (`Distance`, `{factor}|tval|pval|fdr_pval|coef|coef_{scheme}`),
  `resids_{tract}.csv` (`K × N_sub`), `pval-*_thr{NN}/` + `tval-*_thr{NN}/`
  volumes, `tvals_*_{tract}_{factor}_{NN}.poly3d`, `tcounts-*.csv` — so
  `plot_fornix_ba35_results_v4.py`, `create_pajek_graphs`, and `aggregate_stats`
  work against it with at most a directory-name change;
- carries the `effect_scales` coef variants (`coef`, `coef_std`,
  `coef_std_binary`) through, so the effect-size figures keep working;
- sets **`df = N_sub - rank(X)`** (per node, after outlier trimming) — fixed
  here regardless of the other knobs;
- reports **all non-intercept factors simultaneously** (same as the current
  code), each adjusted for the others;
- in metric-first mode, **does not write per-voxel stat NIfTIs** — 1-D traces
  only (decision taken).

Polyline placement (`utils.resample_polyline_by_distance`, proportional arc
length) is orthogonal and already settled on `native-rois`. With
length-proportional shells the shell centres are already in arc-length mm, so
placement becomes exact rather than a rescaling.

---

## 3. Knob 1 — shell scheme

| | **Flood-fill** (legacy) | **Length-proportional** (`length_proportional_shells.py`) |
|---|---|---|
| Definition | BFS from seed through the Gaussian tube; shell = hop count | fine uniform resample of the centreline (0.5 mm); each tube voxel → nearest arc-length position (cKDTree); bin into fixed-width slabs |
| Shell count *K* | emergent from flood-fill topology | `K = round(L / width)` — falls out of tract length |
| Shell width | variable, unphysical | fixed physical mm (**2.0–2.5 recommended**; sweep in the module docstring: < 2 mm risks empty shells, ≥ 2 mm keeps min population ≳ 10) |
| Population balance | poor (CV ≈ 0.5+) | good by construction |
| Cross-tract comparability | none (*K* and width differ arbitrarily) | shell = same physical distance in every tract |
| Interpretability | "shell 17 of 37" | "18 mm along the tract" |
| Interaction with `max` | large shells inflate `max` → chance extremes / sign flips | removed |
| Prototyped? | yes (legacy) | yes — for voxelwise-collapse + RFT (`length_proportional_shells.py` `__main__`) |

**RFT note** (from the prototype's own corrected reasoning): shells *narrower*
than `fwhm_axial` are **not** a validity problem — the RFT step estimates the
trace's empirical smoothness from residuals and corrects cluster inference for
whatever correlation actually exists. The only hard constraint is non-empty
shells.

---

## 4. Knob 2 — trace construction

| | **Voxelwise → collapse** (legacy) | **Metric-first → node-wise GLM** |
|---|---|---|
| Step order | per-voxel OLS → pick per-shell `argmax\|t\|` (or mean/median) | per subject: aggregate metric within shell → per-shell OLS across subjects |
| Per-subject shell value | n/a (collapse is on the *t*-map) | `wmean` (P(tract)^w weights, reuses `max_weight_factor`), `mean`, `median`, or `trimmed` |
| Node statistic | `max\|t\|` over correlated voxels — **not** *t*-distributed | genuine per-node OLS *t* from a well-specified model |
| RFT `STAT="T"` valid? | no (motivates permutation / nonparam here) | yes (modulo residual normality + smoothness) |
| Circularity | shell location chosen using the test statistic | none — aggregation is in metric space, before inference |
| Focal effects | `max` preserves them; `mean` erases them (verified on LC_L→BA35_L: mean → 0 sig clusters, max → consistent AGE/AD/MCI clusters) | shell-averaging dilutes a few-voxel effect; mitigate with `wmean` + optional stricter `profile_core_threshold`; `max` aggregation offered but reintroduces the peak-pick problem (default off, warned) |
| df | `N_sub - 1` (bug) | `N_sub - rank(X)`, per node |
| Voxel stat maps | written | **not** written (decision) |
| Cost | one voxelwise fit + collapse | aggregate once/subject + *K* small fits — cheaper |
| Node-axis heteroskedasticity | present, ignored | present (shell voxel-count varies); options: accept (usually mild), weight by √n_eff, or prefer nonparam (no marginal-variance stationarity assumption) |

---

## 5. Knob 3 — inference (any subset can run in one pass)

Each requested method writes its own suffixed outputs (`-rft`, `-np`, `-perm`).

### 5a. RFT `STAT="T"` — baseline

`utils.get_tvalue_rft1d_clusters` unchanged. Fast, analytic, smooth tail far past
permutation resolution (and `logpvals` past float64 underflow — the figures use
this). **Assumes** node = central *t*, smooth Gaussian residual field,
well-estimated stationary FWHM, good lattice approximation. **Legitimate under
metric-first construction; mis-specified under voxelwise + `max`.**

### 5b. `spm1d.stats.nonparam` (Option A)

`spm1d.stats.nonparam.regress` (continuous factor) / `ttest2` (2-level) on the
per-subject `K × N_sub` traces, then `.inference(alpha, iterations=n_permutations,
tail=…)`. Sign-flip / permutation null; **no assumption on the node statistic's
distribution**, no FWHM estimation, no df dependence.

- **Costs:** *p* floored at `1/(n_perm+1)` — no `p ≪ 1e-6` for the figures;
  small-*N* tail granularity (N = 16 → 2¹⁶ sign flips).
- **Nuisance is not exact.** With "all factors simultaneously", each factor's
  test partials the others by ter-Braak-style residualisation of `Y` and the
  regressor on the remaining columns. Approximate — document it. Anything beyond
  intercept + one predictor is really PALM territory.
- **TFCE** available via a `tfce: true` switch where the installed spm1d
  supports it, else fall back to a cluster-forming threshold.
- Adds `spm1d` to the environment (pulls `rft1d`, already present).

### 5c. Selection-aware permutation (Option B)

The **only** option that makes the voxelwise + `max` statistic's inference
*exactly* valid. Per permutation: Freedman–Lane / sign-flip the factor of
interest against the voxelwise design → refit per-voxel OLS over the tract mask
(vectorised: `B = pinv(X) @ Y_allvox`) → rebuild the trace (`argmax|t|` per
shell, chosen shell scheme) → cluster-form → record the max cluster mass across
the tract, and across all tracts for the family. Observed cluster mass vs that
null → FWER *p*.

- spm1d cannot wrap this (node location is re-selected each permutation) —
  hand-rolled loop.
- Needs per-subject per-voxel data retained/reloaded at the trace stage.
- **Compute:** `n_perm × n_tracts` trace rebuilds. Vectorised OLS makes one
  rebuild cheap; the loop is the cost. Substrate options: a SLURM array over
  permutation blocks writing partial null CSVs + a reduce step, or in-process
  `joblib` over tracts × perm-blocks. Reproducible via `permutation_seed` +
  block index.
- Effectively merges `fit_glms` + `extract_distance_traces` into a
  permutation-aware pass *inside the new module* (legacy code untouched).

---

## 6. How the knobs interact

- `shell_scheme × trace_construction` are fully independent → 4 combos.
- `trace_construction` drives which inference is *defensible*:
  - **metric-first** → RFT and nonparam both valid; permutation optional
    (tighter FWER, assumption-light).
  - **voxelwise** → RFT and nonparam both mis-specified at the node;
    **permutation (5c) is the honest choice**, or accept RFT as "standard but
    approximate".
- `shell_scheme` is orthogonal to inference (RFT self-calibrates to trace
  smoothness; permutation preserves whatever correlation exists).
- **Prototyped today:** length-proportional + voxelwise-collapse + RFT
  (`length_proportional_shells.py` `__main__`, LC_L→BA35_L).

---

## 7. Config surface (illustrative — react to this, not final)

```jsonc
"profiles": {
  "output_dir": "profiles_arclen_metricfirst_wmean", // name it for the knobs (13.1);
                                           // separate tree, never collides with glms_*
  "shell_scheme": "length_proportional",  // "flood_fill" | "length_proportional"
  "shell_width_mm": 2.0,                   // length_proportional only
  "resample_step_mm": 0.5,
  "trace_construction": "metric_first",    // "voxelwise" | "metric_first"
  "profile_aggregation": "wmean",          // metric_first: "mean"|"wmean"|"median"|"trimmed" ("max" allowed, warned)
  "profile_core_threshold": null,          // optional stricter tract-prob cutoff for aggregation
  "max_weight_factor": 1.0,
  "inference_method": ["rft", "nonparam"], // any subset of rft | nonparam | permutation
  "n_permutations": 5000,
  "two_tailed": true,
  "tfce": false,
  "permutation_scheme": "freedman_lane",   // permutation: "freedman_lane" | "sign_flip"
  "permutation_seed": 0,
  "min_clust": 3,
  "pval_alpha": 0.05,
  "fdr_method": "fdr_tsbky",
  "fdr_alpha": 0.05
}
```

---

## 8. Comparison / validation plan

Reference cases: the 16-subject `config_glm_cb_native_sample.json` (Ent↔pACC),
and the prototype's LC_L→BA35_L nonbdp461 lateral-route GLM (legacy numbers
already in `dijkstra_explainer.md` §9).

1. **Legacy baseline** — run the untouched `DwiTractsGlm` path; archive
   `stats_*.csv`, cluster calls, FWHM.
2. **Knob sweep** — for each
   `{flood_fill, length_proportional} × {voxelwise, metric_first} × {rft, nonparam, permutation}`,
   compare trace shape, per-shell *t*, estimated FWHM, significant-shell /
   cluster counts, cluster *p*-values.
3. **Known-artifact check** — confirm the legacy AGE `t ≈ +2.6` sign flip at the
   BA35 end disappears with length-proportional shells and does not reappear
   under any metric-first combo.
4. **Synthetic** — inject a Gaussian bump at a known arc-length position into
   `Y` for *k* subjects, scaled by a covariate; verify recovery by each combo;
   under pure noise, measure empirical FWER (expect ≈ α for metric-first + RFT
   and + nonparam; quantify the voxelwise + max + RFT inflation).
5. **Reproducibility** — fixed `permutation_seed`; nonparam / permutation
   *p*-values stable across reruns.

---

## 9. Recommendation

- **Default for the new path:** `metric_first` construction + `length_proportional`
  shells (width 2.0–2.5 mm) + `inference_method: ["rft", "nonparam"]` — RFT for
  effect magnitude and the sub-1e-6 tail the figures need, nonparam for the
  honest FWER call on the same profiles.
- **Selection-aware permutation (5c):** build it, but scope it as the validity
  tool for the *legacy* voxelwise + `max` path and for answering reviewers — not
  the day-to-day default.
- **Legacy `DwiTractsGlm`:** leave entirely untouched as the comparison baseline
  until the sweep in §8 justifies switching.
- **df fix** (`N_sub - rank(X)`): adopt in the new module unconditionally; note
  the discrepancy when comparing to legacy RFT.

---

## 10. Open decisions

1. Module / driver names (`DwiTractsProfiles` / `run_profiles_generic.py`?) and
   whether the `profiles` block lives in the GLM config or its own file.
2. `profile_aggregation` default — `wmean` vs `median` / `trimmed` (robustness to
   warp / segmentation outliers vs. familiarity). Settle per metric via the §8
   sweep.
3. Node-axis heteroskedasticity — accept, √n_eff weighting, or lean on nonparam.
4. Whether metric-first should optionally still emit per-voxel maps for QA
   (current decision: no) — revisit if the brain-overview figures need them.
5. Option B compute substrate — SLURM array vs in-process joblib — and
   permutation budget (resolution vs wall-clock).
6. nonparam nuisance handling — accept ter-Braak residualisation for "all
   factors simultaneously", or restrict nonparam to single-factor configs and
   send multi-covariate needs to PALM.

---

## 11. Implementation (2026-09-03)

Implemented. Nothing in `dwitracts/glm.py` was touched.

| File | Role |
|---|---|
| `dwitracts/profiles.py` | `DwiTractsProfiles` + module-level numerics (`ols_fit`, `tvals_for_column`, `cluster_masses`, `tfce_1d`, shell/arc-length helpers) |
| `run_profiles_generic.py` | driver; takes the *same* config `run_glm_generic.py` takes, plus a `profiles` block. `--no-inference` / `--inference-only` |
| `project/config_profiles_cb_native_sample.json` | worked example (§8 reference case) |
| `project/test_profiles.py` | checks; `--fwer` adds the §8.4 false-positive simulation |

Two entry points: `compute_profiles()` builds the traces and writes the
summary contract; `run_inference()` runs every method in
`inference_method` over them. `run()` does both.

### 11.1 Deviations from the draft above

1. **`nonparam` is native, not `spm1d`.** `spm1d` is not installed in the
   `dwi-tracts` env, and the multi-covariate case needed ter-Braak-style
   residualisation bolted on regardless. `_nonparam_traces` implements the
   same test directly: Freedman–Lane (or sign-flip) null, max-cluster-mass
   FWER, plus a native 1-D TFCE (`tfce_1d`) for `tfce: true`. No new
   dependency. The `p >= 1/(n_perm+1)` floor and the approximate nuisance
   handling are unchanged — both are properties of the test, not of spm1d.
2. **Permutation methods report their own trace.** The null cannot honour a
   per-node outlier-trimming rule that depends on the (permuted) data, so
   the permuted statistic is an *untrimmed* refit. That trace is written as
   `{factor}|np_tval` / `{factor}|perm_tval` next to the trimmed
   `{factor}|tval`, rather than silently conflating the two.
3. **New config keys** beyond §7: `cluster_forming_p` (default 0.01,
   two-tailed) sets the cluster-forming height for the permutation methods —
   RFT derives its own from `pval_alpha`; `permutation_family`
   (`tract` | `network`) chooses whether the max-statistic null is per tract
   or pooled across the network (auto-falls back to `tract` when subject
   availability differs between tracts); `write_voxel_maps` (default false)
   re-enables per-voxel stat NIfTIs for QA (open decision 4).
4. **Polylines are namespaced by `output_dir`** —
   `polylines/{output_dir}/{rft|np|perm}/{glm}/` — because the legacy RFT
   path writes `polylines/rft/{glm}/tvals_rft_*.poly3d` with byte-identical
   filenames. Point the plotting side's `rft_dir` at the new path.
5. **`nonparam` requires `metric_first`** (`initialize()` rejects the
   combination): it needs per-subject node profiles, which the voxelwise
   construction never forms. Voxelwise gets `permutation` (5c).

### 11.2 Open decisions, as resolved in code

1. `DwiTractsProfiles` / `run_profiles_generic.py`; the `profiles` block
   lives in the GLM config (one config runs both paths).
2. `profile_aggregation` defaults to `wmean`; `median`/`trimmed` available,
   `max` allowed but warns. Settle per metric via the §8 sweep.
3. Heteroskedasticity: accepted (no √n_eff weighting). `nonparam` is the
   escape hatch — it assumes no marginal-variance stationarity.
4. Voxel maps: off, behind `write_voxel_maps`.
5. Compute substrate: in-process, vectorised (`tvals_for_column` hoists
   `pinv`; one matmul per permutation over all voxels). No SLURM array —
   see the measured cost below.
6. nonparam nuisance: Freedman–Lane for all factors simultaneously,
   documented as approximate in `_freedman_lane_parts`.

### 11.3 Validation performed

`python project/test_profiles.py --fwer` — all pass (`dwi-tracts` env):

* `ols_fit` matches `statsmodels` on coef/t/p; `df = n - rank(X)` including
  rank-deficient designs; t invariant to X/y rescaling (the property the
  `effect_scales` variants rely on).
* Cluster/TFCE primitives: masses, `min_clust`, sign changes splitting a
  cluster, TFCE extent monotonicity.
* Shell geometry: arc length, uniform resampling, exact arc-length
  sampling, non-empty length-proportional shells on a flared synthetic tube.
* §8.4 synthetic: a Gaussian bump at node 12 scaled by AGE is recovered and
  localised (±3 nodes) by both cluster and TFCE modes; a null covariate in
  the same design produces no cluster; identical seed reproduces identical
  p-values.
* §8.4 FWER: 200 pure-noise simulations, empirical FWER **0.010** at
  α = 0.05 — valid, conservative (as expected for a p = 0.01 cluster-forming
  threshold on a 25-node trace).

End-to-end on the §8 reference case (`config_glm_cb_native_sample`,
16 subjects, Ent↔pACC, `/share/ConnLS/ADNI/tract_stats/CB-native`):

* **Legacy parity.** `flood_fill` + `voxelwise` reproduces the legacy trace
  *exactly* — 106 nodes, `corr(t_legacy, t_new) = 1.0000`, `max|Δt| = 0.000`
  against `glms_sample/.../summary-max_thr50/stats_*.csv`. RFT on it gives
  p = 2.019e-4 vs the legacy 2.294e-4; the entire difference is the df fix
  (13 = 16 − rank(X) vs the legacy 15 = 16 − 1). This is the baseline the
  §8 sweep should be run from.
* **The anticonservatism is visible.** On that same voxelwise + max trace,
  RFT calls the AGE cluster at p = 2.0e-4 while the selection-aware
  permutation (5c, 200 draws) calls it at p = 0.025 — two orders of
  magnitude, exactly the direction §1.1 predicts.
* **metric_first agreement.** `length_proportional` (2.5 mm, K = 57) +
  `wmean`: RFT and `nonparam` find the *same* AGE cluster, p = 1.1e-5 (RFT)
  vs p = 0.011 (nonparam, floored by 1000 permutations) — the §5b resolution
  trade-off, as designed. FEMALE null under both.
* Cost: 2 tracts × 16 subjects — profiles + RFT + 1000-perm nonparam in 7 s;
  voxelwise + 200-perm selection-aware permutation in 17 s.

**Not yet run:** the full §8.2 knob sweep and the §8.3 known-artifact check
on LC_L→BA35_L (nonbdp461), which need the larger cohort.

### 11.4 Environment

Run under `module load conda-img && source activate dwi-tracts` (Python
3.7). The `scn` env has no working `rft1d` — the repo-root `rft1d/` source
checkout shadows the import as an empty namespace package — so RFT
inference silently yields nothing there. `_rft_traces` now prints why
instead of emitting an empty tree.

---

## 12. Full-sample run — LC↔BA35 lateral, nonbdp461 (2026-09-03)

Reference case `project/config_glm_lateral_v2_LR.json` (457 subjects with
data of 474 matched; factors AGE/FEMALE/AD/MCI; `tract_threshold` 0.5).
Two configs, four method trees, all rendered through the normal figure set:

| config | output_dir | shells | trace | inference | wall |
|---|---|---|---|---|---|
| `config_profiles_lateral_v2_LR.json` | `profiles_arclen_metricfirst_wmean` | length-prop 2.5 mm | metric-first `wmean` | rft, nonparam | 1m50 |
| `config_profiles_lateral_v2_LR_voxelwise.json` | `profiles_floodfill_voxelwise_max` | flood-fill | voxelwise `max` | rft, permutation (5000) | 5m26 |
| `config_profiles_lateral_v2_LR_max.json` | `profiles_arclen_metricfirst_max` | length-prop 2.5 mm | metric-first `max` | rft, nonparam | 1m50 |

Figures: `run_profiles_figures.py <config>` → one tree per method under
`{output_dir}/{glm}__{output_dir}_{sfx}/figures/`, plus per-method
`compare_label`s (`TSA_lateral_v2_LR_{rft,np,perm}`) landing beside the
legacy `TSA_lateral_v2_LR` in `method_comparison/`.

### 12.1 Significant nodes / best cluster p, by method

`n_sig/K`, `p` = smallest surviving cluster p.

**LC_L→BA35_L**

| method | K | AGE | FEMALE | AD | MCI |
|---|---|---|---|---|---|
| legacy flood+vox+RFT | 32 | 24/32 1.1e-08 | 11/32 6.5e-07 | 13/32 **0.0** | 14/32 4.1e-07 |
| flood+vox+RFT (new) | 32 | 23/32 8.5e-10 | 11/32 1.1e-07 | 13/32 **0.0** | 14/32 6.9e-08 |
| flood+vox+**perm** | 32 | 21/32 4.0e-04 | 9/32 1.3e-02 | 13/32 6.0e-04 | 19/32 7.0e-03 |
| lenprop+mf+RFT | 24 | 4/24 1.7e-04 | 4/24 8.8e-04 | — | 3/24 1.4e-03 |
| lenprop+mf+**nonparam** | 24 | 9/24 6.0e-04 | — | — | 9/24 1.3e-02 |

**LC_R→BA35_R**

| method | K | AGE | FEMALE | AD | MCI |
|---|---|---|---|---|---|
| legacy flood+vox+RFT | 36 | 27/36 **0.0** | 19/36 3.1e-07 | 14/36 4.2e-11 | 12/36 1.6e-07 |
| flood+vox+RFT (new) | 36 | 27/36 **0.0** | 19/36 4.3e-08 | 14/36 1.7e-12 | 12/36 2.5e-08 |
| flood+vox+**perm** | 36 | 24/36 2.0e-04 | 17/36 1.6e-03 | 14/36 6.0e-04 | 15/36 3.6e-03 |
| lenprop+mf+RFT | 28 | 12/28 4.8e-06 | 4/28 4.6e-04 | 5/28 3.0e-05 | 4/28 2.2e-03 |
| lenprop+mf+**nonparam** | 28 | 16/28 2.0e-04 | 7/28 2.8e-03 | 5/28 1.6e-03 | 4/28 7.8e-03 |

### 12.2 What the run establishes

1. **Legacy parity holds at full scale.** flood-fill + voxelwise reproduces
   the legacy trace and cluster extents (23–24/32 vs 24/32; identical 27/36,
   19/36, 14/36, 12/36 on the right). Remaining p differences are the df fix
   alone (452 = 457 − 5, vs the legacy 456 = 457 − 1).
2. **The legacy p-values are the thing that was wrong, not the extents.**
   On the *identical* trace, RFT reports p = 0.0 / 1e-8 … 1e-12 where the
   exactly-valid selection-aware permutation reports p = 6e-4 … 1.3e-2 —
   four to ten orders of magnitude. Extents barely move. This is §1.1's
   prediction confirmed on real data at n = 457: `max|t|` over a shell is
   not a central t, and RFT's analytic tail on it is fiction.
3. **Permutation p-values here are at their resolution floor.** 5000 draws
   floor p at 1/5001 = 2.0e-4; AGE hits exactly that on both tracts (and
   4.0e-4 = 2/5001 on LC_L). Read those as "p ≤ 2e-4", not as point
   estimates — raise `n_permutations` if a tighter bound is needed.
4. **§8.3 known-artifact check: passed.** The legacy AGE trace on
   LC_L→BA35_L flips to t = **+2.93** at the BA35 end immediately after a
   run of ≈ −3; the length-proportional + metric-first trace has no positive
   t anywhere in its last quartile (max +t = 0.00). The spurious flip is
   gone, exactly as §1.2 predicted.
5. **metric-first + `wmean` is markedly more conservative in EXTENT** —
   AGE 4/24 vs 24/32 on LC_L, and AD disappears entirely there. Some of that
   is the legitimate removal of the `max` inflation, but some is §4's
   documented dilution: averaging within a shell erases a spatially
   concentrated effect. This is the open decision 2 trade-off, now
   quantified. Before adopting `wmean` as the default, try
   `profile_core_threshold` (restrict aggregation to the tract core) or
   compare `median`/`trimmed`, and check whether the lost AD effect on LC_L
   is focal signal or the artifact it was always meant to remove.

### 12.3 Performance note

The permutations were never the bottleneck (~0.002 s per voxelwise refit).
The original cost was `_selection_aware_traces` re-reading every subject
volume once per FACTOR per tract (457 × 4 × 2 = 3656 NIfTI reads); volumes
are now cached across factors for the lifetime of one `run_inference()`
call. Neither config needs SLURM at this network's size.

### 12.4 Shell width, restated against the THRESHOLDED tube

§3's "2.0–2.5 mm recommended" was measured on the unthresholded tube
(`V_norm > 0`, ~19k voxels). The GLM runs on the tube at `tract_threshold`,
which is far thinner — 793 voxels for LC_L→BA35_L at 0.5. Measured there:

    w=2.0 K=30 min=  0  | w=3.0 K=20 min= 0  | w=5.0 K=12 min=52
    w=2.5 K=24 min=  0  | w=4.0 K=15 min=19  | w=6.0 K=10 min=70

LC_L→BA35_L has an empty shell at every width from 2.0 to 3.0 mm. That is
**not** a width problem to be widened away: it is a genuine gap where no
voxel survives thresholding. Such a shell is kept as a t = 0 / p = 1 node,
which falls below any cluster-forming threshold and correctly BREAKS a
cluster that would otherwise span the gap — there is no evidence there to
join the two sides with. Widening to 4 mm merely coarsens past the gap.

---

## 13. Output-tree naming, and the `max`-aggregation follow-up

### 13.1 Naming

`profiles.output_dir` now spells out the knobs rather than a version number:
`profiles_{shell_scheme}_{trace_construction}_{aggregation}` —

    profiles_arclen_metricfirst_wmean
    profiles_arclen_metricfirst_max
    profiles_floodfill_voxelwise_max

The inference method is *below* that level in every case (`tcounts-rft.csv` /
`tcounts-np.csv` / `tcounts-perm.csv`, `pval-{sfx}_thr{NN}/`, and the
`{glm}__{output_dir}_{sfx}/figures/` trees), so one tree carries every method
that is defensible for its construction. The earlier `profiles_v1` /
`profiles_vox` names were replaced wholesale; nothing refers to them.

### 13.2 `permutation` under `metric_first` is now rejected, not aliased

Selection-aware permutation earns its cost only when the trace construction
reads the GROUP statistic. `voxelwise` + `max` re-picks its argmax voxel from
the permuted t-map, so the selection must live inside the null.
`metric_first` aggregation — **including `profile_aggregation="max"`, which
takes each SUBJECT's own peak voxel on that subject's own metric** — never
touches the covariates, so re-aggregating under a permutation returns a
bit-identical profile matrix and the test degenerates to exactly `nonparam`.

`initialize()` now rejects the combination with that explanation. It was
previously worse than redundant: `_selection_aware_traces` ignores
`trace_construction` and would have rebuilt a *voxelwise* argmax|t| trace,
testing something different from the trace in its own summary CSV.

`profile_aggregation="max"` still warns, but for the right reason — not
circularity (there is none), but that a per-subject max is an order
statistic: upward-biased and skewed, so node errors are non-normal and
`rft` on it is mis-specified. `nonparam` needs only exchangeability.

### 13.3 Does `max` aggregation rescue the effects `wmean` lost?

No. `n_sig/K`, best cluster p:

**LC_L→BA35_L**

| tree [method] | K | AGE | FEMALE | AD | MCI |
|---|---|---|---|---|---|
| legacy floodfill+vox+max | 32 | 24/32 1e-08 | 11/32 6e-07 | 13/32 **0e+00** | 14/32 4e-07 |
| floodfill_voxelwise_max [perm] | 32 | 21/32 4e-04 | 9/32 1e-02 | 13/32 6e-04 | 19/32 7e-03 |
| arclen_metricfirst_wmean [np] | 24 | 9/24 6e-04 | — | **—** | 9/24 1e-02 |
| arclen_metricfirst_max [np] | 24 | 7/24 2e-04 | 3/24 6e-03 | **—** | — |

**LC_R→BA35_R**

| tree [method] | K | AGE | FEMALE | AD | MCI |
|---|---|---|---|---|---|
| legacy floodfill+vox+max | 36 | 27/36 0e+00 | 19/36 3e-07 | 14/36 4e-11 | 12/36 2e-07 |
| floodfill_voxelwise_max [perm] | 36 | 24/36 2e-04 | 17/36 2e-03 | 14/36 6e-04 | 15/36 4e-03 |
| arclen_metricfirst_wmean [np] | 28 | 16/28 2e-04 | 7/28 3e-03 | 5/28 2e-03 | 4/28 8e-03 |
| arclen_metricfirst_max [np] | 28 | 8/28 2e-04 | 5/28 8e-04 | 4/28 6e-03 | 3/28 1e-02 |

Reading:

1. **AD on LC_L is absent under BOTH metric-first aggregations.** `wmean`
   (average the shell) and `max` (each subject's own peak) bracket the shell
   from opposite ends, and neither finds it, while the voxelwise+max
   construction reports 13/32 nodes at p = 0.0. That is evidence the effect
   is a property of the argmax-over-the-group-t-map construction rather than
   of the metric — but it is not proof. `metric_first`+`max` lets each
   subject peak at a DIFFERENT anatomical voxel, which scrambles a
   spatially-consistent few-voxel effect rather than preserving it. The
   clean discriminator is `profile_core_threshold` (narrow the shell while
   keeping the location consistent across subjects); that has not been run.
2. **`max` is not uniformly more sensitive than `wmean`** — on LC_R it is
   flatly worse for AGE (8/28 vs 16/28) and MCI, and it drops MCI on LC_L
   entirely. It buys FEMALE on LC_L. There is no case here for making it the
   default.
3. **The `rft`/`nonparam` divergence on the `max` tree is the
   mis-specification, visible.** AGE on LC_R: `rft` p = 1e-13 against
   `nonparam` p = 2e-4 (its floor). RFT is not entitled to that tail on a
   per-subject-max node statistic. The `nonparam` value is a bound, not a
   point estimate, so the true gap cannot be quantified at 5000 draws.

---

## 14. The full 2×2 knob grid — attributing the change

§12 and §13 compared only the DIAGONAL of the `shell_scheme` ×
`trace_construction` grid (flood-fill+voxelwise vs arc-length+metric-first),
which confounds the two knobs: any difference could be either. §6 says the
knobs are independent, so both off-diagonal cells were run to break the
confound:

    profiles_arclen_voxelwise_max          (config_..._arclen_vox.json)      5m19
    profiles_floodfill_metricfirst_wmean   (config_..._floodfill_mf.json)    1m50

`n_sig/K`, best cluster p, **valid inference only** — `permutation` for the
voxelwise cells, `nonparam` for the metric-first cells:

**LC_L→BA35_L**

| cell | K | AGE | FEMALE | AD | MCI |
|---|---|---|---|---|---|
| flood-fill · voxelwise max | 32 | 21/32 4e-04 | 9/32 1e-02 | 13/32 6e-04 | 19/32 7e-03 |
| arc-length · voxelwise max | 24 | 13/24 2e-04 | 9/24 9e-03 | 5/24 6e-03 | 12/24 7e-03 |
| flood-fill · metric-first | 32 | 9/32 4e-03 | — | 4/32 2e-02 | 14/32 9e-03 |
| arc-length · metric-first | 24 | 9/24 6e-04 | — | — | 9/24 1e-02 |

**LC_R→BA35_R**

| cell | K | AGE | FEMALE | AD | MCI |
|---|---|---|---|---|---|
| flood-fill · voxelwise max | 36 | 24/36 2e-04 | 17/36 2e-03 | 14/36 6e-04 | 15/36 4e-03 |
| arc-length · voxelwise max | 28 | 23/28 2e-04 | 15/28 2e-03 | 7/28 1e-03 | 7/28 4e-03 |
| flood-fill · metric-first | 36 | 20/36 2e-04 | 10/36 8e-04 | 6/36 2e-03 | 8/36 9e-03 |
| arc-length · metric-first | 28 | 16/28 2e-04 | 7/28 3e-03 | 5/28 2e-03 | 4/28 8e-03 |

### 14.1 Attribution

**Both knobs shrink the effects independently, and they compound.** Taking
AD on LC_L (13/32 in the legacy corner): changing only the shells gives
5/24; changing only the construction gives 4/32; changing both gives
nothing. Neither knob alone explains it — so the earlier §13.3 reading
("absent under both metric-first aggregations") was true but attributed too
much to the construction. The shell scheme is doing comparable work.

Per factor on LC_L:

- **AD** — both knobs cut it hard (13 → 5 shells-only, → 4 construction-only,
  → 0 together). Consistent with a substantial artifact component in the
  legacy corner, but it does not vanish until both are changed.
- **FEMALE** — purely a construction effect: 9/32 → 9/24 under the shell
  change alone (untouched), gone under either metric-first cell.
- **AGE, MCI** — shrink under both knobs but survive everywhere.

**LC_R is robust**: every factor survives in all four cells, with extents
declining gently. Whatever is fragile about the left tract is not a general
property of the pipeline.

### 14.2 Caveat on comparing extents across shell schemes

`n_sig/K` is not directly comparable between schemes: K differs (32/36 vs
24/28) and, as §12/§13 note, a flood-fill "shell" spans a median 5.0 mm of
arc length against a 2.5 mm arc-length slab, so the two count different
things. Cluster p-values are comparable; node counts should be read as
fractions, and even then only within a scheme.

---

## 15. Matched resolution, unweighted mean, and metadata export (2026-09-03)

### 15.1 New exports — provenance travels with the figures

`DwiTractsProfiles.export_metadata()` writes two CSVs into every summary dir,
and `run_profiles_figures.py` copies them into each method's `figures/` dir so
a PDF taken out of the tree still records how it was made:

* **`shell_geometry.csv`** — per tract: scheme, construction, aggregation, K,
  shell width, resample step, arc length, n_voxels, voxels/shell
  min/max/mean/median, **n_empty_shells**, tract_threshold, n_subjects, df
  median/min/max, per-tract FWHM.
* **`inference_meta-{rft,np,perm}.csv`** — per (factor, tract): n_sig_nodes,
  n_clusters, best p and log-p, **per-tract and pooled FWHM**, df,
  cluster-forming t, alpha, min_clust, n_permutations, **p_floor**,
  permutation scheme/family, TFCE, two-tailed, FDR settings.

It recomputes everything from `cache/*.npz`, `resids_*.csv` and `stats_*.csv`
rather than from live fit state, so `run_profiles_generic.py <cfg>
--export-only` backfills an existing tree in seconds without repeating a
single permutation. It is also called automatically at the end of
`run_inference()`.

Immediately useful: the pooled FWHM the RFT stage uses is a NETWORK mean, but
the per-tract values differ materially (flood-fill+voxelwise: 3.254 for LC_L,
3.639 for LC_R, pooled 3.446). The left tract is tested against a smoothness
estimate that is partly the right tract's. Now visible in the export.

### 15.2 Shell width matched to flood-fill

§12.4's width sweep was about avoiding empty shells; this one matches
flood-fill's *resolution* so `n_sig/K` becomes comparable across schemes
(the §14.2 caveat). Measured against flood-fill (LC_L K=32 mean 24.8;
LC_R K=36 mean 26.6):

    w=1.85mm -> LC_L K=32 mean 24.8 (exact)  | LC_R K=38 mean 25.2
    w=2.50mm -> LC_L K=24 mean 33.0          | LC_R K=28 mean 34.1

`profiles_arclenfine_*` uses **1.85 mm**, at both constructions.

### 15.3 Results — this revises §14's attribution again

AD, `n_sig/K` and best p, valid inference only:

| | LC_L | LC_R |
|---|---|---|
| flood-fill · voxelwise max | 13/32 6e-04 | 14/36 6e-04 |
| **arc-length 1.85 · voxelwise max** | **11/32 4e-03** | **7/38 2e-03** |
| arc-length 2.5 · voxelwise max | 5/24 6e-03 | 7/28 1e-03 |
| flood-fill · metric-first wmean | 4/32 2e-02 | 6/36 2e-03 |
| **arc-length 1.85 · metric-first wmean** | **—** | **6/38 3e-03** |
| arc-length 2.5 · metric-first wmean | — | 5/28 2e-03 |
| flood-fill · metric-first MEAN | 4/32 2e-02 | 6/36 2e-03 |
| arc-length 2.5 · metric-first MEAN | — | 5/28 2e-03 |
| arc-length 2.5 · metric-first max | — | 4/28 6e-03 |

1. **On LC_L, most of what §14 attributed to the SHELL SCHEME was
   RESOLUTION.** At matched resolution the scheme change barely moves AD
   (13/32 -> 11/32); it is the coarsening to 2.5 mm that takes it to 5/24.
   §14's "each knob independently cuts it by roughly two-thirds" was wrong
   about which knob: for the shell axis on LC_L the active variable is K,
   not the binning geometry.
2. **On LC_R the scheme itself does matter**: AD 14/36 -> 7/38 at matched
   resolution, and MCI 15/36 -> 7/38. The two tracts disagree about which
   knob is doing the work, which is itself a reason not to generalise from
   one tract.
3. **Trace construction remains the largest single factor** and is the one
   effect that holds across both tracts and both resolutions.

### 15.4 P(tract) weighting does essentially nothing

`mean` vs `wmean` traces are near-identical — corr 0.991-0.9996, max |Δt|
0.77 across both tracts for AGE and AD — and give the same clusters almost
everywhere (flood-fill LC_L AD: 4/32 both; arc-length 2.5 LC_L: none both).
`max_weight_factor` is not a knob worth tuning for this network, and open
decision 2 collapses: **`mean` and `wmean` are interchangeable here; `max` is
the only aggregation that behaves differently, and it behaves worse.**

Note this does NOT contradict `length_proportional_shells.py`'s finding that
"mean produced ZERO significant clusters". That prototype's `mean` averaged
the voxelwise **t-map** within a shell; `profile_aggregation="mean"` averages
the **metric** and then fits one GLM per node. Different operations — the
first destroys the effect, the second does not.

---

## 16. `profile_core_threshold` — the AD/LC_L question, settled

The open question from 13.3/14.1/15.3: is the legacy AD effect on LC_L
(13/32 nodes, p = 6e-04) a focal metric effect that shell aggregation
dilutes, or an artifact of picking the extreme voxel from the group t-map?
`profile_core_threshold` is the discriminator, because it narrows the shell
WITHOUT breaking spatial consistency across subjects (unlike
`profile_aggregation="max"`, which lets each subject peak somewhere
different).

P(tract) inside the mask runs 0.50-1.00, median ~0.70; core 0.7 keeps 51% of
voxels (13/shell), 0.8 keeps 33% (9/shell), 0.9 keeps 16% (4.8/shell).

### 16.1 Result — the effect does not come back

`n_sig/K`, best p, `nonparam` (LC_L):

| tree | fallback shells | AGE | FEMALE | AD | MCI |
|---|---|---|---|---|---|
| arclen1.85 mf wmean core=none | 0 | 12/32 6e-04 | — | **—** | 10/32 1e-02 |
| arclen1.85 mf wmean core=0.7 | 1 | 12/32 6e-04 | 4/32 7e-03 | **—** | 10/32 1e-02 |
| arclen1.85 mf wmean core=0.8 | 2 | 12/32 6e-04 | 4/32 6e-03 | **—** | 7/32 1e-02 |
| arclen1.85 mf wmean core=0.9 | 2 | 11/32 6e-04 | 4/32 6e-03 | **—** | 7/32 1e-02 |
| flood-fill mf wmean core=none | 0 | 9/32 4e-03 | — | 4/32 2e-02 | 14/32 9e-03 |
| flood-fill mf wmean core=0.7 | 0 | 9/32 5e-03 | 3/32 3e-02 | 4/32 2e-02 | 8/32 2e-02 |
| *(ref)* flood-fill VOXELWISE max | 0 | 21/32 4e-04 | 9/32 1e-02 | 13/32 6e-04 | 19/32 7e-03 |

AD is flat across every core threshold — absent on arc-length at all of
none/0.7/0.8/0.9, and pinned at exactly 4/32 p=2e-02 on flood-fill. On LC_R
it is equally flat (6/38 -> 6/38 -> 6/38 -> 5/38).

### 16.2 Closing the periphery loophole

Core thresholding would remove a focal effect that lived in the tract
PERIPHERY rather than the core. It does not, because the voxels the legacy
method selects are core voxels:

    LC_L AD argmax voxels: P(tract) median 0.86 (p25 0.73, p75 0.92)
                           88% have P >= 0.7   [mask-wide: median 0.70, 51%]
    LC_R AD argmax voxels: P(tract) median 0.93, 86% have P >= 0.7

So `core=0.7` RETAINS 88% of the very voxels the voxelwise+max trace picked,
and still finds nothing on LC_L. (Caveat: the argmax is itself P(tract)-
weighted at `max_weight_factor=1`, so it is biased toward high-P voxels by
construction — this shows the core-restricted aggregation keeps the selected
voxels, not that the effect is independently core-located.)

### 16.3 Conclusion

**Four independent aggregation schemes fail to recover it** — P-weighted mean,
unweighted mean, per-subject max, and core-restriction at three thresholds —
while the group-t argmax reports 13/32 at p = 6e-04. The LC_L AD effect as
the legacy pipeline states it is a **selection artifact**.

It is not that there is no AD effect on the left tract: flood-fill
metric-first puts it at 4/32, p = 2e-02. The legacy construction inflates a
weak, short effect into a long, highly significant one.

`n_shells_core_fallback` is now in `shell_geometry.csv`: a shell with no
core voxel silently falls back to its full membership, so a trace with many
fallbacks is not the core trace it claims to be. Counts here are 0-2 of
32-38, so these traces are genuinely core-restricted.

---

## 17. Correction to §16 — voxelwise+max IS valid under permutation

§16 concluded the LC_L AD effect was a "selection artifact". **That
conclusion was wrong**, and §16.3 should be read subject to this section.

### 17.1 The error

§16 conflated two separate claims:

1. *RFT on a voxelwise+max trace is mis-specified* — TRUE, unchanged. The
   node is an extreme-value statistic, not a central t, so the legacy
   p = 0.0 / 1e-12 figures are fiction (§12.2).
2. *Therefore the effect is not real* — DOES NOT FOLLOW.

The selection-aware permutation (5c) needs **only exchangeability** of the
Freedman-Lane residual rows under H0. It needs nothing about the node
statistic's distribution, because the argmax selection is re-run inside every
permutation: the null trace carries exactly the inflation the observed trace
carries. So **flood-fill + voxelwise + max + `perm` is a valid test**, and its
AD p = 6e-04 on LC_L is a valid detection.

### 17.2 What the disagreement actually is

metric-first (4/32, p = 2e-02) and voxelwise+max+perm (13/32, p = 6e-04) are
BOTH valid. They differ in **what they target**, not in validity:

* voxelwise + max asks "is there a voxel-level effect anywhere in this shell",
  and is sensitive to an effect occupying a few voxels of a ~25-voxel shell.
* metric-first asks "did the shell AVERAGE move", and dilutes a focal effect
  by roughly the fraction of the shell it occupies. `profile_core_threshold`
  narrows the shell but still averages ~13 voxels at core 0.7 — enough to
  dilute a 2-3 voxel effect substantially.

So §16's four "failures to recover" are not evidence of absence; they are
what a shell-average test does to a focal effect. The knob sweep therefore
does not license "artifact" as a conclusion — the honest statement is that
the AD/LC_L effect is **focal**, detected by the focal-sensitive test and
diluted by the averaging ones.

### 17.3 What this means for the recommendation

§9's "leave the legacy path as baseline only" is too strong. A defensible
position is: **keep flood-fill + voxelwise + max as the primary analysis,
and replace RFT with `perm` as its inference.** That keeps the legacy path's
focal sensitivity while making the p-values honest. `profiles_floodfill_
voxelwise_max` with `inference_method: ["permutation"]` is exactly that
configuration and is already run.

Two caveats that remain against it, neither fatal:

1. **Exchangeability is not free for a group contrast.** Permutation of
   residuals assumes exchangeable errors under H0. If AD subjects have
   larger residual variance than controls (plausible for a degenerating
   tract), the AD/MCI contrasts are not exactly exchangeable and the test is
   approximate. This does not affect AGE (continuous). Worth checking
   residual variance by group before relying on the AD p-value.
2. **Validity is not interpretability.** A valid test of "the per-shell
   argmax" says an effect exists somewhere in those shells; it does not say
   the selected voxels form a coherent anatomical structure. See 17.4.

### 17.4 Spatial coherence of the selected voxels

The remaining discriminator between "coherent focal fascicle" and "validly
detected but anatomically incoherent": do the argmax voxels of the
significant shells trace a continuous path through the tube, or hop across
it? Measured as the median step between consecutive selected voxels, against
a null of picking a random voxel from each of the same shells. Results in
17.5 (script: `$CLAUDE_JOB_DIR/tmp/coherence.py`).

### 17.5 The selected voxels DO form a coherent path

Voxels are **1.0 mm isotropic**, so on this grid a face-adjacent step is
1.000 mm, edge-adjacent 1.414, corner-adjacent 1.732. Median step between
the argmax voxels of consecutive significant shells:

| tract | factor | shells | median step | max jump | weighted-null median | pctile |
|---|---|---|---|---|---|---|
| LC_L | AD | 13 | **1.73** | 9.22 | 3.08 [2.17-3.99] | 0.003 |
| LC_L | AGE | 21 | **1.41** | 8.54 | 2.91 [2.24-3.46] | 0.000 |
| LC_L | MCI | 19 | **1.73** | 10.86 | 3.08 [2.34-3.91] | 0.000 |
| LC_L | FEMALE | 9 | **1.73** | 18.03 | 3.43 [2.34-4.79] | 0.003 |
| LC_R | AD | 14 | **1.73** | 8.66 | 3.00 [2.24-3.74] | 0.020 |
| LC_R | AGE | 24 | **1.73** | 10.82 | 3.00 [2.24-3.74] | 0.000 |
| LC_R | MCI | 15 | **1.41** | 11.22 | 2.72 [2.03-3.74] | 0.000 |
| LC_R | FEMALE | 17 | **1.57** | 5.83 | 2.72 [2.24-3.53] | 0.000 |

The observed median step is exactly corner- or edge-adjacency: **the argmax
voxels are literally neighbouring voxels from one shell to the next.**

The P(tract)-WEIGHTED null (sampling with probability proportional to the
same weights the argmax uses) is indistinguishable from the uniform null
(3.08 vs 3.08 for LC_L AD; 2.91 vs 3.00 for AGE), so the coherence is NOT an
artifact of the weighting pulling selection toward a spatially-clustered
core. Observed sits at percentile 0.000-0.020 under the fair null.

Caveat: the path is not perfectly unbroken — max jumps of 8-18 mm mean one or
two discontinuities per cluster. Median adjacency with an occasional break.

**So the voxelwise+max trace is tracking a contiguous sub-bundle**, not
chasing an unrelated voxel in each shell. Combined with 17.1 (the permutation
null is valid), the LC_L AD effect is best read as a real, spatially coherent,
FOCAL effect that shell-averaging dilutes -- not as an artifact.

### 17.6 Exchangeability, checked

17.3's caveat 1, measured on the metric-first residuals (n: CTL 281, MCI 146,
AD 34):

    LC_L  residual SD  CTL 0.0408  MCI 0.0409  AD 0.0348   AD/CTL 0.852
    LC_R  residual SD  CTL 0.0414  MCI 0.0389  AD 0.0338   AD/CTL 0.817
    per-node AD/CTL SD ratio: median 0.86 (LC_L) / 0.86 (LC_R)

Variance is ~15% LOWER in AD, not higher — the opposite of what was
speculated. With the SMALLER group (n=34) carrying the SMALLER variance, a
pooled-error test is **conservative**, so the AD permutation p-values are if
anything understated rather than inflated. The exchangeability caveat does
not threaten the AD result.

(Two riders: n=34 makes the AD variance estimate noisy, and reduced variance
in AD may partly reflect the effect itself, e.g. a floor in TSA.)

### 17.7 Revised recommendation

Supersedes §9 for this network:

* **Primary: flood-fill + voxelwise + max, inference = `permutation`.** Keeps
  the focal sensitivity that metric-first gives up; the permutation null makes
  the p-values honest. Already available as
  `profiles_floodfill_voxelwise_max` / `tcounts-perm.csv`.
* **Never report RFT on a voxelwise+max trace.** That is the actual defect,
  and it is confined to the inference step.
* **metric-first + `nonparam` as the corroborating analysis**, understood as a
  less focal-sensitive test rather than a more correct one. Where the two
  disagree in extent, the difference estimates how focal the effect is.
* **Length-proportional shells remain preferable for INTERPRETATION** (nodes
  in mm, comparable across tracts, no 5 mm oblique smearing per §12/§15) even
  though they are not what rescues validity.
