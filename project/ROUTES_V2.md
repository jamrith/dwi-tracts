# v2 route extraction (medial / lateral LC-BA35)

Replaces the pipeline's greedy per-shell polyline (`main.py
estimate_unidirectional_tracts`) and the shell-component tracer + snap-fix
(`method_shell_components` / `truncate_routes_at_target` /
`snap_to_target_multi`) with a globally optimal, direction-aware search for
LC-BA35 tract geometry. Motivation, derivation, and figures built from live
data: `route_dijkstra_out/route_method_explainer.html`.

## Architecture

```
route_dijkstra.py            core algorithm (standalone, imports only numpy/
                              scipy/nibabel -- no dwitracts package coupling)
  extract_routes()             direction-augmented Dijkstra: nodes = (voxel,
                                incoming direction), hard 60deg/step turn cap,
                                cost = density ridge + 1/(EDT+eps) clearance +
                                turn + tiny length tie-breaker. Multiple
                                routes via a Gaussian suppression field
                                (peak +10/mm, sigma=4mm) around routes already
                                found, zeroed near seed/target ROIs. Optional
                                V_dirfield (probtrackx local-direction term),
                                V_sink (restrict termination -- see Decisions),
                                repel_routes_mm (alternating joint
                                optimization), capture (QA introspection).
  recenter_polyline()           perpendicular-slab, density-weighted-centroid
                                pull toward the shell medial axis; endpoints
                                fixed.

run_route_injection_v2.py    driver (config-driven, current)
  compute_named_routes()       extract -> smooth (pipeline-identical MA) ->
                                recenter -> restore_endpoints -> name by
                                ascending distance to the pipeline's own
                                greedy baseline polyline
  run_one()                    for one (cohort, route name): loop over every
                                pair configured for that route, inject each
                                via inject_custom_tract (ADNI-root, unchanged),
                                then ONE combined tracts config / TSA pass
                                covering all pairs together (never per-
                                hemisphere)

run_route_injection_v2_job.sh  sbatch wrapper (fsl-img + conda dwi-tracts)
```

`inject_custom_tract.py` (ADNI root, outside this repo) is unmodified --
v2 only replaces how the polyline is found; everything downstream (Gaussian
tube, bidir combine, dist_bidir, avrdir) reuses main.py's own math exactly,
so TSA/GLM cannot tell an injected tract from a native one.

## Config schema

Declared per seed/target pair in a `"polylines"` block on the dwi-tracts
tracts config JSON (both `config_tracts_lc_ba35_nonbdp_thr007.json` and
`config_tracts_lc_native_ba35_only_tatumonly_dorsal.json` have one):

```json
"polylines": {
   "threshold": 0.07,
   "route_names": ["lateral", "medial"],
   "n_routes": { "LC_L,BA35_L": 2, "LC_R,BA35_R": 2 },
   "dijkstra": {},
   "recenter": {}
}
```

- `route_names` maps names to routes by ascending distance from the
  pipeline's greedy baseline (nearest = "lateral" = re-finds the native
  corridor; farther = "medial" = the distinct second corridor). Extras
  beyond the list auto-name `route3`, `route4`, ...
- `n_routes` is per `"roi_a,roi_b"` pair; absent or 0 skips that pair (its
  native pipeline tract stands unchanged).
- `dijkstra` / `recenter` are kwarg-override dicts passed straight to
  `extract_routes` / `recenter_polyline`.
- Pseudo-ROI naming: `BA35_L` + `"medial"` -> `BA35medial_L`
  (`pseudo_roi()` in the driver; any `..._L`/`..._R` ROI works generically).

## Running

```
sbatch ... run_route_injection_v2_job.sh --cohort {nonbdp461,tatumonly} --route {lateral,medial}
sbatch ... run_glm_job.sh project/config_glm_{medial,lateral}_v2_LR.json          # nonbdp461
sbatch ... run_glm_job.sh project/config_glm_lc_ba35_tatumonly_{medial,lateral}_v2_LR.json
```

One driver run covers BOTH hemispheres for one (cohort, route) -- never
split L/R. GLM configs' `preproc_config_file` must point at a **matching**
two-hemisphere network JSON (see Pitfall below) -- generated at
`qa_diag/route_v2_configs/{cohort}/LC-BA35{route}-combined.json` +
`config_preprocess_{route}_combined.json`.

## Decisions on record

- **Sequential (not alternating) multi-route extraction.** Tested:
  alternating refinement to a fixed point moves the medial route <0.7mm and
  the fixed point is order-independent (reversed-start converges to the
  same pair, 0.000mm difference) -- so the first-mover asymmetry is real but
  small, and sequential keeps the first route pinned to the density ridge
  rather than yielding to an artificial repulsion term. `repel_routes_mm`
  is implemented and available if this is revisited
  (`run_route_alternation_test.py`).
- **Gaussian, not hard-cliff, route suppression.** `exp(-d^2/2sigma^2)`
  falloff so the second route settles at the natural centre of the
  remaining shell rather than hugging a suppression-corridor boundary.
- **No core-sink termination (decided 2026-08-16).** Routes terminate at
  the FIRST real-ROI voxel reached, which can be well short of the ROI's
  own centroid if the streamline density only enters at one edge (observed:
  LC-BA35 routes stop near the BA35/amygdala-hippocampal-head border, not
  reaching perirhinal/TEC proper -- density only overlaps ~15-20% of BA35
  voxels). A core-sink prototype (`V_sink` param, restrict termination to
  voxels near the ROI medoid) exists and works
  (`make_core_sink_compare.py` -> `route_dijkstra_out/routes_core_sink_compare.html`)
  but was explicitly NOT adopted: endpoints stay anchored to where the
  tractography data itself enters the ROI, rather than extending through
  ROI territory with no streamline support.
- **Endpoint restoration.** Pipeline smoothing + recentring can drift the
  terminal vertex up to ~3.5mm off the ROI surface even under stop-at-entry
  behaviour. `restore_endpoints()` in the driver re-anchors each processed
  route on its raw graph path's true endpoint (inside the ROI by
  construction) after smoothing/recentring.

## Pitfall already hit once

The GLM stage (`DwiTractsGlm.initialize`, `dwitracts/glm.py`) does **not**
derive its seed/target pairs from the tracts config's `roi_pairs` -- it
re-reads `preproc_config_file` -> `probtrackx.roi_list`, a *separate*
network JSON. Pointing a new combined-LR GLM config at a stale,
hemisphere-specific or old-naming network JSON silently makes it enumerate
every pairwise ROI combination (including nonsense cross-hemisphere pairs),
find zero valid tracts, and crash with a `ZeroDivisionError` deep in
`extract_distance_traces_rft1d` -- not an obvious error at the point it's
raised. Always generate a matching two-hemisphere network JSON (one
seed/single-target entry per hemisphere, correct pseudo-ROI names) alongside
any new route/cohort GLM config.

## Current vs legacy scripts

| Script | Status |
|---|---|
| `route_dijkstra.py` | current -- core algorithm |
| `run_route_injection_v2.py` (+ `_job.sh`) | current -- config-driven driver |
| `run_dorsal_injection_v2.py` (+ `_job.sh`) | superseded by `run_route_injection_v2.py` (hardcoded dorsal-only, per-hemisphere; kept for provenance) |
| `run_dorsal_injection_snapfix.py` | superseded -- pre-v2 shell-tracer + snap-to-ROI fix (461 cohort) |
| `run_dorsal_injection_tatumonly.py` | superseded -- same, tatum-only cohort |
| `run_route_dijkstra_prototype.py`, `run_route_dijkstra_v2.py` | one-off QA scripts from method development, not part of the pipeline |
| `run_route_alternation_test.py` | QA/decision-record script (first-mover analysis above), not part of the pipeline |
| `make_route_method_explainer.py` | doc generator -- regenerate after any `route_dijkstra.py`/config-schema change |
| `make_routes_3d_html.py`, `make_routes_3d_html_thresh.py`, `make_core_sink_compare.py`, `plot_routes_serial_sections.py` | QA visualization generators, standalone |

Legacy on-disk outputs (`BA35dorsal_*`-named tracts/TSA/GLMs in
`tract_stats_nonbdp_thr007` and `tract_stats_tatumonly_dorsal`, from the
snapfix and v2-dorsal runs) are superseded by the `BA35medial_*` /
`BA35lateral_*` outputs but not deleted.
