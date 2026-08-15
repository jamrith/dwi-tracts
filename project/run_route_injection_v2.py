"""
run_route_injection_v2.py

v2 route injection + TSA, driven by a "polylines" block in the dwi-tracts
tracts config JSON. The block declares, per seed/target ROI pair, how many
distinct polylines to fit, what the routes are called, and (optionally) the
extraction parameters:

    "polylines": {
       "threshold": 0.07,                     # density mask threshold
       "route_names": ["lateral", "medial"],  # in order of distance from the
                                              # pipeline's greedy baseline
                                              # polyline (nearest first);
                                              # extras become "route3", ...
       "n_routes": {                          # per "roi_a,roi_b" pair
          "LC_L,BA35_L": 2,
          "LC_R,BA35_R": 2
       },
       "dijkstra": {},                        # kwarg overrides for
                                              # route_dijkstra.extract_routes
       "recenter": {}                         # kwarg overrides for
                                              # recenter_polyline
    }

For every pair, extract_routes (direction-augmented Dijkstra, clearance
cost, Gaussian route suppression) fits the requested number of routes; each
is smoothed, recentred, and injected as a pseudo-ROI tract named by
inserting the route name into the target ROI (BA35_L + "medial" ->
BA35medial_L). Route names map to routes by ascending mean distance to the
greedy baseline, so "lateral" re-finds the native corridor and "medial" is
the distinct second corridor.

Hemispheres/pairs are NEVER processed separately downstream: one run per
(cohort, route name) injects that route for ALL pairs, then a single tracts
config / regress dir / TSA pass covers them together, so GLMs and figures
combine L+R.

Pairs with n_routes < 1 (or absent) are skipped -- the pipeline's own
native tract stands for them.
"""
import os
import re
import sys
import copy
import json
import time
import argparse

import numpy as np
import nibabel as nib
from scipy.spatial import cKDTree

sys.path.insert(0, "/gpfs01/imgshare/ConnLS/ADNI/dwi-tracts")
sys.path.insert(0, "/gpfs01/imgshare/ConnLS/ADNI/dwi-tracts/project")
sys.path.insert(0, "/gpfs01/imgshare/ConnLS/ADNI")
from dwitracts import utils as dwiutils
from dwitracts.main import DwiTracts
from route_dijkstra import extract_routes, recenter_polyline
from inject_custom_tract import inject_custom_tract

PROJECT_ROOT = "/gpfs01/imgshare/ConnLS/ADNI/dwi-tracts"
DATA_ROOT = "/gpfs01/imgshare/ConnLS/ADNI"
ROIS_DIR = os.path.join(PROJECT_ROOT, "data/rois/mtl-fix")

COHORTS = {
    "nonbdp461": {
        "src_config": os.path.join(PROJECT_ROOT, "project", "config_tracts_lc_ba35_nonbdp_thr007.json"),
        "net_root": os.path.join(DATA_ROOT, "tract_stats_nonbdp_thr007"),
        "config_out_dir": os.path.join(PROJECT_ROOT, "project", "qa_diag", "route_v2_configs", "nonbdp461"),
        "regress_dir": "regress_dwi_{route}",
        "fix_share_path": True,
    },
    "tatumonly": {
        "src_config": os.path.join(PROJECT_ROOT, "project", "config_tracts_lc_native_ba35_only_tatumonly_dorsal.json"),
        "net_root": os.path.join(DATA_ROOT, "tract_stats_tatumonly_dorsal"),
        "config_out_dir": os.path.join(PROJECT_ROOT, "project", "qa_diag", "route_v2_configs", "tatumonly"),
        "regress_dir": "regress_dwi_{route}_tatumonly",
        "fix_share_path": False,
    },
}

DEFAULT_ROUTE_NAMES = ["lateral", "medial"]


def parse_poly3d(path):
    with open(path) as f:
        lines = f.read().strip().split("\n")
    n_pts = int(lines[1].split()[0])
    return np.array([list(map(float, lines[2 + i].split())) for i in range(n_pts)])


def mean_curve_dist(curve_a, curve_b):
    d_ab, _ = cKDTree(curve_b).query(curve_a)
    d_ba, _ = cKDTree(curve_a).query(curve_b)
    return 0.5 * (d_ab.mean() + d_ba.mean())


def smooth_like_pipeline(route_mm, header):
    T = np.linalg.inv(header.get_sform())
    route_vox = nib.affines.apply_affine(T, route_mm)
    route_vox = np.round(dwiutils.smooth_polyline_ma(np.round(route_vox), 3))
    return dwiutils.smooth_polyline_ma(dwiutils.voxel_to_world(route_vox, header), 7)


def restore_endpoints(proc_mm, raw_mm, tol=0.1):
    """Re-anchor the processed (smoothed + recentred) polyline on the raw
    graph path's endpoints, which lie inside the seed/target ROI by
    construction. Pipeline smoothing and recentring can drift the terminal
    vertex a couple mm off the ROI surface; this restores it without
    touching the rest of the route."""
    out = proc_mm
    if np.linalg.norm(out[0] - raw_mm[0]) > tol:
        out = np.vstack([raw_mm[0][None, :], out])
    if np.linalg.norm(out[-1] - raw_mm[-1]) > tol:
        out = np.vstack([out, raw_mm[-1][None, :]])
    return out


def pseudo_roi(roi_b, route_name):
    """BA35_L + 'medial' -> BA35medial_L; no _L/_R suffix -> plain append."""
    m = re.match(r"^(.*)_(L|R)$", roi_b)
    if m:
        return "{0}{1}_{2}".format(m.group(1), route_name, m.group(2))
    return roi_b + route_name


def route_name_list(pl_params, n):
    names = list(pl_params.get("route_names", DEFAULT_ROUTE_NAMES))
    while len(names) < n:
        names.append("route{0}".format(len(names) + 1))
    return names[:n]


def compute_named_routes(pair, net_dir, pl_params, log=print):
    """Fit n_routes polylines for one 'roi_a,roi_b' pair; return
    ({name: route_mm}, diag). Names map to routes by ascending mean
    distance to the pipeline's greedy baseline polyline."""
    roi_a, roi_b = pair.split(",")
    n = int(pl_params["n_routes"][pair])
    threshold = pl_params.get("threshold", 0.07)
    names = route_name_list(pl_params, n)

    img = nib.load(os.path.join(net_dir, "average",
                                "avr_min_tract_counts_{0}_{1}.nii.gz".format(roi_a, roi_b)))
    V_dens = np.squeeze(img.get_fdata())
    V_seed = np.squeeze(nib.load(os.path.join(ROIS_DIR, roi_a + ".nii.gz")).get_fdata())
    V_target = np.squeeze(nib.load(os.path.join(ROIS_DIR, roi_b + ".nii.gz")).get_fdata())
    baseline = parse_poly3d(os.path.join(net_dir, "polylines",
                                         "maxes_{0}_{1}_sm3.poly3d".format(roi_a, roi_b)))

    t0 = time.time()
    routes, ediags = extract_routes(V_dens, V_seed, V_target, img.header,
                                    threshold, n_routes=n, verbose=True,
                                    **pl_params.get("dijkstra", {}))
    log("    [{0}] extracted {1}/{2} routes in {3:.1f}s".format(pair, len(routes), n, time.time() - t0))
    diag = {"extract_diags": ediags}
    if len(routes) < n:
        diag["skipped_reason"] = "found {0} route(s), need {1}".format(len(routes), n)
        return None, diag

    sm = [smooth_like_pipeline(r, img.header) for r in routes]
    rc = [recenter_polyline(r, V_dens, img.header, threshold,
                            **pl_params.get("recenter", {})) for r in sm]
    # smoothing/recentring can drift the terminal vertex off the ROI surface
    # (observed up to ~3.5mm); restore the raw graph path's true endpoints,
    # which lie inside seed/target by construction
    rc = [restore_endpoints(r, raw) for r, raw in zip(rc, routes)]

    fits = [mean_curve_dist(r, baseline) for r in rc]
    order = np.argsort(fits)  # nearest to greedy baseline first
    named = {names[j]: rc[int(order[j])] for j in range(n)}
    diag["fit_to_baseline_mm"] = {names[j]: float(fits[int(order[j])]) for j in range(n)}
    if n >= 2:
        diag["separation_mm"] = float(mean_curve_dist(named[names[0]], named[names[1]]))
    return named, diag


def make_tracts_config(cohort, route, pairs):
    """One tracts config covering ALL pairs' pseudo tracts for this route."""
    C = COHORTS[cohort]
    with open(C["src_config"]) as f:
        params = json.load(f)

    os.makedirs(C["config_out_dir"], exist_ok=True)
    rois_file = os.path.join(C["config_out_dir"], "route_{0}_all_pairs.rois".format(route))
    with open(rois_file, "w") as f:
        for pair in pairs:
            roi_a, roi_b = pair.split(",")
            f.write("{0}\n{1}\n".format(roi_a, pseudo_roi(roi_b, route)))

    params = copy.deepcopy(params)
    gen = params["general"]
    gen["rois_list"] = rois_file
    gen["roi_pairs"] = os.path.join(PROJECT_ROOT, gen["roi_pairs"]) \
        if not os.path.isabs(gen["roi_pairs"]) else gen["roi_pairs"]
    gen["subjects_file"] = os.path.join(PROJECT_ROOT, gen["subjects_file"])
    gen["preproc_config"] = os.path.join(PROJECT_ROOT, gen["preproc_config"])
    gen["temp_dir"] = gen["temp_dir"].rstrip("/") + "_{0}_inject_LR".format(route)
    if C["fix_share_path"]:
        gen["temp_dir"] = gen["temp_dir"].replace("/share/ConnLS/ADNI", DATA_ROOT)
    gen["clobber"] = True
    gen["verbose"] = True
    gen["debug"] = False
    params["dwi_regressions"]["regress_dir"] = C["regress_dir"].format(route=route)
    params["dwi_regressions"]["force_regress"] = True

    out_config = os.path.join(C["config_out_dir"],
                              "config_tracts_{0}_{1}_LR_v2.json".format(route, cohort))
    with open(out_config, "w") as f:
        json.dump(params, f, indent=3)
    return out_config


def run_one(cohort, route, log=print):
    C = COHORTS[cohort]
    with open(C["src_config"]) as f:
        tract_params = json.load(f)
    if "polylines" not in tract_params:
        return {"status": "no_polylines_block_in_config"}
    pl_params = tract_params["polylines"]

    network_name = tract_params["general"]["network_name"]
    gauss_params = tract_params["gaussians"]
    net_dir = os.path.join(C["net_root"], network_name)

    avdir_threshold = tract_params["average_directions"]["threshold"]
    subjects_file = os.path.join(PROJECT_ROOT, tract_params["general"]["subjects_file"])
    with open(subjects_file) as f:
        subjects = [l.strip() for l in f if l.strip()]

    # pairs whose configured n_routes includes this route name
    pairs = []
    for pair, n in pl_params["n_routes"].items():
        if route in route_name_list(pl_params, int(n)):
            pairs.append(pair)
    if not pairs:
        return {"status": "route_not_configured_for_any_pair", "route": route}
    pairs = sorted(pairs)

    roi_suffix = "nii"
    if not os.path.isfile(os.path.join(ROIS_DIR, pairs[0].split(",")[0] + ".nii")):
        roi_suffix = "nii.gz"

    expected_tracts = []
    diags = {}
    for pair in pairs:
        roi_a, roi_b = pair.split(",")
        roi_b_pseudo = pseudo_roi(roi_b, route)
        tract_name = "{0}_{1}".format(roi_a, roi_b_pseudo)
        expected_tracts.append(tract_name)

        log("  [{0}] Extracting v2 routes...".format(pair))
        named, diag = compute_named_routes(pair, net_dir, pl_params, log=log)
        diags[pair] = {k: v for k, v in diag.items() if k != "extract_diags"}
        log("    diagnostics: {0}".format(diags[pair]))
        if named is None:
            log("  ** ABORT {0}: {1}".format(tract_name, diag.get("skipped_reason")))
            return {"status": "no_route", "pair": pair, "diags": diags}
        route_mm = named[route]

        os.makedirs(C["config_out_dir"], exist_ok=True)
        poly_out = os.path.join(C["config_out_dir"],
                                "route_{0}_{1}_{2}.poly3d".format(route, roi_a, roi_b))
        dwiutils.write_polyline_mgui(route_mm, poly_out,
                                     name="route_{0}_{1}_{2}".format(route, roi_a, roi_b))

        V_img = nib.load(os.path.join(ROIS_DIR, "{0}.{1}".format(roi_a, roi_suffix)))
        readonly_dist_file = os.path.join(net_dir, "dist",
                                          "tract_dist_{0}_{1}.nii.gz".format(roi_a, roi_b))
        dist_cache_dir = os.path.join(PROJECT_ROOT, "project", "qa_diag",
                                      "route_v2_dist_cache", cohort)

        log("  [{0}] Injecting {1} (avrdir over {2} subjects)...".format(
            pair, tract_name, len(subjects)))
        t0 = time.time()
        inject_custom_tract(
            polyline_world=route_mm,
            roi_a=roi_a, roi_b=roi_b_pseudo, tract_name=tract_name,
            avr_dir=os.path.join(net_dir, "average"),
            rois_dir=ROIS_DIR, roi_suffix=roi_suffix,
            dist_cache_dir=dist_cache_dir,
            output_dir=net_dir,
            V_img=V_img,
            gauss_params=gauss_params,
            seed_dilate=gauss_params["seed_dilate"],
            readonly_dist_file=readonly_dist_file,
            real_roi_b=roi_b,
            compute_avrdir=True,
            subjects=subjects,
            project_dir=DATA_ROOT,
            deriv_dir=tract_params["general"]["deriv_dir"],
            dwi_dir=tract_params["general"]["dwi_dir"],
            network_name=network_name,
            prefix=tract_params["general"]["prefix"],
            avdir_threshold=avdir_threshold,
            verbose=True,
        )
        log("  [{0}] Injection done in {1:.1f}s".format(pair, time.time() - t0))

    config_path = make_tracts_config(cohort, route, pairs)
    with open(config_path) as f:
        route_params = json.load(f)

    my_dwi = DwiTracts(route_params)
    log("  Initializing DwiTracts for combined TSA (config={0})...".format(config_path))
    if not my_dwi.initialize():
        return {"status": "init_failed", "diags": diags}

    tracts_final_bidir = getattr(my_dwi, "tracts_final_bidir", [])
    if sorted(tracts_final_bidir) != sorted(expected_tracts):
        log("  ** SAFETY ABORT: expected {0}, got {1}.".format(expected_tracts, tracts_final_bidir))
        return {"status": "safety_abort_unexpected_tracts",
                "tracts_final_bidir": tracts_final_bidir, "diags": diags}

    log("  Running compute_tsa() for {0} across {1} subjects...".format(
        expected_tracts, len(my_dwi.subjects)))
    t0 = time.time()
    tsa_ok = my_dwi.compute_tsa(verbose=True, clobber=True)
    tsa_elapsed = time.time() - t0
    log("  compute_tsa done in {0:.1f}s (ok={1})".format(tsa_elapsed, tsa_ok))

    return {"status": "tsa_done" if tsa_ok else "tsa_failed",
            "diags": diags, "tsa_elapsed_s": tsa_elapsed, "config_path": config_path}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", required=True, choices=sorted(COHORTS))
    parser.add_argument("--route", required=True,
                        help="route name from the config's polylines.route_names "
                             "(e.g. medial, lateral)")
    args = parser.parse_args()

    def log(msg):
        print("[route-v2-{0}-{1}] {2}".format(args.cohort, args.route, msg), flush=True)

    result = run_one(args.cohort, args.route, log=log)
    log("RESULT: {0}".format(json.dumps(result, default=str)))
