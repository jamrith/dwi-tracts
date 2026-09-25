"""
make_ashs_average_3d_html.py

Full-screen, single-panel version of make_ashs_3d_html.py's group-average
panel: ASHS MTL subfield probability isosurfaces (fraction of subjects with
each label at each voxel, >= GROUP_THRESHOLD) in MNI space, one toggleable
mesh per subfield (L+R together), with toggleable per-hemisphere labels.
BA35 is labelled TEC. Nimbus Sans throughout.

Usage: make_ashs_average_3d_html.py [--view <json|file>] [ashs_mni_batch_dir] [out.html]
  ashs_mni_batch_dir holds <session>_ashs_mni.nii.gz for every session in
  its ok_sessions.list (warp_ashs_to_mni_batch.py output).
"""
import os
import sys

import numpy as np
import nibabel as nib
import plotly.graph_objects as go

import final_3d_common as fc

BATCH_DIR = os.path.join(fc.PROJECT_ROOT, "project", "qa_diag", "ashs_mni_batch_461")
OUT_HTML = os.path.join(fc.OUT_DIR, "ashs_average_3d.html")
GROUP_THRESHOLD = 0.3  # fraction of subjects required for the isosurface

# label id -> (display name, hex colour), from the ashs-fastashs_beta atlas
# snap/snaplabels.txt; BA35 is displayed as TEC
LABELS = {
    1: ("CA1", "#ff0000"),
    2: ("CA2", "#00ff00"),
    3: ("DG", "#8040ff"),
    4: ("CA3", "#ffff00"),
    7: ("misc", "#e74200"),
    8: ("SUB", "#f056e0"),
    10: ("ERC", "#d2b48c"),
    11: ("TEC", "#37b4ff"),
    12: ("BA36", "#0000ff"),
    13: ("PHC", "#f38746"),
    14: ("sulcus", "#2e8b57"),
}


def build_group_probability(batch_dir, sessions):
    counts, affine, n = None, None, 0
    for sess in sessions:
        p = os.path.join(batch_dir, "{0}_ashs_mni.nii.gz".format(sess))
        if not os.path.isfile(p):
            continue
        img = nib.load(p)
        V = np.asarray(img.get_fdata(), dtype=np.int16)
        if counts is None:
            affine = img.affine
            counts = {lid: np.zeros(V.shape, dtype=np.float32) for lid in LABELS}
        for lid in LABELS:
            counts[lid] += (V == lid)
        n += 1
    return affine, {lid: c / n for lid, c in counts.items()}, n


def main():
    view = fc.pop_view_arg(sys.argv)
    batch_dir = sys.argv[1] if len(sys.argv) > 1 else BATCH_DIR
    out_html = sys.argv[2] if len(sys.argv) > 2 else OUT_HTML
    with open(os.path.join(batch_dir, "ok_sessions.list")) as f:
        sessions = [l.strip() for l in f if l.strip()]
    affine, prob, n = build_group_probability(batch_dir, sessions)
    print("group average over {0} subjects".format(n))

    fig = go.Figure()
    label_xyz, label_txt = [], []
    for label_id, (name, color) in LABELS.items():
        V = prob[label_id]
        if V.max() < GROUP_THRESHOLD:
            continue
        x_mm = nib.affines.apply_affine(affine, np.column_stack(
            [np.arange(V.shape[0]), np.zeros(V.shape[0]), np.zeros(V.shape[0])]))[:, 0]
        shown = False
        for hemi_sel in (x_mm < 0, x_mm >= 0):
            Vh = np.where(hemi_sel[:, None, None], V, 0)
            verts, faces = fc.mesh_from_volume(Vh, GROUP_THRESHOLD, affine)
            if verts is None:
                continue
            fig.add_trace(fc.mesh_trace(go, verts, faces, color=color, opacity=0.5,
                                        name=name, legendgroup=name,
                                        showlegend=not shown))
            shown = True
            centroid = nib.affines.apply_affine(
                affine, np.argwhere(Vh >= GROUP_THRESHOLD).mean(axis=0))
            label_xyz.append(fc.label_anchor(verts, centroid))
            label_txt.append(name)

    label_idx = fc.add_label_trace(fig, go, np.array(label_xyz), label_txt)
    fig.update_layout(
        scene=fc.scene_layout(camera=dict(eye=dict(x=-1.3, y=0.5, z=0.9))),
        legend=fc.legend_box("ASHS subfields (N={0}, p &#8805; {1:g})"
                             .format(n, GROUP_THRESHOLD), y=0.99),
        updatemenus=[fc.label_toggle_menu(label_idx)],
        margin=dict(l=0, r=0, t=0, b=0))
    fc.write_fullscreen_html(fig, out_html, view=view)


if __name__ == "__main__":
    main()
