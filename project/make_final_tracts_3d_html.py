"""
make_final_tracts_3d_html.py

Definitive interactive 3D figure of every final tract (MNI space):
  - routes: the pipeline polylines (maxes_<tract>_sm3.poly3d) of the most
    recent tract_stats runs; LC-TEC has two separately toggleable v2 routes
    (lateral, medial) over one shared shell
  - shells: avr_min_tract_counts isosurfaces at the route-extraction
    threshold 0.07
  - ROIs: seed/target masks from data/rois/mtl-fix, with toggleable labels
  - MNI152 glass brain for context
Three legends (routes / shells / ROIs); L and R hemispheres share an entry.

Usage: make_final_tracts_3d_html.py [out.html]
"""
import os
import sys

import plotly.graph_objects as go

import final_3d_common as fc

OUT_HTML = os.path.join(fc.OUT_DIR, "final_tracts_3d.html")


def main():
    out_html = sys.argv[1] if len(sys.argv) > 1 else OUT_HTML
    fig = go.Figure()
    fc.add_brain(fig, go, legend="legend3")

    shells_done, shell_groups_done = set(), set()
    for e in fc.TRACTS:
        if e["shell"] in shells_done:
            continue
        shells_done.add(e["shell"])
        verts, faces = fc.shell_mesh(e)
        fig.add_trace(fc.mesh_trace(
            go, verts, faces, color=fc.SHELL_COLORS[e["shell_group"]], opacity=0.13,
            name=e["shell_group"], legendgroup="shell-" + e["shell_group"],
            showlegend=e["shell_group"] not in shell_groups_done, legend="legend2"))
        shell_groups_done.add(e["shell_group"])
        print("shell", e["shell"], len(faces), "faces")

    seen = set()
    for e in fc.TRACTS:
        r = fc.route_polyline(e)
        fig.add_trace(go.Scatter3d(
            x=r[:, 0], y=r[:, 1], z=r[:, 2], mode="lines",
            line=dict(color=e["color"], width=8), name=e["group"],
            legendgroup="route-" + e["group"], showlegend=e["group"] not in seen,
            hovertemplate="{0}<br>{1}<extra></extra>".format(e["group"], e["tract"]),
            legend="legend"))
        seen.add(e["group"])

    label_idx = fc.add_rois(fig, go, legend="legend3")

    fig.update_layout(
        scene=fc.scene_layout(),
        legend=fc.legend_box("Routes", y=0.99),
        legend2=fc.legend_box("Shells (p &#8805; {0:g})".format(fc.SHELL_THRESHOLD), y=0.66),
        legend3=fc.legend_box("ROIs", y=0.40),
        updatemenus=[fc.label_toggle_menu(label_idx)],
        margin=dict(l=0, r=0, t=0, b=0))
    fc.write_fullscreen_html(fig, out_html)


if __name__ == "__main__":
    main()
