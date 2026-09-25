"""
make_final_tsa_avrdir_3d_html.py

All-tract version of plot_tract_tsa_avrdir.py's "avrdir glyphs by TSA" layer:
one whisker per tract voxel, centred on the voxel and aligned with the
group-average local orientation (avrdir/avrdir_<tract>), coloured by the
group-mean TSA (tsa_avr/tsa_avr_<tract>) on ONE shared colour scale, so
tracts are directly comparable. ROIs (with toggleable labels), routes
(hidden by default) and an MNI152 glass brain give context.

Voxels: tract_final_bidir_<tract> >= 0.1, the same mask that defines the
tract-average TSA predictors and the pipeline's along-tract default.
LC-TEC lateral and medial are separate entries: they overlap but their
tsa_avr/avrdir maps differ in the shared voxels.

Usage: make_final_tsa_avrdir_3d_html.py [--glyph-spacing MM] [--glyph-length MM] [out.html]
"""
import argparse
import os

import numpy as np
import nibabel as nib
import plotly.graph_objects as go

import final_3d_common as fc

OUT_HTML = os.path.join(fc.OUT_DIR, "final_tsa_avrdir_3d.html")
TRACT_THRESHOLD = 0.1
COLORSCALE = "Viridis"


def load_tract(e):
    mask_img = nib.load(fc.tract_path(e["net"], "final",
                                      "tract_final_bidir_{0}.nii.gz".format(e["tract"])))
    weight = np.squeeze(mask_img.get_fdata())
    tsa = np.squeeze(nib.load(fc.tract_path(
        e["net"], "tsa_avr", "tsa_avr_{0}.nii.gz".format(e["tract"]))).get_fdata())
    avrdir = nib.load(fc.tract_path(
        e["net"], "avrdir", "avrdir_{0}.nii.gz".format(e["tract"]))).get_fdata()
    norm = np.linalg.norm(avrdir, axis=-1)
    mask = (weight >= TRACT_THRESHOLD) & np.isfinite(tsa) & (tsa != 0) & (norm > 0)
    ijk = np.argwhere(mask)
    return dict(xyz=nib.affines.apply_affine(mask_img.affine, ijk),
                weight=weight[mask], tsa=tsa[mask],
                dir=avrdir[mask] / norm[mask][:, None])


def thin(xyz, weight, spacing):
    """Keep the highest-weight voxel per spacing-mm cell (0 = keep all)."""
    if spacing <= 0:
        return np.arange(len(xyz))
    best = {}
    for i, cell in enumerate(map(tuple, np.floor(xyz / spacing).astype(int))):
        if cell not in best or weight[i] > weight[best[cell]]:
            best[cell] = i
    return np.array(sorted(best.values()))


def whisker_coords(xyz, d, half):
    pts = np.full((len(xyz), 3, 3), np.nan)
    pts[:, 0] = xyz - half * d
    pts[:, 1] = xyz + half * d
    flat = pts.reshape(-1, 3).astype(object)
    flat[2::3] = None
    return flat[:, 0], flat[:, 1], flat[:, 2]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_html", nargs="?", default=OUT_HTML)
    ap.add_argument("--glyph-spacing", type=float, default=0.0)
    ap.add_argument("--glyph-length", type=float, default=2.0)
    ap.add_argument("--glyph-width", type=float, default=5.0)
    args = ap.parse_args()

    data = [(e, load_tract(e)) for e in fc.TRACTS]
    all_tsa = np.concatenate([d["tsa"] for _, d in data])
    cmin, cmax = np.percentile(all_tsa, [2, 98])
    print("shared TSA colour range {0:.4f}..{1:.4f} over {2} voxels"
          .format(cmin, cmax, len(all_tsa)))

    fig = go.Figure()
    fc.add_brain(fig, go, legend="legend2")

    seen = set()
    for i, (e, d) in enumerate(data):
        keep = thin(d["xyz"], d["weight"], args.glyph_spacing)
        x, y, z = whisker_coords(d["xyz"][keep], d["dir"][keep], args.glyph_length / 2)
        per_vox = np.column_stack([d["tsa"][keep], d["weight"][keep]])
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z, mode="lines",
            line=dict(color=np.repeat(d["tsa"][keep], 3), width=args.glyph_width,
                      colorscale=COLORSCALE, cmin=cmin, cmax=cmax,
                      showscale=(i == 0),
                      colorbar=dict(title=dict(text="Mean TSA", side="right",
                                               font=dict(family=fc.FONT_STACK,
                                                         size=fc.AXIS_TITLE_SIZE)),
                                    tickfont=dict(family=fc.FONT_STACK,
                                                  size=fc.TICK_SIZE),
                                    x=0.0, xanchor="left", y=0.5, len=0.5,
                                    thickness=16)),
            customdata=np.repeat(per_vox, 3, axis=0),
            hovertemplate=("<b>{0}</b><br>TSA=%{{customdata[0]:.3f}}"
                           "<br>tract weight=%{{customdata[1]:.2f}}<extra></extra>")
            .format(e["group"]),
            name=e["group"], legendgroup="tsa-" + e["group"],
            showlegend=e["group"] not in seen, legend="legend"))
        seen.add(e["group"])
        print(e["tract"], len(keep), "whiskers")

    seen = set()
    for e in fc.TRACTS:
        r = fc.route_polyline(e)
        fig.add_trace(go.Scatter3d(
            x=r[:, 0], y=r[:, 1], z=r[:, 2], mode="lines",
            line=dict(color=e["color"], width=6), name=e["group"] + " route",
            legendgroup="route-" + e["group"], showlegend=e["group"] not in seen,
            visible="legendonly", hoverinfo="name", legend="legend3"))
        seen.add(e["group"])

    label_idx = fc.add_rois(fig, go, legend="legend2", opacity=0.3)

    fig.update_layout(
        scene=fc.scene_layout(),
        legend=fc.legend_box("TSA whiskers (tract &#8805; {0:g})".format(TRACT_THRESHOLD),
                             y=0.99),
        legend2=fc.legend_box("ROIs", y=0.70),
        legend3=fc.legend_box("Routes", y=0.30),
        updatemenus=[fc.label_toggle_menu(label_idx)],
        margin=dict(l=0, r=0, t=0, b=0))
    fc.write_fullscreen_html(fig, args.out_html)


if __name__ == "__main__":
    main()
