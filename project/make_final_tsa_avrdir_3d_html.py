"""
make_final_tsa_avrdir_3d_html.py

All-tract version of plot_tract_tsa_avrdir.py's "avrdir glyphs by TSA" layer:
one whisker per tract voxel, centred on the voxel and aligned with the
group-average local orientation (avrdir/avrdir_<tract>), coloured by the
group-mean TSA (tsa_avr/tsa_avr_<tract>). By default the colour range is
the 2-98th TSA percentile of the tracts currently shown (legend clicks
rescale it); a "Colour range" toggle switches to one fixed range shared by
all tracts. ROIs (with toggleable labels), routes
(hidden by default) and an MNI152 glass brain give context.

Voxels: tract_final_bidir_<tract> >= 0.1, the same mask that defines the
tract-average TSA predictors and the pipeline's along-tract default.
LC-TEC lateral and medial are separate entries: they overlap but their
tsa_avr/avrdir maps differ in the shared voxels.

Usage: make_final_tsa_avrdir_3d_html.py [--view <json|file>] [--glyph-spacing MM] [--glyph-length MM] [out.html]
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
WHISKER_META = "tsa-whisker"
RANGE_PCT = (2, 98)

# Controls box. "Whisker width" slider restyles line.width of every whisker
# trace. "Colour range" toggle: "Shown tracts" rescales cmin/cmax of every whisker
# trace to the RANGE_PCT percentiles of the TSA values of the whisker traces
# currently visible (updated on every legend click); "All tracts" restores
# the fixed shared range. The colorbar follows the first visible tract.
_AUTO_RANGE_JS = r"""
(function () {
  var gd = document.getElementById('{plot_id}');
  var META = '%(meta)s', LO = %(lo)s, HI = %(hi)s, WIDTH = %(width)s;
  var mode = 'shown', busy = false, allRange = null;
  function whiskers() {
    var idx = [];
    gd.data.forEach(function (t, i) { if (t.meta === META) idx.push(i); });
    return idx;
  }
  function pct(sorted, q) {
    var pos = (sorted.length - 1) * q / 100, lo = Math.floor(pos), hi = Math.ceil(pos);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
  }
  function rangeOf(idx) {
    var vals = [];
    idx.forEach(function (i) {
      Array.prototype.forEach.call(gd.data[i].line.color, function (v) {
        if (v !== null && isFinite(v)) vals.push(+v);
      });
    });
    if (!vals.length) return null;
    vals.sort(function (a, b) { return a - b; });
    return [pct(vals, LO), pct(vals, HI)];
  }
  function update() {
    var idx = whiskers();
    var shown = idx.filter(function (i) { return gd.data[i].visible === undefined || gd.data[i].visible === true; });
    if (!allRange) allRange = rangeOf(idx);
    var rng = (mode === 'shown' ? rangeOf(shown) : null) || allRange;
    var first = shown.length ? shown[0] : idx[0];
    busy = true;
    Plotly.restyle(gd, {'line.cmin': rng[0], 'line.cmax': rng[1],
                        'line.showscale': idx.map(function (i) { return i === first; })}, idx)
      .then(function () { busy = false; label.textContent = rng[0].toFixed(3) + ' to ' + rng[1].toFixed(3); },
            function () { busy = false; });
  }
  var box = document.createElement('div');
  box.style.cssText = 'position:fixed;left:10px;top:50px;z-index:1000;' +
    'background:rgba(255,255,255,0.95);border:1px solid #c8cdd2;border-radius:4px;padding:5px 8px;' +
    "font:13px 'Nimbus Sans',Helvetica,Arial,sans-serif;color:#111";
  box.innerHTML = '<b>Colour range</b> ' +
    '<label><input type="radio" name="cr" value="shown" checked> Shown tracts</label> ' +
    '<label><input type="radio" name="cr" value="all"> All tracts</label>' +
    '<div id="cr-label" style="color:#555;margin-top:2px"></div>' +
    '<div style="margin-top:4px"><b>Whisker width</b> ' +
    '<input id="ww" type="range" min="1" max="20" step="0.5" value="' + WIDTH + '" ' +
    'style="vertical-align:middle;width:130px"> <span id="ww-val">' + WIDTH + ' px</span></div>';
  document.body.appendChild(box);
  var label = box.querySelector('#cr-label');
  Array.prototype.forEach.call(box.querySelectorAll('input[name=cr]'), function (el) {
    el.addEventListener('change', function () { mode = el.value; update(); });
  });
  var ww = box.querySelector('#ww'), wwVal = box.querySelector('#ww-val');
  ww.addEventListener('input', function () {
    wwVal.textContent = ww.value + ' px';
    busy = true;
    Plotly.restyle(gd, {'line.width': +ww.value}, whiskers())
      .then(function () { busy = false; }, function () { busy = false; });
  });
  gd.on('plotly_restyle', function (ev) {
    if (busy || !ev || !ev[0] || !('visible' in ev[0])) return;
    update();
  });
  update();
})();
"""


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
    ap.add_argument("--view", type=fc.load_view, default=None,
                    help="camera/box JSON (or file) copied from the HTML View panel")
    args = ap.parse_args()

    data = [(e, load_tract(e)) for e in fc.TRACTS]
    all_tsa = np.concatenate([d["tsa"] for _, d in data])
    cmin, cmax = np.percentile(all_tsa, RANGE_PCT)
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
            name=e["group"], legendgroup="tsa-" + e["group"], meta=WHISKER_META,
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
    fc.write_fullscreen_html(fig, args.out_html, view=args.view,
                            extra_js=[_AUTO_RANGE_JS % dict(
                                meta=WHISKER_META, lo=RANGE_PCT[0], hi=RANGE_PCT[1],
                                width=args.glyph_width)])


if __name__ == "__main__":
    main()
