"""
Shared registry and helpers for the definitive 3D HTML figures:
  make_final_tracts_3d_html.py   routes + 0.07 shells + ROIs (all final tracts)
  make_final_tsa_avrdir_3d_html.py   mean-TSA-coloured orientation whiskers
  make_ashs_average_3d_html.py   full-screen ASHS group-average subfields

The tract registry points at the most recent tract_stats runs (LC-BA35-only,
UF-native, CB-native, Fornix-native-ca1, ForcepsMajor-native). LC-BA35 is
labelled LC-TEC everywhere: one tract/shell with two separately toggleable
v2 routes (lateral, medial). All text is Nimbus Sans, embedded in the HTML
as base64 @font-face so it renders on machines without the font installed.
"""
import base64
import os

import numpy as np
import nibabel as nib
from scipy import ndimage
from skimage import measure

PROJECT_ROOT = "/gpfs01/imgshare/ConnLS/ADNI/dwi-tracts"
TRACT_STATS = "/gpfs01/imgshare/ConnLS/ADNI/tract_stats"
ROIS_DIR = os.path.join(PROJECT_ROOT, "data", "rois", "mtl-fix")
OUT_DIR = os.path.join(PROJECT_ROOT, "project", "route_dijkstra_out")
MNI_BRAIN = "/share/ConnLS/ADNI/fsl_mod/data/standard/MNI152_T1_1mm_brain.nii.gz"
FONT_DIR = os.path.expanduser("~/.fonts")
FONT_FAMILY = "Nimbus Sans"
FONT_STACK = "'Nimbus Sans', 'Nimbus Sans L', Helvetica, Arial, sans-serif"

SHELL_THRESHOLD = 0.07

# Text sizes (px). ROI labels and legends set their own sizes explicitly.
BASE_FONT_SIZE = 18
AXIS_TITLE_SIZE = 18
TICK_SIZE = 15
BUTTON_SIZE = 15

# One entry per drawable tract (hemisphere-specific). `group` is the legend
# entry (L and R toggle together); `shell` names the avr_min_tract_counts
# pair, shared by LC-TEC lateral and medial.
TRACTS = [
    dict(group="LC-TEC lateral", net="LC-BA35-only", tract="LC_L_BA35lateral_L",
         shell="LC_L_BA35_L", shell_group="LC-TEC", color="#1f4fd6"),
    dict(group="LC-TEC lateral", net="LC-BA35-only", tract="LC_R_BA35lateral_R",
         shell="LC_R_BA35_R", shell_group="LC-TEC", color="#1f4fd6"),
    dict(group="LC-TEC medial", net="LC-BA35-only", tract="LC_L_BA35medial_L",
         shell="LC_L_BA35_L", shell_group="LC-TEC", color="#d62728"),
    dict(group="LC-TEC medial", net="LC-BA35-only", tract="LC_R_BA35medial_R",
         shell="LC_R_BA35_R", shell_group="LC-TEC", color="#d62728"),
    dict(group="Fornix", net="Fornix-native-ca1", tract="CA1_L_hypothal_lh",
         shell="CA1_L_hypothal_lh", shell_group="Fornix", color="#2e8b57"),
    dict(group="Fornix", net="Fornix-native-ca1", tract="CA1_R_hypothal_rh",
         shell="CA1_R_hypothal_rh", shell_group="Fornix", color="#2e8b57"),
    dict(group="UF", net="UF-native", tract="Ent_L_ofc_anterior_lh",
         shell="Ent_L_ofc_anterior_lh", shell_group="UF", color="#ff7f0e"),
    dict(group="UF", net="UF-native", tract="Ent_R_ofc_anterior_rh",
         shell="Ent_R_ofc_anterior_rh", shell_group="UF", color="#ff7f0e"),
    dict(group="CB", net="CB-native", tract="Ent_L_area_p24ab_pacc_lh",
         shell="Ent_L_area_p24ab_pacc_lh", shell_group="CB", color="#9467bd"),
    dict(group="CB", net="CB-native", tract="Ent_R_area_p24ab_pacc_rh",
         shell="Ent_R_area_p24ab_pacc_rh", shell_group="CB", color="#9467bd"),
    dict(group="Forceps major", net="ForcepsMajor-native",
         tract="occipital_lh_occipital_rh", shell="occipital_lh_occipital_rh",
         shell_group="Forceps major", color="#8c564b"),
]

SHELL_COLORS = {"LC-TEC": "#5a5fd6", "Fornix": "#2e8b57", "UF": "#ff7f0e",
                "CB": "#9467bd", "Forceps major": "#8c564b"}

# ROI display label -> (mask files, colour)
ROIS = [
    ("LC", ("LC_L", "LC_R"), "#202020"),
    ("TEC", ("BA35_L", "BA35_R"), "#37b4ff"),
    ("ERC", ("Ent_L", "Ent_R"), "#c9a46b"),
    ("CA1", ("CA1_L", "CA1_R"), "#e0301e"),
    ("Mammillary bodies", ("hypothal_lh", "hypothal_rh"), "#8b4513"),
    ("OFC", ("ofc_anterior_lh", "ofc_anterior_rh"), "#e377c2"),
    ("pACC", ("area_p24ab_pacc_lh", "area_p24ab_pacc_rh"), "#bcbd22"),
    ("Occipital", ("occipital_lh", "occipital_rh"), "#17becf"),
]

LIGHTING = dict(ambient=0.6, diffuse=0.7, specular=0.1)


def tract_path(net, *parts):
    return os.path.join(TRACT_STATS, net, *parts)


def parse_poly3d(path):
    with open(path) as f:
        lines = f.read().strip().split("\n")
    n_pts = int(lines[1].split()[0])
    return np.array([list(map(float, lines[2 + i].split())) for i in range(n_pts)])


def route_polyline(entry):
    return parse_poly3d(tract_path(entry["net"], "polylines",
                                   "maxes_{0}_sm3.poly3d".format(entry["tract"])))


def mesh_from_volume(V, level, affine, crop_pad=2, step_size=1):
    nz = np.argwhere(V > level)
    if len(nz) < 4:
        return None, None
    lo = np.maximum(nz.min(axis=0) - crop_pad, 0)
    hi = np.minimum(nz.max(axis=0) + crop_pad + 1, np.array(V.shape))
    sub = V[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    try:
        verts, faces, _, _ = measure.marching_cubes(sub, level, step_size=step_size)
    except (ValueError, RuntimeError):
        return None, None
    verts = nib.affines.apply_affine(affine, verts + lo[None, :])
    return verts, faces


def shell_mesh(entry, level=SHELL_THRESHOLD):
    img = nib.load(tract_path(entry["net"], "average",
                              "avr_min_tract_counts_{0}.nii.gz".format(entry["shell"])))
    return mesh_from_volume(np.squeeze(img.get_fdata()), level, img.header.get_sform())


def roi_mesh(name, smooth_fwhm=1.5, level=0.4):
    """Display-smoothed isosurface of a binary ROI mask plus its centroid (mm).
    Smoothing is applied only to this plotting copy."""
    img = nib.load(os.path.join(ROIS_DIR, name + ".nii.gz"))
    binary = np.squeeze(img.get_fdata()) > 0.5
    centroid = nib.affines.apply_affine(img.affine, np.argwhere(binary).mean(axis=0))
    V = binary.astype(np.float32)
    if smooth_fwhm > 0:
        sigma = smooth_fwhm / 2.3548 / nib.affines.voxel_sizes(img.affine)
        V = ndimage.gaussian_filter(V, sigma)
    verts, faces = mesh_from_volume(V, level, img.affine)
    return verts, faces, centroid


def brain_mesh():
    img = nib.load(MNI_BRAIN)
    V = ndimage.gaussian_filter(np.squeeze(img.get_fdata()).astype(np.float32), 1.5)
    return mesh_from_volume(V, 0.15 * float(V.max()), img.affine, step_size=3)


def mesh_trace(go, verts, faces, **kw):
    kw.setdefault("flatshading", True)
    kw.setdefault("lighting", LIGHTING)
    kw.setdefault("hoverinfo", "name")
    return go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                     i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], **kw)


def add_brain(fig, go, legend="legend"):
    verts, faces = brain_mesh()
    fig.add_trace(mesh_trace(go, verts, faces, color="#b8bec6", opacity=0.07,
                             name="MNI152 brain", hoverinfo="skip", legend=legend,
                             flatshading=False,
                             lighting=dict(ambient=0.8, diffuse=0.3, specular=0.0)))


def add_rois(fig, go, legend="legend", opacity=0.5):
    """One mesh per ROI (L+R share a legend entry) and one text trace of
    labels placed just outside each ROI. Returns the label trace index."""
    label_xyz, label_txt = [], []
    for label, names, color in ROIS:
        for hi, name in enumerate(names):
            verts, faces, centroid = roi_mesh(name)
            if verts is None:
                continue
            fig.add_trace(mesh_trace(go, verts, faces, color=color, opacity=opacity,
                                     name=label, legendgroup="roi-" + label,
                                     showlegend=(hi == 0), legend=legend))
            label_xyz.append(label_anchor(verts, centroid))
            label_txt.append(label)
    return add_label_trace(fig, go, np.array(label_xyz), label_txt, legend)


def label_anchor(verts, centroid, pad_mm=2.0):
    """Label position: above the ROI's top surface, at its centroid's x/y."""
    return np.array([centroid[0], centroid[1], verts[:, 2].max() + pad_mm])


def add_label_trace(fig, go, xyz, text, legend="legend", size=14):
    fig.add_trace(go.Scatter3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="text", text=text,
        textposition="top center",
        textfont=dict(family=FONT_STACK, size=size, color="#111111"),
        name="ROI labels", hoverinfo="skip", legend=legend, showlegend=True))
    return len(fig.data) - 1


def label_toggle_menu(label_idx, x=0.01, y=0.99):
    return dict(type="buttons", direction="right", x=x, y=y, xanchor="left",
                yanchor="top", showactive=True, active=0,
                font=dict(family=FONT_STACK, size=BUTTON_SIZE),
                buttons=[dict(label="Labels on", method="restyle",
                              args=[{"visible": True}, [label_idx]]),
                         dict(label="Labels off", method="restyle",
                              args=[{"visible": False}, [label_idx]])])


def scene_layout(**extra):
    axis = dict(title_font=dict(family=FONT_STACK, size=AXIS_TITLE_SIZE),
                tickfont=dict(family=FONT_STACK, size=TICK_SIZE),
                backgroundcolor="white", gridcolor="#e3e3e3", showbackground=False)
    scene = dict(aspectmode="data", bgcolor="white",
                 xaxis=dict(axis, title="x (mm, R+)"),
                 yaxis=dict(axis, title="y (mm, A+)"),
                 zaxis=dict(axis, title="z (mm, S+)"),
                 camera=dict(eye=dict(x=-1.6, y=0.9, z=0.6)))
    scene.update(extra)
    return scene


def legend_box(title, y, x=1.0):
    return dict(title=dict(text="<b>{0}</b>".format(title),
                           font=dict(family=FONT_STACK, size=13)),
                font=dict(family=FONT_STACK, size=12), itemsizing="constant",
                x=x, xanchor="right", y=y, yanchor="top",
                bgcolor="rgba(255,255,255,0.85)", bordercolor="#d0d0d0", borderwidth=1)


def _font_face_css():
    faces = []
    for fname, weight, style in (("NimbusSans-Regular.otf", "normal", "normal"),
                                 ("NimbusSans-Bold.otf", "bold", "normal"),
                                 ("NimbusSans-Italic.otf", "normal", "italic")):
        path = os.path.join(FONT_DIR, fname)
        if not os.path.isfile(path):
            continue
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        faces.append("@font-face{{font-family:'{0}';src:url(data:font/otf;base64,{1}) "
                     "format('opentype');font-weight:{2};font-style:{3};}}"
                     .format(FONT_FAMILY, b64, weight, style))
    return "\n".join(faces)


# Re-render once the embedded font has loaded: WebGL scene text is rasterised
# at first draw and would otherwise keep the fallback font.
_FONT_RERENDER_JS = """
var gd = document.getElementById('{plot_id}');
if (document.fonts && document.fonts.load) {
  Promise.all([document.fonts.load("12px 'Nimbus Sans'"),
               document.fonts.load("bold 12px 'Nimbus Sans'")]).then(function () {
    Plotly.react(gd, gd.data, gd.layout);
  });
}
"""


def write_fullscreen_html(fig, out_path):
    """Self-contained, full-window HTML with Nimbus Sans embedded."""
    fig.update_layout(font=dict(family=FONT_STACK, size=BASE_FONT_SIZE, color="#111111"),
                      paper_bgcolor="white", autosize=True)
    html = fig.to_html(include_plotlyjs=True, full_html=True,
                       default_width="100%", default_height="100vh",
                       config={"responsive": True, "displaylogo": False,
                               "toImageButtonOptions": dict(format="png", scale=3)},
                       post_script=_FONT_RERENDER_JS)
    style = ("<style>{0}\nhtml,body{{margin:0;padding:0;height:100%;overflow:hidden;"
             "background:#fff;font-family:{1};}}</style>").format(_font_face_css(), FONT_STACK)
    html = html.replace("<head>", "<head>\n" + style, 1)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(html)
    print("wrote {0} ({1:.1f} MB)".format(out_path, os.path.getsize(out_path) / 1e6))
