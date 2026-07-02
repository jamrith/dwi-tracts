"""
Create two exclusion/stop masks for the LC-limbic probtrackx network:

1. thalamus_inner_stop (lh / rh / bilateral)
   Union of all 30 Julich Thalamus-* PM nuclei thresholded at mean probability
   of the combined bilateral map, then eroded 2 mm.  Acts as a barrier to
   prevent fornix streamlines from passing *through* the thalamus.

2. temporal_sulcus_stop
   Geometric slab covering the temporal stem / STS corridor bilaterally.
   Blocks direct uncinate-type shortcuts from ENT/HIPP to ACC, forcing
   streamlines to travel via the cingulum bundle (which runs more medially
   at |X| < 20 mm and superiorly at Z > 5 mm).

All outputs are written to exclusion_masks/ in FSL MNI152 1 mm space
(182 x 218 x 182).  The Julich PMs live in Colin27/MNI152 1 mm space
(193 x 229 x 193) and are resampled via flirt -applyxfm -usesqform.

Usage
-----
  python create_stop_masks.py [--dry-run]

Dependencies: nibabel, numpy, scipy, FSL (flirt on PATH)
"""

import argparse
import os
import subprocess
import tempfile

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation, binary_fill_holes

# ── paths ─────────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
EXCL_DIR = os.path.join(HERE, "exclusion_masks")
JULICH_PM = "/gpfs01/imgshare/ConnLS/ADNI/julich-3.1/probabilistic-maps_PMs_207-areas"
FSL_MNI_REF = os.environ.get("FSLDIR", "/software/imaging/fsl/6.0.6.3") + "/data/standard/MNI152_T1_1mm.nii.gz"

# All Julich thalamus nucleus directories
THAL_NUCLEI = [
    "Thalamus-AM", "Thalamus-AV", "Thalamus-CL", "Thalamus-CM",
    "Thalamus-LD", "Thalamus-LP", "Thalamus-Li", "Thalamus-MD",
    "Thalamus-MV", "Thalamus-PUa", "Thalamus-PUi", "Thalamus-PUl",
    "Thalamus-PUm", "Thalamus-Pf", "Thalamus-Po", "Thalamus-Pv",
    "Thalamus-Rt", "Thalamus-Sg", "Thalamus-VA", "Thalamus-VAmc",
    "Thalamus-VIM", "Thalamus-VLA", "Thalamus-VLP", "Thalamus-VM",
    "Thalamus-VPL", "Thalamus-VPM", "Thalamus-VPMpc", "Thalamus-VPi",
    "Thalamus-ZI", "Thalamus-sPf",
]

# Temporal sulcus slab geometry (MNI mm coordinates)
# Covers the temporal stem / STS corridor where uncinate-type fibers run.
# Cingulum bundle runs at |X| < 20 mm and Z > 5 mm, so it is NOT blocked.
TS_X_ABS_MIN = 30   # medial boundary (distance from midline); >29 avoids OFC
TS_X_ABS_MAX = 55   # lateral boundary
TS_Y_MIN     = -15  # posterior boundary (mm)
TS_Y_MAX     = 30   # anterior boundary (mm)
TS_Z_MIN     = -25  # inferior boundary (mm)
TS_Z_MAX     =  5   # superior boundary (mm)

# Uncinate fasciculus stop: thin sagittal slab centred at X=±24 mm, running
# A→P along the lateral aspect of the anterior putamen/caudate head.
# Blocks UF fibres crossing medially from temporal lobe to OFC without
# touching any seed or target ROI.
UF_X_CTR     = 24   # centre of slab (mm from midline), applied ± each hemisphere
UF_X_HW      =  3   # half-width of slab (mm); total thickness = 2*UF_X_HW
UF_Y_MIN     = -12  # posterior boundary (mm)
UF_Y_MAX     = 22   # anterior boundary (mm)
UF_Z_MIN     = -35  # inferior boundary (mm)
UF_Z_MAX     = 25   # superior boundary (mm)


# ── helpers ───────────────────────────────────────────────────────────────────

def load_hemi_union(nuclei, hemi):
    """Max-merge all PM volumes for one hemisphere across all thalamus nuclei."""
    ref_img = None
    combined = None
    missing = []
    for nuc in nuclei:
        path = os.path.join(JULICH_PM, nuc, f"{nuc}_{hemi}_MNI152.nii.gz")
        if not os.path.exists(path):
            missing.append(path)
            continue
        img = nib.load(path)
        data = img.get_fdata(dtype=np.float32)
        if combined is None:
            ref_img = img
            combined = data
        else:
            combined = np.maximum(combined, data)
    if missing:
        print(f"  WARNING: {len(missing)} PM files not found (first: {missing[0]})")
    return combined, ref_img


def flirt_resample(src_path, ref_path, out_path):
    """Resample src to ref space using flirt -applyxfm -usesqform (identity xfm)."""
    fsl_bin = os.path.join(os.environ.get("FSLDIR", "/software/imaging/fsl/6.0.6.3"), "bin")
    cmd = [
        os.path.join(fsl_bin, "flirt"),
        "-in",  src_path,
        "-ref", ref_path,
        "-out", out_path,
        "-applyxfm", "-usesqform",
        "-interp", "nearestneighbour",
    ]
    subprocess.run(cmd, check=True)


def smooth_and_erode_mask(data, erode_iters=2, close_iters=2):
    """
    Smooth a binary mask and erode its interior:
      1. Fill internal holes
      2. Morphological closing (dilate then erode) to smooth ragged edges
      3. Final erosion to pull boundary inward
    """
    struct = np.ones((3, 3, 3), dtype=bool)
    mask = binary_fill_holes(data > 0)
    # closing: smooth surface without growing volume
    mask = binary_dilation(mask,  structure=struct, iterations=close_iters)
    mask = binary_erosion(mask,   structure=struct, iterations=close_iters)
    # inward erosion to get inner region
    mask = binary_erosion(mask,   structure=struct, iterations=erode_iters)
    return mask.astype(np.int16)


def save_nifti(data, ref_img, path):
    img = nib.Nifti1Image(data.astype(np.int16), ref_img.affine, ref_img.header)
    img.header.set_data_dtype(np.int16)
    nib.save(img, path)
    n = int((data > 0).sum())
    print(f"  Saved {os.path.basename(path)}  ({n} voxels)")


# ── mask 1: thalamus inner stop ───────────────────────────────────────────────

MPM_DIR = os.path.join(os.path.dirname(JULICH_PM), "maximum-probability-maps_MPMs_207-areas")


def thalamus_grayvalues_from_xml(hemi):
    """Parse the MPM XML and return grayvalues for all Thalamus-* structures."""
    import xml.etree.ElementTree as ET
    xml_path = os.path.join(MPM_DIR, f"JulichBrainAtlas_3.1_207areas_MPM_{hemi}_MNI152.xml")
    tree = ET.parse(xml_path)
    grayvals = []
    for elem in tree.getroot().iter("Structure"):
        if elem.text and "Thalamus" in elem.text:
            grayvals.append(int(elem.attrib["grayvalue"]))
    return grayvals


def make_thalamus_inner_stop(dry_run=False):
    print("\n=== Thalamus inner stop mask (MPM) ===")

    for hemi, label in [("lh", "lh"), ("rh", "rh")]:
        mpm_path = os.path.join(MPM_DIR, f"JulichBrainAtlas_3.1_207areas_MPM_{hemi}_MNI152.nii.gz")
        mpm_img  = nib.load(mpm_path)
        mpm_data = mpm_img.get_fdata(dtype=np.float32)

        grayvals = thalamus_grayvalues_from_xml(hemi)
        print(f"  {hemi}: {len(grayvals)} thalamus labels in MPM")

        binary = np.zeros(mpm_data.shape, dtype=bool)
        for gv in grayvals:
            binary |= (mpm_data == gv)
        print(f"  {hemi}: macro thalamus = {int(binary.sum())} voxels (193³ space)")

        if not dry_run:
            with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
                tmp_path = tmp.name
            save_nifti(binary.astype(np.int16), mpm_img, tmp_path)

            out_mni = os.path.join(EXCL_DIR, f"thalamus_inner_stop_{label}_pre.nii.gz")
            flirt_resample(tmp_path, FSL_MNI_REF, out_mni)
            os.unlink(tmp_path)

            # Smooth edges and erode in MNI space
            mni_img = nib.load(out_mni)
            eroded = smooth_and_erode_mask(mni_img.get_fdata(), erode_iters=2, close_iters=2)
            out_eroded = os.path.join(EXCL_DIR, f"thalamus_inner_stop_{label}.nii.gz")
            save_nifti(eroded, mni_img, out_eroded)
            os.unlink(out_mni)

    if not dry_run:
        # Bilateral = union of lh + rh
        lh_img = nib.load(os.path.join(EXCL_DIR, "thalamus_inner_stop_lh.nii.gz"))
        rh_img = nib.load(os.path.join(EXCL_DIR, "thalamus_inner_stop_rh.nii.gz"))
        bilateral = np.maximum(lh_img.get_fdata(), rh_img.get_fdata()).astype(np.int16)
        out_bi = os.path.join(EXCL_DIR, "thalamus_inner_stop_bilateral.nii.gz")
        save_nifti(bilateral, lh_img, out_bi)


# ── mask 2: uncinate fasciculus stop ─────────────────────────────────────────

def make_uf_stop(dry_run=False):
    """
    Thin coronal slab blocking the uncinate fasciculus hook at the inferior
    aspect of the anterior putamen / caudate head.

    MNI coordinate bounds (mm):
      X: ±[UF_X_ABS_MIN, UF_X_ABS_MAX]
      Y: [UF_Y_MIN, UF_Y_MAX]
      Z: [UF_Z_MIN, UF_Z_MAX]
    """
    print("\n=== Uncinate fasciculus stop mask ===")
    print(f"  MNI bounds: X=±{UF_X_CTR}±{UF_X_HW}mm, "
          f"Y=[{UF_Y_MIN},{UF_Y_MAX}], Z=[{UF_Z_MIN},{UF_Z_MAX}]")

    ref_img = nib.load(os.path.join(HERE, "Ent_L.nii.gz"))
    affine = ref_img.affine
    shape = ref_img.shape

    i, j, k = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    vox = np.column_stack([i.ravel(), j.ravel(), k.ravel()])
    mni = nib.affines.apply_affine(affine, vox)
    x, y, z = mni[:, 0], mni[:, 1], mni[:, 2]

    in_y = (y >= UF_Y_MIN) & (y <= UF_Y_MAX)
    in_z = (z >= UF_Z_MIN) & (z <= UF_Z_MAX)

    lh = (in_y & in_z &
          (x <= -(UF_X_CTR - UF_X_HW)) & (x >= -(UF_X_CTR + UF_X_HW))
          ).reshape(shape).astype(np.int16)

    rh = (in_y & in_z &
          (x >= (UF_X_CTR - UF_X_HW)) & (x <= (UF_X_CTR + UF_X_HW))
          ).reshape(shape).astype(np.int16)

    bilateral = np.maximum(lh, rh)

    print(f"  bilateral slab: {int(bilateral.sum())} voxels")

    if not dry_run:
        save_nifti(bilateral, ref_img, os.path.join(EXCL_DIR, "uf_stop.nii.gz"))
        save_nifti(lh,        ref_img, os.path.join(EXCL_DIR, "uf_stop_lh.nii.gz"))
        save_nifti(rh,        ref_img, os.path.join(EXCL_DIR, "uf_stop_rh.nii.gz"))


# ── mask 3: temporal sulcus stop ─────────────────────────────────────────────

def make_temporal_sulcus_stop(dry_run=False):
    """
    Geometric slab covering the temporal stem / STS corridor bilaterally.

    MNI coordinate bounds (mm):
      X: ±[TS_X_ABS_MIN, TS_X_ABS_MAX]  (lateral to midline)
      Y: [TS_Y_MIN, TS_Y_MAX]
      Z: [TS_Z_MIN, TS_Z_MAX]

    Tune these constants at the top of the script if the geometry needs
    adjustment after inspecting tractography results.
    """
    print("\n=== Temporal sulcus stop mask ===")
    print(f"  MNI bounds: |X|=[{TS_X_ABS_MIN},{TS_X_ABS_MAX}], "
          f"Y=[{TS_Y_MIN},{TS_Y_MAX}], Z=[{TS_Z_MIN},{TS_Z_MAX}]")

    # Use an existing ROI as the spatial reference (FSL MNI 182³)
    ref_img = nib.load(os.path.join(HERE, "Ent_L.nii.gz"))
    affine = ref_img.affine
    shape = ref_img.shape

    # Build voxel coordinate grid
    i, j, k = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    vox = np.column_stack([i.ravel(), j.ravel(), k.ravel()])
    mni = nib.affines.apply_affine(affine, vox)

    x, y, z = mni[:, 0], mni[:, 1], mni[:, 2]

    mask = (
        (np.abs(x) >= TS_X_ABS_MIN) & (np.abs(x) <= TS_X_ABS_MAX) &
        (y >= TS_Y_MIN) & (y <= TS_Y_MAX) &
        (z >= TS_Z_MIN) & (z <= TS_Z_MAX)
    ).reshape(shape).astype(np.int16)

    print(f"  bilateral slab: {int(mask.sum())} voxels")

    if not dry_run:
        out = os.path.join(EXCL_DIR, "temporal_sulcus_stop.nii.gz")
        save_nifti(mask, ref_img, out)

        # Left-only and right-only variants (FSL radiological: X<0 is right in image,
        # but affine diagonal[0]=-1 means MNI X<0 = left hemisphere)
        lh_mask = ((x <= -TS_X_ABS_MIN) & (x >= -TS_X_ABS_MAX) &
                   (y >= TS_Y_MIN) & (y <= TS_Y_MAX) &
                   (z >= TS_Z_MIN) & (z <= TS_Z_MAX)).reshape(shape).astype(np.int16)
        rh_mask = ((x >= TS_X_ABS_MIN) & (x <= TS_X_ABS_MAX) &
                   (y >= TS_Y_MIN) & (y <= TS_Y_MAX) &
                   (z >= TS_Z_MIN) & (z <= TS_Z_MAX)).reshape(shape).astype(np.int16)
        save_nifti(lh_mask, ref_img, os.path.join(EXCL_DIR, "temporal_sulcus_stop_lh.nii.gz"))
        save_nifti(rh_mask, ref_img, os.path.join(EXCL_DIR, "temporal_sulcus_stop_rh.nii.gz"))


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Print statistics without writing files")
    args = parser.parse_args()

    if args.dry_run:
        print("DRY RUN — no files will be written")

    os.makedirs(EXCL_DIR, exist_ok=True)
    make_thalamus_inner_stop(dry_run=args.dry_run)
    make_temporal_sulcus_stop(dry_run=args.dry_run)
    make_uf_stop(dry_run=args.dry_run)

    print("\nDone.")
