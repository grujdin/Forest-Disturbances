#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ndvi_windthrow_bitemp_v2.py

Reproduces the *original* EO Browser custom script logic (sigNDVI RGB + alpha keep mask),
ignoring SCL, with hard-coded parameters. Adds requested outputs:
  - loss_filtered_min8px_255.tif
  - loss_filtered_min20px_255.tif
  - loss_filtered_min50px_255.tif
  - loss_filtered_min50_overlay_red50_rgba_blackoutside.tif

Inputs can be:
  - EO Browser exported ZIPs containing B04/B08 "Raw" GeoTIFFs/JP2
  - Directories containing B04/B08 files

Usage (optional args; if omitted, edit PRE_DATASET/POST_DATASET below):
  python ndvi_windthrow_bitemp_v2.py --pre PRE.zip --post POST.zip --outdir out

Dependencies:
  pip install rasterio numpy
"""

from __future__ import annotations

import argparse
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.features import sieve

# -----------------------------
# HARD-CODED PARAMETERS (core)
# -----------------------------
NDVI_MIN = 0.64 #0.73 (usually), 0.64 (polygon, 2 multi-polygons)
NDVI_MAX = 1.0

S4 = 0.02  # assumed uncertainty for B04
S8 = 0.03  # assumed uncertainty for B08

PRE_OPACITY = 0.55  # used only for the "grey intermediary" blend preview

# If your bands are DN scaled 0..10000, we auto-convert to reflectance (0..1) for sigNDVI darkness.
AUTO_DN_SCALE = True
DN_SCALE = 10000.0  # Sentinel-2 L2A typical reflectance scaling

# -----------------------------
# Defaults (edit if you want)
# -----------------------------
#PRE_DATASET = r"D:/Users/JOHN/Downloads/Forest Disturbance Classification vs NRDE/Latest Picioru Calului/2017-09-11-Picioru_Calului.zip"
#POST_DATASET = r"D:/Users/JOHN/Downloads/Forest Disturbance Classification vs NRDE/Latest Picioru Calului/2017-10-01-Picioru_Calului.zip"
#OUTDIR = r"D:/Users/JOHN/Downloads/Forest Disturbance Classification vs NRDE/Latest Picioru Calului/Outputs2"

#PRE_DATASET = r"D:/Users/JOHN/Downloads/Forest Disturbance Classification vs NRDE/Latest_Stana_de_Vale/AOI_Bitemp_Extracts/2017-11-23/2017-07-31_Full.zip"
#POST_DATASET = r"D:/Users/JOHN/Downloads/Forest Disturbance Classification vs NRDE/Latest_Stana_de_Vale/AOI_Bitemp_Extracts/2017-11-23/2017-11-23_Extract.zip"
#OUTDIR = r"D:/Users/JOHN/Downloads/Forest Disturbance Classification vs NRDE/Latest_Stana_de_Vale/AOI_Bitemp_Extracts/2017-11-23/Outputs"

#PRE_DATASET = r"D:/Forest_Disturbance/imagery_zip/Comandau_BV/Comandau_2020_07_08.zip"
#POST_DATASET = r"D:/Forest_Disturbance/imagery_zip/Comandau_BV/Comandau_2021_08_08.zip"
#OUTDIR = r"D:/Forest_Disturbance/imagery_zip/Comandau_BV/Outputs"

#PRE_DATASET = r"D:/Forest_Disturbance/imagery_zip/Comandau_BV/Comandau_2018_08_14.zip"
#POST_DATASET = r"D:/Forest_Disturbance/imagery_zip/Comandau_BV/Comandau_2019_08_24.zip"
#OUTDIR = r"D:/Forest_Disturbance/imagery_zip/Comandau_BV/Outputs"

PRE_DATASET = r"D:/Forest_Disturbance/imagery_zip/Comandau_BV/Comandau_2018_08_14.zip"
POST_DATASET = r"D:/Forest_Disturbance/imagery_zip/Comandau_BV/Comandau_2019_08_24.zip"
OUTDIR = r"D:/Forest_Disturbance/imagery_zip/Comandau_BV/Outputs"

def _maybe_to_ref(arr: np.ndarray) -> np.ndarray:
    """Convert to float32 reflectance if data looks DN-scaled."""
    arrf = arr.astype(np.float32)
    if not AUTO_DN_SCALE:
        return arrf
    # Heuristic: if values routinely exceed 1.5, assume 0..10000 scaling
    vmax = np.nanpercentile(arrf, 99.9)
    if vmax > 1.5:
        arrf = arrf / DN_SCALE
    return arrf


def _read_band(path: Path) -> Tuple[np.ndarray, dict]:
    """Read first band as float32 array and return (arr, profile)."""
    with rasterio.open(path) as ds:
        arr = ds.read(1)
        profile = ds.profile.copy()
        nodata = ds.nodata
        # Build validity mask: treat nodata as invalid; also treat zeros as valid (keep original behaviour)
        if nodata is not None:
            valid = arr != nodata
        else:
            valid = np.ones(arr.shape, dtype=bool)
    return arr, profile


def _align_to(ref_profile: dict, src_path: Path) -> np.ndarray:
    """Reproject/resample src raster to match ref_profile grid."""
    with rasterio.open(src_path) as src:
        src_arr = src.read(1)
        dst = np.zeros((ref_profile["height"], ref_profile["width"]), dtype=src_arr.dtype)
        reproject(
            source=src_arr,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_profile["transform"],
            dst_crs=ref_profile["crs"],
            resampling=Resampling.nearest,
        )
    return dst


def find_band_paths(dataset_path: Path) -> Tuple[Path, Path]:
    """Return (B04_path, B08_path) for a dataset path (ZIP or directory)."""
    if dataset_path.is_file() and dataset_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(dataset_path, "r") as z:
            names = z.namelist()

            def pick(patterns):
                for n in names:
                    ln = n.lower()
                    if all(p in ln for p in patterns) and ln.endswith((".tif", ".tiff", ".jp2")):
                        return n
                return None

            b04 = pick(["b04", "raw"])
            b08 = pick(["b08", "raw"])

            if b04 is None:
                for n in names:
                    if "b04" in n.lower() and n.lower().endswith((".tif", ".tiff", ".jp2")):
                        b04 = n
                        break
            if b08 is None:
                for n in names:
                    if re.search(r"\bb08\b", n.lower()) and n.lower().endswith((".tif", ".tiff", ".jp2")):
                        b08 = n
                        break
                if b08 is None:
                    for n in names:
                        if "b08" in n.lower() and n.lower().endswith((".tif", ".tiff", ".jp2")):
                            b08 = n
                            break

            if b04 is None or b08 is None:
                raise FileNotFoundError(f"Could not find B04/B08 inside {dataset_path.name}")

        tmpdir = Path(tempfile.mkdtemp(prefix="s2zip_"))
        # Extract to fixed Windows-safe names (avoid ':' in original names)
        ext04 = Path(b04).suffix.lower()
        ext08 = Path(b08).suffix.lower()
        b04_path = tmpdir / f"B04_raw{ext04}"
        b08_path = tmpdir / f"B08_raw{ext08}"

        with zipfile.ZipFile(dataset_path, "r") as z:
            b04_path.write_bytes(z.read(b04))
            b08_path.write_bytes(z.read(b08))

        return b04_path, b08_path

    if dataset_path.is_dir():
        files = list(dataset_path.rglob("*"))

        def pick_dir(tag: str) -> Optional[Path]:
            cand = [
                p for p in files
                if p.is_file()
                and tag in p.name.lower()
                and p.suffix.lower() in [".tif", ".tiff", ".jp2"]
            ]
            if not cand:
                return None
            raw = [p for p in cand if "raw" in p.name.lower()]
            return raw[0] if raw else cand[0]

        b04 = pick_dir("b04")
        b08 = pick_dir("b08")
        if b04 is None or b08 is None:
            raise FileNotFoundError(f"Could not find B04/B08 in folder {dataset_path}")
        return b04, b08

    raise FileNotFoundError(f"Dataset path not found: {dataset_path}")


def clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def compute_signdvi_rgb_and_keep(b04: np.ndarray, b08: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implements the EO script:
      rgb = [0.9*clamp(1-ndvi)*darkness, 0.8*clamp(ndvi)*darkness, 0.1*darkness]
      keep = NDVI in [NDVI_MIN, NDVI_MAX]
    Returns:
      rgb_uint8: (H,W,3) uint8
      keep_bool: (H,W) bool
    """
    b04r = _maybe_to_ref(b04)
    b08r = _maybe_to_ref(b08)

    sum_ = b08r + b04r
    # avoid divide by zero
    sum_safe = np.where(sum_ == 0, np.nan, sum_)

    ndvi = (b08r - b04r) / sum_safe
    ndvi = np.nan_to_num(ndvi, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # s_ndvi as in script
    s_ndvi = (2.0 / (sum_safe * sum_safe)) * np.sqrt((b08r * b08r * S4 * S4) + (b04r * b04r * S8 * S8))
    s_ndvi = np.nan_to_num(s_ndvi, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    darkness = clamp01(1.0 - 2.0 * s_ndvi)

    r = 0.9 * clamp01(1.0 - ndvi) * darkness
    g = 0.8 * clamp01(ndvi) * darkness
    b = 0.1 * darkness

    rgb = np.stack([r, g, b], axis=-1)
    rgb_u8 = (clamp01(rgb) * 255.0 + 0.5).astype(np.uint8)

    keep = (ndvi >= NDVI_MIN) & (ndvi <= NDVI_MAX)

    return rgb_u8, keep


def write_geotiff(path: Path, arr: np.ndarray, profile: dict, nodata=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    prof = profile.copy()
    if arr.ndim == 2:
        prof.update(count=1)
    else:
        prof.update(count=arr.shape[0])
    prof.update(dtype=str(arr.dtype))
    if nodata is not None:
        prof.update(nodata=nodata)
    with rasterio.open(path, "w", **prof) as dst:
        if arr.ndim == 2:
            dst.write(arr, 1)
        else:
            dst.write(arr)


def filter_min_patch(mask255: np.ndarray, min_px: int, connectivity: int = 8) -> np.ndarray:
    """
    Remove small connected components from a 0/255 mask.
    Uses rasterio.features.sieve on a 0/1 mask, then rescales to 0/255.
    """
    mask01 = (mask255 > 0).astype(np.uint8)
    # sieve will also fill tiny 0-holes inside 1-regions (usually OK)
    sieved = sieve(mask01, size=min_px, connectivity=connectivity).astype(np.uint8)
    return (sieved * 255).astype(np.uint8)


def make_overlay_red50(mask255: np.ndarray) -> np.ndarray:
    """
    RGBA overlay: red at 50% where mask=255, black elsewhere.
    'blackoutside' means RGB=0 outside; alpha=0 outside.
    """
    h, w = mask255.shape
    rgba = np.zeros((4, h, w), dtype=np.uint8)
    on = mask255 > 0
    rgba[0, on] = 255  # R
    rgba[1, on] = 0
    rgba[2, on] = 0
    rgba[3, on] = 128  # 50% alpha
    return rgba


def alpha_blend_over_white(rgba_bottom: np.ndarray, rgba_top: np.ndarray, top_opacity: float = 1.0) -> np.ndarray:
    """
    Alpha blend two RGBA images (uint8) over white background.
    rgba_* expected shape: (4,H,W), values 0..255.
    Returns RGB uint8 (3,H,W).
    """
    # Convert to float 0..1
    bot_rgb = rgba_bottom[:3].astype(np.float32) / 255.0
    bot_a = rgba_bottom[3].astype(np.float32) / 255.0

    top_rgb = rgba_top[:3].astype(np.float32) / 255.0
    top_a = (rgba_top[3].astype(np.float32) / 255.0) * float(top_opacity)

    # Composite bottom over white
    white = np.ones_like(bot_rgb)
    bot_over_white = bot_rgb * bot_a + white * (1.0 - bot_a)
    bot_a_over_white = np.ones_like(bot_a)  # after composing onto white, alpha is 1 everywhere

    # Now composite top over that
    out_rgb = top_rgb * top_a + bot_over_white * (1.0 - top_a)

    out_u8 = (np.clip(out_rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return out_u8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", type=str, default=PRE_DATASET, help="Pre dataset (zip or folder)")
    ap.add_argument("--post", type=str, default=POST_DATASET, help="Post dataset (zip or folder)")
    ap.add_argument("--outdir", type=str, default=OUTDIR, help="Output directory")
    args = ap.parse_args()

    pre_path = Path(args.pre)
    post_path = Path(args.post)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pre_b04_path, pre_b08_path = find_band_paths(pre_path)
    post_b04_path, post_b08_path = find_band_paths(post_path)

    pre_b04, profile = _read_band(pre_b04_path)
    pre_b08, _ = _read_band(pre_b08_path)

    # Align post to pre grid if needed
    post_b04, post_prof = _read_band(post_b04_path)
    post_b08, _ = _read_band(post_b08_path)

    # If shapes differ, resample/reproject post bands onto pre grid
    if post_b04.shape != pre_b04.shape or post_prof.get("transform") != profile.get("transform") or post_prof.get("crs") != profile.get("crs"):
        post_b04 = _align_to(profile, post_b04_path)
        post_b08 = _align_to(profile, post_b08_path)

    # Compute RGB + keep masks (core)
    pre_rgb, keep_pre = compute_signdvi_rgb_and_keep(pre_b04, pre_b08)
    post_rgb, keep_post = compute_signdvi_rgb_and_keep(post_b04, post_b08)

    # Build RGBA (uint8)
    pre_rgba = np.zeros((4, pre_rgb.shape[0], pre_rgb.shape[1]), dtype=np.uint8)
    post_rgba = np.zeros_like(pre_rgba)
    pre_rgba[:3] = np.transpose(pre_rgb, (2, 0, 1))
    post_rgba[:3] = np.transpose(post_rgb, (2, 0, 1))
    pre_rgba[3] = (keep_pre.astype(np.uint8) * 255)
    post_rgba[3] = (keep_post.astype(np.uint8) * 255)

    # Binary masks 0/255
    keep_pre_255 = pre_rgba[3].copy()
    keep_post_255 = post_rgba[3].copy()
    holes_pre_255 = (255 - keep_pre_255).astype(np.uint8)
    holes_post_255 = (255 - keep_post_255).astype(np.uint8)

    loss_255 = ((keep_pre) & (~keep_post)).astype(np.uint8) * 255
    gain_255 = ((~keep_pre) & (keep_post)).astype(np.uint8) * 255
    stable_255 = ((keep_pre) & (keep_post)).astype(np.uint8) * 255

    # Requested filtered masks
    loss_f8_255 = filter_min_patch(loss_255, min_px=25)
    loss_f20_255 = filter_min_patch(loss_255, min_px=38)
    loss_f50_255 = filter_min_patch(loss_255, min_px=50)

    # Requested overlay for min50
    loss_f50_overlay_red50 = make_overlay_red50(loss_f50_255)

    # "Grey intermediary" blend that mimics EO Browser Compare (post opacity 1, pre opacity PRE_OPACITY)
    # We blend PRE (top) over POST (bottom) over white, using PRE_OPACITY as layer opacity.
    blend_rgb = alpha_blend_over_white(post_rgba, pre_rgba, top_opacity=PRE_OPACITY)

    # Optional: a very clear visual where LOSS pixels are painted grey on top of POST RGB
    post_with_loss_grey = np.transpose(post_rgb, (2, 0, 1)).copy()
    on = loss_f50_255 > 0
    post_with_loss_grey[0, on] = 180
    post_with_loss_grey[1, on] = 180
    post_with_loss_grey[2, on] = 180

    # ✅ Pre base (unmasked) with loss painted grey  (THIS is what you asked for)
    pre_with_loss_grey = np.transpose(pre_rgb, (2, 0, 1)).copy()
    pre_with_loss_grey[:, on] = 180

    # Keep the "clean" pre version too (white outside keep_pre)
    pre_with_loss_grey_clean = np.transpose(pre_rgb, (2, 0, 1)).copy()
    pre_with_loss_grey_clean[:, keep_pre_255 == 0] = 255
    pre_with_loss_grey_clean[:, on] = 180


    # Write outputs
    # Masks
    write_geotiff(outdir / "keep_pre_255.tif", keep_pre_255.astype(np.uint8), profile, nodata=0)
    write_geotiff(outdir / "keep_post_255.tif", keep_post_255.astype(np.uint8), profile, nodata=0)
    write_geotiff(outdir / "holes_pre_notkept_255.tif", holes_pre_255, profile, nodata=0)
    write_geotiff(outdir / "holes_post_notkept_255.tif", holes_post_255, profile, nodata=0)
    write_geotiff(outdir / "loss_pre_kept_post_notkept_255.tif", loss_255.astype(np.uint8), profile, nodata=0)
    write_geotiff(outdir / "gain_pre_notkept_post_kept_255.tif", gain_255.astype(np.uint8), profile, nodata=0)
    write_geotiff(outdir / "stable_kept_both_255.tif", stable_255.astype(np.uint8), profile, nodata=0)

    # Requested filtered masks
    write_geotiff(outdir / "loss_filtered_min25px_255.tif", loss_f8_255, profile, nodata=0)
    write_geotiff(outdir / "loss_filtered_min38px_255.tif", loss_f20_255, profile, nodata=0)
    write_geotiff(outdir / "loss_filtered_min50px_255.tif", loss_f50_255, profile, nodata=0)

    # RGBA layers
    write_geotiff(outdir / "pre_rgba_signdvi.tif", pre_rgba, profile)
    write_geotiff(outdir / "post_rgba_signdvi.tif", post_rgba, profile)
    write_geotiff(outdir / "loss_filtered_min50_overlay_red50_rgba_blackoutside.tif", loss_f50_overlay_red50, profile)

    # Visual blends
    write_geotiff(outdir / "blend_pre055_over_post1_rgb_uint8.tif", blend_rgb, profile)  # RGB as bands 1..3
    write_geotiff(outdir / "post_with_loss_grey_min50_rgb_uint8.tif", post_with_loss_grey, profile)
    write_geotiff(outdir / "pre_with_loss_grey_min50_clean_rgb_uint8.tif", pre_with_loss_grey_clean, profile)

    print("Done. Outputs written to:", outdir.resolve())


if __name__ == "__main__":
    main()
