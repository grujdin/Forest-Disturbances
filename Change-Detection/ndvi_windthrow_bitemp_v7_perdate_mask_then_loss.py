#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ndvi_windthrow_bitemp_v7_perdate_mask_then_loss.py

Clean version built from the original v2 workflow, with cloud and cloud-shadow
screening added in the simplest possible way:

1) compute the original v2 keep mask for PRE and POST from B04/B08 only
2) compute a clear-valid mask for PRE and POST from the uploaded cloud/shadow evalscripts
3) apply each date's clear-valid mask to that date's keep mask
4) compute loss / gain / stable only on pixels that are clear in both dates

This keeps the original NDVI/sigNDVI calculation untouched and uses the masks only
as a validity screen.

Main idea
---------
    keep_pre_masked  = keep_pre_raw  & pre_clear_valid
    keep_post_masked = keep_post_raw & post_clear_valid
    comparison_valid = pre_clear_valid & post_clear_valid

    loss   = keep_pre_masked & (~keep_post_masked) & comparison_valid
    gain   = (~keep_pre_masked) & keep_post_masked & comparison_valid
    stable = keep_pre_masked & keep_post_masked & comparison_valid

This is the logic the user explicitly requested.

Inputs can be:
  - EO Browser exported ZIPs containing the required Raw bands
  - Directories containing the required band files

Required bands for cloud/shadow masks:
  B02, B03, B04, B08, B8A, B11, B12

Dependencies:
  pip install rasterio numpy
"""

from __future__ import annotations

import argparse
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import sieve
from rasterio.warp import reproject

# -----------------------------
# HARD-CODED PARAMETERS (core)
# -----------------------------
NDVI_MIN = 0.64
NDVI_MAX = 1.0

S4 = 0.02
S8 = 0.03

PRE_OPACITY = 0.55

AUTO_DN_SCALE = True
DN_SCALE = 10000.0

# -----------------------------
# Cloud / shadow evalscript thresholds
# -----------------------------
SHADOW_DARKNESS_MAX = 0.4
SHADOW_SUM_MAX = 0.20
VIS_MAX = 0.12
B8A_MAX = 0.12
B11_MAX = 0.08
B12_MAX = 0.06

REQUIRED_BANDS = ("B02", "B03", "B04", "B08", "B8A", "B11", "B12")
RASTER_EXTS = (".tif", ".tiff", ".jp2")

# -----------------------------
# Defaults (edit if you want)
# -----------------------------
PRE_DATASET = r"D:/Forest_Disturbance/imagery_zip/Comandau_BV/Comandau_2017_08_04.zip"
POST_DATASET = r"D:/Forest_Disturbance/imagery_zip/Comandau_BV/Comandau_2025_11_02.zip"
OUTDIR = r"D:/Forest_Disturbance/imagery_zip/Comandau_BV/Outputs"


# -----------------------------------------------------------------------------
# Low-level helpers
# -----------------------------------------------------------------------------
def _maybe_to_ref(arr: np.ndarray) -> np.ndarray:
    """Convert to float32 reflectance if data looks DN-scaled."""
    arrf = arr.astype(np.float32)
    if not AUTO_DN_SCALE:
        return arrf
    vmax = np.nanpercentile(arrf, 99.9)
    if vmax > 1.5:
        arrf = arrf / DN_SCALE
    return arrf


def clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def idx(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    s = a + b
    out = np.zeros_like(a, dtype=np.float32)
    np.divide(a - b, s, out=out, where=s != 0)
    return out


def ratio(a: np.ndarray, b: np.ndarray, fallback: float) -> np.ndarray:
    out = np.full_like(a, fallback, dtype=np.float32)
    np.divide(a, b, out=out, where=b != 0)
    return out


def _band_name_matches(path_name: str, tag: str) -> bool:
    stem = Path(path_name).stem.lower()
    tag_l = tag.lower()
    pattern = rf"(^|[^a-z0-9]){re.escape(tag_l)}([^a-z0-9]|$)"
    return re.search(pattern, stem) is not None


def _pick_zip_member(names: Iterable[str], tag: str) -> Optional[str]:
    names = list(names)
    cands = [n for n in names if Path(n).suffix.lower() in RASTER_EXTS and _band_name_matches(Path(n).name, tag)]
    if not cands:
        return None
    raw = [n for n in cands if "raw" in Path(n).name.lower()]
    return raw[0] if raw else cands[0]


def _pick_dir_file(files: Iterable[Path], tag: str) -> Optional[Path]:
    files = list(files)
    cands = [p for p in files if p.is_file() and p.suffix.lower() in RASTER_EXTS and _band_name_matches(p.name, tag)]
    if not cands:
        return None
    raw = [p for p in cands if "raw" in p.name.lower()]
    return raw[0] if raw else cands[0]


def find_band_paths(dataset_path: Path, tags: Iterable[str] = REQUIRED_BANDS) -> Dict[str, Path]:
    """Return a dict like {'B02': path, ..., 'B12': path} for ZIP or directory."""
    tags = tuple(tags)

    if dataset_path.is_file() and dataset_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(dataset_path, "r") as z:
            names = z.namelist()
            members: Dict[str, str] = {}
            for tag in tags:
                picked = _pick_zip_member(names, tag)
                if picked is None:
                    raise FileNotFoundError(f"Could not find {tag} inside {dataset_path.name}")
                members[tag] = picked

        tmpdir = Path(tempfile.mkdtemp(prefix="s2zip_"))
        out: Dict[str, Path] = {}
        with zipfile.ZipFile(dataset_path, "r") as z:
            for tag, member in members.items():
                ext = Path(member).suffix.lower()
                band_path = tmpdir / f"{tag}_raw{ext}"
                band_path.write_bytes(z.read(member))
                out[tag] = band_path
        return out

    if dataset_path.is_dir():
        files = [p for p in dataset_path.rglob("*") if p.is_file() and p.suffix.lower() in RASTER_EXTS]
        out: Dict[str, Path] = {}
        for tag in tags:
            picked = _pick_dir_file(files, tag)
            if picked is None:
                raise FileNotFoundError(f"Could not find {tag} in folder {dataset_path}")
            out[tag] = picked
        return out

    raise FileNotFoundError(f"Dataset path not found: {dataset_path}")


def _read_band(path: Path) -> Tuple[np.ndarray, dict, np.ndarray]:
    """Read first band and return (array, profile, dataset-valid-mask)."""
    with rasterio.open(path) as ds:
        arr = ds.read(1)
        profile = ds.profile.copy()
        valid = ds.read_masks(1) > 0
    return arr, profile, valid


def _same_grid(profile_a: dict, profile_b: dict) -> bool:
    return (
        profile_a.get("height") == profile_b.get("height")
        and profile_a.get("width") == profile_b.get("width")
        and profile_a.get("transform") == profile_b.get("transform")
        and profile_a.get("crs") == profile_b.get("crs")
    )


def _align_array_to(ref_profile: dict, src_path: Path, resampling: Resampling = Resampling.nearest) -> np.ndarray:
    """Reproject/resample source values to match ref_profile."""
    with rasterio.open(src_path) as src:
        src_arr = src.read(1)
        dst = np.zeros((ref_profile["height"], ref_profile["width"]), dtype=np.float32)
        reproject(
            source=src_arr.astype(np.float32),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_profile["transform"],
            dst_crs=ref_profile["crs"],
            resampling=resampling,
        )
    return dst


def _align_valid_to(ref_profile: dict, src_path: Path) -> np.ndarray:
    """Reproject/resample source dataset mask to match ref_profile."""
    with rasterio.open(src_path) as src:
        src_valid = (src.read_masks(1) > 0).astype(np.uint8)
        dst_valid = np.zeros((ref_profile["height"], ref_profile["width"]), dtype=np.uint8)
        reproject(
            source=src_valid,
            destination=dst_valid,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_profile["transform"],
            dst_crs=ref_profile["crs"],
            resampling=Resampling.nearest,
        )
    return dst_valid > 0


# -----------------------------------------------------------------------------
# Core v2 logic (unchanged math)
# -----------------------------------------------------------------------------
def compute_signdvi_rgb_and_keep(b04: np.ndarray, b08: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Original v2 logic:
      rgb = [0.9*clamp(1-ndvi)*darkness, 0.8*clamp(ndvi)*darkness, 0.1*darkness]
      keep = NDVI in [NDVI_MIN, NDVI_MAX]
    Returns:
      rgb_uint8: (H,W,3) uint8
      keep_bool: (H,W) bool
    """
    b04r = _maybe_to_ref(b04)
    b08r = _maybe_to_ref(b08)

    sum_ = b08r + b04r
    sum_safe = np.where(sum_ == 0, np.nan, sum_)

    ndvi = (b08r - b04r) / sum_safe
    ndvi = np.nan_to_num(ndvi, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

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


# -----------------------------------------------------------------------------
# Cloud / shadow masks from the uploaded evalscripts
# -----------------------------------------------------------------------------
def shadow_info(b4: np.ndarray, s4: float, b8: np.ndarray, s8: float) -> Tuple[np.ndarray, np.ndarray]:
    sum_ = b8 + b4
    s_ndvi = np.zeros_like(sum_, dtype=np.float32)
    valid = sum_ > 0
    if np.any(valid):
        s_ndvi[valid] = (
            2.0 / (sum_[valid] * sum_[valid])
            * np.sqrt((b8[valid] * b8[valid] * s4 * s4) + (b4[valid] * b4[valid] * s8 * s8))
        )
    darkness = clamp01(1.0 - 2.0 * s_ndvi)
    return darkness.astype(np.float32), sum_.astype(np.float32)


def compute_cloud_valid_mask(bands: Dict[str, np.ndarray], data_mask: np.ndarray) -> np.ndarray:
    """
    Cloud evalscript semantics:
      1 = valid scene
      0 = cloud or outside valid footprint
    """
    b02 = bands["B02"]
    b03 = bands["B03"]
    b04 = bands["B04"]
    b08 = bands["B08"]
    b8a = bands["B8A"]
    b11 = bands["B11"]

    ndvi = idx(b08, b04)
    ndsi = idx(b03, b11)
    ndgr = idx(b03, b04)

    b2b11 = ratio(b02, b11, 999.0)
    b8b11 = ratio(b8a, b11, 999.0)
    b4b11 = ratio(b04, b11, 999.0)
    vis = (b02 + b03 + b04) / 3.0

    water = ((b02 - b04) > 0.034) & (b8a < b04) & (b02 < 0.20)
    snow = (b03 > 0.20) & (b11 < 0.15) & (ratio(b04, b11, 999.0) > 4.0)

    bcy_cloud = ((b03 > 0.175) & (ndgr > 0)) | (b03 > 0.39)
    sen2cor_like_cloud = (
        (vis > 0.12)
        & (b04 > 0.06)
        & (ndsi > -0.24)
        & (ndvi < 0.42)
        & (b2b11 > 0.70)
        & (b8b11 > 0.90)
        & (b4b11 < 6.0)
        & (~water)
    )

    cloud = (~snow) & (bcy_cloud | sen2cor_like_cloud)
    return data_mask & (~cloud)


def compute_shadow_valid_mask(bands: Dict[str, np.ndarray], data_mask: np.ndarray) -> np.ndarray:
    """
    Shadow evalscript semantics:
      1 = valid scene
      0 = shadow
    Note: outside-scene pixels also become 1 in the original evalscript.
    """
    b02 = bands["B02"]
    b03 = bands["B03"]
    b04 = bands["B04"]
    b08 = bands["B08"]
    b8a = bands["B8A"]
    b11 = bands["B11"]
    b12 = bands["B12"]

    darkness, sum_ = shadow_info(b04, 0.02, b08, 0.03)
    vis = (b02 + b03 + b04) / 3.0

    water = ((b02 - b04) > 0.034) & (b8a < b04) & (b02 < 0.20)
    snow = (b03 > 0.20) & (b11 < 0.15) & (ratio(b04, b11, 999.0) > 4.0)

    shadow_by_index = (darkness < SHADOW_DARKNESS_MAX) & (sum_ < SHADOW_SUM_MAX)
    dark_spec = (
        (b02 < 0.15)
        & (b03 < 0.13)
        & (b04 < 0.13)
        & (b8a < B8A_MAX)
        & (b11 < B11_MAX)
        & (b12 < B12_MAX)
        & (vis < VIS_MAX)
    )

    shadow = (~water) & (~snow) & shadow_by_index & dark_spec
    return (~shadow) | (~data_mask)


# -----------------------------------------------------------------------------
# I/O helpers and visualization helpers
# -----------------------------------------------------------------------------
def write_geotiff(path: Path, arr: np.ndarray, profile: dict, nodata=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 1 if arr.ndim == 2 else arr.shape[0]
    prof = {
        "driver": "GTiff",
        "height": profile["height"],
        "width": profile["width"],
        "count": count,
        "dtype": str(arr.dtype),
        "crs": profile["crs"],
        "transform": profile["transform"],
        "compress": "LZW",
    }
    if nodata is not None:
        prof["nodata"] = nodata
    with rasterio.open(path, "w", **prof) as dst:
        if arr.ndim == 2:
            dst.write(arr, 1)
        else:
            dst.write(arr)


def filter_min_patch(mask255: np.ndarray, min_px: int, connectivity: int = 8) -> np.ndarray:
    mask01 = (mask255 > 0).astype(np.uint8)
    sieved = sieve(mask01, size=min_px, connectivity=connectivity).astype(np.uint8)
    return (sieved * 255).astype(np.uint8)


def make_overlay_red50(mask255: np.ndarray) -> np.ndarray:
    h, w = mask255.shape
    rgba = np.zeros((4, h, w), dtype=np.uint8)
    on = mask255 > 0
    rgba[0, on] = 255
    rgba[3, on] = 128
    return rgba


def alpha_blend_over_white(rgba_bottom: np.ndarray, rgba_top: np.ndarray, top_opacity: float = 1.0) -> np.ndarray:
    bot_rgb = rgba_bottom[:3].astype(np.float32) / 255.0
    bot_a = rgba_bottom[3].astype(np.float32) / 255.0

    top_rgb = rgba_top[:3].astype(np.float32) / 255.0
    top_a = (rgba_top[3].astype(np.float32) / 255.0) * float(top_opacity)

    white = np.ones_like(bot_rgb)
    bot_over_white = bot_rgb * bot_a + white * (1.0 - bot_a)
    out_rgb = top_rgb * top_a + bot_over_white * (1.0 - top_a)
    return (np.clip(out_rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def rgba_from_rgb_keep(rgb_hwc: np.ndarray, keep: np.ndarray) -> np.ndarray:
    rgba = np.zeros((4, rgb_hwc.shape[0], rgb_hwc.shape[1]), dtype=np.uint8)
    rgba[:3] = np.transpose(rgb_hwc, (2, 0, 1))
    rgba[3] = keep.astype(np.uint8) * 255
    return rgba


def white_mask_rgb(rgb_hwc: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.transpose(rgb_hwc, (2, 0, 1)).copy()
    out[:, ~valid] = 255
    return out


def grey_overlay(rgb_hwc: np.ndarray, grey_mask255: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> np.ndarray:
    out = np.transpose(rgb_hwc, (2, 0, 1)).copy()
    if valid_mask is not None:
        out[:, ~valid_mask] = 255
    on = grey_mask255 > 0
    out[:, on] = 180
    return out


# -----------------------------------------------------------------------------
# Dataset loading for the requested workflow
# -----------------------------------------------------------------------------
def load_core_b04_b08(dataset_path: Path, ref_profile: Optional[dict] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict, Dict[str, Path]]:
    """
    Load B04/B08 exactly for the v2 core workflow, aligning to ref_profile only when needed.

    Returns
    -------
    b04, b08 : arrays on target grid
    data_mask : valid mask from B04/B08 dataset masks on target grid
    profile : target grid profile
    paths : all located band paths for later cloud/shadow loading
    """
    paths = find_band_paths(dataset_path, REQUIRED_BANDS)

    b04, b04_profile, b04_valid = _read_band(paths["B04"])
    b08, b08_profile, b08_valid = _read_band(paths["B08"])

    # Choose target grid. PRE keeps its own B04 grid. POST is aligned to PRE grid if supplied.
    target_profile = b04_profile if ref_profile is None else ref_profile

    if ref_profile is None and _same_grid(b04_profile, b08_profile):
        out_b04 = b04
        out_b08 = b08
        out_valid = b04_valid & b08_valid
    else:
        # Bring both B04 and B08 onto the target grid so the core logic stays consistent.
        if _same_grid(b04_profile, target_profile):
            out_b04 = b04
            out_b04_valid = b04_valid
        else:
            out_b04 = _align_array_to(target_profile, paths["B04"], resampling=Resampling.nearest)
            out_b04_valid = _align_valid_to(target_profile, paths["B04"])

        if _same_grid(b08_profile, target_profile):
            out_b08 = b08
            out_b08_valid = b08_valid
        else:
            out_b08 = _align_array_to(target_profile, paths["B08"], resampling=Resampling.nearest)
            out_b08_valid = _align_valid_to(target_profile, paths["B08"])

        out_valid = out_b04_valid & out_b08_valid

    return out_b04, out_b08, out_valid, target_profile, paths


def load_cloudshadow_bands(paths: Dict[str, Path], ref_profile: dict) -> Dict[str, np.ndarray]:
    """
    Load all bands needed for cloud/shadow masking on the target grid.
    Uses nearest resampling to preserve threshold behavior.
    """
    out: Dict[str, np.ndarray] = {}
    for tag, path in paths.items():
        _, prof, _ = _read_band(path)
        if _same_grid(prof, ref_profile):
            arr, _, _ = _read_band(path)
            out[tag] = _maybe_to_ref(arr)
        else:
            out[tag] = _maybe_to_ref(_align_array_to(ref_profile, path, resampling=Resampling.nearest))
    return out


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 1) Load the core v2 B04/B08 workflow exactly on one common grid.
    # ------------------------------------------------------------------
    pre_b04, pre_b08, pre_data_mask, profile, pre_paths = load_core_b04_b08(pre_path, ref_profile=None)
    post_b04, post_b08, post_data_mask, _, post_paths = load_core_b04_b08(post_path, ref_profile=profile)

    # ------------------------------------------------------------------
    # 2) Compute the original v2 NDVI keep masks and RGBs, untouched.
    # ------------------------------------------------------------------
    pre_rgb, keep_pre_raw = compute_signdvi_rgb_and_keep(pre_b04, pre_b08)
    post_rgb, keep_post_raw = compute_signdvi_rgb_and_keep(post_b04, post_b08)

    # Raw/original-style outputs kept for comparison.
    pre_rgba_raw = rgba_from_rgb_keep(pre_rgb, keep_pre_raw)
    post_rgba_raw = rgba_from_rgb_keep(post_rgb, keep_post_raw)
    keep_pre_raw_255 = pre_rgba_raw[3].copy()
    keep_post_raw_255 = post_rgba_raw[3].copy()
    holes_pre_raw_255 = (255 - keep_pre_raw_255).astype(np.uint8)
    holes_post_raw_255 = (255 - keep_post_raw_255).astype(np.uint8)

    loss_raw_255 = ((keep_pre_raw) & (~keep_post_raw)).astype(np.uint8) * 255
    gain_raw_255 = ((~keep_pre_raw) & (keep_post_raw)).astype(np.uint8) * 255
    stable_raw_255 = ((keep_pre_raw) & (keep_post_raw)).astype(np.uint8) * 255

    loss_raw_f25_255 = filter_min_patch(loss_raw_255, min_px=25)
    loss_raw_f38_255 = filter_min_patch(loss_raw_255, min_px=38)
    loss_raw_f50_255 = filter_min_patch(loss_raw_255, min_px=50)
    loss_raw_overlay = make_overlay_red50(loss_raw_f50_255)
    blend_raw_rgb = alpha_blend_over_white(post_rgba_raw, pre_rgba_raw, top_opacity=PRE_OPACITY)
    post_with_loss_grey_raw = grey_overlay(post_rgb, loss_raw_f50_255)
    pre_with_loss_grey_raw = grey_overlay(pre_rgb, loss_raw_f50_255)
    pre_with_loss_grey_raw_clean = np.transpose(pre_rgb, (2, 0, 1)).copy()
    pre_with_loss_grey_raw_clean[:, keep_pre_raw_255 == 0] = 255
    pre_with_loss_grey_raw_clean[:, loss_raw_f50_255 > 0] = 180

    # ------------------------------------------------------------------
    # 3) Compute per-date cloud/shadow valid masks from evalscript logic.
    # ------------------------------------------------------------------
    pre_cs_bands = load_cloudshadow_bands(pre_paths, profile)
    post_cs_bands = load_cloudshadow_bands(post_paths, profile)

    pre_cloud_valid = compute_cloud_valid_mask(pre_cs_bands, pre_data_mask)
    post_cloud_valid = compute_cloud_valid_mask(post_cs_bands, post_data_mask)
    pre_shadow_valid = compute_shadow_valid_mask(pre_cs_bands, pre_data_mask)
    post_shadow_valid = compute_shadow_valid_mask(post_cs_bands, post_data_mask)

    pre_clear_valid = pre_data_mask & pre_cloud_valid & pre_shadow_valid
    post_clear_valid = post_data_mask & post_cloud_valid & post_shadow_valid
    comparison_valid = pre_clear_valid & post_clear_valid

    pre_cloud_valid_255 = pre_cloud_valid.astype(np.uint8) * 255
    post_cloud_valid_255 = post_cloud_valid.astype(np.uint8) * 255
    pre_shadow_valid_255 = pre_shadow_valid.astype(np.uint8) * 255
    post_shadow_valid_255 = post_shadow_valid.astype(np.uint8) * 255
    pre_clear_valid_255 = pre_clear_valid.astype(np.uint8) * 255
    post_clear_valid_255 = post_clear_valid.astype(np.uint8) * 255
    comparison_valid_255 = comparison_valid.astype(np.uint8) * 255

    # ------------------------------------------------------------------
    # 4) User-requested logic: mask each date first, then compare.
    # ------------------------------------------------------------------
    keep_pre = keep_pre_raw & pre_clear_valid
    keep_post = keep_post_raw & post_clear_valid

    # Important: comparison_valid is still required, otherwise invalid POST pixels
    # would become false losses after masking.
    loss_255 = (keep_pre & (~keep_post) & comparison_valid).astype(np.uint8) * 255
    gain_255 = ((~keep_pre) & keep_post & comparison_valid).astype(np.uint8) * 255
    stable_255 = (keep_pre & keep_post & comparison_valid).astype(np.uint8) * 255

    pre_rgba = rgba_from_rgb_keep(pre_rgb, keep_pre)
    post_rgba = rgba_from_rgb_keep(post_rgb, keep_post)

    keep_pre_255 = pre_rgba[3].copy()
    keep_post_255 = post_rgba[3].copy()
    holes_pre_255 = (255 - keep_pre_255).astype(np.uint8)
    holes_post_255 = (255 - keep_post_255).astype(np.uint8)

    loss_f25_255 = filter_min_patch(loss_255, min_px=25)
    loss_f38_255 = filter_min_patch(loss_255, min_px=38)
    loss_f50_255 = filter_min_patch(loss_255, min_px=50)
    loss_f50_overlay = make_overlay_red50(loss_f50_255)

    blend_rgb = alpha_blend_over_white(post_rgba, pre_rgba, top_opacity=PRE_OPACITY)

    # Main previews use the masked/valid images, so POST clouds/shadows are not shown.
    post_with_loss_grey = grey_overlay(post_rgb, loss_f50_255, valid_mask=post_clear_valid)
    pre_with_loss_grey = grey_overlay(pre_rgb, loss_f50_255, valid_mask=pre_clear_valid)
    pre_with_loss_grey_clean = np.transpose(pre_rgb, (2, 0, 1)).copy()
    pre_with_loss_grey_clean[:, ~pre_clear_valid] = 255
    pre_with_loss_grey_clean[:, keep_pre_255 == 0] = 255
    pre_with_loss_grey_clean[:, loss_f50_255 > 0] = 180

    post_rgb_nocloud_noshadow = white_mask_rgb(post_rgb, post_clear_valid)
    pre_rgb_nocloud_noshadow = white_mask_rgb(pre_rgb, pre_clear_valid)

    # ------------------------------------------------------------------
    # 5) Write outputs.
    # ------------------------------------------------------------------
    # Main output names now correspond to the per-date-masked workflow.
    write_geotiff(outdir / "keep_pre_255.tif", keep_pre_255.astype(np.uint8), profile, nodata=0)
    write_geotiff(outdir / "keep_post_255.tif", keep_post_255.astype(np.uint8), profile, nodata=0)
    write_geotiff(outdir / "holes_pre_notkept_255.tif", holes_pre_255, profile, nodata=0)
    write_geotiff(outdir / "holes_post_notkept_255.tif", holes_post_255, profile, nodata=0)
    write_geotiff(outdir / "loss_pre_kept_post_notkept_255.tif", loss_255.astype(np.uint8), profile, nodata=0)
    write_geotiff(outdir / "gain_pre_notkept_post_kept_255.tif", gain_255.astype(np.uint8), profile, nodata=0)
    write_geotiff(outdir / "stable_kept_both_255.tif", stable_255.astype(np.uint8), profile, nodata=0)

    write_geotiff(outdir / "loss_filtered_min25px_255.tif", loss_f25_255, profile, nodata=0)
    write_geotiff(outdir / "loss_filtered_min38px_255.tif", loss_f38_255, profile, nodata=0)
    write_geotiff(outdir / "loss_filtered_min50px_255.tif", loss_f50_255, profile, nodata=0)

    write_geotiff(outdir / "pre_rgba_signdvi.tif", pre_rgba, profile)
    write_geotiff(outdir / "post_rgba_signdvi.tif", post_rgba, profile)
    write_geotiff(outdir / "loss_filtered_min50_overlay_red50_rgba_blackoutside.tif", loss_f50_overlay, profile)

    write_geotiff(outdir / "blend_pre055_over_post1_rgb_uint8.tif", blend_rgb, profile)
    write_geotiff(outdir / "post_with_loss_grey_min50_rgb_uint8.tif", post_with_loss_grey, profile)
    write_geotiff(outdir / "pre_with_loss_grey_min50_rgb_uint8.tif", pre_with_loss_grey, profile)
    write_geotiff(outdir / "pre_with_loss_grey_min50_clean_rgb_uint8.tif", pre_with_loss_grey_clean, profile)

    # Diagnostics for cloud/shadow screening.
    write_geotiff(outdir / "pre_cloud_valid_255.tif", pre_cloud_valid_255, profile, nodata=0)
    write_geotiff(outdir / "post_cloud_valid_255.tif", post_cloud_valid_255, profile, nodata=0)
    write_geotiff(outdir / "pre_shadow_valid_255.tif", pre_shadow_valid_255, profile, nodata=0)
    write_geotiff(outdir / "post_shadow_valid_255.tif", post_shadow_valid_255, profile, nodata=0)
    write_geotiff(outdir / "pre_clear_valid_nocloud_noshadow_255.tif", pre_clear_valid_255, profile, nodata=0)
    write_geotiff(outdir / "post_clear_valid_nocloud_noshadow_255.tif", post_clear_valid_255, profile, nodata=0)
    write_geotiff(outdir / "comparison_clear_valid_bothdates_255.tif", comparison_valid_255, profile, nodata=0)
    write_geotiff(outdir / "pre_rgb_nocloud_noshadow_rgb_uint8.tif", pre_rgb_nocloud_noshadow, profile)
    write_geotiff(outdir / "post_rgb_nocloud_noshadow_rgb_uint8.tif", post_rgb_nocloud_noshadow, profile)

    # Raw/original v2 outputs preserved under *_raw names for direct comparison.
    write_geotiff(outdir / "keep_pre_raw_255.tif", keep_pre_raw_255.astype(np.uint8), profile, nodata=0)
    write_geotiff(outdir / "keep_post_raw_255.tif", keep_post_raw_255.astype(np.uint8), profile, nodata=0)
    write_geotiff(outdir / "holes_pre_raw_notkept_255.tif", holes_pre_raw_255, profile, nodata=0)
    write_geotiff(outdir / "holes_post_raw_notkept_255.tif", holes_post_raw_255, profile, nodata=0)
    write_geotiff(outdir / "loss_raw_pre_kept_post_notkept_255.tif", loss_raw_255.astype(np.uint8), profile, nodata=0)
    write_geotiff(outdir / "gain_raw_pre_notkept_post_kept_255.tif", gain_raw_255.astype(np.uint8), profile, nodata=0)
    write_geotiff(outdir / "stable_raw_kept_both_255.tif", stable_raw_255.astype(np.uint8), profile, nodata=0)
    write_geotiff(outdir / "loss_raw_filtered_min25px_255.tif", loss_raw_f25_255, profile, nodata=0)
    write_geotiff(outdir / "loss_raw_filtered_min38px_255.tif", loss_raw_f38_255, profile, nodata=0)
    write_geotiff(outdir / "loss_raw_filtered_min50px_255.tif", loss_raw_f50_255, profile, nodata=0)
    write_geotiff(outdir / "pre_rgba_signdvi_raw.tif", pre_rgba_raw, profile)
    write_geotiff(outdir / "post_rgba_signdvi_raw.tif", post_rgba_raw, profile)
    write_geotiff(outdir / "loss_raw_filtered_min50_overlay_red50_rgba_blackoutside.tif", loss_raw_overlay, profile)
    write_geotiff(outdir / "blend_pre055_over_post1_raw_rgb_uint8.tif", blend_raw_rgb, profile)
    write_geotiff(outdir / "post_with_loss_grey_min50_raw_rgb_uint8.tif", post_with_loss_grey_raw, profile)
    write_geotiff(outdir / "pre_with_loss_grey_min50_raw_rgb_uint8.tif", pre_with_loss_grey_raw, profile)
    write_geotiff(outdir / "pre_with_loss_grey_min50_clean_raw_rgb_uint8.tif", pre_with_loss_grey_raw_clean, profile)

    print("Done. Outputs written to:", outdir.resolve())


if __name__ == "__main__":
    main()
