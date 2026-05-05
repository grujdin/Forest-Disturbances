#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ndvi_windthrow_bitemp_v11_sclfix_sep_ndvi_cloudedge.py

Improved version of the v10 workflow with three key changes:

1) separate NDVI min/max thresholds for PRE and POST
2) robust cloud/shadow removal using the EO Browser Scene Classification Map (SCL)
   when it is available in the ZIP/folder; otherwise the script falls back to the
   previous evalscript-based cloud/shadow masks
3) cloud-edge cleanup for EO Browser rendered SCL exports by decoding the SCL palette
   to class IDs first and invalidating ambiguous edge pixels (especially class 7,
   and optionally class 5) when they occur directly around cloud cores

Why this change
---------------
For some EO Browser "Raw" exports, the absolute reflectance scaling of the bands
may not match the absolute thresholds used by the cloud evalscripts well enough.
That can make the evalscript cloud mask unstable. When the EO Browser SCL layer is
present, it is generally the safer source for removing clouds / cloud shadows /
cirrus / snow footprints from the final outputs.

In some rendered EO Browser SCL TIFFs, cloud fringes are not always encoded only as
class 8/9/10/11. Around the cloud edge, some pixels can appear as class 7
("unclassified") and a few as class 5 ("bare"). Those pixels survived in the
previous SCL implementation and showed up as residual bright artefacts.

Main logic
----------
    keep_pre_raw  = NDVI_KEEP(PRE_B04,  PRE_B08,  PRE_NDVI_MIN,  PRE_NDVI_MAX)
    keep_post_raw = NDVI_KEEP(POST_B04, POST_B08, POST_NDVI_MIN, POST_NDVI_MAX)

    clear_pre  = pre_data_mask  & pre_valid_scene_mask
    clear_post = post_data_mask & post_valid_scene_mask

    keep_pre  = keep_pre_raw  & clear_pre
    keep_post = keep_post_raw & clear_post

    comparison_valid = clear_pre & clear_post

    loss   = keep_pre & (~keep_post) & comparison_valid
    gain   = (~keep_pre) & keep_post & comparison_valid
    stable = keep_pre & keep_post & comparison_valid

Inputs can be:
  - EO Browser exported ZIPs containing the required Raw bands
  - Directories containing the required band files

Required bands for the fallback evalscript masks:
  B02, B03, B04, B08, B8A, B11, B12

Optional extra file for robust masking:
  Scene_classification_map.tiff  (EO Browser SCL visualization export)

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
PRE_NDVI_MIN = 0.35
PRE_NDVI_MAX = 1.0

POST_NDVI_MIN = 0.35
POST_NDVI_MAX = 1.0

S4 = 0.02
S8 = 0.03

PRE_OPACITY = 0.55

AUTO_DN_SCALE = True
DN_SCALE = 10000.0

# -----------------------------
# Cloud / shadow evalscript thresholds
# (used only as fallback, or diagnostics)
# -----------------------------
SHADOW_DARKNESS_MAX = 0.4
SHADOW_SUM_MAX = 0.20
VIS_MAX = 0.12
B8A_MAX = 0.12
B11_MAX = 0.08
B12_MAX = 0.06

# -----------------------------
# Preferred valid-scene masking
# -----------------------------
USE_SCL_IF_AVAILABLE = True
SCL_DILATE_PX = 2   # final safety dilation around all invalid SCL pixels

# Ambiguous cloud-edge handling:
# In some EO Browser SCL exports, cloud fringes are not all class 8/9/10/11.
# Some are class 7 (unclassified) and a few class 5 (bare) pixels directly
# adjacent to cloud pixels. Those survived in the original script.
SCL_EDGE_EXPAND_PX = 3
SCL_INVALIDATE_UNCLASSIFIED_NEAR_CLOUD = True
SCL_INVALIDATE_BARE_NEAR_CLOUD = True
SCL_INVALIDATE_TOPOGRAPHIC_SHADOW_NEAR_CLOUD = False
SCL_INVALIDATE_UNCLASSIFIED_GLOBALLY = False
INCLUDE_SCL_TOPOGRAPHIC_SHADOW = False
SCL_COLOR_MATCH_TOL = 3

REQUIRED_BANDS = ("B02", "B03", "B04", "B08", "B8A", "B11", "B12")
RASTER_EXTS = (".tif", ".tiff", ".jp2")

# -----------------------------
# Defaults (edit if you want)
# -----------------------------
PRE_DATASET = r"D:/Forest_Disturbance/imagery_zip/Comandau_BV/Comandau_2023_10_12.zip"
POST_DATASET = r"D:/Forest_Disturbance/imagery_zip/Comandau_BV/Comandau_2024_10_26.zip"
OUTDIR = r"D:/Forest_Disturbance/imagery_zip/Comandau_BV/Outputs_2023_10_12_2024_10_26_v7_improved_0.35_0.35_new_version_Oct_Oct"


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


def _pick_zip_member_contains(names: Iterable[str], patterns: Iterable[str]) -> Optional[str]:
    names = list(names)
    pats = [p.lower() for p in patterns]
    cands = []
    for n in names:
        ln = Path(n).name.lower()
        if Path(n).suffix.lower() in RASTER_EXTS and any(p in ln for p in pats):
            cands.append(n)
    return cands[0] if cands else None


def _pick_dir_file_contains(files: Iterable[Path], patterns: Iterable[str]) -> Optional[Path]:
    files = list(files)
    pats = [p.lower() for p in patterns]
    cands = []
    for p in files:
        ln = p.name.lower()
        if p.is_file() and p.suffix.lower() in RASTER_EXTS and any(pt in ln for pt in pats):
            cands.append(p)
    return cands[0] if cands else None


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


def find_optional_scl_path(dataset_path: Path) -> Optional[Path]:
    """
    Find the EO Browser Scene Classification Map export if it exists.
    Accepts both a raw numeric SCL raster and the EO Browser rendered SCL TIFF.
    """
    patterns = ["scene_classification_map", "scl"]

    if dataset_path.is_file() and dataset_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(dataset_path, "r") as z:
            names = z.namelist()
            member = _pick_zip_member_contains(names, patterns)
            if member is None:
                return None

        tmpdir = Path(tempfile.mkdtemp(prefix="s2scl_"))
        ext = Path(member).suffix.lower()
        out = tmpdir / f"SCL{ext}"
        with zipfile.ZipFile(dataset_path, "r") as z:
            out.write_bytes(z.read(member))
        return out

    if dataset_path.is_dir():
        files = [p for p in dataset_path.rglob("*") if p.is_file() and p.suffix.lower() in RASTER_EXTS]
        return _pick_dir_file_contains(files, patterns)

    return None


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


def _align_multiband_to(ref_profile: dict, src_path: Path, band_indexes: Optional[Iterable[int]] = None) -> np.ndarray:
    with rasterio.open(src_path) as src:
        if band_indexes is None:
            band_indexes = range(1, src.count + 1)
        band_indexes = list(band_indexes)
        dst = np.zeros((len(band_indexes), ref_profile["height"], ref_profile["width"]), dtype=np.float32)
        for out_i, band_i in enumerate(band_indexes):
            src_arr = src.read(band_i)
            reproject(
                source=src_arr.astype(np.float32),
                destination=dst[out_i],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_profile["transform"],
                dst_crs=ref_profile["crs"],
                resampling=Resampling.nearest,
            )
    return dst


def binary_dilate(mask: np.ndarray, radius_px: int) -> np.ndarray:
    """Simple square-pixel dilation without extra dependencies."""
    if radius_px <= 0:
        return mask.copy()
    out = mask.copy()
    h, w = mask.shape
    for dy in range(-radius_px, radius_px + 1):
        for dx in range(-radius_px, radius_px + 1):
            if dy == 0 and dx == 0:
                continue
            y0_src = max(0, -dy)
            y1_src = min(h, h - dy)
            x0_src = max(0, -dx)
            x1_src = min(w, w - dx)
            y0_dst = max(0, dy)
            y1_dst = min(h, h + dy)
            x0_dst = max(0, dx)
            x1_dst = min(w, w + dx)
            out[y0_dst:y1_dst, x0_dst:x1_dst] |= mask[y0_src:y1_src, x0_src:x1_src]
    return out


# -----------------------------------------------------------------------------
# Core v2 logic (same math, but with per-date thresholds)
# -----------------------------------------------------------------------------
def compute_signdvi_rgb_and_keep(
    b04: np.ndarray,
    b08: np.ndarray,
    ndvi_min: float,
    ndvi_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Original v2 logic:
      rgb = [0.9*clamp(1-ndvi)*darkness, 0.8*clamp(ndvi)*darkness, 0.1*darkness]
      keep = NDVI in [ndvi_min, ndvi_max]
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

    keep = (ndvi >= ndvi_min) & (ndvi <= ndvi_max)
    return rgb_u8, keep


# -----------------------------------------------------------------------------
# Fallback cloud / shadow masks from the evalscripts
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


def compute_cloud_valid_mask_evalscript(bands: Dict[str, np.ndarray], data_mask: np.ndarray) -> np.ndarray:
    """
    Evalscript semantics:
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


def compute_shadow_valid_mask_evalscript(bands: Dict[str, np.ndarray], data_mask: np.ndarray) -> np.ndarray:
    """
    Evalscript semantics:
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
# SCL-based valid-scene mask
# -----------------------------------------------------------------------------
def _scl_color_u16(rgb8: Tuple[int, int, int]) -> Tuple[int, int, int]:
    return tuple(int(round(v * 257)) for v in rgb8)


SCL_COLORS_U8 = {
    "no_data": (0, 0, 0),
    "saturated": (255, 0, 0),
    "topographic_shadow": (47, 47, 47),
    "cloud_shadow": (100, 50, 0),
    "vegetation": (0, 160, 0),
    "bare": (255, 230, 90),
    "water": (0, 0, 255),
    "unclassified": (128, 128, 128),
    "cloud_medium": (192, 192, 192),
    "cloud_high": (255, 255, 255),
    "thin_cirrus": (100, 200, 255),
    "snow": (255, 150, 255),
}
SCL_COLORS_U16 = {k: _scl_color_u16(v) for k, v in SCL_COLORS_U8.items()}

SCL_CLASS_BY_NAME = {
    "no_data": 0,
    "saturated": 1,
    "topographic_shadow": 2,
    "cloud_shadow": 3,
    "vegetation": 4,
    "bare": 5,
    "water": 6,
    "unclassified": 7,
    "cloud_medium": 8,
    "cloud_high": 9,
    "thin_cirrus": 10,
    "snow": 11,
}


def _color_match(rgb: np.ndarray, color: Tuple[int, int, int], tol: int = 2) -> np.ndarray:
    return np.all(np.abs(rgb.astype(np.int32) - np.array(color, dtype=np.int32)) <= tol, axis=2)


def _decode_rendered_scl_classes(rgb: np.ndarray) -> np.ndarray:
    """
    Decode an EO Browser rendered SCL RGB/RGBA raster into numeric SCL class IDs.

    Supports both 8-bit and 16-bit palette exports.
    Unknown colors remain 255.
    """
    vmax = float(np.nanmax(rgb)) if rgb.size else 0.0
    use_u16 = vmax > 255.5

    palette = {}
    for name, cid in SCL_CLASS_BY_NAME.items():
        if use_u16:
            palette[cid] = np.array(SCL_COLORS_U16[name], dtype=np.int32)
        else:
            palette[cid] = np.array(SCL_COLORS_U8[name], dtype=np.int32)

    scl = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    rgb_i32 = rgb.astype(np.int32)
    tol = int(SCL_COLOR_MATCH_TOL)

    for cid, color in palette.items():
        match = np.all(np.abs(rgb_i32 - color) <= tol, axis=2)
        scl[match] = cid

    return scl


def _load_scl_classes(scl_path: Path, ref_profile: dict) -> np.ndarray:
    """
    Load SCL as numeric class IDs on the reference grid.

    Returns values in Sentinel-2 SCL convention:
      0 no data, 1 saturated, 2 topo shadow, 3 cloud shadow,
      4 vegetation, 5 bare, 6 water, 7 unclassified,
      8 cloud med, 9 cloud high, 10 cirrus, 11 snow.
    Unknown colors in rendered exports remain 255.
    """
    with rasterio.open(scl_path) as ds:
        prof = ds.profile.copy()

        if ds.count == 1:
            arr = ds.read(1)
            if not _same_grid(prof, ref_profile):
                dst = np.zeros((ref_profile["height"], ref_profile["width"]), dtype=arr.dtype)
                reproject(
                    source=arr,
                    destination=dst,
                    src_transform=ds.transform,
                    src_crs=ds.crs,
                    dst_transform=ref_profile["transform"],
                    dst_crs=ref_profile["crs"],
                    resampling=Resampling.nearest,
                )
                arr = dst
            return arr.astype(np.uint8)

        rgb = _align_multiband_to(ref_profile, scl_path, band_indexes=[1, 2, 3]).transpose(1, 2, 0)
        return _decode_rendered_scl_classes(rgb)



def load_scl_clear_valid_mask(scl_path: Path, ref_profile: dict, data_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a valid-scene mask from the SCL raster.

    Improvements over the original version:
      - decodes the SCL into class IDs first
      - masks the standard cloud/shadow/snow classes
      - also masks ambiguous cloud-edge pixels (class 7 and, optionally, class 5)
        when they sit next to cloud/shadow core pixels

    This is important for EO Browser exports where cloud fringes are sometimes
    labeled as "unclassified" or include a few "bare" pixels at the cloud edge.
    """
    scl = _load_scl_classes(scl_path, ref_profile)

    # Standard invalid SCL classes.
    invalid = np.isin(
        scl,
        [
            SCL_CLASS_BY_NAME["no_data"],
            SCL_CLASS_BY_NAME["saturated"],
            SCL_CLASS_BY_NAME["cloud_shadow"],
            SCL_CLASS_BY_NAME["cloud_medium"],
            SCL_CLASS_BY_NAME["cloud_high"],
            SCL_CLASS_BY_NAME["thin_cirrus"],
            SCL_CLASS_BY_NAME["snow"],
        ],
    )

    if INCLUDE_SCL_TOPOGRAPHIC_SHADOW:
        invalid |= scl == SCL_CLASS_BY_NAME["topographic_shadow"]

    if SCL_INVALIDATE_UNCLASSIFIED_GLOBALLY:
        invalid |= scl == SCL_CLASS_BY_NAME["unclassified"]

    # Expand only the cloud/shadow/snow core, not generic no-data boundaries.
    cloud_core = np.isin(
        scl,
        [
            SCL_CLASS_BY_NAME["cloud_shadow"],
            SCL_CLASS_BY_NAME["cloud_medium"],
            SCL_CLASS_BY_NAME["cloud_high"],
            SCL_CLASS_BY_NAME["thin_cirrus"],
            SCL_CLASS_BY_NAME["snow"],
        ],
    )

    if SCL_EDGE_EXPAND_PX > 0:
        near_cloud_core = binary_dilate(cloud_core, SCL_EDGE_EXPAND_PX)
    else:
        near_cloud_core = cloud_core

    ambiguous_edge = np.zeros_like(invalid, dtype=bool)

    if SCL_INVALIDATE_UNCLASSIFIED_NEAR_CLOUD:
        ambiguous_edge |= (scl == SCL_CLASS_BY_NAME["unclassified"]) & near_cloud_core

    if SCL_INVALIDATE_BARE_NEAR_CLOUD:
        ambiguous_edge |= (scl == SCL_CLASS_BY_NAME["bare"]) & near_cloud_core

    if (not INCLUDE_SCL_TOPOGRAPHIC_SHADOW) and SCL_INVALIDATE_TOPOGRAPHIC_SHADOW_NEAR_CLOUD:
        ambiguous_edge |= (scl == SCL_CLASS_BY_NAME["topographic_shadow"]) & near_cloud_core

    invalid |= ambiguous_edge

    if SCL_DILATE_PX > 0:
        invalid = binary_dilate(invalid, SCL_DILATE_PX)

    invalid &= data_mask
    clear_valid = data_mask & (~invalid)
    return clear_valid, invalid


# -----------------------------------------------------------------------------
# I/O helpers and visualization helpers
# -----------------------------------------------------------------------------
def write_geotiff(path: Path, arr: np.ndarray, profile: dict, nodata=None, valid_mask: Optional[np.ndarray] = None):
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
        if valid_mask is not None:
            dst.write_mask(valid_mask.astype(np.uint8) * 255)


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


def rgba_from_rgb_keep(rgb_hwc: np.ndarray, keep: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> np.ndarray:
    rgba = np.zeros((4, rgb_hwc.shape[0], rgb_hwc.shape[1]), dtype=np.uint8)
    rgba[:3] = np.transpose(rgb_hwc, (2, 0, 1))
    if valid_mask is None:
        rgba[3] = keep.astype(np.uint8) * 255
    else:
        rgba[3] = (keep & valid_mask).astype(np.uint8) * 255
    return rgba


def transparent_rgba_from_rgb(rgb_hwc: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    rgba = np.zeros((4, rgb_hwc.shape[0], rgb_hwc.shape[1]), dtype=np.uint8)
    rgba[:3] = np.transpose(rgb_hwc, (2, 0, 1))
    rgba[3] = valid_mask.astype(np.uint8) * 255
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

    target_profile = b04_profile if ref_profile is None else ref_profile

    if ref_profile is None and _same_grid(b04_profile, b08_profile):
        out_b04 = b04
        out_b08 = b08
        out_valid = b04_valid & b08_valid
    else:
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
    Load all bands needed for fallback evalscript cloud/shadow masking on the target grid.
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
    # 2) Compute the original v2 NDVI keep masks and RGBs, now with
    #    separate PRE and POST thresholds.
    # ------------------------------------------------------------------
    pre_rgb, keep_pre_raw = compute_signdvi_rgb_and_keep(pre_b04, pre_b08, PRE_NDVI_MIN, PRE_NDVI_MAX)
    post_rgb, keep_post_raw = compute_signdvi_rgb_and_keep(post_b04, post_b08, POST_NDVI_MIN, POST_NDVI_MAX)

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
    # 3) Compute valid-scene masks.
    #    Prefer SCL if available; otherwise use the evalscript masks.
    # ------------------------------------------------------------------
    pre_scl_path = find_optional_scl_path(pre_path)
    post_scl_path = find_optional_scl_path(post_path)

    pre_cs_bands = load_cloudshadow_bands(pre_paths, profile)
    post_cs_bands = load_cloudshadow_bands(post_paths, profile)

    # Fallback / diagnostic evalscript masks
    pre_cloud_valid_eval = compute_cloud_valid_mask_evalscript(pre_cs_bands, pre_data_mask)
    post_cloud_valid_eval = compute_cloud_valid_mask_evalscript(post_cs_bands, post_data_mask)
    pre_shadow_valid_eval = compute_shadow_valid_mask_evalscript(pre_cs_bands, pre_data_mask)
    post_shadow_valid_eval = compute_shadow_valid_mask_evalscript(post_cs_bands, post_data_mask)
    pre_clear_valid_eval = pre_data_mask & pre_cloud_valid_eval & pre_shadow_valid_eval
    post_clear_valid_eval = post_data_mask & post_cloud_valid_eval & post_shadow_valid_eval

    pre_cloud_valid_255 = pre_cloud_valid_eval.astype(np.uint8) * 255
    post_cloud_valid_255 = post_cloud_valid_eval.astype(np.uint8) * 255
    pre_shadow_valid_255 = pre_shadow_valid_eval.astype(np.uint8) * 255
    post_shadow_valid_255 = post_shadow_valid_eval.astype(np.uint8) * 255

    # Preferred SCL masks if available
    pre_scl_clear_valid = np.ones_like(pre_data_mask, dtype=bool)
    post_scl_clear_valid = np.ones_like(post_data_mask, dtype=bool)
    pre_scl_invalid = np.zeros_like(pre_data_mask, dtype=bool)
    post_scl_invalid = np.zeros_like(post_data_mask, dtype=bool)

    if USE_SCL_IF_AVAILABLE and pre_scl_path is not None:
        pre_scl_clear_valid, pre_scl_invalid = load_scl_clear_valid_mask(pre_scl_path, profile, pre_data_mask)
    if USE_SCL_IF_AVAILABLE and post_scl_path is not None:
        post_scl_clear_valid, post_scl_invalid = load_scl_clear_valid_mask(post_scl_path, profile, post_data_mask)

    # Final valid-scene masks
    if USE_SCL_IF_AVAILABLE and pre_scl_path is not None:
        pre_clear_valid = pre_scl_clear_valid
    else:
        pre_clear_valid = pre_clear_valid_eval

    if USE_SCL_IF_AVAILABLE and post_scl_path is not None:
        post_clear_valid = post_scl_clear_valid
    else:
        post_clear_valid = post_clear_valid_eval

    comparison_valid = pre_clear_valid & post_clear_valid

    pre_clear_valid_255 = pre_clear_valid.astype(np.uint8) * 255
    post_clear_valid_255 = post_clear_valid.astype(np.uint8) * 255
    comparison_valid_255 = comparison_valid.astype(np.uint8) * 255

    pre_scl_invalid_255 = pre_scl_invalid.astype(np.uint8) * 255
    post_scl_invalid_255 = post_scl_invalid.astype(np.uint8) * 255
    pre_scl_clear_valid_255 = pre_scl_clear_valid.astype(np.uint8) * 255
    post_scl_clear_valid_255 = post_scl_clear_valid.astype(np.uint8) * 255

    # ------------------------------------------------------------------
    # 4) Per-date-masked change logic.
    # ------------------------------------------------------------------
    keep_pre = keep_pre_raw & pre_clear_valid
    keep_post = keep_post_raw & post_clear_valid

    loss_255 = (keep_pre & (~keep_post) & comparison_valid).astype(np.uint8) * 255
    gain_255 = ((~keep_pre) & keep_post & comparison_valid).astype(np.uint8) * 255
    stable_255 = (keep_pre & keep_post & comparison_valid).astype(np.uint8) * 255

    pre_rgba = rgba_from_rgb_keep(pre_rgb, keep_pre, valid_mask=pre_clear_valid)
    post_rgba = rgba_from_rgb_keep(post_rgb, keep_post, valid_mask=post_clear_valid)

    keep_pre_255 = pre_rgba[3].copy()
    keep_post_255 = post_rgba[3].copy()
    holes_pre_255 = (255 - keep_pre_255).astype(np.uint8)
    holes_post_255 = (255 - keep_post_255).astype(np.uint8)

    loss_f25_255 = filter_min_patch(loss_255, min_px=25)
    loss_f38_255 = filter_min_patch(loss_255, min_px=38)
    loss_f50_255 = filter_min_patch(loss_255, min_px=50)
    loss_f50_overlay = make_overlay_red50(loss_f50_255)

    blend_rgb = alpha_blend_over_white(post_rgba, pre_rgba, top_opacity=PRE_OPACITY)

    # Main previews use only valid/clear pixels.
    post_with_loss_grey = grey_overlay(post_rgb, loss_f50_255, valid_mask=post_clear_valid)
    pre_with_loss_grey = grey_overlay(pre_rgb, loss_f50_255, valid_mask=pre_clear_valid)
    pre_with_loss_grey_clean = np.transpose(pre_rgb, (2, 0, 1)).copy()
    pre_with_loss_grey_clean[:, ~pre_clear_valid] = 255
    pre_with_loss_grey_clean[:, keep_pre_255 == 0] = 255
    pre_with_loss_grey_clean[:, loss_f50_255 > 0] = 180

    post_rgb_nocloud_noshadow = white_mask_rgb(post_rgb, post_clear_valid)
    pre_rgb_nocloud_noshadow = white_mask_rgb(pre_rgb, pre_clear_valid)

    # RGBA transparent versions (visually safest for cloud-free display)
    post_rgb_clear_rgba = transparent_rgba_from_rgb(post_rgb, post_clear_valid)
    pre_rgb_clear_rgba = transparent_rgba_from_rgb(pre_rgb, pre_clear_valid)
    post_with_loss_grey_rgba = transparent_rgba_from_rgb(np.transpose(post_with_loss_grey, (1, 2, 0)), post_clear_valid)
    pre_with_loss_grey_rgba = transparent_rgba_from_rgb(np.transpose(pre_with_loss_grey, (1, 2, 0)), pre_clear_valid)

    # ------------------------------------------------------------------
    # 5) Write outputs.
    # ------------------------------------------------------------------
    # Main output names correspond to the preferred (SCL if available) masked workflow.
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

    write_geotiff(outdir / "pre_rgba_signdvi.tif", pre_rgba, profile, valid_mask=pre_clear_valid)
    write_geotiff(outdir / "post_rgba_signdvi.tif", post_rgba, profile, valid_mask=post_clear_valid)
    write_geotiff(outdir / "loss_filtered_min50_overlay_red50_rgba_blackoutside.tif", loss_f50_overlay, profile, valid_mask=comparison_valid)

    write_geotiff(outdir / "blend_pre055_over_post1_rgb_uint8.tif", blend_rgb, profile, valid_mask=comparison_valid)
    write_geotiff(outdir / "post_with_loss_grey_min50_rgb_uint8.tif", post_with_loss_grey, profile, valid_mask=post_clear_valid)
    write_geotiff(outdir / "pre_with_loss_grey_min50_rgb_uint8.tif", pre_with_loss_grey, profile, valid_mask=pre_clear_valid)
    write_geotiff(outdir / "pre_with_loss_grey_min50_clean_rgb_uint8.tif", pre_with_loss_grey_clean, profile, valid_mask=pre_clear_valid)

    # Transparent versions for cloud-free visual display
    write_geotiff(outdir / "pre_rgb_nocloud_noshadow_rgba_uint8.tif", pre_rgb_clear_rgba, profile, valid_mask=pre_clear_valid)
    write_geotiff(outdir / "post_rgb_nocloud_noshadow_rgba_uint8.tif", post_rgb_clear_rgba, profile, valid_mask=post_clear_valid)
    write_geotiff(outdir / "pre_with_loss_grey_min50_rgba_uint8.tif", pre_with_loss_grey_rgba, profile, valid_mask=pre_clear_valid)
    write_geotiff(outdir / "post_with_loss_grey_min50_rgba_uint8.tif", post_with_loss_grey_rgba, profile, valid_mask=post_clear_valid)

    # White-background RGB previews
    write_geotiff(outdir / "pre_rgb_nocloud_noshadow_rgb_uint8.tif", pre_rgb_nocloud_noshadow, profile, valid_mask=pre_clear_valid)
    write_geotiff(outdir / "post_rgb_nocloud_noshadow_rgb_uint8.tif", post_rgb_nocloud_noshadow, profile, valid_mask=post_clear_valid)

    # Diagnostics for fallback evalscript screening
    write_geotiff(outdir / "pre_cloud_valid_255.tif", pre_cloud_valid_255, profile, nodata=0)
    write_geotiff(outdir / "post_cloud_valid_255.tif", post_cloud_valid_255, profile, nodata=0)
    write_geotiff(outdir / "pre_shadow_valid_255.tif", pre_shadow_valid_255, profile, nodata=0)
    write_geotiff(outdir / "post_shadow_valid_255.tif", post_shadow_valid_255, profile, nodata=0)
    write_geotiff(outdir / "pre_clear_valid_evalscript_255.tif", pre_clear_valid_eval.astype(np.uint8) * 255, profile, nodata=0)
    write_geotiff(outdir / "post_clear_valid_evalscript_255.tif", post_clear_valid_eval.astype(np.uint8) * 255, profile, nodata=0)

    # Diagnostics for preferred final masks
    write_geotiff(outdir / "pre_clear_valid_nocloud_noshadow_255.tif", pre_clear_valid_255, profile, nodata=0)
    write_geotiff(outdir / "post_clear_valid_nocloud_noshadow_255.tif", post_clear_valid_255, profile, nodata=0)
    write_geotiff(outdir / "comparison_clear_valid_bothdates_255.tif", comparison_valid_255, profile, nodata=0)

    # SCL diagnostics when available
    if USE_SCL_IF_AVAILABLE:
        write_geotiff(outdir / "pre_scl_invalid_255.tif", pre_scl_invalid_255, profile, nodata=0)
        write_geotiff(outdir / "post_scl_invalid_255.tif", post_scl_invalid_255, profile, nodata=0)
        write_geotiff(outdir / "pre_scl_clear_valid_255.tif", pre_scl_clear_valid_255, profile, nodata=0)
        write_geotiff(outdir / "post_scl_clear_valid_255.tif", post_scl_clear_valid_255, profile, nodata=0)

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
    print(f"PRE NDVI thresholds:  [{PRE_NDVI_MIN}, {PRE_NDVI_MAX}]")
    print(f"POST NDVI thresholds: [{POST_NDVI_MIN}, {POST_NDVI_MAX}]")
    print(f"Using SCL for PRE clear mask:  {USE_SCL_IF_AVAILABLE and pre_scl_path is not None}")
    print(f"Using SCL for POST clear mask: {USE_SCL_IF_AVAILABLE and post_scl_path is not None}")
    print(f"SCL edge expand px: {SCL_EDGE_EXPAND_PX}")
    print(f"Invalidate unclassified near cloud: {SCL_INVALIDATE_UNCLASSIFIED_NEAR_CLOUD}")
    print(f"Invalidate bare near cloud: {SCL_INVALIDATE_BARE_NEAR_CLOUD}")


if __name__ == "__main__":
    main()
