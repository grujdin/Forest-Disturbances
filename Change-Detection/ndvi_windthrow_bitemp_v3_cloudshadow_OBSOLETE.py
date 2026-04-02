
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ndvi_windthrow_bitemp_v3_cloudshadow.py

Extends the original bitemporal sigNDVI workflow with cloud and cloud-shadow
removal modeled after the two EO Browser evalscripts provided by the user.

What changed
------------
1) Reads all bands needed by the evalscripts:
     B02, B03, B04, B08, B8A, B11, B12
2) Reproduces the cloud mask evalscript:
     output 1 = valid scene, 0 = cloud / outside valid footprint
3) Reproduces the shadow mask evalscript:
     output 1 = valid scene, 0 = shadow
4) Uses a combined clear-scene mask per date:
     clear = dataMask AND cloud_valid AND shadow_valid
5) Uses a comparison-valid mask for change detection:
     comparison_valid = pre_clear AND post_clear
   so cloudy / shadowed pixels do not become false loss / gain.

Inputs can be:
  - EO Browser exported ZIPs containing the required "Raw" bands
  - Directories containing the required band files

Usage (optional args; if omitted, edit PRE_DATASET/POST_DATASET below):
  python ndvi_windthrow_bitemp_v3_cloudshadow.py --pre PRE.zip --post POST.zip --outdir out

Dependencies:
  pip install rasterio numpy
"""

from __future__ import annotations

import argparse
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import sieve
from rasterio.warp import reproject

# -----------------------------
# HARD-CODED PARAMETERS (core)
# -----------------------------
NDVI_MIN = 0.64  # 0.73 (usually), 0.64 (polygon, 2 multi-polygons)
NDVI_MAX = 1.0

S4 = 0.02  # assumed uncertainty for B04
S8 = 0.03  # assumed uncertainty for B08

PRE_OPACITY = 0.55  # used only for the "grey intermediary" blend preview

# If your bands are DN scaled 0..10000, we auto-convert to reflectance (0..1)
AUTO_DN_SCALE = True
DN_SCALE = 10000.0  # Sentinel-2 L2A typical reflectance scaling

# Cloud / shadow evalscript thresholds
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
    vmax = np.nanpercentile(arrf, 99.9)
    if vmax > 1.5:
        arrf = arrf / DN_SCALE
    return arrf


def _is_raster_name(name: str) -> bool:
    return name.lower().endswith(RASTER_EXTS)


def _pick_name(names, tag: str) -> Optional[str]:
    tag_l = tag.lower()
    cands = [n for n in names if _is_raster_name(n) and tag_l in Path(n).name.lower()]
    if not cands:
        return None
    raw = [n for n in cands if "raw" in Path(n).name.lower()]
    return raw[0] if raw else cands[0]


def find_band_paths(dataset_path: Path) -> Dict[str, Path]:
    """Return dict like {'B02': path, ..., 'B12': path} for ZIP or directory."""
    if dataset_path.is_file() and dataset_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(dataset_path, "r") as z:
            names = z.namelist()
            members = {}
            for tag in REQUIRED_BANDS:
                picked = _pick_name(names, tag)
                if picked is None:
                    raise FileNotFoundError(f"Could not find {tag} inside {dataset_path.name}")
                members[tag] = picked

        tmpdir = Path(tempfile.mkdtemp(prefix="s2zip_"))
        out = {}
        with zipfile.ZipFile(dataset_path, "r") as z:
            for tag, member in members.items():
                ext = Path(member).suffix.lower()
                band_path = tmpdir / f"{tag}_raw{ext}"
                band_path.write_bytes(z.read(member))
                out[tag] = band_path
        return out

    if dataset_path.is_dir():
        files = [p for p in dataset_path.rglob("*") if p.is_file() and p.suffix.lower() in RASTER_EXTS]
        out = {}
        for tag in REQUIRED_BANDS:
            cands = [p for p in files if tag.lower() in p.name.lower()]
            if not cands:
                raise FileNotFoundError(f"Could not find {tag} in folder {dataset_path}")
            raw = [p for p in cands if "raw" in p.name.lower()]
            out[tag] = raw[0] if raw else cands[0]
        return out

    raise FileNotFoundError(f"Dataset path not found: {dataset_path}")


def _read_band(path: Path) -> Tuple[np.ndarray, dict, np.ndarray]:
    """Read first band and return (array, profile, valid_mask_from_dataset_mask)."""
    with rasterio.open(path) as ds:
        arr = ds.read(1)
        profile = ds.profile.copy()
        valid = ds.read_masks(1) > 0
    return arr, profile, valid


def _align_array_to(ref_profile: dict, src_path: Path, resampling: Resampling = Resampling.bilinear) -> np.ndarray:
    """Reproject/resample src raster values to match ref_profile grid."""
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
    """Reproject/resample the source valid-data mask to match ref_profile grid."""
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


def load_dataset(dataset_path: Path, ref_profile: Optional[dict] = None) -> Tuple[Dict[str, np.ndarray], np.ndarray, dict]:
    """
    Load all required bands as reflectance arrays on a common grid.

    Returns
    -------
    bands_ref : dict[str, np.ndarray]
        Reflectance arrays on the common grid.
    data_mask : np.ndarray[bool]
        Combined valid-data mask across all required bands.
    profile : dict
        Raster profile of the common grid.
    """
    paths = find_band_paths(dataset_path)

    if ref_profile is None:
        _, ref_profile, _ = _read_band(paths["B04"])

    bands = {}
    data_mask = np.ones((ref_profile["height"], ref_profile["width"]), dtype=bool)

    for tag, path in paths.items():
        same_grid = False
        if tag == "B04" and ref_profile is not None:
            _, native_profile, _ = _read_band(path)
            same_grid = (
                native_profile.get("height") == ref_profile.get("height")
                and native_profile.get("width") == ref_profile.get("width")
                and native_profile.get("transform") == ref_profile.get("transform")
                and native_profile.get("crs") == ref_profile.get("crs")
            )

        if same_grid:
            arr, _, valid = _read_band(path)
            arr = arr.astype(np.float32)
        else:
            arr = _align_array_to(ref_profile, path, resampling=Resampling.bilinear)
            valid = _align_valid_to(ref_profile, path)

        bands[tag] = _maybe_to_ref(arr)
        data_mask &= valid

    return bands, data_mask, ref_profile


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


def shadow_info(b4: np.ndarray, s4: float, b8: np.ndarray, s8: float) -> Tuple[np.ndarray, np.ndarray]:
    """Replicates the shadow evalscript helper, returning (darkness, sum)."""
    sum_ = b8 + b4
    s_ndvi = np.zeros_like(sum_, dtype=np.float32)
    valid = sum_ > 0
    if np.any(valid):
        s_ndvi[valid] = (
            2.0 / (sum_[valid] * sum_[valid]) *
            np.sqrt((b8[valid] * b8[valid] * s4 * s4) + (b4[valid] * b4[valid] * s8 * s8))
        )
    darkness = clamp01(1.0 - 2.0 * s_ndvi)
    return darkness.astype(np.float32), sum_.astype(np.float32)


def compute_signdvi_rgb_and_keep(b04: np.ndarray, b08: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implements the EO script:
      rgb = [0.9*clamp(1-ndvi)*darkness, 0.8*clamp(ndvi)*darkness, 0.1*darkness]
      keep = NDVI in [NDVI_MIN, NDVI_MAX]
    """
    sum_ = b08 + b04
    sum_safe = np.where(sum_ == 0, np.nan, sum_)

    ndvi = (b08 - b04) / sum_safe
    ndvi = np.nan_to_num(ndvi, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    s_ndvi = (2.0 / (sum_safe * sum_safe)) * np.sqrt((b08 * b08 * S4 * S4) + (b04 * b04 * S8 * S8))
    s_ndvi = np.nan_to_num(s_ndvi, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    darkness = clamp01(1.0 - 2.0 * s_ndvi)

    r = 0.9 * clamp01(1.0 - ndvi) * darkness
    g = 0.8 * clamp01(ndvi) * darkness
    b = 0.1 * darkness

    rgb = np.stack([r, g, b], axis=-1)
    rgb_u8 = (clamp01(rgb) * 255.0 + 0.5).astype(np.uint8)

    keep = (ndvi >= NDVI_MIN) & (ndvi <= NDVI_MAX)
    return rgb_u8, keep


def compute_cloud_valid_mask(bands: Dict[str, np.ndarray], data_mask: np.ndarray) -> np.ndarray:
    """
    Reproduces the uploaded cloud evalscript:
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

    water = (
        (b02 - b04 > 0.034) &
        (b8a < b04) &
        (b02 < 0.20)
    )

    snow = (
        (b03 > 0.20) &
        (b11 < 0.15) &
        (ratio(b04, b11, 999.0) > 4.0)
    )

    bcy_cloud = ((b03 > 0.175) & (ndgr > 0)) | (b03 > 0.39)

    sen2cor_like_cloud = (
        (vis > 0.12) &
        (b04 > 0.06) &
        (ndsi > -0.24) &
        (ndvi < 0.42) &
        (b2b11 > 0.70) &
        (b8b11 > 0.90) &
        (b4b11 < 6.0) &
        (~water)
    )

    cloud = (~snow) & (bcy_cloud | sen2cor_like_cloud)
    valid_scene = data_mask & (~cloud)
    return valid_scene


def compute_shadow_valid_mask(bands: Dict[str, np.ndarray], data_mask: np.ndarray) -> np.ndarray:
    """
    Reproduces the uploaded shadow evalscript:
      1 = valid scene
      0 = shadow
    Note: the original evalscript returns 1 outside the scene.
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

    water = (
        (b02 - b04 > 0.034) &
        (b8a < b04) &
        (b02 < 0.20)
    )

    snow = (
        (b03 > 0.20) &
        (b11 < 0.15) &
        (ratio(b04, b11, 999.0) > 4.0)
    )

    shadow_by_index = (
        (darkness < SHADOW_DARKNESS_MAX) &
        (sum_ < SHADOW_SUM_MAX)
    )

    dark_spec = (
        (b02 < 0.15) &
        (b03 < 0.13) &
        (b04 < 0.13) &
        (b8a < B8A_MAX) &
        (b11 < B11_MAX) &
        (b12 < B12_MAX) &
        (vis < VIS_MAX)
    )

    shadow = (
        (~water) &
        (~snow) &
        shadow_by_index &
        dark_spec
    )

    # Exact evalscript behavior: outside dataMask => 1
    valid_scene = (~shadow) | (~data_mask)
    return valid_scene


def write_geotiff(path: Path, arr: np.ndarray, profile: dict, nodata=None):
    """Write a GeoTIFF using the spatial reference of profile."""
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
    """
    Remove small connected components from a 0/255 mask.
    Uses rasterio.features.sieve on a 0/1 mask, then rescales to 0/255.
    """
    mask01 = (mask255 > 0).astype(np.uint8)
    sieved = sieve(mask01, size=min_px, connectivity=connectivity).astype(np.uint8)
    return (sieved * 255).astype(np.uint8)


def make_overlay_red50(mask255: np.ndarray) -> np.ndarray:
    """
    RGBA overlay: red at 50% where mask=255, transparent black elsewhere.
    """
    h, w = mask255.shape
    rgba = np.zeros((4, h, w), dtype=np.uint8)
    on = mask255 > 0
    rgba[0, on] = 255
    rgba[3, on] = 128
    return rgba


def alpha_blend_over_white(rgba_bottom: np.ndarray, rgba_top: np.ndarray, top_opacity: float = 1.0) -> np.ndarray:
    """
    Alpha blend two RGBA images (uint8) over white background.
    rgba_* expected shape: (4,H,W), values 0..255.
    Returns RGB uint8 (3,H,W).
    """
    bot_rgb = rgba_bottom[:3].astype(np.float32) / 255.0
    bot_a = rgba_bottom[3].astype(np.float32) / 255.0

    top_rgb = rgba_top[:3].astype(np.float32) / 255.0
    top_a = (rgba_top[3].astype(np.float32) / 255.0) * float(top_opacity)

    white = np.ones_like(bot_rgb)
    bot_over_white = bot_rgb * bot_a + white * (1.0 - bot_a)
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

    # Load all required bands; post is aligned to pre grid.
    pre_bands, pre_data_mask, profile = load_dataset(pre_path)
    post_bands, post_data_mask, _ = load_dataset(post_path, ref_profile=profile)

    # Cloud / shadow masks replicated from the uploaded evalscripts.
    pre_cloud_valid = compute_cloud_valid_mask(pre_bands, pre_data_mask)
    post_cloud_valid = compute_cloud_valid_mask(post_bands, post_data_mask)

    pre_shadow_valid = compute_shadow_valid_mask(pre_bands, pre_data_mask)
    post_shadow_valid = compute_shadow_valid_mask(post_bands, post_data_mask)

    pre_clear_valid = pre_data_mask & pre_cloud_valid & pre_shadow_valid
    post_clear_valid = post_data_mask & post_cloud_valid & post_shadow_valid

    # Only compare pixels that are clear in both dates.
    comparison_valid = pre_clear_valid & post_clear_valid

    # Core sigNDVI RGB + per-date keep masks.
    pre_rgb, pre_keep_ndvi = compute_signdvi_rgb_and_keep(pre_bands["B04"], pre_bands["B08"])
    post_rgb, post_keep_ndvi = compute_signdvi_rgb_and_keep(post_bands["B04"], post_bands["B08"])

    # Per-date visual keep: require that the date itself is clear.
    keep_pre = pre_keep_ndvi & pre_clear_valid
    keep_post = post_keep_ndvi & post_clear_valid

    # RGBA for visualisation
    pre_rgba = np.zeros((4, pre_rgb.shape[0], pre_rgb.shape[1]), dtype=np.uint8)
    post_rgba = np.zeros_like(pre_rgba)
    pre_rgba[:3] = np.transpose(pre_rgb, (2, 0, 1))
    post_rgba[:3] = np.transpose(post_rgb, (2, 0, 1))
    pre_rgba[3] = keep_pre.astype(np.uint8) * 255
    post_rgba[3] = keep_post.astype(np.uint8) * 255

    # Binary masks 0/255
    keep_pre_255 = pre_rgba[3].copy()
    keep_post_255 = post_rgba[3].copy()
    holes_pre_255 = (255 - keep_pre_255).astype(np.uint8)
    holes_post_255 = (255 - keep_post_255).astype(np.uint8)

    pre_cloud_valid_255 = pre_cloud_valid.astype(np.uint8) * 255
    post_cloud_valid_255 = post_cloud_valid.astype(np.uint8) * 255
    pre_shadow_valid_255 = pre_shadow_valid.astype(np.uint8) * 255
    post_shadow_valid_255 = post_shadow_valid.astype(np.uint8) * 255
    pre_clear_valid_255 = pre_clear_valid.astype(np.uint8) * 255
    post_clear_valid_255 = post_clear_valid.astype(np.uint8) * 255
    comparison_valid_255 = comparison_valid.astype(np.uint8) * 255

    # Change masks: only on pixels that are clear in both dates.
    loss_255 = (pre_keep_ndvi & (~post_keep_ndvi) & comparison_valid).astype(np.uint8) * 255
    gain_255 = ((~pre_keep_ndvi) & post_keep_ndvi & comparison_valid).astype(np.uint8) * 255
    stable_255 = (pre_keep_ndvi & post_keep_ndvi & comparison_valid).astype(np.uint8) * 255

    # Requested filtered masks
    loss_f25_255 = filter_min_patch(loss_255, min_px=25)
    loss_f38_255 = filter_min_patch(loss_255, min_px=38)
    loss_f50_255 = filter_min_patch(loss_255, min_px=50)

    # Requested overlay for min50
    loss_f50_overlay_red50 = make_overlay_red50(loss_f50_255)

    # Grey intermediary blend
    blend_rgb = alpha_blend_over_white(post_rgba, pre_rgba, top_opacity=PRE_OPACITY)

    # Visuals
    post_with_loss_grey = np.transpose(post_rgb, (2, 0, 1)).copy()
    on = loss_f50_255 > 0
    post_with_loss_grey[:, on] = 180
    post_with_loss_grey[:, ~post_clear_valid] = 255

    pre_with_loss_grey = np.transpose(pre_rgb, (2, 0, 1)).copy()
    pre_with_loss_grey[:, on] = 180

    pre_with_loss_grey_clean = np.transpose(pre_rgb, (2, 0, 1)).copy()
    pre_with_loss_grey_clean[:, keep_pre_255 == 0] = 255
    pre_with_loss_grey_clean[:, on] = 180

    # Write outputs
    write_geotiff(outdir / "keep_pre_255.tif", keep_pre_255.astype(np.uint8), profile, nodata=0)
    write_geotiff(outdir / "keep_post_255.tif", keep_post_255.astype(np.uint8), profile, nodata=0)
    write_geotiff(outdir / "holes_pre_notkept_255.tif", holes_pre_255, profile, nodata=0)
    write_geotiff(outdir / "holes_post_notkept_255.tif", holes_post_255, profile, nodata=0)

    write_geotiff(outdir / "pre_cloud_valid_255.tif", pre_cloud_valid_255, profile, nodata=0)
    write_geotiff(outdir / "post_cloud_valid_255.tif", post_cloud_valid_255, profile, nodata=0)
    write_geotiff(outdir / "pre_shadow_valid_255.tif", pre_shadow_valid_255, profile, nodata=0)
    write_geotiff(outdir / "post_shadow_valid_255.tif", post_shadow_valid_255, profile, nodata=0)
    write_geotiff(outdir / "pre_clear_valid_nocloud_noshadow_255.tif", pre_clear_valid_255, profile, nodata=0)
    write_geotiff(outdir / "post_clear_valid_nocloud_noshadow_255.tif", post_clear_valid_255, profile, nodata=0)
    write_geotiff(outdir / "comparison_clear_valid_bothdates_255.tif", comparison_valid_255, profile, nodata=0)

    write_geotiff(outdir / "loss_pre_kept_post_notkept_255.tif", loss_255.astype(np.uint8), profile, nodata=0)
    write_geotiff(outdir / "gain_pre_notkept_post_kept_255.tif", gain_255.astype(np.uint8), profile, nodata=0)
    write_geotiff(outdir / "stable_kept_both_255.tif", stable_255.astype(np.uint8), profile, nodata=0)

    write_geotiff(outdir / "loss_filtered_min25px_255.tif", loss_f25_255, profile, nodata=0)
    write_geotiff(outdir / "loss_filtered_min38px_255.tif", loss_f38_255, profile, nodata=0)
    write_geotiff(outdir / "loss_filtered_min50px_255.tif", loss_f50_255, profile, nodata=0)

    write_geotiff(outdir / "pre_rgba_signdvi.tif", pre_rgba, profile)
    write_geotiff(outdir / "post_rgba_signdvi.tif", post_rgba, profile)
    write_geotiff(outdir / "loss_filtered_min50_overlay_red50_rgba_blackoutside.tif", loss_f50_overlay_red50, profile)

    write_geotiff(outdir / "blend_pre055_over_post1_rgb_uint8.tif", blend_rgb, profile)
    write_geotiff(outdir / "post_with_loss_grey_min50_rgb_uint8.tif", post_with_loss_grey, profile)
    write_geotiff(outdir / "pre_with_loss_grey_min50_rgb_uint8.tif", pre_with_loss_grey, profile)
    write_geotiff(outdir / "pre_with_loss_grey_min50_clean_rgb_uint8.tif", pre_with_loss_grey_clean, profile)

    print("Done. Outputs written to:", outdir.resolve())


if __name__ == "__main__":
    main()
