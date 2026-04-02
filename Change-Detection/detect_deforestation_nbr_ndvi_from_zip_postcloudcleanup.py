#!/usr/bin/env python3
from __future__ import annotations

"""
detect_deforestation_nbr_ndvi_from_zip_postcloudcleanup.py

Threshold-based deforestation / non-vegetation detection for Sentinel-2 EO Browser
exports, using NBR as the main discriminator and NDVI as an optional extra filter.

This version keeps SCL out of the detection step itself:
  - NDVI/NBR masks are computed from the spectral bands only
  - SCL is NOT used to define the analysis-valid land mask
  - if SCL exists, it is used only as a very light *final cleanup* to remove
    obvious cloud / cloud-shadow / cirrus footprints from the output masks

Inputs can be:
  - EO Browser ZIP exports
  - folders containing the extracted rasters

Required per date:
  - B04
  - B08
  - B12
Optional per date:
  - SCL / Scene_classification_map

Main outputs
------------
<OUT_PREFIX>_post_nonvegetation_nbr_only_mask_1_nodata.tif
    1 where the POST date looks non-vegetated from NBR alone,
    computed without SCL in the detection step and then cleaned only from
    obvious POST-date cloud pixels if SCL is available.

<OUT_PREFIX>_post_nonvegetation_nbr_ndvi_mask_1_nodata.tif
    1 where the POST date looks non-vegetated from NBR and NDVI together,
    computed without SCL in the detection step and then cleaned only from
    obvious POST-date cloud pixels if SCL is available.

<OUT_PREFIX>_deforestation_mask_1_nodata.tif
    1 where the PRE date is vegetation-like and the POST date is non-vegetated,
    computed without SCL in the detection step and then cleaned only from
    obvious cloud pixels on the corresponding date(s) if SCL is available.
"""

import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import sieve
from rasterio.warp import reproject


# ============================================================
# USER SETTINGS — EDIT THESE VALUES, THEN RUN THE SCRIPT.
# Input can be either:
#   - an EO Browser ZIP export
#   - a folder containing the extracted rasters
# ============================================================
AUTUMN_DATASET = r"D:/Forest_Disturbance/imagery_zip/Comandau_BV/Comandau_2017_10_18.zip"
SPRING_DATASET = r"D:/Forest_Disturbance/imagery_zip/Comandau_BV/Comandau_2018_04_21.zip"
OUT_PREFIX = r"D:/Forest_Disturbance/imagery_zip/Comandau_BV/deforestation2"

# PRE date must still look vegetated / forest-like.
PRE_VEG_NDVI_MIN = 0.60
PRE_VEG_NBR_MIN = 0.50

# POST date non-vegetation threshold based mainly on NBR.
# Increase POST_NONVEG_NBR_MAX to detect more disturbed / open pixels.
# Decrease it to be more conservative.
POST_NONVEG_NBR_MAX = 0.35

# Optional NDVI cross-check on the POST date.
USE_POST_NDVI_FILTER = True
POST_NONVEG_NDVI_MAX = 0.50

# Remove small isolated detections.
MIN_PATCH_PX = 9

# Optional SCL use only AFTER detection, for minimal cleanup.
USE_SCL_POST_CLEANUP = True
CLEANUP_DILATE_PX = 0
CLEANUP_REMOVE_CLOUD_SHADOW = True
CLEANUP_REMOVE_CLOUD_MEDIUM = True
CLEANUP_REMOVE_CLOUD_HIGH = True
CLEANUP_REMOVE_THIN_CIRRUS = True
CLEANUP_REMOVE_SNOW = False

# Write masks before cloud cleanup too, for comparison.
WRITE_PRE_CLEANUP_OUTPUTS = True
# ============================================================


RASTER_EXTS = (".tif", ".tiff", ".jp2")
REQUIRED_TAGS = ("B04", "B08", "B12")
SCL_PATTERNS = ("scene_classification_map", "_scl", "scl")
SCL_COLOR_MATCH_TOL = 3

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


@dataclass(frozen=True)
class Settings:
    autumn_dataset: str
    spring_dataset: str
    out_prefix: str
    pre_veg_ndvi_min: float = 0.60
    pre_veg_nbr_min: float = 0.50
    post_nonveg_nbr_max: float = 0.35
    use_post_ndvi_filter: bool = True
    post_nonveg_ndvi_max: float = 0.50
    min_patch_px: int = 9
    use_scl_post_cleanup: bool = True
    cleanup_dilate_px: int = 0
    cleanup_remove_cloud_shadow: bool = True
    cleanup_remove_cloud_medium: bool = True
    cleanup_remove_cloud_high: bool = True
    cleanup_remove_thin_cirrus: bool = True
    cleanup_remove_snow: bool = False
    write_pre_cleanup_outputs: bool = True


class DatasetLocator:
    def __init__(self) -> None:
        self._tmpdirs: list[Path] = []

    def cleanup(self) -> None:
        for p in self._tmpdirs:
            shutil.rmtree(p, ignore_errors=True)
        self._tmpdirs.clear()

    def _make_tmpdir(self, prefix: str) -> Path:
        p = Path(tempfile.mkdtemp(prefix=prefix))
        self._tmpdirs.append(p)
        return p

    @staticmethod
    def _band_name_matches(path_name: str, tag: str) -> bool:
        stem = Path(path_name).stem.lower()
        tag_l = tag.lower()
        pattern = rf"(^|[^a-z0-9]){re.escape(tag_l)}([^a-z0-9]|$)"
        return re.search(pattern, stem) is not None

    def _pick_zip_member(self, names: Iterable[str], tag: str) -> Optional[str]:
        names = list(names)
        cands = [
            n
            for n in names
            if Path(n).suffix.lower() in RASTER_EXTS and self._band_name_matches(Path(n).name, tag)
        ]
        if not cands:
            return None
        raw = [n for n in cands if "raw" in Path(n).name.lower()]
        return raw[0] if raw else cands[0]

    def _pick_dir_file(self, files: Iterable[Path], tag: str) -> Optional[Path]:
        files = list(files)
        cands = [
            p
            for p in files
            if p.is_file() and p.suffix.lower() in RASTER_EXTS and self._band_name_matches(p.name, tag)
        ]
        if not cands:
            return None
        raw = [p for p in cands if "raw" in p.name.lower()]
        return raw[0] if raw else cands[0]

    @staticmethod
    def _pick_contains_from_names(names: Iterable[str], patterns: Iterable[str]) -> Optional[str]:
        pats = [p.lower() for p in patterns]
        cands = []
        for n in names:
            ln = Path(n).name.lower()
            if Path(n).suffix.lower() in RASTER_EXTS and any(p in ln for p in pats):
                cands.append(n)
        if not cands:
            return None
        exact_scene = [n for n in cands if "scene_classification_map" in Path(n).name.lower()]
        return exact_scene[0] if exact_scene else cands[0]

    @staticmethod
    def _pick_contains_from_files(files: Iterable[Path], patterns: Iterable[str]) -> Optional[Path]:
        pats = [p.lower() for p in patterns]
        cands = []
        for p in files:
            ln = p.name.lower()
            if p.is_file() and p.suffix.lower() in RASTER_EXTS and any(pt in ln for pt in pats):
                cands.append(p)
        if not cands:
            return None
        exact_scene = [p for p in cands if "scene_classification_map" in p.name.lower()]
        return exact_scene[0] if exact_scene else cands[0]

    def locate_dataset(self, dataset_path: Path) -> Dict[str, Path]:
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        if dataset_path.is_file() and dataset_path.suffix.lower() == ".zip":
            with zipfile.ZipFile(dataset_path, "r") as z:
                names = z.namelist()
                picked: Dict[str, str] = {}
                for tag in REQUIRED_TAGS:
                    member = self._pick_zip_member(names, tag)
                    if member is None:
                        raise FileNotFoundError(f"Could not find {tag} inside {dataset_path.name}")
                    picked[tag] = member
                scl_member = self._pick_contains_from_names(names, SCL_PATTERNS)
                if scl_member is not None:
                    picked["SCL"] = scl_member

            tmpdir = self._make_tmpdir("s2_defor_zip_")
            out: Dict[str, Path] = {}
            with zipfile.ZipFile(dataset_path, "r") as z:
                for tag, member in picked.items():
                    ext = Path(member).suffix.lower()
                    out_path = tmpdir / f"{tag}{ext}"
                    out_path.write_bytes(z.read(member))
                    out[tag] = out_path
            return out

        if dataset_path.is_dir():
            files = [p for p in dataset_path.rglob("*") if p.is_file() and p.suffix.lower() in RASTER_EXTS]
            out: Dict[str, Path] = {}
            for tag in REQUIRED_TAGS:
                picked = self._pick_dir_file(files, tag)
                if picked is None:
                    raise FileNotFoundError(f"Could not find {tag} in folder {dataset_path}")
                out[tag] = picked
            scl_path = self._pick_contains_from_files(files, SCL_PATTERNS)
            if scl_path is not None:
                out["SCL"] = scl_path
            return out

        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")


def get_settings() -> Settings:
    return Settings(
        autumn_dataset=AUTUMN_DATASET,
        spring_dataset=SPRING_DATASET,
        out_prefix=OUT_PREFIX,
        pre_veg_ndvi_min=PRE_VEG_NDVI_MIN,
        pre_veg_nbr_min=PRE_VEG_NBR_MIN,
        post_nonveg_nbr_max=POST_NONVEG_NBR_MAX,
        use_post_ndvi_filter=USE_POST_NDVI_FILTER,
        post_nonveg_ndvi_max=POST_NONVEG_NDVI_MAX,
        min_patch_px=MIN_PATCH_PX,
        use_scl_post_cleanup=USE_SCL_POST_CLEANUP,
        cleanup_dilate_px=CLEANUP_DILATE_PX,
        cleanup_remove_cloud_shadow=CLEANUP_REMOVE_CLOUD_SHADOW,
        cleanup_remove_cloud_medium=CLEANUP_REMOVE_CLOUD_MEDIUM,
        cleanup_remove_cloud_high=CLEANUP_REMOVE_CLOUD_HIGH,
        cleanup_remove_thin_cirrus=CLEANUP_REMOVE_THIN_CIRRUS,
        cleanup_remove_snow=CLEANUP_REMOVE_SNOW,
        write_pre_cleanup_outputs=WRITE_PRE_CLEANUP_OUTPUTS,
    )


def validate_settings(cfg: Settings) -> None:
    placeholders = {
        "/path/to/autumn_dataset.zip",
        "/path/to/spring_dataset.zip",
        "/path/to/out/deforestation",
    }

    values = {
        "AUTUMN_DATASET": cfg.autumn_dataset,
        "SPRING_DATASET": cfg.spring_dataset,
        "OUT_PREFIX": cfg.out_prefix,
    }

    for name, value in values.items():
        if not value or value.strip() in placeholders:
            raise ValueError(f"{name} is still a placeholder. Edit the USER SETTINGS block at the top of the script.")

    for name, value in {
        "AUTUMN_DATASET": cfg.autumn_dataset,
        "SPRING_DATASET": cfg.spring_dataset,
    }.items():
        if not Path(value).exists():
            raise FileNotFoundError(f"{name} does not exist: {value}")

    if cfg.min_patch_px < 1:
        raise ValueError("MIN_PATCH_PX must be >= 1")
    if cfg.cleanup_dilate_px < 0:
        raise ValueError("CLEANUP_DILATE_PX must be >= 0")


def read_profile(path: str | Path):
    with rasterio.open(path) as src:
        return src.profile.copy(), src.transform, src.crs, src.width, src.height


def read_resampled(path: str | Path, ref_profile, resampling, dst_dtype=np.float32, dst_nodata=np.nan):
    dst = np.full((ref_profile["height"], ref_profile["width"]), dst_nodata, dtype=dst_dtype)
    with rasterio.open(path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=ref_profile["transform"],
            dst_crs=ref_profile["crs"],
            dst_nodata=dst_nodata,
            resampling=resampling,
        )
    return dst


def safe_index(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32, copy=False)
    b = b.astype(np.float32, copy=False)
    den = a + b
    out = np.full(a.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(a) & np.isfinite(b) & (np.abs(den) > 1e-6)
    out[valid] = (a[valid] - b[valid]) / den[valid]
    return out


def binary_dilate(mask: np.ndarray, radius_px: int) -> np.ndarray:
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


def _decode_rendered_scl_classes(rgb: np.ndarray) -> np.ndarray:
    vmax = float(np.nanmax(rgb)) if rgb.size else 0.0
    use_u16 = vmax > 255.5

    palette = {}
    for name, cid in SCL_CLASS_BY_NAME.items():
        color = np.array(SCL_COLORS_U8[name], dtype=np.int32)
        if use_u16:
            color = color * 257
        palette[cid] = color

    scl = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    tol = int(SCL_COLOR_MATCH_TOL)
    rgb_i = rgb.astype(np.int32)

    for cid, color in palette.items():
        match = np.all(np.abs(rgb_i - color) <= tol, axis=2)
        scl[match] = cid

    return scl


def load_scl_classes(path: str | Path, ref_profile) -> np.ndarray:
    with rasterio.open(path) as src:
        if src.count == 1:
            arr = read_resampled(path, ref_profile, Resampling.nearest, dst_dtype=np.uint8, dst_nodata=0)
            return arr.astype(np.uint8)

        rgb = np.zeros((3, ref_profile["height"], ref_profile["width"]), dtype=np.float32)
        for i, band_i in enumerate([1, 2, 3]):
            reproject(
                source=rasterio.band(src, band_i),
                destination=rgb[i],
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=ref_profile["transform"],
                dst_crs=ref_profile["crs"],
                dst_nodata=0,
                resampling=Resampling.nearest,
            )
    return _decode_rendered_scl_classes(rgb.transpose(1, 2, 0))


def build_scl_cloud_cleanup_mask(scl: np.ndarray, cfg: Settings) -> np.ndarray:
    cleanup = np.zeros_like(scl, dtype=bool)

    if cfg.cleanup_remove_cloud_shadow:
        cleanup |= scl == SCL_CLASS_BY_NAME["cloud_shadow"]
    if cfg.cleanup_remove_cloud_medium:
        cleanup |= scl == SCL_CLASS_BY_NAME["cloud_medium"]
    if cfg.cleanup_remove_cloud_high:
        cleanup |= scl == SCL_CLASS_BY_NAME["cloud_high"]
    if cfg.cleanup_remove_thin_cirrus:
        cleanup |= scl == SCL_CLASS_BY_NAME["thin_cirrus"]
    if cfg.cleanup_remove_snow:
        cleanup |= scl == SCL_CLASS_BY_NAME["snow"]

    if cfg.cleanup_dilate_px > 0:
        cleanup = binary_dilate(cleanup, cfg.cleanup_dilate_px)

    return cleanup


def write_binary_nodata_mask(path: str | Path, mask: np.ndarray, ref_profile) -> None:
    out = np.full(mask.shape, 255, dtype=np.uint8)
    out[mask] = 1

    profile = ref_profile.copy()
    profile.update(
        driver="GTiff",
        count=1,
        dtype="uint8",
        nodata=255,
        compress="deflate",
    )

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out, 1)


def write_float32(path: str | Path, arr: np.ndarray, valid: np.ndarray, ref_profile) -> None:
    out = np.full(arr.shape, np.nan, dtype=np.float32)
    out[valid] = arr[valid].astype(np.float32)

    profile = ref_profile.copy()
    profile.update(
        driver="GTiff",
        count=1,
        dtype="float32",
        nodata=np.nan,
        compress="deflate",
    )

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out, 1)


def sieve_bool(mask: np.ndarray, min_patch_px: int) -> np.ndarray:
    if min_patch_px <= 1:
        return mask
    cleaned = sieve(mask.astype(np.uint8), size=min_patch_px, connectivity=8).astype(bool)
    return cleaned


def mask_count(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))


def main() -> None:
    cfg = get_settings()
    validate_settings(cfg)

    out_prefix = Path(cfg.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    locator = DatasetLocator()
    try:
        autumn_files = locator.locate_dataset(Path(cfg.autumn_dataset))
        spring_files = locator.locate_dataset(Path(cfg.spring_dataset))

        ref_path = autumn_files.get("SCL", autumn_files["B12"])
        ref_profile, _, _, _, _ = read_profile(ref_path)

        b04_a = read_resampled(autumn_files["B04"], ref_profile, Resampling.bilinear)
        b08_a = read_resampled(autumn_files["B08"], ref_profile, Resampling.bilinear)
        b12_a = read_resampled(autumn_files["B12"], ref_profile, Resampling.bilinear)

        b04_s = read_resampled(spring_files["B04"], ref_profile, Resampling.bilinear)
        b08_s = read_resampled(spring_files["B08"], ref_profile, Resampling.bilinear)
        b12_s = read_resampled(spring_files["B12"], ref_profile, Resampling.bilinear)

        scl_a = load_scl_classes(autumn_files["SCL"], ref_profile) if "SCL" in autumn_files else None
        scl_s = load_scl_classes(spring_files["SCL"], ref_profile) if "SCL" in spring_files else None

        ndvi_a = safe_index(b08_a, b04_a)
        ndvi_s = safe_index(b08_s, b04_s)
        nbr_a = safe_index(b08_a, b12_a)
        nbr_s = safe_index(b08_s, b12_s)
        dnbr = nbr_s - nbr_a

        spectral_valid_a = np.isfinite(ndvi_a) & np.isfinite(nbr_a)
        spectral_valid_s = np.isfinite(ndvi_s) & np.isfinite(nbr_s)
        spectral_valid_both = spectral_valid_a & spectral_valid_s

        if mask_count(spectral_valid_both) < 100:
            raise RuntimeError("Too few valid pixels after index computation.")

        # Detection step: no SCL restriction at all.
        pre_veg_like_raw = spectral_valid_a & (ndvi_a >= cfg.pre_veg_ndvi_min) & (nbr_a >= cfg.pre_veg_nbr_min)
        post_nonveg_nbr_only_raw = spectral_valid_s & (nbr_s <= cfg.post_nonveg_nbr_max)
        post_nonveg_nbr_ndvi_raw = post_nonveg_nbr_only_raw.copy()
        if cfg.use_post_ndvi_filter:
            post_nonveg_nbr_ndvi_raw &= ndvi_s <= cfg.post_nonveg_ndvi_max
        deforestation_raw = pre_veg_like_raw & post_nonveg_nbr_ndvi_raw & spectral_valid_both

        # Optional SCL use only as the very last cleanup step.
        cleanup_invalid_a = np.zeros_like(spectral_valid_a, dtype=bool)
        cleanup_invalid_s = np.zeros_like(spectral_valid_s, dtype=bool)
        if cfg.use_scl_post_cleanup and scl_a is not None:
            cleanup_invalid_a = build_scl_cloud_cleanup_mask(scl_a, cfg)
        if cfg.use_scl_post_cleanup and scl_s is not None:
            cleanup_invalid_s = build_scl_cloud_cleanup_mask(scl_s, cfg)

        pre_veg_like = pre_veg_like_raw & (~cleanup_invalid_a) & spectral_valid_a
        post_nonveg_nbr_only = post_nonveg_nbr_only_raw & (~cleanup_invalid_s) & spectral_valid_s
        post_nonveg_nbr_ndvi = post_nonveg_nbr_ndvi_raw & (~cleanup_invalid_s) & spectral_valid_s
        deforestation = deforestation_raw & (~cleanup_invalid_a) & (~cleanup_invalid_s) & spectral_valid_both

        # Patch filtering after cleanup.
        pre_veg_like = sieve_bool(pre_veg_like, cfg.min_patch_px) & spectral_valid_a
        post_nonveg_nbr_only = sieve_bool(post_nonveg_nbr_only, cfg.min_patch_px) & spectral_valid_s
        post_nonveg_nbr_ndvi = sieve_bool(post_nonveg_nbr_ndvi, cfg.min_patch_px) & spectral_valid_s
        deforestation = sieve_bool(deforestation, cfg.min_patch_px) & spectral_valid_both

        pre_veg_like_raw_clean = sieve_bool(pre_veg_like_raw, cfg.min_patch_px) & spectral_valid_a
        post_nonveg_nbr_only_raw_clean = sieve_bool(post_nonveg_nbr_only_raw, cfg.min_patch_px) & spectral_valid_s
        post_nonveg_nbr_ndvi_raw_clean = sieve_bool(post_nonveg_nbr_ndvi_raw, cfg.min_patch_px) & spectral_valid_s
        deforestation_raw_clean = sieve_bool(deforestation_raw, cfg.min_patch_px) & spectral_valid_both

        # Main outputs: no-SCL detection + last-step cloud cleanup.
        write_binary_nodata_mask(f"{out_prefix}_post_nonvegetation_nbr_only_mask_1_nodata.tif", post_nonveg_nbr_only, ref_profile)
        write_binary_nodata_mask(f"{out_prefix}_post_nonvegetation_nbr_ndvi_mask_1_nodata.tif", post_nonveg_nbr_ndvi, ref_profile)
        write_binary_nodata_mask(f"{out_prefix}_pre_vegetation_like_mask_1_nodata.tif", pre_veg_like, ref_profile)
        write_binary_nodata_mask(f"{out_prefix}_deforestation_mask_1_nodata.tif", deforestation, ref_profile)

        # Diagnostics.
        write_binary_nodata_mask(f"{out_prefix}_autumn_spectral_valid_mask_1_nodata.tif", spectral_valid_a, ref_profile)
        write_binary_nodata_mask(f"{out_prefix}_spring_spectral_valid_mask_1_nodata.tif", spectral_valid_s, ref_profile)
        write_binary_nodata_mask(f"{out_prefix}_spectral_valid_both_dates_mask_1_nodata.tif", spectral_valid_both, ref_profile)
        write_binary_nodata_mask(f"{out_prefix}_autumn_cloud_cleanup_mask_1_nodata.tif", cleanup_invalid_a, ref_profile)
        write_binary_nodata_mask(f"{out_prefix}_spring_cloud_cleanup_mask_1_nodata.tif", cleanup_invalid_s, ref_profile)

        if cfg.write_pre_cleanup_outputs:
            write_binary_nodata_mask(
                f"{out_prefix}_post_nonvegetation_nbr_only_before_cleanup_mask_1_nodata.tif",
                post_nonveg_nbr_only_raw_clean,
                ref_profile,
            )
            write_binary_nodata_mask(
                f"{out_prefix}_post_nonvegetation_nbr_ndvi_before_cleanup_mask_1_nodata.tif",
                post_nonveg_nbr_ndvi_raw_clean,
                ref_profile,
            )
            write_binary_nodata_mask(
                f"{out_prefix}_pre_vegetation_like_before_cleanup_mask_1_nodata.tif",
                pre_veg_like_raw_clean,
                ref_profile,
            )
            write_binary_nodata_mask(
                f"{out_prefix}_deforestation_before_cleanup_mask_1_nodata.tif",
                deforestation_raw_clean,
                ref_profile,
            )

        write_float32(f"{out_prefix}_ndvi_autumn.tif", ndvi_a, spectral_valid_a, ref_profile)
        write_float32(f"{out_prefix}_ndvi_spring.tif", ndvi_s, spectral_valid_s, ref_profile)
        write_float32(f"{out_prefix}_nbr_autumn.tif", nbr_a, spectral_valid_a, ref_profile)
        write_float32(f"{out_prefix}_nbr_spring.tif", nbr_s, spectral_valid_s, ref_profile)
        write_float32(f"{out_prefix}_dnbr_spring_minus_autumn.tif", dnbr, spectral_valid_both, ref_profile)

        valid_count = mask_count(spectral_valid_both)
        pre_veg_count = mask_count(pre_veg_like)
        post_nonveg_nbr_only_count = mask_count(post_nonveg_nbr_only)
        post_nonveg_nbr_ndvi_count = mask_count(post_nonveg_nbr_ndvi)
        defor_count = mask_count(deforestation)

        print("Done.")
        print(f"Autumn dataset:      {cfg.autumn_dataset}")
        print(f"Spring dataset:      {cfg.spring_dataset}")
        print(f"Autumn B04:          {autumn_files['B04'].name}")
        print(f"Autumn B08:          {autumn_files['B08'].name}")
        print(f"Autumn B12:          {autumn_files['B12'].name}")
        print(f"Autumn SCL:          {autumn_files['SCL'].name if 'SCL' in autumn_files else 'not found / not used'}")
        print(f"Spring B04:          {spring_files['B04'].name}")
        print(f"Spring B08:          {spring_files['B08'].name}")
        print(f"Spring B12:          {spring_files['B12'].name}")
        print(f"Spring SCL:          {spring_files['SCL'].name if 'SCL' in spring_files else 'not found / not used'}")
        print()
        print(f"Spectral-valid pixels on both dates:        {valid_count}")
        print(f"PRE vegetation-like pixels:                 {pre_veg_count} ({100.0 * pre_veg_count / valid_count:.2f}% of spectral-valid)")
        print(f"POST non-veg (NBR only):                    {post_nonveg_nbr_only_count} ({100.0 * post_nonveg_nbr_only_count / valid_count:.2f}% of spectral-valid)")
        print(f"POST non-veg (NBR+NDVI):                    {post_nonveg_nbr_ndvi_count} ({100.0 * post_nonveg_nbr_ndvi_count / valid_count:.2f}% of spectral-valid)")
        print(f"Deforestation pixels:                       {defor_count} ({100.0 * defor_count / valid_count:.2f}% of spectral-valid)")
        print()
        print("Cloud-cleanup counts:")
        print(f"  Autumn cleanup pixels:                    {mask_count(cleanup_invalid_a)}")
        print(f"  Spring cleanup pixels:                    {mask_count(cleanup_invalid_s)}")
        if cfg.write_pre_cleanup_outputs:
            print(f"  POST non-veg NBR+NDVI before cleanup:     {mask_count(post_nonveg_nbr_ndvi_raw_clean)}")
            print(f"  Deforestation before cleanup:             {mask_count(deforestation_raw_clean)}")
        print()
        print("Used thresholds:")
        print(f"  PRE_VEG_NDVI_MIN:                         {cfg.pre_veg_ndvi_min:.2f}")
        print(f"  PRE_VEG_NBR_MIN:                          {cfg.pre_veg_nbr_min:.2f}")
        print(f"  POST_NONVEG_NBR_MAX:                      {cfg.post_nonveg_nbr_max:.2f}")
        print(f"  USE_POST_NDVI_FILTER:                     {cfg.use_post_ndvi_filter}")
        if cfg.use_post_ndvi_filter:
            print(f"  POST_NONVEG_NDVI_MAX:                     {cfg.post_nonveg_ndvi_max:.2f}")
        print(f"  MIN_PATCH_PX:                             {cfg.min_patch_px}")
        print(f"  USE_SCL_POST_CLEANUP:                     {cfg.use_scl_post_cleanup}")
        print(f"  CLEANUP_DILATE_PX:                        {cfg.cleanup_dilate_px}")
        print(f"  CLEANUP_REMOVE_CLOUD_SHADOW:              {cfg.cleanup_remove_cloud_shadow}")
        print(f"  CLEANUP_REMOVE_CLOUD_MEDIUM:              {cfg.cleanup_remove_cloud_medium}")
        print(f"  CLEANUP_REMOVE_CLOUD_HIGH:                {cfg.cleanup_remove_cloud_high}")
        print(f"  CLEANUP_REMOVE_THIN_CIRRUS:               {cfg.cleanup_remove_thin_cirrus}")
        print(f"  CLEANUP_REMOVE_SNOW:                      {cfg.cleanup_remove_snow}")
    finally:
        locator.cleanup()


if __name__ == "__main__":
    main()
