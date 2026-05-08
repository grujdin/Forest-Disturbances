"""
Algorithm #5. Construct EO predictor vectors for disturbance objects.

Purpose
-------
This script builds the object-level EO predictor vector described in the article:

    x = [x_EO, x_delta, x_shape, x_topo, m]

where:
- x_EO     : current optical/SAR state summaries over the object
- x_delta  : pre/post temporal deltas and rate-normalized deltas
- x_shape  : geometric and shape predictors
- x_topo   : DEM-derived elevation, slope, aspect, and exposure predictors
- m        : missingness, valid-pixel, date-support, and sensor-availability indicators

It is designed to be run after Algorithm #3 and Algorithm #4. It reuses:
- Algorithm #4 object_features.csv / object_features_all_intervals.csv
- Algorithm #4 object-id rasters
- Phase 3 Sentinel-2 index-stack cache
- Phase 5 Sentinel-1 descriptor cache, if available
- a zipped Copernicus DEM raster

Hardcoded by design, following the previous pipeline scripts.
"""
from __future__ import annotations

import ast
import json
import math
import re
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import rasterio.windows
from rasterio.enums import Resampling
from rasterio.transform import array_bounds
from rasterio.warp import calculate_default_transform, reproject, transform_bounds
from pyproj import CRS, Transformer
from shapely.geometry import shape
from shapely.ops import transform as shapely_transform, unary_union

try:
    import geopandas as gpd
except Exception:
    gpd = None

# =============================================================================
# HARD-CODED CONFIG
# =============================================================================

# Latest Phase 5 v4 products.
ALGORITHM4_ROOT = Path("D:/Forest_Disturbance/outputs/rdf_loss_objects_graphdb_annual_s1s2_blacklist_and_caution_eligible_phase5")
ALGORITHM3_ROOT = Path("D:/Forest_Disturbance/outputs/semantic_change_detection_s1s2_blacklist_and_caution_eligible")
SEMANTIC_MASK_ROOT = Path("D:/Forest_Disturbance/outputs/semantic_masks_s1s2_empirical_envelopes_blacklist_and_caution_eligible")
S2_CACHE_ROOT = Path("D:/Forest_Disturbance/outputs/sdv_phase3_preprocessing_cache")
S1_CACHE_ROOT = Path("D:/Forest_Disturbance/outputs/sdv_phase5_sentinel1_descriptor_cache")

DEM_ZIP = Path("D:/Forest_Disturbance/imagery_zip/SdV_DEM_Copernicus_30.zip")

OUTPUT_ROOT = Path("D:/Forest_Disturbance/outputs/eo_predictor_vectors_s1s2_topo_geom_algorithm5")

# Common projected CRS for geometry/topography, matching the article and previous scripts.
PROJECTED_CRS = "EPSG:32634"  # WGS 84 / UTM Zone 34N
DEM_PROJECTED_RESOLUTION_M = 30.0
FORCE_REBUILD_TOPO_CACHE = False

# Algorithm #4 interval raster names.
RASTER_SUBDIR = "rasters"
RAW_OBJECT_ID_NAME = "raw_loss_object_id.tif"
CORE_OBJECT_ID_NAME = "strict_core_object_id.tif"
FRINGE_OBJECT_ID_NAME = "fringe_object_id.tif"
FINAL_STATUS_CODE_NAME = "final_status_code.tif"
RETAINED_OBJECT_ID_NAME = "retained_feasible_object_id.tif"

# Sentinel-2 cached index-stack bands.
S2_INDEX_BANDS = ("NDVI", "NDMI", "NBR", "NDRE", "NDSI")
S2_CACHE_FLOAT_NODATA = -9999.0

# Sentinel-1 descriptors. Use auto discovery from inventory when possible.
S1_DESCRIPTOR_EXCLUDE = {"S1_VALID"}
S1_MAX_MATCH_DAYS = 45
S1_CACHE_FLOAT_NODATA = -9999.0

# Topographic aspect exposure azimuths from Appendix D: 30° and 120°.
EXPOSURE_AZIMUTHS_DEG = (30.0, 120.0)

# Output controls.
WRITE_PER_INTERVAL_OUTPUTS = True
WRITE_GPKG = True
WRITE_FEATURE_DICTIONARY = True
MAKE_TOPO_CACHE_QUICKLOOKS = False

DATE_RE = re.compile(r"(20\d{2})[-_](\d{2})[-_](\d{2})")

# =============================================================================
# Logging and generic helpers
# =============================================================================

def log(msg: str) -> None:
    from datetime import datetime
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def safe_float(x, default=np.nan) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def interval_label(pre_date: str, post_date: str) -> str:
    return f"{pre_date}_to_{post_date}"


def interval_id(pre_date: str, post_date: str) -> str:
    return f"{pre_date.replace('-', '_')}_{post_date.replace('-', '_')}"


def parse_date(value) -> pd.Timestamp:
    if pd.isna(value):
        return pd.NaT
    return pd.Timestamp(str(value)[:10])


def same_grid(profile_a: dict, profile_b: dict) -> bool:
    return (
        profile_a.get("height") == profile_b.get("height")
        and profile_a.get("width") == profile_b.get("width")
        and profile_a.get("transform") == profile_b.get("transform")
        and str(profile_a.get("crs")) == str(profile_b.get("crs"))
    )


def mask_to_grid(mask: np.ndarray, src_profile: dict, dst_profile: dict) -> np.ndarray:
    """Reproject a boolean mask from src_profile to dst_profile using nearest neighbour."""
    if same_grid(src_profile, dst_profile):
        return mask.astype(bool)
    dst = np.zeros((dst_profile["height"], dst_profile["width"]), dtype=np.uint8)
    reproject(
        source=mask.astype(np.uint8),
        destination=dst,
        src_transform=src_profile["transform"],
        src_crs=src_profile["crs"],
        dst_transform=dst_profile["transform"],
        dst_crs=dst_profile["crs"],
        resampling=Resampling.nearest,
        src_nodata=0,
        dst_nodata=0,
    )
    return dst.astype(bool)


def read_single(path: Path) -> Tuple[np.ndarray, dict]:
    with rasterio.open(path) as ds:
        arr = ds.read(1)
        profile = ds.profile.copy()
    return arr, profile


def read_float_stack(path: Path, requested: Optional[Sequence[str]] = None, nodata: float = -9999.0) -> Tuple[Dict[str, np.ndarray], dict]:
    """Read a named Float32 stack written by the previous cache scripts."""
    with rasterio.open(path) as ds:
        profile = ds.profile.copy()
        desc = list(ds.descriptions)
        names = [d if d else f"band_{i}" for i, d in enumerate(desc, start=1)]
        if requested is None:
            requested = names
        name_to_band = {str(n): i for i, n in enumerate(names, start=1)}
        out: Dict[str, np.ndarray] = {}
        for name in requested:
            if name not in name_to_band:
                # Fallback for old/no-description S2 stacks.
                if name in S2_INDEX_BANDS and len(names) >= S2_INDEX_BANDS.index(name) + 1:
                    band_i = S2_INDEX_BANDS.index(name) + 1
                else:
                    continue
            else:
                band_i = name_to_band[name]
            arr = ds.read(band_i).astype(np.float32)
            nd = ds.nodata if ds.nodata is not None else nodata
            arr[np.isclose(arr, float(nd))] = np.nan
            out[str(name)] = arr
    return out, profile


def zonal_stats(arr: np.ndarray, mask: np.ndarray, prefix: str) -> Dict[str, float]:
    """Robust object-level statistics for a raster array."""
    vals = arr[mask]
    finite = vals[np.isfinite(vals)]
    total = int(mask.sum())
    if total == 0:
        return {
            f"{prefix}_mean": np.nan, f"{prefix}_median": np.nan, f"{prefix}_std": np.nan,
            f"{prefix}_q10": np.nan, f"{prefix}_q90": np.nan, f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan, f"{prefix}_valid_px": 0, f"{prefix}_valid_frac": np.nan,
        }
    if finite.size == 0:
        return {
            f"{prefix}_mean": np.nan, f"{prefix}_median": np.nan, f"{prefix}_std": np.nan,
            f"{prefix}_q10": np.nan, f"{prefix}_q90": np.nan, f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan, f"{prefix}_valid_px": 0, f"{prefix}_valid_frac": 0.0,
        }
    return {
        f"{prefix}_mean": float(np.nanmean(finite)),
        f"{prefix}_median": float(np.nanmedian(finite)),
        f"{prefix}_std": float(np.nanstd(finite)),
        f"{prefix}_q10": float(np.nanquantile(finite, 0.10)),
        f"{prefix}_q90": float(np.nanquantile(finite, 0.90)),
        f"{prefix}_min": float(np.nanmin(finite)),
        f"{prefix}_max": float(np.nanmax(finite)),
        f"{prefix}_valid_px": int(finite.size),
        f"{prefix}_valid_frac": float(finite.size / total),
    }


def binary_fraction(mask: np.ndarray, condition: np.ndarray, name: str) -> Dict[str, float]:
    n = int(mask.sum())
    if n == 0:
        return {name: np.nan}
    return {name: float(np.sum(mask & condition) / n)}

# =============================================================================
# DEM / topography cache
# =============================================================================

def find_dem_member(dem_zip: Path) -> str:
    if dem_zip.suffix.lower() not in {".zip"}:
        return ""
    with zipfile.ZipFile(dem_zip, "r") as zf:
        candidates = [n for n in zf.namelist() if n.lower().endswith((".tif", ".tiff")) and not n.lower().endswith((".ovr", ".aux.xml"))]
    if not candidates:
        raise FileNotFoundError(f"No GeoTIFF DEM member found inside {dem_zip}")
    # Prefer names that contain DEM, Copernicus, or elevation; otherwise first TIFF.
    ranked = sorted(candidates, key=lambda n: (0 if any(k in n.lower() for k in ["dem", "copernicus", "elev", "height"]) else 1, n))
    return ranked[0]


def dem_vsi_path(dem_zip: Path) -> str:
    if dem_zip.suffix.lower() == ".zip":
        member = find_dem_member(dem_zip)
        return f"/vsizip/{dem_zip}/{member}"
    return str(dem_zip)


def bounded_dem_window(src: rasterio.io.DatasetReader, ref_profile: dict, pad_pixels: int = 5) -> rasterio.windows.Window:
    """Window the DEM to the semantic/object raster extent, with a small padding."""
    ref_crs = CRS.from_user_input(ref_profile["crs"])
    src_crs = CRS.from_user_input(src.crs)
    bounds = array_bounds(ref_profile["height"], ref_profile["width"], ref_profile["transform"])
    if str(ref_crs) != str(src_crs):
        bounds = transform_bounds(ref_crs, src_crs, *bounds, densify_pts=21)
    win = rasterio.windows.from_bounds(*bounds, transform=src.transform)
    win = win.round_offsets().round_lengths()
    win = rasterio.windows.Window(
        win.col_off - pad_pixels,
        win.row_off - pad_pixels,
        win.width + 2 * pad_pixels,
        win.height + 2 * pad_pixels,
    )
    full = rasterio.windows.Window(0, 0, src.width, src.height)
    try:
        win = win.intersection(full)
    except Exception:
        win = full
    return win


def shift_nan(arr: np.ndarray, dr: int, dc: int) -> np.ndarray:
    out = np.full(arr.shape, np.nan, dtype=np.float32)
    h, w = arr.shape
    r_src0 = max(0, -dr)
    r_src1 = min(h, h - dr)
    c_src0 = max(0, -dc)
    c_src1 = min(w, w - dc)
    r_dst0 = max(0, dr)
    r_dst1 = min(h, h + dr)
    c_dst0 = max(0, dc)
    c_dst1 = min(w, w + dc)
    if r_src1 > r_src0 and c_src1 > c_src0:
        out[r_dst0:r_dst1, c_dst0:c_dst1] = arr[r_src0:r_src1, c_src0:c_src1]
    return out


def nanmean_filter3(arr: np.ndarray) -> np.ndarray:
    """3 × 3 nan-aware mean without RuntimeWarning on all-nodata windows.

    np.nanmean emits a warning for pixels whose full 3 × 3 neighbourhood is
    NaN. Those pixels occur normally along DEM nodata margins or outside the
    DEM/analysis overlap. For topographic predictors, the correct output for
    such cells is simply NaN, not a warning.
    """
    stack = np.stack([shift_nan(arr, dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)], axis=0)
    finite = np.isfinite(stack)
    count = finite.sum(axis=0).astype(np.float32)
    total = np.where(finite, stack, 0.0).sum(axis=0, dtype=np.float32)
    out = np.full(arr.shape, np.nan, dtype=np.float32)
    np.divide(total, count, out=out, where=count > 0)
    return out.astype(np.float32)


def nanstd_filter3(arr: np.ndarray) -> np.ndarray:
    """3 × 3 nan-aware population std without warnings on all-nodata windows."""
    stack = np.stack([shift_nan(arr, dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)], axis=0)
    finite = np.isfinite(stack)
    count = finite.sum(axis=0).astype(np.float32)

    total = np.where(finite, stack, 0.0).sum(axis=0, dtype=np.float32)
    mean = np.full(arr.shape, np.nan, dtype=np.float32)
    np.divide(total, count, out=mean, where=count > 0)

    diff2 = np.where(finite, (stack - mean[None, :, :]) ** 2, 0.0).sum(axis=0, dtype=np.float32)
    var = np.full(arr.shape, np.nan, dtype=np.float32)
    np.divide(diff2, count, out=var, where=count > 0)
    return np.sqrt(var).astype(np.float32)


def derive_slope_aspect(dem: np.ndarray, transform) -> Tuple[np.ndarray, np.ndarray]:
    """Compute slope and GIS-like downslope aspect from a projected DEM grid."""
    xres = abs(float(transform.a))
    yres = abs(float(transform.e))
    valid = np.isfinite(dem)
    # Fill NaNs locally enough for gradient calculation, then mask them back.
    filled = dem.copy().astype(np.float32)
    if np.any(~valid):
        mean_val = float(np.nanmean(filled)) if np.any(valid) else 0.0
        filled[~valid] = mean_val
    dz_drow_south, dz_dx_east = np.gradient(filled, yres, xres)
    slope_rad = np.arctan(np.sqrt(dz_dx_east ** 2 + dz_drow_south ** 2))
    slope_deg = np.degrees(slope_rad).astype(np.float32)
    # Downslope vector = (-dz/dx_east, +dz/drow_south). Azimuth clockwise from north.
    aspect_rad = np.arctan2(-dz_dx_east, dz_drow_south)
    aspect_deg = (np.degrees(aspect_rad) + 360.0) % 360.0
    aspect_deg = aspect_deg.astype(np.float32)
    # Exclude invalid and nearly flat cells from aspect.
    slope_deg[~valid] = np.nan
    aspect_deg[~valid | (slope_deg < 0.5)] = np.nan
    return slope_deg, aspect_deg


def write_float_raster(path: Path, arr: np.ndarray, profile: dict, nodata: float = -9999.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    p = profile.copy()
    p.update(dtype=rasterio.float32, count=1, nodata=float(nodata), compress="deflate", predictor=3)
    out = np.where(np.isfinite(arr), arr, nodata).astype(np.float32)
    with rasterio.open(path, "w", **p) as dst:
        dst.write(out, 1)


def build_topography_cache(dem_zip: Path, ref_profile: dict, out_dir: Path) -> Tuple[Dict[str, np.ndarray], dict, pd.DataFrame]:
    """Create or reuse DEM-derived topographic layers in PROJECTED_CRS."""
    out_dir.mkdir(parents=True, exist_ok=True)
    topo_paths = {
        "elevation": out_dir / "dem_elevation_m_utm34n.tif",
        "slope_deg": out_dir / "dem_slope_deg_utm34n.tif",
        "aspect_deg": out_dir / "dem_aspect_deg_utm34n.tif",
        "northness": out_dir / "dem_northness_utm34n.tif",
        "eastness": out_dir / "dem_eastness_utm34n.tif",
        "d30": out_dir / "dem_d30_utm34n.tif",
        "d120": out_dir / "dem_d120_utm34n.tif",
        "tpi3": out_dir / "dem_tpi3_utm34n.tif",
        "roughness3": out_dir / "dem_roughness3_utm34n.tif",
    }
    if all(p.exists() for p in topo_paths.values()) and not FORCE_REBUILD_TOPO_CACHE:
        layers = {}
        profile = None
        for name, p in topo_paths.items():
            arr, prof = read_single(p)
            nd = prof.get("nodata", -9999.0)
            arr = arr.astype(np.float32)
            arr[np.isclose(arr, float(nd))] = np.nan
            layers[name] = arr
            profile = prof
        diag = pd.DataFrame([{"source": "existing_cache", "dem_zip": str(dem_zip), "topo_cache": str(out_dir)}])
        return layers, profile, diag

    if not dem_zip.exists():
        raise FileNotFoundError(f"DEM ZIP/path not found: {dem_zip}")

    vsi = dem_vsi_path(dem_zip)
    with rasterio.open(vsi) as src:
        win = bounded_dem_window(src, ref_profile)
        src_transform = src.window_transform(win)
        src_bounds = rasterio.windows.bounds(win, src.transform)
        src_arr = src.read(1, window=win, masked=True).astype(np.float32)
        src_nodata = src.nodata if src.nodata is not None else -9999.0
        src_filled = src_arr.filled(float(src_nodata)).astype(np.float32)

        dst_transform, dst_width, dst_height = calculate_default_transform(
            src.crs,
            PROJECTED_CRS,
            int(win.width),
            int(win.height),
            *src_bounds,
            resolution=DEM_PROJECTED_RESOLUTION_M,
        )
        dst = np.full((dst_height, dst_width), -9999.0, dtype=np.float32)
        reproject(
            source=src_filled,
            destination=dst,
            src_transform=src_transform,
            src_crs=src.crs,
            src_nodata=src_nodata,
            dst_transform=dst_transform,
            dst_crs=PROJECTED_CRS,
            dst_nodata=-9999.0,
            resampling=Resampling.bilinear,
        )
        dem = dst.astype(np.float32)
        dem[np.isclose(dem, -9999.0)] = np.nan
        profile = {
            "driver": "GTiff",
            "height": dem.shape[0],
            "width": dem.shape[1],
            "count": 1,
            "dtype": rasterio.float32,
            "crs": PROJECTED_CRS,
            "transform": dst_transform,
            "nodata": -9999.0,
        }

    slope, aspect = derive_slope_aspect(dem, profile["transform"])
    aspect_rad = np.radians(aspect)
    northness = np.cos(aspect_rad).astype(np.float32)
    eastness = np.sin(aspect_rad).astype(np.float32)
    d30 = np.cos(aspect_rad - np.radians(30.0)).astype(np.float32)
    d120 = np.cos(aspect_rad - np.radians(120.0)).astype(np.float32)
    local_mean = nanmean_filter3(dem)
    tpi3 = (dem - local_mean).astype(np.float32)
    roughness3 = nanstd_filter3(dem)

    layers = {
        "elevation": dem,
        "slope_deg": slope,
        "aspect_deg": aspect,
        "northness": northness,
        "eastness": eastness,
        "d30": d30,
        "d120": d120,
        "tpi3": tpi3,
        "roughness3": roughness3,
    }
    for name, arr in layers.items():
        write_float_raster(topo_paths[name], arr, profile)
    diag = pd.DataFrame([{
        "source": "rebuilt",
        "dem_zip": str(dem_zip),
        "dem_vsi_path": vsi,
        "projected_crs": PROJECTED_CRS,
        "projected_resolution_m": DEM_PROJECTED_RESOLUTION_M,
        "height": dem.shape[0],
        "width": dem.shape[1],
        "valid_elevation_px": int(np.isfinite(dem).sum()),
        "invalid_elevation_px": int((~np.isfinite(dem)).sum()),
        "valid_elevation_frac": float(np.isfinite(dem).mean()) if dem.size else np.nan,
        "all_nodata_3x3_neighbourhood_px": int((~np.isfinite(local_mean)).sum()),
        "all_nodata_3x3_neighbourhood_frac": float((~np.isfinite(local_mean)).mean()) if local_mean.size else np.nan,
        "topo_cache": str(out_dir),
    }])
    return layers, profile, diag

# =============================================================================
# Geometry predictors
# =============================================================================

def transformer_between(src_crs, dst_crs) -> Transformer:
    return Transformer.from_crs(CRS.from_user_input(src_crs), CRS.from_user_input(dst_crs), always_xy=True)


def transform_geometry(geom, src_crs, dst_crs):
    if geom is None or geom.is_empty:
        return geom
    if str(CRS.from_user_input(src_crs)) == str(CRS.from_user_input(dst_crs)):
        return geom
    tr = transformer_between(src_crs, dst_crs)
    return shapely_transform(tr.transform, geom)


def object_geometries_from_id_raster(id_arr: np.ndarray, profile: dict) -> Dict[int, object]:
    geoms_by_id: Dict[int, List[object]] = defaultdict(list)
    mask = id_arr > 0
    if not np.any(mask):
        return {}
    for geom_json, val in rasterio.features.shapes(id_arr.astype(np.int32), mask=mask, transform=profile["transform"]):
        oid = int(val)
        if oid > 0:
            geoms_by_id[oid].append(shape(geom_json))
    out = {}
    for oid, parts in geoms_by_id.items():
        try:
            g = unary_union(parts)
        except Exception:
            g = parts[0]
        out[oid] = g
    return out


def minimum_rotated_rectangle_features(geom_proj) -> Dict[str, float]:
    out = {
        "shape_mrr_long_m": np.nan,
        "shape_mrr_short_m": np.nan,
        "shape_elongation": np.nan,
        "shape_orientation_deg": np.nan,
    }
    if geom_proj is None or geom_proj.is_empty:
        return out
    try:
        mrr = geom_proj.minimum_rotated_rectangle
        coords = list(mrr.exterior.coords)
        if len(coords) < 4:
            return out
        side_lengths = []
        side_angles = []
        for i in range(4):
            x1, y1 = coords[i]
            x2, y2 = coords[i + 1]
            dx = x2 - x1
            dy = y2 - y1
            length = math.hypot(dx, dy)
            side_lengths.append(length)
            # Azimuth clockwise from north; orientation is axial, so modulo 180.
            az = (math.degrees(math.atan2(dx, dy)) + 360.0) % 180.0
            side_angles.append(az)
        if not side_lengths:
            return out
        long_i = int(np.argmax(side_lengths))
        long_len = float(side_lengths[long_i])
        short_candidates = [l for l in side_lengths if l < long_len * 0.999 or len(side_lengths) == 4]
        short_len = float(np.partition(np.asarray(side_lengths), 1)[1]) if len(side_lengths) >= 2 else np.nan
        # Unique rectangle sides appear twice; the two shortest/longest are enough.
        sorted_lengths = sorted(side_lengths)
        short_len = float(sorted_lengths[0])
        if short_len <= 0:
            elong = np.nan
        else:
            elong = long_len / short_len
        out.update({
            "shape_mrr_long_m": long_len,
            "shape_mrr_short_m": short_len,
            "shape_elongation": float(elong),
            "shape_orientation_deg": float(side_angles[long_i]),
        })
    except Exception:
        pass
    return out


def count_parts_and_holes(geom) -> Tuple[int, int]:
    if geom is None or geom.is_empty:
        return 0, 0
    if geom.geom_type == "Polygon":
        return 1, len(geom.interiors)
    if geom.geom_type == "MultiPolygon":
        parts = list(geom.geoms)
        return len(parts), sum(len(p.interiors) for p in parts)
    return 1, 0


def geometry_predictors(geom, src_crs, projected_crs=PROJECTED_CRS, prefix="shape") -> Dict[str, float]:
    geom_proj = transform_geometry(geom, src_crs, projected_crs)
    out: Dict[str, float] = {}
    if geom_proj is None or geom_proj.is_empty:
        keys = [
            "area_m2", "area_ha", "perimeter_m", "compactness", "equiv_diameter_m",
            "perimeter_area_ratio", "hull_area_m2", "hull_perimeter_m", "solidity",
            "convexity", "edge_complexity", "bbox_width_m", "bbox_height_m", "bbox_aspect_ratio",
            "centroid_x", "centroid_y", "n_parts", "n_holes",
        ]
        return {f"{prefix}_{k}": np.nan for k in keys}
    area = float(geom_proj.area)
    perim = float(geom_proj.length)
    hull = geom_proj.convex_hull
    hull_area = float(hull.area) if hull and not hull.is_empty else np.nan
    hull_perim = float(hull.length) if hull and not hull.is_empty else np.nan
    compactness = float(4.0 * math.pi * area / (perim ** 2)) if perim > 0 else np.nan
    equiv_diam = float(math.sqrt(4.0 * area / math.pi)) if area > 0 else np.nan
    solidity = float(area / hull_area) if hull_area and hull_area > 0 else np.nan
    convexity = float(hull_perim / perim) if perim > 0 and hull_perim and hull_perim > 0 else np.nan
    edge_complexity = float(perim / hull_perim) if hull_perim and hull_perim > 0 else np.nan
    minx, miny, maxx, maxy = geom_proj.bounds
    bbox_w = maxx - minx
    bbox_h = maxy - miny
    n_parts, n_holes = count_parts_and_holes(geom_proj)
    centroid = geom_proj.centroid
    out.update({
        f"{prefix}_area_m2": area,
        f"{prefix}_area_ha": area / 10000.0,
        f"{prefix}_perimeter_m": perim,
        f"{prefix}_compactness": compactness,
        f"{prefix}_equiv_diameter_m": equiv_diam,
        f"{prefix}_perimeter_area_ratio": perim / area if area > 0 else np.nan,
        f"{prefix}_hull_area_m2": hull_area,
        f"{prefix}_hull_perimeter_m": hull_perim,
        f"{prefix}_solidity": solidity,
        f"{prefix}_convexity": convexity,
        f"{prefix}_edge_complexity": edge_complexity,
        f"{prefix}_bbox_width_m": float(bbox_w),
        f"{prefix}_bbox_height_m": float(bbox_h),
        f"{prefix}_bbox_aspect_ratio": float(max(bbox_w, bbox_h) / min(bbox_w, bbox_h)) if min(bbox_w, bbox_h) > 0 else np.nan,
        f"{prefix}_centroid_x": float(centroid.x),
        f"{prefix}_centroid_y": float(centroid.y),
        f"{prefix}_n_parts": int(n_parts),
        f"{prefix}_n_holes": int(n_holes),
    })
    out.update(minimum_rotated_rectangle_features(geom_proj))
    return out

# =============================================================================
# Input discovery
# =============================================================================

def load_algorithm4_features() -> pd.DataFrame:
    all_path = ALGORITHM4_ROOT / "object_features_all_intervals.csv"
    if all_path.exists():
        df = pd.read_csv(all_path)
        return df
    rows = []
    for p in sorted(ALGORITHM4_ROOT.glob("*_to_*")):
        fp = p / "object_features.csv"
        if fp.exists():
            rows.append(pd.read_csv(fp))
    if not rows:
        raise FileNotFoundError(f"Could not find Algorithm #4 object_features files in {ALGORITHM4_ROOT}")
    return pd.concat(rows, ignore_index=True)


def interval_folder(row: pd.Series) -> Path:
    if "interval_label" in row and isinstance(row["interval_label"], str) and row["interval_label"]:
        return ALGORITHM4_ROOT / row["interval_label"]
    return ALGORITHM4_ROOT / interval_label(str(row["pre_date"]), str(row["post_date"]))


def load_s2_inventory() -> Tuple[pd.DataFrame, pd.DataFrame]:
    inv_path = S2_CACHE_ROOT / "scene_inventory.csv"
    if not inv_path.exists():
        raise FileNotFoundError(f"Missing Sentinel-2 cache inventory: {inv_path}")
    inv = pd.read_csv(inv_path)
    inv["date"] = pd.to_datetime(inv["date"], errors="coerce")
    qual_path = S2_CACHE_ROOT / "scene_quality_summary.csv"
    if qual_path.exists():
        qual = pd.read_csv(qual_path)
        qual["date"] = pd.to_datetime(qual["date"], errors="coerce")
    else:
        qual = pd.DataFrame()
    return inv, qual


def select_s2_scene_for_date(date_value, inv: pd.DataFrame, qual: pd.DataFrame) -> Optional[pd.Series]:
    date = parse_date(date_value)
    if pd.isna(date):
        return None
    subset = inv[inv["date"].dt.date == date.date()].copy()
    subset = subset[subset["index_stack_path"].astype(str).ne("")]
    if subset.empty:
        return None
    if not qual.empty and "clear_frac" in qual.columns:
        q = qual[["scene", "clear_frac", "veg_frac", "snow_frac", "cloud_frac", "shadow_frac"]].copy()
        subset = subset.merge(q, on="scene", how="left")
    else:
        for c in ["clear_frac", "veg_frac", "snow_frac", "cloud_frac", "shadow_frac"]:
            subset[c] = np.nan
    # Prefer high clear fraction and filenames marked no_clouds.
    subset["name_score"] = subset["scene"].astype(str).str.lower().map(lambda s: 2 if "no_cloud" in s else (1 if "low" in s else 0))
    subset["clear_sort"] = subset["clear_frac"].fillna(-1.0)
    subset = subset.sort_values(["clear_sort", "name_score", "scene"], ascending=[False, False, True])
    return subset.iloc[0]


def load_s1_inventory() -> pd.DataFrame:
    inv_path = S1_CACHE_ROOT / "s1_scene_inventory.csv"
    if not inv_path.exists():
        return pd.DataFrame()
    inv = pd.read_csv(inv_path)
    inv["date"] = pd.to_datetime(inv["date"], errors="coerce")
    return inv


def find_nearest_s1(date_value, inv: pd.DataFrame) -> Optional[pd.Series]:
    if inv.empty:
        return None
    date = parse_date(date_value)
    if pd.isna(date):
        return None
    temp = inv.dropna(subset=["date"]).copy()
    if temp.empty:
        return None
    temp["abs_offset_days"] = (temp["date"] - date).abs().dt.days
    temp = temp[temp["abs_offset_days"] <= S1_MAX_MATCH_DAYS]
    if temp.empty:
        return None
    temp = temp.sort_values(["abs_offset_days", "date", "scene"])
    return temp.iloc[0]


def available_s1_descriptors(row: pd.Series) -> List[str]:
    value = str(row.get("descriptor_bands_available", ""))
    names = [x.strip() for x in value.split(",") if x.strip()]
    return [n for n in names if n not in S1_DESCRIPTOR_EXCLUDE]


def read_s1_stack(row: pd.Series, descriptors: Sequence[str]) -> Tuple[Dict[str, np.ndarray], dict]:
    # Prefer available stack if present; otherwise fixed stack.
    path = str(row.get("s1_available_stack_path", "")) or str(row.get("s1_stack_path", ""))
    if not path:
        return {}, {}
    p = Path(path)
    if not p.exists():
        return {}, {}
    return read_float_stack(p, descriptors, nodata=S1_CACHE_FLOAT_NODATA)

# =============================================================================
# Predictor computation
# =============================================================================

class RasterCache:
    def __init__(self):
        self.s2: Dict[str, Tuple[Dict[str, np.ndarray], dict, pd.Series]] = {}
        self.s1: Dict[str, Tuple[Dict[str, np.ndarray], dict, pd.Series]] = {}

    def get_s2(self, date_value, inv: pd.DataFrame, qual: pd.DataFrame):
        key = str(parse_date(date_value).date()) if not pd.isna(parse_date(date_value)) else ""
        if key in self.s2:
            return self.s2[key]
        row = select_s2_scene_for_date(date_value, inv, qual)
        if row is None:
            self.s2[key] = ({}, {}, pd.Series(dtype=object))
            return self.s2[key]
        path = Path(str(row["index_stack_path"]))
        if not path.exists():
            self.s2[key] = ({}, {}, row)
            return self.s2[key]
        stack, profile = read_float_stack(path, S2_INDEX_BANDS, nodata=S2_CACHE_FLOAT_NODATA)
        self.s2[key] = (stack, profile, row)
        return self.s2[key]

    def get_s1(self, date_value, inv: pd.DataFrame):
        key = str(parse_date(date_value).date()) if not pd.isna(parse_date(date_value)) else ""
        if key in self.s1:
            return self.s1[key]
        row = find_nearest_s1(date_value, inv)
        if row is None:
            self.s1[key] = ({}, {}, pd.Series(dtype=object))
            return self.s1[key]
        desc = available_s1_descriptors(row)
        stack, profile = read_s1_stack(row, desc)
        self.s1[key] = (stack, profile, row)
        return self.s1[key]


def s2_predictors_for_mask(mask: np.ndarray, mask_profile: dict, pre_date, post_date, inv: pd.DataFrame, qual: pd.DataFrame, cache: RasterCache, days_between: float) -> Dict[str, object]:
    out: Dict[str, object] = {}
    pre_stack, pre_profile, pre_row = cache.get_s2(pre_date, inv, qual)
    post_stack, post_profile, post_row = cache.get_s2(post_date, inv, qual)
    out["m_s2_pre_available"] = bool(pre_stack)
    out["m_s2_post_available"] = bool(post_stack)
    out["support_s2_pre_scene"] = str(pre_row.get("scene", "")) if len(pre_row) else ""
    out["support_s2_post_scene"] = str(post_row.get("scene", "")) if len(post_row) else ""
    out["support_days_between"] = float(days_between) if math.isfinite(days_between) else np.nan
    if not pre_stack or not post_stack:
        return out
    pre_mask = mask_to_grid(mask, mask_profile, pre_profile)
    post_mask = mask_to_grid(mask, mask_profile, post_profile)
    for idx in S2_INDEX_BANDS:
        if idx not in pre_stack or idx not in post_stack:
            continue
        low = idx.lower()
        pre_stats = zonal_stats(pre_stack[idx], pre_mask, f"eo_s2_pre_{low}")
        post_stats = zonal_stats(post_stack[idx], post_mask, f"eo_s2_post_{low}")
        out.update(pre_stats)
        out.update(post_stats)
        # Delta on the post grid; if grids differ, reproject pre array to post grid.
        if same_grid(pre_profile, post_profile):
            pre_arr_for_delta = pre_stack[idx]
        else:
            pre_arr_for_delta = np.full((post_profile["height"], post_profile["width"]), np.nan, dtype=np.float32)
            src = np.where(np.isfinite(pre_stack[idx]), pre_stack[idx], S2_CACHE_FLOAT_NODATA).astype(np.float32)
            dst = np.full_like(pre_arr_for_delta, S2_CACHE_FLOAT_NODATA, dtype=np.float32)
            reproject(
                source=src,
                destination=dst,
                src_transform=pre_profile["transform"],
                src_crs=pre_profile["crs"],
                src_nodata=S2_CACHE_FLOAT_NODATA,
                dst_transform=post_profile["transform"],
                dst_crs=post_profile["crs"],
                dst_nodata=S2_CACHE_FLOAT_NODATA,
                resampling=Resampling.bilinear,
            )
            dst[np.isclose(dst, S2_CACHE_FLOAT_NODATA)] = np.nan
            pre_arr_for_delta = dst
        delta = post_stack[idx] - pre_arr_for_delta
        d_stats = zonal_stats(delta, post_mask, f"delta_s2_{low}")
        out.update(d_stats)
        mean_delta = d_stats.get(f"delta_s2_{low}_mean", np.nan)
        med_delta = d_stats.get(f"delta_s2_{low}_median", np.nan)
        if math.isfinite(days_between) and days_between > 0:
            out[f"delta_s2_{low}_mean_per_year"] = float(mean_delta * 365.25 / days_between) if math.isfinite(mean_delta) else np.nan
            out[f"delta_s2_{low}_median_per_year"] = float(med_delta * 365.25 / days_between) if math.isfinite(med_delta) else np.nan
        else:
            out[f"delta_s2_{low}_mean_per_year"] = np.nan
            out[f"delta_s2_{low}_median_per_year"] = np.nan
    return out


def s1_predictors_for_mask(mask: np.ndarray, mask_profile: dict, pre_date, post_date, inv: pd.DataFrame, cache: RasterCache, days_between: float) -> Dict[str, object]:
    out: Dict[str, object] = {}
    if inv.empty:
        out["m_s1_pre_available"] = False
        out["m_s1_post_available"] = False
        return out
    pre_stack, pre_profile, pre_row = cache.get_s1(pre_date, inv)
    post_stack, post_profile, post_row = cache.get_s1(post_date, inv)
    out["m_s1_pre_available"] = bool(pre_stack)
    out["m_s1_post_available"] = bool(post_stack)
    out["support_s1_pre_scene"] = str(pre_row.get("scene", "")) if len(pre_row) else ""
    out["support_s1_post_scene"] = str(post_row.get("scene", "")) if len(post_row) else ""
    out["support_s1_pre_offset_days"] = safe_float(pre_row.get("abs_offset_days", pre_row.get("matched_offset_days", np.nan))) if len(pre_row) else np.nan
    out["support_s1_post_offset_days"] = safe_float(post_row.get("abs_offset_days", post_row.get("matched_offset_days", np.nan))) if len(post_row) else np.nan
    common = sorted(set(pre_stack.keys()) & set(post_stack.keys()) - S1_DESCRIPTOR_EXCLUDE)
    if not pre_stack or not post_stack or not common:
        return out
    pre_mask = mask_to_grid(mask, mask_profile, pre_profile)
    post_mask = mask_to_grid(mask, mask_profile, post_profile)
    for desc in common:
        safe = desc.lower()
        out.update(zonal_stats(pre_stack[desc], pre_mask, f"eo_s1_pre_{safe}"))
        out.update(zonal_stats(post_stack[desc], post_mask, f"eo_s1_post_{safe}"))
        if same_grid(pre_profile, post_profile):
            pre_arr_for_delta = pre_stack[desc]
        else:
            dst = np.full((post_profile["height"], post_profile["width"]), S1_CACHE_FLOAT_NODATA, dtype=np.float32)
            src = np.where(np.isfinite(pre_stack[desc]), pre_stack[desc], S1_CACHE_FLOAT_NODATA).astype(np.float32)
            reproject(
                source=src,
                destination=dst,
                src_transform=pre_profile["transform"],
                src_crs=pre_profile["crs"],
                src_nodata=S1_CACHE_FLOAT_NODATA,
                dst_transform=post_profile["transform"],
                dst_crs=post_profile["crs"],
                dst_nodata=S1_CACHE_FLOAT_NODATA,
                resampling=Resampling.bilinear,
            )
            dst[np.isclose(dst, S1_CACHE_FLOAT_NODATA)] = np.nan
            pre_arr_for_delta = dst
        delta = post_stack[desc] - pre_arr_for_delta
        d_stats = zonal_stats(delta, post_mask, f"delta_s1_{safe}")
        out.update(d_stats)
        mean_delta = d_stats.get(f"delta_s1_{safe}_mean", np.nan)
        if math.isfinite(days_between) and days_between > 0:
            out[f"delta_s1_{safe}_mean_per_year"] = float(mean_delta * 365.25 / days_between) if math.isfinite(mean_delta) else np.nan
        else:
            out[f"delta_s1_{safe}_mean_per_year"] = np.nan
    return out


def topo_predictors_for_geom(geom, geom_crs, topo_layers: Dict[str, np.ndarray], topo_profile: dict) -> Dict[str, object]:
    out: Dict[str, object] = {"m_dem_available": bool(topo_layers), "m_topo_valid_frac": np.nan}
    if not topo_layers or topo_profile is None or geom is None or geom.is_empty:
        return out
    geom_topo = transform_geometry(geom, geom_crs, topo_profile["crs"])
    if geom_topo is None or geom_topo.is_empty:
        return out
    mask = rasterio.features.rasterize(
        [(geom_topo, 1)],
        out_shape=(topo_profile["height"], topo_profile["width"]),
        transform=topo_profile["transform"],
        fill=0,
        dtype="uint8",
        all_touched=False,
    ).astype(bool)
    n = int(mask.sum())
    out["topo_support_px"] = n
    if n == 0:
        return out
    # Main statistics.
    stats_map = {
        "elevation": "elev_m",
        "slope_deg": "slope_deg",
        "northness": "northness",
        "eastness": "eastness",
        "d30": "d30",
        "d120": "d120",
        "tpi3": "tpi3",
        "roughness3": "roughness3",
    }
    for key, short in stats_map.items():
        if key in topo_layers:
            out.update(zonal_stats(topo_layers[key], mask, f"topo_{short}"))
    # Aspect mean is circular. Report it from mean north/east if possible.
    north = out.get("topo_northness_mean", np.nan)
    east = out.get("topo_eastness_mean", np.nan)
    if math.isfinite(north) and math.isfinite(east):
        out["topo_aspect_mean_deg"] = float((math.degrees(math.atan2(east, north)) + 360.0) % 360.0)
        out["topo_aspect_vector_strength"] = float(math.sqrt(north * north + east * east))
    else:
        out["topo_aspect_mean_deg"] = np.nan
        out["topo_aspect_vector_strength"] = np.nan
    elev_valid = out.get("topo_elev_m_valid_px", 0)
    out["m_topo_valid_frac"] = float(elev_valid / n) if n else np.nan
    return out


def semantic_support_predictors(mask: np.ndarray, mask_profile: dict, pre_date: str, post_date: str) -> Dict[str, object]:
    """Add semantic vote/anomaly/confidence support from Algorithm #2 masks."""
    out: Dict[str, object] = {}
    for label, date in [("pre", pre_date), ("post", post_date)]:
        folder = SEMANTIC_MASK_ROOT / str(date)
        vote_path = folder / "semantic_vote_count.tif"
        anomaly_path = folder / "semantic_anomaly_score.tif"
        confidence_path = folder / "semantic_envelope_confidence.tif"
        if vote_path.exists():
            arr, prof = read_single(vote_path)
            m = mask_to_grid(mask, mask_profile, prof)
            vote = arr.astype(np.float32)
            vote[vote == 255] = np.nan
            out.update(zonal_stats(vote, m, f"eo_semantic_{label}_vote_count"))
            out.update(binary_fraction(m, vote >= 2, f"eo_semantic_{label}_keep_frac"))
            out.update(binary_fraction(m, vote < 2, f"eo_semantic_{label}_drop_frac"))
        else:
            out[f"m_semantic_{label}_vote_available"] = False
        if anomaly_path.exists():
            arr, prof = read_single(anomaly_path)
            arr = arr.astype(np.float32)
            nd = prof.get("nodata", -9999.0)
            arr[np.isclose(arr, float(nd))] = np.nan
            m = mask_to_grid(mask, mask_profile, prof)
            out.update(zonal_stats(arr, m, f"eo_semantic_{label}_anomaly"))
        if confidence_path.exists():
            arr, prof = read_single(confidence_path)
            m = mask_to_grid(mask, mask_profile, prof)
            denom = float(m.sum())
            if denom > 0:
                for code, cname in [(1, "high"), (2, "medium"), (3, "low"), (4, "missing")]:
                    out[f"eo_semantic_{label}_confidence_{cname}_frac"] = float(np.sum(m & (arr == code)) / denom)
    return out

# =============================================================================
# Feature dictionary
# =============================================================================

def feature_dictionary_rows(columns: Sequence[str]) -> pd.DataFrame:
    rows = []
    def block_for(col: str) -> str:
        if col.startswith("eo_s2_") or col.startswith("eo_s1_") or col.startswith("eo_semantic_"):
            return "x_EO_current_state"
        if col.startswith("delta_") or "_delta_" in col or col.startswith("rate_"):
            return "x_delta_change"
        if col.startswith("shape_") or col in {"compactness", "core_ratio", "fringe_ratio"}:
            return "x_shape_geometry"
        if col.startswith("topo_"):
            return "x_topo_environment"
        if col.startswith("m_") or col.startswith("support_"):
            return "m_support_missingness"
        if col.startswith("raw_") or col.startswith("core_") or col.startswith("fringe_"):
            return "algorithm4_object_descriptor"
        return "identifier_or_context"
    def source_for(col: str) -> str:
        if col.startswith("topo_") or col.startswith("m_dem"):
            return "DEM Copernicus 30 m"
        if col.startswith("shape_"):
            return "Algorithm #4 object geometry"
        if col.startswith("eo_s2_") or col.startswith("delta_s2_"):
            return "Phase 3 Sentinel-2 index cache"
        if col.startswith("eo_s1_") or col.startswith("delta_s1_"):
            return "Phase 5 Sentinel-1 descriptor cache"
        if col.startswith("eo_semantic_"):
            return "Algorithm #2 semantic masks"
        if col.startswith("raw_") or col.startswith("core_") or col.startswith("fringe_"):
            return "Algorithm #4 features"
        return "metadata"
    def unit_for(col: str) -> str:
        if col.endswith("_ha") or "area_ha" in col:
            return "ha"
        if col.endswith("_m") or "_m_" in col or col.endswith("_m2"):
            return "m or m²"
        if "deg" in col or "aspect" in col or "orientation" in col:
            return "degrees"
        if "frac" in col or "ratio" in col or "compactness" in col or "solidity" in col or "convexity" in col:
            return "dimensionless"
        if "ndvi" in col or "ndmi" in col or "nbr" in col or "ndre" in col or "ndsi" in col:
            return "index value"
        if "s1" in col:
            return "backscatter descriptor units"
        return "mixed"
    for c in columns:
        rows.append({
            "feature": c,
            "feature_block": block_for(c),
            "source": source_for(c),
            "unit": unit_for(c),
            "description": "Automatically generated EO-vector predictor or metadata field.",
        })
    return pd.DataFrame(rows)

# =============================================================================
# Main processing
# =============================================================================

def process_interval(df_interval: pd.DataFrame, raster_dir: Path, s2_inv: pd.DataFrame, s2_qual: pd.DataFrame, s1_inv: pd.DataFrame, topo_layers, topo_profile, raster_cache: RasterCache) -> Tuple[pd.DataFrame, Optional["gpd.GeoDataFrame"]]:
    raw_path = raster_dir / RAW_OBJECT_ID_NAME
    core_path = raster_dir / CORE_OBJECT_ID_NAME
    fringe_path = raster_dir / FRINGE_OBJECT_ID_NAME
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw object-id raster: {raw_path}")
    raw_id, raw_profile = read_single(raw_path)
    raw_id = raw_id.astype(np.int32)
    core_id = None
    fringe_id = None
    if core_path.exists():
        core_id, _ = read_single(core_path)
        core_id = core_id.astype(np.int32)
    if fringe_path.exists():
        fringe_id, _ = read_single(fringe_path)
        fringe_id = fringe_id.astype(np.int32)

    geom_by_id = object_geometries_from_id_raster(raw_id, raw_profile)
    src_crs = raw_profile["crs"]
    rows: List[Dict[str, object]] = []
    geoms = []

    for _, r in df_interval.iterrows():
        oid = int(r["object_id"])
        raw_mask = raw_id == oid
        if not np.any(raw_mask):
            continue
        core_mask = (core_id == oid) if core_id is not None else np.zeros_like(raw_mask, dtype=bool)
        fringe_mask = (fringe_id == oid) if fringe_id is not None else (raw_mask & (~core_mask))
        geom = geom_by_id.get(oid)
        pre_date = str(r["pre_date"])
        post_date = str(r["post_date"])
        days_between = float((pd.Timestamp(post_date) - pd.Timestamp(pre_date)).days)

        rec = r.to_dict()
        # Primary geometry predictors are computed over the raw candidate object.
        rec.update(geometry_predictors(geom, src_crs, PROJECTED_CRS, prefix="shape"))
        rec.update(topo_predictors_for_geom(geom, src_crs, topo_layers, topo_profile))
        rec.update(s2_predictors_for_mask(raw_mask, raw_profile, pre_date, post_date, s2_inv, s2_qual, raster_cache, days_between))
        rec.update(s1_predictors_for_mask(raw_mask, raw_profile, pre_date, post_date, s1_inv, raster_cache, days_between))
        rec.update(semantic_support_predictors(raw_mask, raw_profile, pre_date, post_date))
        rec["m_core_available"] = bool(np.any(core_mask))
        rec["m_fringe_available"] = bool(np.any(fringe_mask))
        rec["support_raw_object_px"] = int(raw_mask.sum())
        rec["support_core_object_px"] = int(core_mask.sum())
        rec["support_fringe_object_px"] = int(fringe_mask.sum())
        rows.append(rec)
        geoms.append(geom)

    out_df = pd.DataFrame(rows)
    gdf = None
    if WRITE_GPKG and gpd is not None and rows:
        gdf = gpd.GeoDataFrame(out_df.copy(), geometry=geoms, crs=src_crs)
        try:
            gdf["geometry"] = gdf.geometry.make_valid()
        except Exception:
            gdf["geometry"] = gdf.geometry.buffer(0)
    return out_df, gdf


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    log("Starting EO predictor-vector construction")
    feat = load_algorithm4_features()
    required = {"pre_date", "post_date", "object_id"}
    missing = required - set(feat.columns)
    if missing:
        raise ValueError(f"Algorithm #4 feature table is missing required columns: {sorted(missing)}")
    if "interval_label" not in feat.columns:
        feat["interval_label"] = [interval_label(str(a), str(b)) for a, b in zip(feat["pre_date"], feat["post_date"])]
    if "object_uid" not in feat.columns:
        feat["object_uid"] = [f"lossObj_{interval_id(str(a), str(b))}_{int(oid):04d}" for a, b, oid in zip(feat["pre_date"], feat["post_date"], feat["object_id"])]

    s2_inv, s2_qual = load_s2_inventory()
    s1_inv = load_s1_inventory()
    raster_cache = RasterCache()

    # Reference profile from the first interval object raster.
    first_label = str(feat.iloc[0]["interval_label"])
    first_raw = ALGORITHM4_ROOT / first_label / RASTER_SUBDIR / RAW_OBJECT_ID_NAME
    if not first_raw.exists():
        raise FileNotFoundError(f"Cannot locate first Algorithm #4 raw object raster: {first_raw}")
    _, first_profile = read_single(first_raw)
    topo_layers, topo_profile, topo_diag = build_topography_cache(DEM_ZIP, first_profile, OUTPUT_ROOT / "topography_cache")
    topo_diag.to_csv(OUTPUT_ROOT / "topography_cache_diagnostics.csv", index=False)

    all_rows = []
    all_gdfs = []
    for ilabel, group in feat.groupby("interval_label", sort=True):
        raster_dir = ALGORITHM4_ROOT / str(ilabel) / RASTER_SUBDIR
        log(f"Processing interval {ilabel}: {len(group)} objects")
        interval_df, interval_gdf = process_interval(group.copy(), raster_dir, s2_inv, s2_qual, s1_inv, topo_layers, topo_profile, raster_cache)
        if WRITE_PER_INTERVAL_OUTPUTS:
            out_dir = OUTPUT_ROOT / str(ilabel)
            out_dir.mkdir(parents=True, exist_ok=True)
            interval_df.to_csv(out_dir / "eo_predictor_vector.csv", index=False)
            if WRITE_GPKG and interval_gdf is not None and not interval_gdf.empty:
                try:
                    interval_gdf.to_file(out_dir / "eo_predictor_vector.gpkg", layer="eo_vector", driver="GPKG")
                except Exception as exc:
                    log(f"Warning: could not write per-interval GeoPackage for {ilabel}: {exc}")
        all_rows.append(interval_df)
        if interval_gdf is not None and not interval_gdf.empty:
            all_gdfs.append(interval_gdf)

    vector_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    vector_df.to_csv(OUTPUT_ROOT / "eo_predictor_vector_all_intervals.csv", index=False)
    log(f"Wrote EO predictor CSV with {len(vector_df)} objects")

    if WRITE_GPKG and gpd is not None and all_gdfs:
        gdf_all = gpd.GeoDataFrame(pd.concat(all_gdfs, ignore_index=True), crs=all_gdfs[0].crs)
        try:
            gdf_all.to_file(OUTPUT_ROOT / "eo_predictor_vector_all_intervals.gpkg", layer="eo_vector", driver="GPKG")
            log("Wrote EO predictor GeoPackage")
        except Exception as exc:
            log(f"Warning: could not write global GeoPackage: {exc}")

    if WRITE_FEATURE_DICTIONARY:
        feature_dictionary_rows(vector_df.columns).to_csv(OUTPUT_ROOT / "eo_predictor_feature_dictionary.csv", index=False)

    # Compact summary by feature block.
    if not vector_df.empty:
        fd = feature_dictionary_rows(vector_df.columns)
        summary = fd.groupby("feature_block", as_index=False).agg(n_features=("feature", "count"))
        summary.to_csv(OUTPUT_ROOT / "eo_predictor_feature_block_summary.csv", index=False)

    readme = OUTPUT_ROOT / "README.txt"
    readme.write_text(
        "EO predictor vector construction for Algorithm #4 candidate loss objects.\n\n"
        "The output vector follows the article feature-block structure:\n"
        "  x = [x_EO, x_delta, x_shape, x_topo, m]\n\n"
        "Main output files:\n"
        "  eo_predictor_vector_all_intervals.csv\n"
        "  eo_predictor_vector_all_intervals.gpkg\n"
        "  eo_predictor_feature_dictionary.csv\n"
        "  topography_cache/*.tif\n\n"
        f"DEM source: {DEM_ZIP}\n"
        f"Algorithm #4 root: {ALGORITHM4_ROOT}\n"
        f"S2 cache root: {S2_CACHE_ROOT}\n"
        f"S1 cache root: {S1_CACHE_ROOT}\n"
        f"Projected CRS for geometry/topography: {PROJECTED_CRS}\n"
        "Topographic variables include elevation, slope, northness, eastness, D30, D120, TPI3, and roughness3.\n"
        "Geometric variables include area, perimeter, compactness, elongation, solidity, convexity, edge complexity, orientation, and part/hole counts.\n",
        encoding="utf-8",
    )
    log(f"Done. Outputs in {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
