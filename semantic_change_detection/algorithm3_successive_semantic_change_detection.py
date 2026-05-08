"""
Algorithm #3 Phase 5. Compare successive semantic masks and attach optional Sentinel-1 pair descriptors.

This version adds stricter control over LOSS and GAIN inflation:
1) optional vote-based gating for loss/gain
2) optional 1-pixel erosion of change masks to keep the disturbance core
3) minimum patch-size filtering before summaries are computed

Input assumption
----------------
Each scene directory contains at least:
- semantic_keep_mask.tif   : 1 keep, 0 drop, 255 nodata
- semantic_vote_count.tif  : 0..3, 255 nodata
- semantic_group_id.tif    : positive analytical group id, 0 outside group/nodata

Change classes
--------------
Within the common comparable domain (both dates valid):
- 0 : stable_drop   (drop -> drop)
- 1 : stable_keep   (keep -> keep)
- 2 : loss          (keep -> drop)
- 3 : gain/recovery (drop -> keep)
- 255 : nodata / not comparable / filtered out

Outputs per pair
----------------
- change_class.tif
- stable_drop_mask.tif
- stable_keep_mask.tif
- loss_mask.tif
- gain_mask.tif
- filtered_out_mask.tif
- vote_delta.tif
- pair_summary.csv
- pair_group_summary.csv
- change_objects_utm34n.gpkg              (optional polygon output per pair)
- successive_change_objects_utm34n.gpkg   (optional global polygon output)
- polygonization_inventory.csv            (optional global polygon inventory)

Hardcoded by design.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple
import re

import numpy as np
import pandas as pd
import rasterio
import rasterio.features

try:
    import geopandas as gpd
    from shapely.geometry import shape
except ImportError:  # polygon export is optional unless WRITE_POLYGONS=True
    gpd = None
    shape = None

from pyproj import Geod

from sdv_shared import (
    S1_DESCRIPTOR_BANDS,
    load_s1_inventory,
    find_nearest_s1_scene,
    read_float32_stack,
    read_uint8_raster,
    write_float32_stack,
)

# =============================================================================
# HARD-CODED CONFIG
# =============================================================================
MASK_ROOT = Path("D:/Forest_Disturbance/outputs/semantic_masks_s1s2_empirical_envelopes_blacklist_and_caution_eligible")
OUTPUT_ROOT = Path("D:/Forest_Disturbance/outputs/semantic_change_detection_s1s2_blacklist_and_caution_eligible")
S1_CACHE_ROOT = Path("D:/Forest_Disturbance/outputs/sdv_phase5_sentinel1_descriptor_cache")
SCENE_DIR_DATE_RE = re.compile(r"^(20\d{2}-\d{2}-\d{2})(?:__.*)?$")

PAIR_MODE = "annual_anchor_sequence"  # "annual_anchor_sequence" or "all_successive"
ANNUAL_ANCHOR_DATES = [
    "2017-08-25",
    "2018-08-25",
    "2019-07-01",
    "2020-08-09",
    "2021-08-09",
    "2022-07-25",
    "2023-08-04",
    "2024-07-29",
    "2025-08-28",
]
KEEP_MASK_NAME = "semantic_keep_mask.tif"
VOTE_MASK_NAME = "semantic_vote_count.tif"
GROUP_MASK_NAME = "semantic_group_id.tif"

# Minimum size of connected components retained.
MIN_CLASS_PATCH_PIXELS = 25
CONNECTIVITY = 8  # 4 or 8

# Whether to apply the component filter to stable classes too.
FILTER_STABLE_CLASSES = False

# Stricter vote-based gating for change masks.
USE_STRICT_VOTE_GATING = True
MIN_PRE_VOTES_FOR_LOSS = 2      # require clearly healthy before; initial value = 2
MAX_POST_VOTES_FOR_LOSS = 0     # require clearly unhealthy after
MIN_VOTE_DROP_FOR_LOSS = 2      # require vote_delta <= -2; initial value = 2
MAX_PRE_VOTES_FOR_GAIN = 0      # require clearly unhealthy before
MIN_POST_VOTES_FOR_GAIN = 2     # require clearly healthy after
MIN_VOTE_RISE_FOR_GAIN = 2      # require vote_delta >= +2

# Optional erosion to keep only the core of change objects.
ERODE_CHANGE_MASKS = False
ERODE_ITERATIONS = 1
ERODE_CONNECTIVITY = 8

# Optional Sentinel-1 pair descriptors. These descriptors are attached to pair
# summaries and written as a delta stack; they do not alter the semantic class
# rules by default.
ENABLE_S1_PAIR_DESCRIPTORS = True
S1_MAX_ABS_DATE_OFFSET_DAYS = 45
# Use "auto" to compute deltas only for descriptors that are present in both
# nearest-date S1 scenes. This avoids all-nodata VV delta bands when the S1 ZIPs
# contain VH only.
S1_DESCRIPTORS_FOR_DELTA = "auto"
S1_DESCRIPTOR_EXCLUDE_FROM_DELTA = {"S1_VALID"}
WRITE_S1_DELTA_STACK = True


# Optional polygon export.
# The polygons are created from the already filtered masks, with an additional
# component-size check of MIN_POLYGON_COMPONENT_PIXELS. By default, stable_keep is
# not polygonized because it can create very large background polygons. Add it to
# POLYGONIZE_CLASSES if it is needed for a specific diagnostic.
WRITE_POLYGONS = True
TARGET_CRS = "EPSG:32634"  # UTM 34N
MIN_POLYGON_COMPONENT_PIXELS = 25  # 5 x 5 pixels
POLYGONIZE_CONNECTIVITY = CONNECTIVITY  # use 8 by default; set to 4 for stricter diagonal separation
POLYGONIZE_CLASSES = ("stable_drop", "loss", "gain")

# Export one combined layer plus one separate layer per class.
WRITE_COMBINED_POLYGON_LAYER = True
WRITE_SEPARATE_CLASS_LAYERS = True
COMBINED_LAYER_NAME = "all_change_objects"

# Optional light geometric simplification after projection to UTM.
# Because the source is raster-derived, the outlines are staircase-like.
# A tolerance around half a pixel (e.g. 5 m for 10 m data) usually gives a
# cleaner shape without materially changing the footprint.
SIMPLIFY_POLYGONS = True
SIMPLIFY_TOLERANCE_M = 5.0
SIMPLIFY_PRESERVE_TOPOLOGY = True

GLOBAL_POLYGON_GPKG = OUTPUT_ROOT / "successive_change_objects_utm34n.gpkg"
POLYGONIZATION_INVENTORY_CSV = OUTPUT_ROOT / "polygonization_inventory.csv"

# Optional temporal confirmation layer. For each previous interval t_i -> t_{i+1},
# this creates a persistence-supported subset by intersecting the previous
# interval loss mask with the following interval stable_drop mask:
#   confirmed_previous_loss = loss(t_i, t_{i+1}) AND stable_drop(t_{i+1}, t_{i+2})
# The last interval cannot be confirmed by a following stable_drop layer.
WRITE_CONFIRMED_PREVIOUS_LOSS = True
CONFIRMED_LOSS_NAME = "loss_confirmed_by_next_stable_drop"
CONFIRMED_LOSS_CLASS_CODE = 4
CONFIRMED_LOSS_MIN_PIXELS = 25
CONFIRMED_LOSS_CONNECTIVITY = CONNECTIVITY
WRITE_UNCONFIRMED_PREVIOUS_LOSS = True
UNCONFIRMED_LOSS_NAME = "loss_not_confirmed_by_next_stable_drop"
UNCONFIRMED_LOSS_CLASS_CODE = 5
GLOBAL_CONFIRMED_LOSS_GPKG = OUTPUT_ROOT / "successive_confirmed_previous_loss_utm34n.gpkg"
CONFIRMED_LOSS_SUMMARY_CSV = OUTPUT_ROOT / "confirmed_previous_loss_summary.csv"

# =============================================================================
# Utilities
# =============================================================================
GEOD = Geod(ellps="WGS84")


def scene_date_from_dir(path: Path) -> pd.Timestamp:
    m = SCENE_DIR_DATE_RE.match(path.name)
    if not m:
        raise ValueError(f"Could not parse date from semantic scene folder: {path.name}")
    return pd.Timestamp(m.group(1))


def discover_scene_dirs(root: Path) -> List[Path]:
    scenes = [p for p in root.iterdir() if p.is_dir() and SCENE_DIR_DATE_RE.match(p.name)]
    return sorted(scenes, key=scene_date_from_dir)


def select_pair_dirs(scene_dirs: List[Path]) -> List[Tuple[Path, Path]]:
    if PAIR_MODE == "all_successive":
        return list(zip(scene_dirs[:-1], scene_dirs[1:]))
    if PAIR_MODE == "annual_anchor_sequence":
        by_date: Dict[str, Path] = {scene_date_from_dir(p).date().isoformat(): p for p in scene_dirs}
        missing = [d for d in ANNUAL_ANCHOR_DATES if d not in by_date]
        if missing:
            raise FileNotFoundError("Missing annual-anchor semantic mask folders: " + ", ".join(missing))
        ordered = [by_date[d] for d in ANNUAL_ANCHOR_DATES]
        return list(zip(ordered[:-1], ordered[1:]))
    raise ValueError("PAIR_MODE must be 'annual_anchor_sequence' or 'all_successive'")


def read_mask(path: Path) -> Tuple[np.ndarray, dict]:
    with rasterio.open(path) as ds:
        arr = ds.read(1)
        profile = ds.profile.copy()
    return arr, profile


def same_grid(profile_a: dict, profile_b: dict) -> bool:
    return (
        profile_a.get("height") == profile_b.get("height")
        and profile_a.get("width") == profile_b.get("width")
        and profile_a.get("transform") == profile_b.get("transform")
        and str(profile_a.get("crs")) == str(profile_b.get("crs"))
    )


def compute_pixel_area_ha_grid(profile: dict) -> np.ndarray:
    h = int(profile["height"])
    w = int(profile["width"])
    transform = profile["transform"]
    crs = profile.get("crs")

    if crs is not None and hasattr(crs, "is_projected") and crs.is_projected:
        px_area_m2 = abs(transform.a * transform.e)
        return np.full((h, w), px_area_m2 / 10000.0, dtype=np.float64)

    areas = np.zeros((h, w), dtype=np.float64)
    dx = transform.a
    dy = transform.e
    x_left = transform.c
    x_right = x_left + dx
    for r in range(h):
        y_top = transform.f + r * dy
        y_bot = y_top + dy
        lon = [x_left, x_right, x_right, x_left]
        lat = [y_top, y_top, y_bot, y_bot]
        area_m2, _ = GEOD.polygon_area_perimeter(lon, lat)
        areas[r, :] = abs(area_m2) / 10000.0
    return areas


def neighbors(r: int, c: int, h: int, w: int, connectivity: int) -> List[Tuple[int, int]]:
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            if connectivity == 4 and abs(dr) + abs(dc) != 1:
                continue
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w:
                out.append((rr, cc))
    return out


def filter_small_components(mask: np.ndarray, min_pixels: int, connectivity: int = 8) -> np.ndarray:
    if min_pixels <= 1:
        return mask.copy()
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    out = np.zeros((h, w), dtype=bool)
    ys, xs = np.where(mask)
    for start_r, start_c in zip(ys, xs):
        if visited[start_r, start_c]:
            continue
        q = deque([(start_r, start_c)])
        visited[start_r, start_c] = True
        comp = []
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for rr, cc in neighbors(r, c, h, w, connectivity):
                if not visited[rr, cc] and mask[rr, cc]:
                    visited[rr, cc] = True
                    q.append((rr, cc))
        if len(comp) >= min_pixels:
            for r, c in comp:
                out[r, c] = True
    return out


def binary_erode(mask: np.ndarray, iterations: int = 1, connectivity: int = 8) -> np.ndarray:
    """Pure-numpy binary erosion with a 3x3 (8-neigh) or cross (4-neigh) structuring element."""
    if iterations <= 0:
        return mask.copy()
    out = mask.copy()
    for _ in range(iterations):
        p = np.pad(out, 1, mode="constant", constant_values=False)
        if connectivity == 8:
            parts = [
                p[:-2, :-2], p[:-2, 1:-1], p[:-2, 2:],
                p[1:-1, :-2], p[1:-1, 1:-1], p[1:-1, 2:],
                p[2:, :-2], p[2:, 1:-1], p[2:, 2:],
            ]
        else:
            parts = [
                p[1:-1, 1:-1],
                p[:-2, 1:-1],
                p[2:, 1:-1],
                p[1:-1, :-2],
                p[1:-1, 2:],
            ]
        out = np.logical_and.reduce(parts)
    return out


def write_u8(path: Path, arr: np.ndarray, profile: dict, nodata: int = 255) -> None:
    p = profile.copy()
    p.update(dtype=rasterio.uint8, count=1, nodata=nodata, compress="deflate")
    with rasterio.open(path, "w", **p) as dst:
        dst.write(arr.astype(np.uint8), 1)


def write_i16(path: Path, arr: np.ndarray, profile: dict, nodata: int = -32768) -> None:
    p = profile.copy()
    p.update(dtype=rasterio.int16, count=1, nodata=nodata, compress="deflate")
    with rasterio.open(path, "w", **p) as dst:
        dst.write(arr.astype(np.int16), 1)


def boolean_mask_to_u8(mask: np.ndarray, domain: np.ndarray) -> np.ndarray:
    out = np.full(mask.shape, 255, dtype=np.uint8)
    out[domain] = 0
    out[mask] = 1
    return out


def label_components_with_stats(mask: np.ndarray,
                                min_pixels: int,
                                connectivity: int = 8) -> Tuple[np.ndarray, Dict[int, int]]:
    """Label connected components and remove components smaller than min_pixels.

    Returns
    -------
    labels : np.ndarray[int32]
        Zero means background. Positive values are sequential component IDs.
    counts : dict[int, int]
        Pixel count per retained component ID.
    """
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    labels = np.zeros((h, w), dtype=np.int32)
    counts: Dict[int, int] = {}
    next_label = 0

    ys, xs = np.where(mask)
    for start_r, start_c in zip(ys, xs):
        if visited[start_r, start_c]:
            continue
        q = deque([(start_r, start_c)])
        visited[start_r, start_c] = True
        comp = []
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for rr, cc in neighbors(r, c, h, w, connectivity):
                if not visited[rr, cc] and mask[rr, cc]:
                    visited[rr, cc] = True
                    q.append((rr, cc))

        if len(comp) >= min_pixels:
            next_label += 1
            for r, c in comp:
                labels[r, c] = next_label
            counts[next_label] = len(comp)

    return labels, counts


def polygonize_class(mask: np.ndarray,
                     class_name: str,
                     class_code: int,
                     pre_date: pd.Timestamp,
                     post_date: pd.Timestamp,
                     pair_name: str,
                     profile: dict,
                     pixel_area_ha: np.ndarray,
                     group_id: np.ndarray,
                     group_lut: Dict[int, str],
                     min_pixels: int,
                     connectivity: int):
    """Polygonize one semantic class and project polygons to TARGET_CRS.

    The function computes both geodetic raster-area sums (area_ha_grid) and
    UTM-based polygon area/perimeter after projection. The geodetic raster-area
    sum should be used when comparing with raster summaries; the UTM area is
    useful for vector/GIS display and geometry diagnostics.
    """
    if not WRITE_POLYGONS:
        return None
    if gpd is None or shape is None:
        raise RuntimeError("WRITE_POLYGONS=True requires geopandas and shapely. Install geopandas shapely fiona pyproj.")

    labels, counts = label_components_with_stats(mask, min_pixels=min_pixels, connectivity=connectivity)
    if not counts:
        return None

    raster_crs = profile.get("crs")
    if raster_crs is None:
        raise ValueError("Raster CRS is missing; cannot project polygons to UTM 34N.")

    # Pre-compute area and group statistics per component.
    area_by_label: Dict[int, float] = {}
    dom_gid_by_label: Dict[int, int] = {}
    dom_gfrac_by_label: Dict[int, float] = {}
    n_groups_by_label: Dict[int, int] = {}

    for lab in counts:
        lab_mask = labels == lab
        area_by_label[lab] = float(pixel_area_ha[lab_mask].sum())
        gids = group_id[lab_mask]
        gids = gids[gids > 0]
        if len(gids):
            vals, cnts = np.unique(gids, return_counts=True)
            imax = int(np.argmax(cnts))
            dom_gid = int(vals[imax])
            dom_gid_by_label[lab] = dom_gid
            dom_gfrac_by_label[lab] = float(cnts[imax] / cnts.sum())
            n_groups_by_label[lab] = int(len(vals))
        else:
            dom_gid_by_label[lab] = 0
            dom_gfrac_by_label[lab] = np.nan
            n_groups_by_label[lab] = 0

    records = []
    geom_records = []
    for geom_mapping, val in rasterio.features.shapes(
        labels.astype(np.int32),
        mask=labels > 0,
        transform=profile["transform"],
        connectivity=connectivity,
    ):
        lab = int(val)
        if lab <= 0:
            continue
        geom = shape(geom_mapping)
        if geom.is_empty:
            continue
        gid = dom_gid_by_label.get(lab, 0)
        group_label = group_lut.get(gid, f"group_{gid}" if gid else "UNKNOWN")
        unique_name = f"SdV_{pre_date.strftime('%Y%m%d')}_{post_date.strftime('%Y%m%d')}_{class_name}_{lab:04d}"
        records.append({
            "object_name": unique_name,
            "pair_name": pair_name,
            "pre_date": pre_date.date().isoformat(),
            "post_date": post_date.date().isoformat(),
            "class_code": class_code,
            "class_name": class_name,
            "component_id": lab,
            "pixel_count": counts.get(lab, 0),
            "area_ha_grid": area_by_label.get(lab, np.nan),
            "dominant_group_id": gid,
            "dominant_group": group_label,
            "dominant_group_frac": dom_gfrac_by_label.get(lab, np.nan),
            "n_groups": n_groups_by_label.get(lab, 0),
            "min_component_pixels": min_pixels,
            "connectivity": connectivity,
        })
        geom_records.append(geom)

    if not records:
        return None

    gdf = gpd.GeoDataFrame(records, geometry=geom_records, crs=raster_crs)
    gdf_utm = gdf.to_crs(TARGET_CRS)

    if SIMPLIFY_POLYGONS and SIMPLIFY_TOLERANCE_M > 0:
        gdf_utm["geometry"] = gdf_utm.geometry.simplify(
            tolerance=SIMPLIFY_TOLERANCE_M,
            preserve_topology=SIMPLIFY_PRESERVE_TOPOLOGY,
        )
        gdf_utm = gdf_utm[~gdf_utm.geometry.is_empty].copy()

    gdf_utm["area_ha_utm"] = gdf_utm.geometry.area / 10000.0
    gdf_utm["perimeter_m"] = gdf_utm.geometry.length
    gdf_utm["simplified"] = bool(SIMPLIFY_POLYGONS)
    gdf_utm["simplify_tol_m"] = float(SIMPLIFY_TOLERANCE_M) if SIMPLIFY_POLYGONS else 0.0
    return gdf_utm


def write_pair_polygons(out_dir: Path,
                        pair_name: str,
                        pre_date: pd.Timestamp,
                        post_date: pd.Timestamp,
                        profile: dict,
                        pixel_area_ha: np.ndarray,
                        group_use: np.ndarray,
                        group_lut: Dict[int, str],
                        masks_by_class: Dict[str, np.ndarray]):
    """Write per-pair polygons and return a GeoDataFrame with all classes.

    Outputs
    -------
    change_objects_utm34n.gpkg with:
      - one combined layer (optional)
      - one separate layer per class (optional)
    change_objects_utm34n_inventory.csv
    """
    if not WRITE_POLYGONS:
        return None

    class_codes = {
        "stable_drop": 0,
        "stable_keep": 1,
        "loss": 2,
        "gain": 3,
    }
    frames = []
    by_class: Dict[str, object] = {}
    for class_name in POLYGONIZE_CLASSES:
        if class_name not in masks_by_class:
            continue
        gdf = polygonize_class(
            masks_by_class[class_name],
            class_name=class_name,
            class_code=class_codes[class_name],
            pre_date=pre_date,
            post_date=post_date,
            pair_name=pair_name,
            profile=profile,
            pixel_area_ha=pixel_area_ha,
            group_id=group_use,
            group_lut=group_lut,
            min_pixels=MIN_POLYGON_COMPONENT_PIXELS,
            connectivity=POLYGONIZE_CONNECTIVITY,
        )
        if gdf is not None and not gdf.empty:
            by_class[class_name] = gdf
            frames.append(gdf)

    if not frames:
        return None

    pair_gdf = pd.concat(frames, ignore_index=True)
    pair_gdf = gpd.GeoDataFrame(pair_gdf, geometry="geometry", crs=TARGET_CRS)

    gpkg_path = out_dir / "change_objects_utm34n.gpkg"
    if gpkg_path.exists():
        gpkg_path.unlink()

    if WRITE_COMBINED_POLYGON_LAYER:
        pair_gdf.to_file(gpkg_path, layer=COMBINED_LAYER_NAME, driver="GPKG")

    if WRITE_SEPARATE_CLASS_LAYERS:
        for class_name in POLYGONIZE_CLASSES:
            gdf = by_class.get(class_name)
            if gdf is None or gdf.empty:
                continue
            gdf.to_file(gpkg_path, layer=class_name, driver="GPKG")

    pair_gdf.drop(columns="geometry").to_csv(out_dir / "change_objects_utm34n_inventory.csv", index=False)
    return pair_gdf


def write_confirmed_loss_raster(path: Path,
                                confirmed_mask: np.ndarray,
                                domain: np.ndarray,
                                profile: dict) -> None:
    """Write a binary confirmation raster: 1 confirmed, 0 not confirmed/background, 255 nodata."""
    out = np.full(confirmed_mask.shape, 255, dtype=np.uint8)
    out[domain] = 0
    out[confirmed_mask] = 1
    write_u8(path, out, profile, nodata=255)


def write_confirmed_previous_loss_products(scene_dirs: List[Path],
                                           pixel_area_ha: np.ndarray,
                                           group_lut: Dict[int, str]):
    """Create persistence-supported previous-loss rasters and polygons.

    For every adjacent pair of intervals:
        previous interval: t_i -> t_{i+1}
        current interval:  t_{i+1} -> t_{i+2}

    the function computes:
        loss_confirmed_by_next_stable_drop = previous loss AND current stable_drop

    The output is written into the previous interval folder because it corrects
    the previous interval loss layer.
    """
    if not WRITE_CONFIRMED_PREVIOUS_LOSS:
        return []

    if len(scene_dirs) < 3:
        return []

    summary_rows = []
    confirmed_frames = []
    unconfirmed_frames = []

    for i in range(len(scene_dirs) - 2):
        prev_scene = scene_dirs[i]
        middle_scene = scene_dirs[i + 1]
        next_scene = scene_dirs[i + 2]

        previous_pair_name = f"{prev_scene.name}_to_{middle_scene.name}"
        current_pair_name = f"{middle_scene.name}_to_{next_scene.name}"

        previous_out_dir = OUTPUT_ROOT / previous_pair_name
        current_out_dir = OUTPUT_ROOT / current_pair_name
        previous_out_dir.mkdir(parents=True, exist_ok=True)

        prev_loss_path = previous_out_dir / "loss_mask.tif"
        curr_stable_drop_path = current_out_dir / "stable_drop_mask.tif"
        if not prev_loss_path.exists() or not curr_stable_drop_path.exists():
            continue

        prev_loss, prof_prev = read_mask(prev_loss_path)
        curr_stable_drop, prof_curr = read_mask(curr_stable_drop_path)
        if not same_grid(prof_prev, prof_curr):
            raise ValueError(f"Grid mismatch between {previous_pair_name} loss and {current_pair_name} stable_drop")

        # Use group IDs from the middle date where possible. This is the date at
        # which the previous loss has just become drop and is also the pre-date
        # for the current stable_drop comparison.
        group_mid, prof_group = read_mask(middle_scene / GROUP_MASK_NAME)
        if not same_grid(prof_prev, prof_group):
            raise ValueError(f"Group mask is not aligned for middle date {middle_scene.name}")

        domain = (prev_loss != 255) & (curr_stable_drop != 255)
        prev_loss_bool = prev_loss == 1
        curr_stable_bool = curr_stable_drop == 1

        confirmed_raw = prev_loss_bool & curr_stable_bool
        confirmed = filter_small_components(
            confirmed_raw,
            min_pixels=CONFIRMED_LOSS_MIN_PIXELS,
            connectivity=CONFIRMED_LOSS_CONNECTIVITY,
        )

        # Optional companion layer: loss candidates from the previous interval
        # that do not overlap current stable_drop. This is a diagnostic layer;
        # it should not automatically be interpreted as false loss.
        unconfirmed_raw = prev_loss_bool & domain & (~curr_stable_bool)
        unconfirmed = filter_small_components(
            unconfirmed_raw,
            min_pixels=CONFIRMED_LOSS_MIN_PIXELS,
            connectivity=CONFIRMED_LOSS_CONNECTIVITY,
        ) if WRITE_UNCONFIRMED_PREVIOUS_LOSS else np.zeros_like(prev_loss_bool, dtype=bool)

        confirmed_raster = previous_out_dir / f"{CONFIRMED_LOSS_NAME}.tif"
        write_confirmed_loss_raster(confirmed_raster, confirmed, domain, prof_prev)

        if WRITE_UNCONFIRMED_PREVIOUS_LOSS:
            unconfirmed_raster = previous_out_dir / f"{UNCONFIRMED_LOSS_NAME}.tif"
            write_confirmed_loss_raster(unconfirmed_raster, unconfirmed, domain, prof_prev)

        confirmed_area = float(pixel_area_ha[confirmed].sum())
        unconfirmed_area = float(pixel_area_ha[unconfirmed].sum()) if WRITE_UNCONFIRMED_PREVIOUS_LOSS else np.nan
        prev_loss_area = float(pixel_area_ha[prev_loss_bool].sum())
        raw_confirmed_area = float(pixel_area_ha[confirmed_raw].sum())

        # Polygonize confirmed layer and write it as a separate layer in the same
        # per-pair GeoPackage, plus as a dedicated global GeoPackage.
        pre_ts = pd.Timestamp(prev_scene.name)
        post_ts = pd.Timestamp(middle_scene.name)

        confirmed_gdf = None
        if WRITE_POLYGONS:
            confirmed_gdf = polygonize_class(
                confirmed,
                class_name=CONFIRMED_LOSS_NAME,
                class_code=CONFIRMED_LOSS_CLASS_CODE,
                pre_date=pre_ts,
                post_date=post_ts,
                pair_name=previous_pair_name,
                profile=prof_prev,
                pixel_area_ha=pixel_area_ha,
                group_id=group_mid,
                group_lut=group_lut,
                min_pixels=CONFIRMED_LOSS_MIN_PIXELS,
                connectivity=CONFIRMED_LOSS_CONNECTIVITY,
            )
            if confirmed_gdf is not None and not confirmed_gdf.empty:
                pair_gpkg = previous_out_dir / "change_objects_utm34n.gpkg"
                confirmed_gdf.to_file(pair_gpkg, layer=CONFIRMED_LOSS_NAME, driver="GPKG")
                confirmed_gdf.drop(columns="geometry").to_csv(
                    previous_out_dir / f"{CONFIRMED_LOSS_NAME}_inventory.csv",
                    index=False,
                )
                confirmed_frames.append(confirmed_gdf)

            if WRITE_UNCONFIRMED_PREVIOUS_LOSS:
                unconfirmed_gdf = polygonize_class(
                    unconfirmed,
                    class_name=UNCONFIRMED_LOSS_NAME,
                    class_code=UNCONFIRMED_LOSS_CLASS_CODE,
                    pre_date=pre_ts,
                    post_date=post_ts,
                    pair_name=previous_pair_name,
                    profile=prof_prev,
                    pixel_area_ha=pixel_area_ha,
                    group_id=group_mid,
                    group_lut=group_lut,
                    min_pixels=CONFIRMED_LOSS_MIN_PIXELS,
                    connectivity=CONFIRMED_LOSS_CONNECTIVITY,
                )
                if unconfirmed_gdf is not None and not unconfirmed_gdf.empty:
                    pair_gpkg = previous_out_dir / "change_objects_utm34n.gpkg"
                    unconfirmed_gdf.to_file(pair_gpkg, layer=UNCONFIRMED_LOSS_NAME, driver="GPKG")
                    unconfirmed_gdf.drop(columns="geometry").to_csv(
                        previous_out_dir / f"{UNCONFIRMED_LOSS_NAME}_inventory.csv",
                        index=False,
                    )
                    unconfirmed_frames.append(unconfirmed_gdf)

        confirmed_components = int(confirmed_gdf.shape[0]) if confirmed_gdf is not None and not confirmed_gdf.empty else 0
        summary_rows.append({
            "previous_pair": previous_pair_name,
            "current_pair": current_pair_name,
            "pre_date": prev_scene.name,
            "post_date": middle_scene.name,
            "next_date": next_scene.name,
            "previous_loss_px": int(prev_loss_bool.sum()),
            "previous_loss_area_ha": prev_loss_area,
            "raw_confirmed_px_before_size_filter": int(confirmed_raw.sum()),
            "raw_confirmed_area_ha_before_size_filter": raw_confirmed_area,
            "confirmed_px": int(confirmed.sum()),
            "confirmed_area_ha": confirmed_area,
            "unconfirmed_px": int(unconfirmed.sum()) if WRITE_UNCONFIRMED_PREVIOUS_LOSS else np.nan,
            "unconfirmed_area_ha": unconfirmed_area,
            "confirmed_components": confirmed_components,
            "confirmation_rule": "previous loss AND current stable_drop",
            "min_component_pixels": CONFIRMED_LOSS_MIN_PIXELS,
            "connectivity": CONFIRMED_LOSS_CONNECTIVITY,
        })

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(CONFIRMED_LOSS_SUMMARY_CSV, index=False)

    if WRITE_POLYGONS and confirmed_frames:
        confirmed_all = pd.concat(confirmed_frames, ignore_index=True)
        confirmed_all = gpd.GeoDataFrame(confirmed_all, geometry="geometry", crs=TARGET_CRS)
        if GLOBAL_CONFIRMED_LOSS_GPKG.exists():
            GLOBAL_CONFIRMED_LOSS_GPKG.unlink()
        confirmed_all.to_file(GLOBAL_CONFIRMED_LOSS_GPKG, layer=CONFIRMED_LOSS_NAME, driver="GPKG")
        confirmed_all.drop(columns="geometry").to_csv(
            OUTPUT_ROOT / f"{CONFIRMED_LOSS_NAME}_inventory.csv",
            index=False,
        )

        if WRITE_UNCONFIRMED_PREVIOUS_LOSS and unconfirmed_frames:
            unconfirmed_all = pd.concat(unconfirmed_frames, ignore_index=True)
            unconfirmed_all = gpd.GeoDataFrame(unconfirmed_all, geometry="geometry", crs=TARGET_CRS)
            unconfirmed_all.to_file(GLOBAL_CONFIRMED_LOSS_GPKG, layer=UNCONFIRMED_LOSS_NAME, driver="GPKG")
            unconfirmed_all.drop(columns="geometry").to_csv(
                OUTPUT_ROOT / f"{UNCONFIRMED_LOSS_NAME}_inventory.csv",
                index=False,
            )

    return summary_rows


def summarize_by_group(group_id: np.ndarray,
                       common_domain_raw: np.ndarray,
                       retained_domain: np.ndarray,
                       stable_keep: np.ndarray,
                       stable_drop: np.ndarray,
                       loss: np.ndarray,
                       gain: np.ndarray,
                       filtered_out: np.ndarray,
                       pixel_area_ha: np.ndarray,
                       group_lut: Dict[int, str]) -> pd.DataFrame:
    rows = []
    gids = sorted([g for g in np.unique(group_id[common_domain_raw]) if g > 0])
    for gid in gids:
        gmask = group_id == gid
        m_raw = common_domain_raw & gmask
        if not np.any(m_raw):
            continue
        m_ret = retained_domain & gmask
        raw_px = int(m_raw.sum())
        retained_px = int(m_ret.sum())
        rows.append({
            "group_id": gid,
            "group_label": group_lut.get(gid, f"group_{gid}"),
            "comparable_px_raw": raw_px,
            "comparable_px_after_filter": retained_px,
            "filtered_out_px": int((filtered_out & gmask).sum()),
            "stable_keep_px": int((stable_keep & gmask).sum()),
            "stable_drop_px": int((stable_drop & gmask).sum()),
            "loss_px": int((loss & gmask).sum()),
            "gain_px": int((gain & gmask).sum()),
            "stable_keep_area_ha": float(pixel_area_ha[stable_keep & gmask].sum()),
            "stable_drop_area_ha": float(pixel_area_ha[stable_drop & gmask].sum()),
            "loss_area_ha": float(pixel_area_ha[loss & gmask].sum()),
            "gain_area_ha": float(pixel_area_ha[gain & gmask].sum()),
            "loss_pct_of_retained": float(100.0 * (loss & gmask).sum() / retained_px) if retained_px else np.nan,
            "gain_pct_of_retained": float(100.0 * (gain & gmask).sum() / retained_px) if retained_px else np.nan,
        })
    return pd.DataFrame(rows)



_S1_INV_CACHE: pd.DataFrame | None = None


def get_s1_inventory_cached() -> pd.DataFrame:
    global _S1_INV_CACHE
    if _S1_INV_CACHE is None:
        _S1_INV_CACHE = load_s1_inventory(S1_CACHE_ROOT)
    return _S1_INV_CACHE


def resolve_s1_descriptors_for_delta(pre_row: pd.Series, post_row: pd.Series, pre_stack: dict, post_stack: dict) -> List[str]:
    cfg = S1_DESCRIPTORS_FOR_DELTA
    if isinstance(cfg, str) and cfg.lower().strip() == "auto":
        pre_names = set()
        post_names = set()
        for txt, store in [(str(pre_row.get("descriptor_bands_available", "")), pre_names), (str(post_row.get("descriptor_bands_available", "")), post_names)]:
            for name in [x.strip() for x in txt.split(",") if x.strip()]:
                if name not in S1_DESCRIPTOR_EXCLUDE_FROM_DELTA:
                    store.add(name)
        if not pre_names:
            pre_names = {name for name, arr in pre_stack.items() if name not in S1_DESCRIPTOR_EXCLUDE_FROM_DELTA and arr is not None and np.isfinite(arr).any()}
        if not post_names:
            post_names = {name for name, arr in post_stack.items() if name not in S1_DESCRIPTOR_EXCLUDE_FROM_DELTA and arr is not None and np.isfinite(arr).any()}
        ordered = []
        for name in S1_DESCRIPTOR_BANDS:
            if name in pre_names and name in post_names and name not in S1_DESCRIPTOR_EXCLUDE_FROM_DELTA:
                ordered.append(name)
        for name in sorted((pre_names & post_names) - set(ordered)):
            ordered.append(name)
        return ordered
    return [str(x) for x in cfg if str(x) not in S1_DESCRIPTOR_EXCLUDE_FROM_DELTA]


def load_s1_pair_descriptor_data(pre_date: pd.Timestamp, post_date: pd.Timestamp, profile: dict):
    """Load nearest Sentinel-1 stacks for a date pair and compute descriptor deltas."""
    meta = {
        "s1_pair_descriptors_enabled": bool(ENABLE_S1_PAIR_DESCRIPTORS),
        "s1_pre_scene": "",
        "s1_post_scene": "",
        "s1_pre_date": "",
        "s1_post_date": "",
        "s1_pre_abs_offset_days": np.nan,
        "s1_post_abs_offset_days": np.nan,
        "s1_pair_available": False,
    }
    if not ENABLE_S1_PAIR_DESCRIPTORS:
        return meta, {}, {}, {}, np.zeros((profile["height"], profile["width"]), dtype=bool)
    inv = get_s1_inventory_cached()
    if inv.empty:
        return meta, {}, {}, {}, np.zeros((profile["height"], profile["width"]), dtype=bool)
    pre_row = find_nearest_s1_scene(inv, pre_date, S1_MAX_ABS_DATE_OFFSET_DAYS)
    post_row = find_nearest_s1_scene(inv, post_date, S1_MAX_ABS_DATE_OFFSET_DAYS)
    if pre_row is None or post_row is None:
        return meta, {}, {}, {}, np.zeros((profile["height"], profile["width"]), dtype=bool)
    pre_stack_path = Path(str(pre_row.get("s1_stack_path", "")))
    post_stack_path = Path(str(post_row.get("s1_stack_path", "")))
    pre_valid_path = Path(str(pre_row.get("s1_valid_mask_path", "")))
    post_valid_path = Path(str(post_row.get("s1_valid_mask_path", "")))
    if not all(p.exists() for p in [pre_stack_path, post_stack_path, pre_valid_path, post_valid_path]):
        return meta, {}, {}, {}, np.zeros((profile["height"], profile["width"]), dtype=bool)
    try:
        pre_stack, pre_profile = read_float32_stack(pre_stack_path, expected_names=S1_DESCRIPTOR_BANDS)
    except Exception:
        pre_stack, pre_profile = read_float32_stack(pre_stack_path)
    try:
        post_stack, post_profile = read_float32_stack(post_stack_path, expected_names=S1_DESCRIPTOR_BANDS)
    except Exception:
        post_stack, post_profile = read_float32_stack(post_stack_path)
    s1_descriptors_for_delta = resolve_s1_descriptors_for_delta(pre_row, post_row, pre_stack, post_stack)
    pre_valid, pre_valid_profile = read_uint8_raster(pre_valid_path)
    post_valid, post_valid_profile = read_uint8_raster(post_valid_path)
    for p in [pre_profile, post_profile, pre_valid_profile, post_valid_profile]:
        if not same_grid(profile, p):
            raise RuntimeError("Sentinel-1 descriptor cache grid is not aligned with Algorithm #3 semantic grid")
    valid_common = (pre_valid == 1) & (post_valid == 1)
    deltas = {}
    for name in s1_descriptors_for_delta:
        a = pre_stack.get(name)
        b = post_stack.get(name)
        if a is None or b is None:
            deltas[name] = np.full(valid_common.shape, np.nan, dtype=np.float32)
        else:
            out = np.full(valid_common.shape, np.nan, dtype=np.float32)
            ok = valid_common & np.isfinite(a) & np.isfinite(b)
            out[ok] = b[ok] - a[ok]
            deltas[name] = out
    meta.update({
        "s1_pre_scene": str(pre_row.get("scene", "")),
        "s1_post_scene": str(post_row.get("scene", "")),
        "s1_pre_date": pd.Timestamp(pre_row["date"]).date().isoformat(),
        "s1_post_date": pd.Timestamp(post_row["date"]).date().isoformat(),
        "s1_pre_abs_offset_days": int(pre_row.get("abs_offset_days", abs((pd.Timestamp(pre_row['date']) - pre_date).days))),
        "s1_post_abs_offset_days": int(post_row.get("abs_offset_days", abs((pd.Timestamp(post_row['date']) - post_date).days))),
        "s1_pair_available": True,
        "s1_descriptors_for_delta": ",".join(s1_descriptors_for_delta),
    })
    return meta, pre_stack, post_stack, deltas, valid_common


def summarize_s1_pair_descriptors(pre_stack: dict, post_stack: dict, deltas: dict, valid_common: np.ndarray, masks_by_class: Dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    if not pre_stack or not post_stack or not deltas:
        return pd.DataFrame()
    for class_name, mask in masks_by_class.items():
        m = mask & valid_common
        for name in list(deltas.keys()):
            pre_arr = pre_stack.get(name)
            post_arr = post_stack.get(name)
            delta_arr = deltas.get(name)
            ok = m
            if pre_arr is not None:
                ok = ok & np.isfinite(pre_arr)
            if post_arr is not None:
                ok = ok & np.isfinite(post_arr)
            if delta_arr is not None:
                ok = ok & np.isfinite(delta_arr)
            if int(ok.sum()) == 0:
                rows.append({"class_name": class_name, "descriptor": name, "valid_px": 0})
                continue
            rows.append({
                "class_name": class_name,
                "descriptor": name,
                "valid_px": int(ok.sum()),
                "pre_mean": float(np.nanmean(pre_arr[ok])),
                "post_mean": float(np.nanmean(post_arr[ok])),
                "delta_mean": float(np.nanmean(delta_arr[ok])),
                "delta_std": float(np.nanstd(delta_arr[ok])),
                "delta_min": float(np.nanmin(delta_arr[ok])),
                "delta_max": float(np.nanmax(delta_arr[ok])),
            })
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    scene_dirs = discover_scene_dirs(MASK_ROOT)
    if len(scene_dirs) < 2:
        raise RuntimeError("Need at least two scene folders to compare semantic observations.")
    pair_dirs = select_pair_dirs(scene_dirs)
    if not pair_dirs:
        raise RuntimeError("No date pairs selected for Algorithm #3.")

    pair_rows = []
    pair_group_rows = []
    all_polygon_frames = []
    group_label_map: Dict[int, str] = {}

    _, first_profile = read_mask(pair_dirs[0][0] / KEEP_MASK_NAME)
    pixel_area_ha = compute_pixel_area_ha_grid(first_profile)

    for pre_dir, post_dir in pair_dirs:
        pre_date = scene_date_from_dir(pre_dir)
        post_date = scene_date_from_dir(post_dir)
        pair_name = f"{pre_date.date().isoformat()}_to_{post_date.date().isoformat()}"
        out_dir = OUTPUT_ROOT / pair_name
        out_dir.mkdir(parents=True, exist_ok=True)

        keep_pre, prof_pre = read_mask(pre_dir / KEEP_MASK_NAME)
        keep_post, prof_post = read_mask(post_dir / KEEP_MASK_NAME)
        if not same_grid(prof_pre, prof_post):
            raise ValueError(f"Grid mismatch between {pre_dir.name} and {post_dir.name}")

        vote_pre, prof_vp = read_mask(pre_dir / VOTE_MASK_NAME)
        vote_post, prof_vq = read_mask(post_dir / VOTE_MASK_NAME)
        group_pre, prof_gp = read_mask(pre_dir / GROUP_MASK_NAME)
        group_post, prof_gq = read_mask(post_dir / GROUP_MASK_NAME)
        for other in (prof_vp, prof_vq, prof_gp, prof_gq):
            if not same_grid(prof_pre, other):
                raise ValueError(f"Mask grids not aligned for pair {pair_name}")

        if np.any((group_pre > 0) != (group_post > 0)) or np.any((group_pre != group_post) & (group_pre > 0) & (group_post > 0)):
            group_use = np.where(group_post > 0, group_post, group_pre)
        else:
            group_use = group_pre

        for group_summary_csv in (pre_dir / "group_scene_summary.csv", post_dir / "group_scene_summary.csv"):
            if group_summary_csv.exists() and group_summary_csv.stat().st_size > 0:
                try:
                    gdf = pd.read_csv(group_summary_csv)
                except Exception:
                    gdf = pd.DataFrame()
                if not gdf.empty and {"group_id", "group_code"}.issubset(gdf.columns):
                    for r in gdf[["group_id", "group_code"]].drop_duplicates().itertuples(index=False):
                        group_label_map[int(r.group_id)] = str(r.group_code)

        common_domain_raw = (keep_pre != 255) & (keep_post != 255) & (group_use > 0)
        good_votes = common_domain_raw & (vote_pre != 255) & (vote_post != 255)
        vote_delta = np.full(common_domain_raw.shape, -32768, dtype=np.int16)
        vp = vote_pre.astype(np.int16)
        vq = vote_post.astype(np.int16)
        vote_delta[good_votes] = vq[good_votes] - vp[good_votes]

        # Raw mutually exclusive classes.
        stable_drop_raw = common_domain_raw & (keep_pre == 0) & (keep_post == 0)
        stable_keep_raw = common_domain_raw & (keep_pre == 1) & (keep_post == 1)
        loss_raw = common_domain_raw & (keep_pre == 1) & (keep_post == 0)
        gain_raw = common_domain_raw & (keep_pre == 0) & (keep_post == 1)

        # Stricter vote-based gating to reduce inflated boundaries.
        if USE_STRICT_VOTE_GATING:
            loss_gate = good_votes & (vp >= MIN_PRE_VOTES_FOR_LOSS) & (vq <= MAX_POST_VOTES_FOR_LOSS) & (vote_delta <= -MIN_VOTE_DROP_FOR_LOSS)
            gain_gate = good_votes & (vp <= MAX_PRE_VOTES_FOR_GAIN) & (vq >= MIN_POST_VOTES_FOR_GAIN) & (vote_delta >= MIN_VOTE_RISE_FOR_GAIN)
            loss_strict = loss_raw & loss_gate
            gain_strict = gain_raw & gain_gate
        else:
            loss_strict = loss_raw.copy()
            gain_strict = gain_raw.copy()

        # Optional erosion to keep the core of change objects.
        if ERODE_CHANGE_MASKS and ERODE_ITERATIONS > 0:
            loss_core = binary_erode(loss_strict, ERODE_ITERATIONS, ERODE_CONNECTIVITY)
            gain_core = binary_erode(gain_strict, ERODE_ITERATIONS, ERODE_CONNECTIVITY)
        else:
            loss_core = loss_strict.copy()
            gain_core = gain_strict.copy()

        if MIN_CLASS_PATCH_PIXELS > 1:
            if FILTER_STABLE_CLASSES:
                stable_drop = filter_small_components(stable_drop_raw, MIN_CLASS_PATCH_PIXELS, CONNECTIVITY)
                stable_keep = filter_small_components(stable_keep_raw, MIN_CLASS_PATCH_PIXELS, CONNECTIVITY)
            else:
                stable_drop = stable_drop_raw.copy()
                stable_keep = stable_keep_raw.copy()
            loss = filter_small_components(loss_core, MIN_CLASS_PATCH_PIXELS, CONNECTIVITY)
            gain = filter_small_components(gain_core, MIN_CLASS_PATCH_PIXELS, CONNECTIVITY)
        else:
            stable_drop = stable_drop_raw.copy()
            stable_keep = stable_keep_raw.copy()
            loss = loss_core.copy()
            gain = gain_core.copy()

        retained_domain = stable_drop | stable_keep | loss | gain
        filtered_out = common_domain_raw & (~retained_domain)

        change_class = np.full(common_domain_raw.shape, 255, dtype=np.uint8)
        change_class[stable_drop] = 0
        change_class[stable_keep] = 1
        change_class[loss] = 2
        change_class[gain] = 3
        write_u8(out_dir / "change_class.tif", change_class, prof_pre)

        write_u8(out_dir / "stable_drop_mask.tif", boolean_mask_to_u8(stable_drop, common_domain_raw), prof_pre)
        write_u8(out_dir / "stable_keep_mask.tif", boolean_mask_to_u8(stable_keep, common_domain_raw), prof_pre)
        write_u8(out_dir / "loss_mask.tif", boolean_mask_to_u8(loss, common_domain_raw), prof_pre)
        write_u8(out_dir / "gain_mask.tif", boolean_mask_to_u8(gain, common_domain_raw), prof_pre)
        write_u8(out_dir / "filtered_out_mask.tif", boolean_mask_to_u8(filtered_out, common_domain_raw), prof_pre)
        write_i16(out_dir / "vote_delta.tif", vote_delta, prof_pre)

        # Reusable intermediate transition masks for Algorithm #4. These allow
        # object/RDF export to reuse exactly the raw and strict masks created here.
        write_u8(out_dir / "raw_loss_keep_to_drop_mask.tif", boolean_mask_to_u8(loss_raw, common_domain_raw), prof_pre)
        write_u8(out_dir / "raw_reentry_drop_to_keep_mask.tif", boolean_mask_to_u8(gain_raw, common_domain_raw), prof_pre)
        write_u8(out_dir / "loss_after_vote_gate_mask.tif", boolean_mask_to_u8(loss_strict, common_domain_raw), prof_pre)
        write_u8(out_dir / "gain_after_vote_gate_mask.tif", boolean_mask_to_u8(gain_strict, common_domain_raw), prof_pre)
        write_u8(out_dir / "loss_after_erosion_mask.tif", boolean_mask_to_u8(loss_core, common_domain_raw), prof_pre)
        write_u8(out_dir / "gain_after_erosion_mask.tif", boolean_mask_to_u8(gain_core, common_domain_raw), prof_pre)

        s1_pair_meta, s1_pre_stack, s1_post_stack, s1_delta_stack, s1_valid_common = load_s1_pair_descriptor_data(pre_date, post_date, prof_pre)
        s1_summary_df = summarize_s1_pair_descriptors(
            s1_pre_stack,
            s1_post_stack,
            s1_delta_stack,
            s1_valid_common,
            {
                "stable_drop": stable_drop,
                "stable_keep": stable_keep,
                "loss": loss,
                "gain": gain,
                "loss_raw": loss_raw,
                "gain_raw": gain_raw,
            },
        )
        if not s1_summary_df.empty:
            s1_summary_df.to_csv(out_dir / "s1_pair_descriptor_summary.csv", index=False)
        if WRITE_S1_DELTA_STACK and s1_pair_meta.get("s1_pair_available") and s1_delta_stack:
            write_float32_stack(out_dir / "s1_descriptor_delta_stack.tif", s1_delta_stack, prof_pre, list(s1_delta_stack.keys()))

        # Optional polygonization of selected classes. The polygons are projected
        # to UTM 34N and isolated regions smaller than MIN_POLYGON_COMPONENT_PIXELS
        # are removed before vector export.
        pair_polygons = write_pair_polygons(
            out_dir=out_dir,
            pair_name=pair_name,
            pre_date=pre_date,
            post_date=post_date,
            profile=prof_pre,
            pixel_area_ha=pixel_area_ha,
            group_use=group_use,
            group_lut=group_label_map,
            masks_by_class={
                "stable_drop": stable_drop,
                "stable_keep": stable_keep,
                "loss": loss,
                "gain": gain,
            },
        )
        if pair_polygons is not None and not pair_polygons.empty:
            all_polygon_frames.append(pair_polygons)

        raw_comparable_px = int(common_domain_raw.sum())
        retained_px = int(retained_domain.sum())
        filtered_out_px = int(filtered_out.sum())
        pair_summary = {
            "pre_date": pre_date.date().isoformat(),
            "post_date": post_date.date().isoformat(),
            "days_between": int((post_date - pre_date).days),
            "comparable_px_raw": raw_comparable_px,
            "comparable_px_after_filter": retained_px,
            "filtered_out_px": filtered_out_px,
            "filtered_out_pct_of_raw": float(100.0 * filtered_out_px / raw_comparable_px) if raw_comparable_px else np.nan,
            "stable_keep_px": int(stable_keep.sum()),
            "stable_drop_px": int(stable_drop.sum()),
            "loss_px": int(loss.sum()),
            "gain_px": int(gain.sum()),
            "loss_raw_keep_to_drop_px": int(loss_raw.sum()),
            "gain_raw_drop_to_keep_px": int(gain_raw.sum()),
            "loss_after_vote_gate_px": int(loss_strict.sum()),
            "gain_after_vote_gate_px": int(gain_strict.sum()),
            "loss_after_erosion_px": int(loss_core.sum()),
            "gain_after_erosion_px": int(gain_core.sum()),
            "stable_keep_area_ha": float(pixel_area_ha[stable_keep].sum()),
            "stable_drop_area_ha": float(pixel_area_ha[stable_drop].sum()),
            "loss_area_ha": float(pixel_area_ha[loss].sum()),
            "gain_area_ha": float(pixel_area_ha[gain].sum()),
            "loss_pct_of_retained": float(100.0 * loss.sum() / retained_px) if retained_px else np.nan,
            "gain_pct_of_retained": float(100.0 * gain.sum() / retained_px) if retained_px else np.nan,
            "min_class_patch_pixels": MIN_CLASS_PATCH_PIXELS,
            "connectivity": CONNECTIVITY,
            "use_strict_vote_gating": USE_STRICT_VOTE_GATING,
            "erode_change_masks": ERODE_CHANGE_MASKS,
            "erode_iterations": ERODE_ITERATIONS,
            "erode_connectivity": ERODE_CONNECTIVITY,
            **s1_pair_meta,
        }
        pd.DataFrame([pair_summary]).to_csv(out_dir / "pair_summary.csv", index=False)
        pair_rows.append(pair_summary)

        pair_group_df = summarize_by_group(
            group_use,
            common_domain_raw,
            retained_domain,
            stable_keep,
            stable_drop,
            loss,
            gain,
            filtered_out,
            pixel_area_ha,
            group_label_map,
        )
        if not pair_group_df.empty:
            pair_group_df.insert(0, "post_date", post_date.date().isoformat())
            pair_group_df.insert(0, "pre_date", pre_date.date().isoformat())
            pair_group_df.to_csv(out_dir / "pair_group_summary.csv", index=False)
            pair_group_rows.append(pair_group_df)

    pd.DataFrame(pair_rows).to_csv(OUTPUT_ROOT / "successive_pair_summary.csv", index=False)
    if pair_group_rows:
        pd.concat(pair_group_rows, ignore_index=True).to_csv(
            OUTPUT_ROOT / "successive_pair_group_summary.csv", index=False
        )

    if WRITE_POLYGONS and all_polygon_frames:
        all_polygons = pd.concat(all_polygon_frames, ignore_index=True)
        all_polygons = gpd.GeoDataFrame(all_polygons, geometry="geometry", crs=TARGET_CRS)
        if GLOBAL_POLYGON_GPKG.exists():
            GLOBAL_POLYGON_GPKG.unlink()

        if WRITE_COMBINED_POLYGON_LAYER:
            all_polygons.to_file(GLOBAL_POLYGON_GPKG, layer=COMBINED_LAYER_NAME, driver="GPKG")

        if WRITE_SEPARATE_CLASS_LAYERS:
            for class_name in POLYGONIZE_CLASSES:
                class_gdf = all_polygons[all_polygons["class_name"] == class_name].copy()
                if class_gdf.empty:
                    continue
                class_gdf.to_file(GLOBAL_POLYGON_GPKG, layer=class_name, driver="GPKG")

        all_polygons.drop(columns="geometry").to_csv(POLYGONIZATION_INVENTORY_CSV, index=False)

    # Optional temporal composite layer: previous loss AND current stable_drop.
    confirmed_loss_rows = write_confirmed_previous_loss_products(
        scene_dirs=[pair_dirs[0][0]] + [p for _, p in pair_dirs],
        pixel_area_ha=pixel_area_ha,
        group_lut=group_label_map,
    )

    readme = OUTPUT_ROOT / "README.txt"
    readme.write_text(
        "Algorithm #3 Phase 5 annual-anchor semantic transition detection from Phase 5 S1/S2 semantic masks plus optional Sentinel-1 pair descriptors.\n\n"
        f"Pair mode: {PAIR_MODE}\n"
        f"Semantic mask root: {MASK_ROOT}\n"

        "Interpretation:\n"
        "- loss (class 2): healthy-compatible at date t, not healthy-compatible at date t+1\n"
        "- gain (class 3): not healthy-compatible at date t, healthy-compatible at date t+1\n"
        "- stable_keep (class 1): healthy-compatible at both dates\n"
        "- stable_drop (class 0): not healthy-compatible at both dates\n\n"
        "Controls against inflated boundaries:\n"
        f"- Strict vote gating = {USE_STRICT_VOTE_GATING}\n"
        f"  loss: pre_votes>={MIN_PRE_VOTES_FOR_LOSS}, post_votes<={MAX_POST_VOTES_FOR_LOSS}, vote_delta<=-{MIN_VOTE_DROP_FOR_LOSS}\n"
        f"  gain: pre_votes<={MAX_PRE_VOTES_FOR_GAIN}, post_votes>={MIN_POST_VOTES_FOR_GAIN}, vote_delta>={MIN_VOTE_RISE_FOR_GAIN}\n"
        f"- Erode change masks = {ERODE_CHANGE_MASKS}, iterations = {ERODE_ITERATIONS}, connectivity = {ERODE_CONNECTIVITY}\n"
        f"- Minimum class patch size = {MIN_CLASS_PATCH_PIXELS} pixels\n"
        f"- Connectivity = {CONNECTIVITY}\n"
        f"- Filter stable classes too = {FILTER_STABLE_CLASSES}\n\n"
        f"Sentinel-1 pair descriptors enabled: {ENABLE_S1_PAIR_DESCRIPTORS}\n"
        f"Sentinel-1 cache root: {S1_CACHE_ROOT}\n"
        f"Sentinel-1 max date offset: {S1_MAX_ABS_DATE_OFFSET_DAYS} days\n"
        f"Sentinel-1 descriptors: {S1_DESCRIPTORS_FOR_DELTA}\n"
        "Raw transition masks for Algorithm #4 reuse: raw_loss_keep_to_drop, raw_reentry_drop_to_keep, loss_after_vote_gate, loss_after_erosion\n\n"
        "Important note:\n"
        "These are semantic changes relative to the healthy forest envelope, not final disturbance types.\n"
        "They are best used as candidate changes for later attribution.\n",
        encoding="utf-8",
    )

    print(f"Done. Outputs in {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
