"""
Build RDF exports and GraphDB-friendly pruning outputs for raw/core/fringe
semantic-loss objects over an annual anchor sequence.

Phase 5 aligns Algorithm #4 with the Phase 3/4 cache-aware workflow, optionally
reuses Algorithm #3 transition rasters, and attaches Sentinel-1 descriptor and
Phase 4 anomaly/confidence features to each loss object.

This version generalizes the one-interval prototype to the eligible-scene annual
sequence and aligns it with the scene-filtered semantic-mask workflow.

Annual sequence
---------------
2017-08-25 -> 2018-08-25
2018-08-25 -> 2019-07-01
2019-07-01 -> 2020-08-09
2020-08-09 -> 2021-08-09
2021-08-09 -> 2022-07-25
2022-07-25 -> 2023-08-04
2023-08-04 -> 2024-07-29
2024-07-29 -> 2025-08-28

Inputs
------
Primary mode reads already generated eligible semantic masks from:
D:/Forest_Disturbance/outputs/semantic_masks_empirical_envelopes_blacklist_and_caution_eligible

If a required date folder is missing, the script can try to generate it by
importing:
C:/Users/JOHN/PycharmProjects/JSTARS/.venv/apply_empirical_group_envelopes_semantic_mask_scene_filtered.py
and discovering the corresponding EO Browser ZIP by date.

Outputs
-------
For each interval:
- object_features.csv
- rule_test_results.csv
- loss_objects_rdf.ttl
- inferred_actions.ttl
- summary.txt
- rasters/raw_loss_object_id.tif
- rasters/strict_core_object_id.tif
- rasters/fringe_object_id.tif
- rasters/raw_core_fringe_class.tif
- rasters/final_status_code.tif
- rasters/retained_feasible_object_id.tif
- rasters/review_object_or_fringe_id.tif
- rasters/rejected_object_id.tif

Global:
- object_features_all_intervals.csv
- rule_test_results_all_intervals.csv
- annual_interval_summary.csv
- graphdb_pruning_rule_catalog.csv
- graphdb_pruning_rules_v1.ttl
- graphdb_pruning_rules_v1.rq
- loss_objects_all_intervals_rdf.ttl
- summary_all_intervals.txt

Object logic
------------
- raw    = raw keep->drop component from annual semantic masks
- core   = strict-core vote-gated loss inside the raw object
- fringe = raw minus core
- feasible geometry is determined by rule action:
    AcceptCorePlusFringe    -> raw geometry
    AcceptCoreOnly          -> core geometry
    AcceptCoreReviewFringe  -> core geometry accepted, fringe reviewed
    RejectNoise/ReviewObject -> no automatic feasible geometry

Persistence logic
-----------------
For interval t_i -> t_{i+1}, the next interval t_{i+1} -> t_{i+2}
is used when available.
- next persistence = overlap with next stable_drop
- next expansion   = overlap with next strict loss
- next support     = overlap with next stable_drop OR next strict loss

This separates persistence of the current drop state from next-year expansion.
"""
from __future__ import annotations

import importlib.util
import math
import re
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import rasterio
import rasterio.features
from pyproj import CRS, Geod, Transformer
from shapely.geometry import shape
from shapely.ops import transform as shp_transform, unary_union

try:
    from sdv_shared import (
        DEFAULT_S1_DESCRIPTOR_BANDS,
        match_nearest_scene_by_date,
        read_float32_stack,
    )
except Exception:
    DEFAULT_S1_DESCRIPTOR_BANDS = (
        "S1_VH_DB", "S1_VH_LINEAR", "S1_VH_DB_MEAN3", "S1_VH_DB_STD3",
        "S1_VH_LINEAR_MEAN3", "S1_VH_LINEAR_STD3",
    )
    match_nearest_scene_by_date = None
    read_float32_stack = None

# =============================================================================
# HARD-CODED CONFIG
# =============================================================================
JOINFIX_SCRIPT = Path("C:/Users/JOHN/PycharmProjects/JSTARS/.venv/apply_empirical_group_envelopes_semantic_mask_phase5.py")
IMAGERY_ZIP_ROOT = Path("D:/Forest_Disturbance/imagery_zip/Stana_de_Vale_S2")
SEMANTIC_MASK_ROOT = Path("D:/Forest_Disturbance/outputs/semantic_masks_s1s2_empirical_envelopes_blacklist_and_caution_eligible")
CHANGE_DETECTION_ROOT = Path("D:/Forest_Disturbance/outputs/semantic_change_detection_s1s2_blacklist_and_caution_eligible")
OUTPUT_ROOT = Path("D:/Forest_Disturbance/outputs/rdf_loss_objects_graphdb_annual_s1s2_blacklist_and_caution_eligible_phase5")

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

# Use existing eligible semantic masks when present. If False, masks are
# regenerated for all dates through JOINFIX_SCRIPT.
USE_EXISTING_MASKS = True
GENERATE_MISSING_MASKS = False

KEEP_MASK_NAME = "semantic_keep_mask.tif"
VOTE_MASK_NAME = "semantic_vote_count.tif"
GROUP_MASK_NAME = "semantic_group_id.tif"
GROUP_SUMMARY_NAME = "group_scene_summary.csv"

# Phase 4 / Phase 5 optional inputs.
ANOMALY_MASK_NAME = "semantic_anomaly_score.tif"
ENVELOPE_CONFIDENCE_MASK_NAME = "semantic_envelope_confidence.tif"
USE_ALGORITHM3_TRANSITION_RASTERS = True
RAW_LOSS_MASK_NAME = "raw_loss_keep_to_drop_mask.tif"
RAW_REENTRY_MASK_NAME = "raw_reentry_drop_to_keep_mask.tif"
STRICT_LOSS_MASK_NAME = "loss_mask.tif"
STABLE_DROP_PAIR_MASK_NAME = "stable_drop_mask.tif"
STABLE_KEEP_PAIR_MASK_NAME = "stable_keep_mask.tif"

USE_S1_DESCRIPTOR_CACHE = True
S1_CACHE_ROOT = Path("D:/Forest_Disturbance/outputs/sdv_phase5_sentinel1_descriptor_cache")
S1_INVENTORY_CSV = S1_CACHE_ROOT / "s1_scene_inventory.csv"
S1_MAX_MATCH_DAYS = 45
# Use "auto" to attach only S1 descriptors that are available in the nearest
# cache scene. This avoids all-nodata VV object attributes when the S1 ZIPs
# contain VH only.
S1_DESCRIPTOR_NAMES = "auto"
S1_DESCRIPTOR_EXCLUDE_FROM_OBJECT_STATS = {"S1_VALID"}

# Strict-core settings used to define the core inside each raw object.
MIN_CLASS_PATCH_PIXELS = 25
CONNECTIVITY = 8
MIN_PRE_VOTES_FOR_LOSS = 2
MAX_POST_VOTES_FOR_LOSS = 0
MIN_VOTE_DROP_FOR_LOSS = 2

# Objectization on the raw loss mask.
MIN_RAW_OBJECT_PIXELS = 25

# Optional erosion; disabled to match the latest annual strict-core results.
ERODE_CHANGE_MASKS = False
ERODE_ITERATIONS = 1
ERODE_CONNECTIVITY = 8

# Geometry transform for area/perimeter/compactness.
PROJECTED_CRS = "EPSG:32634"

# Namespaces for RDF.
NS_FD = "http://example.org/forest-disturbance/"
NS_GEO = "http://www.opengis.net/ont/geosparql#"
NS_XSD = "http://www.w3.org/2001/XMLSchema#"
NS_RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
NS_RDFS = "http://www.w3.org/2000/01/rdf-schema#"

GEOD = Geod(ellps="WGS84")
DATE_RE = re.compile(r"(20\d{2})[-_](\d{2})[-_](\d{2})")

# =============================================================================
# Basic helpers
# =============================================================================
def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module specification from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def interval_id(pre_date: str, post_date: str) -> str:
    return f"{pre_date.replace('-', '_')}_{post_date.replace('-', '_')}"


def interval_label(pre_date: str, post_date: str) -> str:
    return f"{pre_date}_to_{post_date}"


def neighbors(r: int, c: int, h: int, w: int, connectivity: int) -> Iterable[Tuple[int, int]]:
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            if connectivity == 4 and abs(dr) + abs(dc) != 1:
                continue
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w:
                yield rr, cc


def label_components(mask: np.ndarray, connectivity: int = 8) -> np.ndarray:
    h, w = mask.shape
    labels = np.zeros(mask.shape, dtype=np.int32)
    visited = np.zeros(mask.shape, dtype=bool)
    label = 0
    ys, xs = np.where(mask)
    for start_r, start_c in zip(ys, xs):
        if visited[start_r, start_c]:
            continue
        label += 1
        q = deque([(start_r, start_c)])
        visited[start_r, start_c] = True
        labels[start_r, start_c] = label
        while q:
            r, c = q.popleft()
            for rr, cc in neighbors(r, c, h, w, connectivity):
                if not visited[rr, cc] and mask[rr, cc]:
                    visited[rr, cc] = True
                    labels[rr, cc] = label
                    q.append((rr, cc))
    return labels


def filter_small_components(mask: np.ndarray, min_pixels: int, connectivity: int = 8) -> np.ndarray:
    if min_pixels <= 1:
        return mask.copy()
    labels = label_components(mask, connectivity)
    ids, counts = np.unique(labels[labels > 0], return_counts=True)
    keep_ids = set(ids[counts >= min_pixels].tolist())
    if not keep_ids:
        return np.zeros_like(mask, dtype=bool)
    return np.isin(labels, list(keep_ids))


def binary_erode(mask: np.ndarray, iterations: int = 1, connectivity: int = 8) -> np.ndarray:
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
                p[1:-1, 1:-1], p[:-2, 1:-1], p[2:, 1:-1],
                p[1:-1, :-2], p[1:-1, 2:],
            ]
        out = np.logical_and.reduce(parts)
    return out


def read_mask(path: Path) -> Tuple[np.ndarray, dict]:
    with rasterio.open(path) as ds:
        return ds.read(1), ds.profile.copy()


def optional_read_float(path: Path, ref_profile: dict, fill_value: float = np.nan) -> Tuple[np.ndarray, bool]:
    """Read an optional Float32 raster aligned to ref_profile.

    Returns a fill array and False when the file is absent. Nodata values are
    converted to NaN when the file is present.
    """
    path = Path(path)
    shape = (int(ref_profile["height"]), int(ref_profile["width"]))
    if not path.exists():
        return np.full(shape, fill_value, dtype=np.float32), False
    with rasterio.open(path) as ds:
        prof = ds.profile.copy()
        if not same_grid(ref_profile, prof):
            raise ValueError(f"Optional float raster is not aligned with reference grid: {path}")
        arr = ds.read(1).astype(np.float32)
        if ds.nodata is not None:
            arr[np.isclose(arr, float(ds.nodata))] = np.nan
    return arr, True


def optional_read_u8(path: Path, ref_profile: dict, fill_value: int = 255) -> Tuple[np.ndarray, bool]:
    """Read an optional UInt8 raster aligned to ref_profile.

    Returns a fill array and False when the file is absent.
    """
    path = Path(path)
    shape = (int(ref_profile["height"]), int(ref_profile["width"]))
    if not path.exists():
        return np.full(shape, fill_value, dtype=np.uint8), False
    with rasterio.open(path) as ds:
        prof = ds.profile.copy()
        if not same_grid(ref_profile, prof):
            raise ValueError(f"Optional UInt8 raster is not aligned with reference grid: {path}")
        arr = ds.read(1).astype(np.uint8)
    return arr, True


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


def transformer_from_profile(profile: dict, target_crs: str = PROJECTED_CRS) -> Transformer:
    source_crs = profile.get("crs")
    if source_crs is None:
        raise ValueError("Raster profile has no CRS; cannot project geometry safely.")
    return Transformer.from_crs(source_crs, CRS.from_string(target_crs), always_xy=True)


def crs_wkt_uri_from_profile(profile: dict) -> str:
    crs = profile.get("crs")
    if crs is None:
        return ""
    epsg = crs.to_epsg() if hasattr(crs, "to_epsg") else None
    if epsg is not None:
        return f"<http://www.opengis.net/def/crs/EPSG/0/{epsg}> "
    return ""


def mask_to_geometry(mask: np.ndarray, transform) -> object | None:
    if not np.any(mask):
        return None
    geoms = [
        shape(geom)
        for geom, val in rasterio.features.shapes(mask.astype(np.uint8), mask=mask.astype(bool), transform=transform)
        if val == 1
    ]
    if not geoms:
        return None
    if len(geoms) == 1:
        return geoms[0]
    return unary_union(geoms)


def project_geometry(geom, transformer: Transformer):
    return shp_transform(transformer.transform, geom)


def geom_area_ha(geom, transformer: Transformer) -> float:
    if geom is None or geom.is_empty:
        return 0.0
    return project_geometry(geom, transformer).area / 10000.0


def geom_compactness(geom, transformer: Transformer) -> float:
    if geom is None or geom.is_empty:
        return float("nan")
    g = project_geometry(geom, transformer)
    if g.length == 0:
        return float("nan")
    return float(4.0 * math.pi * g.area / (g.length * g.length))


def turtle_string(s: str) -> str:
    return str(s).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def typed_decimal(val: float) -> Optional[str]:
    if pd.isna(val):
        return None
    return f'"{float(val):.8f}"^^xsd:decimal'


def typed_int(val: int) -> str:
    return f'"{int(val)}"^^xsd:integer'


def typed_bool(val: bool) -> str:
    return '"true"^^xsd:boolean' if bool(val) else '"false"^^xsd:boolean'


def typed_date(val: str) -> str:
    return f'"{val}"^^xsd:date'


def decimal_prop(prop: str, val: float) -> Optional[str]:
    lit = typed_decimal(val)
    if lit is None:
        return None
    return f"    fd:{prop} {lit} ;"


def find_zip_for_date(date: str) -> Path:
    candidates = sorted(IMAGERY_ZIP_ROOT.glob(f"*{date}*.zip"))
    if not candidates:
        raise FileNotFoundError(f"No EO Browser ZIP found for {date} in {IMAGERY_ZIP_ROOT}")
    if len(candidates) > 1:
        # Prefer names that indicate good scene quality when multiple ZIPs exist.
        preferred_terms = ["no_clouds", "Low_clouds", "LowMedium_clouds"]
        for term in preferred_terms:
            matches = [p for p in candidates if term.lower() in p.name.lower()]
            if matches:
                return matches[0]
    return candidates[0]



def safe_mean(arr: np.ndarray, mask: np.ndarray) -> float:
    if arr is None or not np.any(mask):
        return np.nan
    vals = np.asarray(arr, dtype=np.float32)[mask]
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if vals.size else np.nan


def finite_fraction(arr: np.ndarray, mask: np.ndarray) -> float:
    if arr is None or not np.any(mask):
        return np.nan
    return float(np.isfinite(arr[mask]).sum() / mask.sum()) if mask.sum() else np.nan


def load_s1_inventory() -> pd.DataFrame:
    if not USE_S1_DESCRIPTOR_CACHE or not S1_INVENTORY_CSV.exists() or match_nearest_scene_by_date is None:
        return pd.DataFrame()
    inv = pd.read_csv(S1_INVENTORY_CSV)
    if "s1_descriptor_stack_path" in inv.columns:
        inv = inv[inv["s1_descriptor_stack_path"].astype(str).ne("")].copy()
        inv["date"] = pd.to_datetime(inv.get("s1_date", inv.get("date", "")), errors="coerce")
        inv["scene"] = inv.get("s1_scene", inv.get("scene", ""))
        inv["descriptor_stack"] = inv["s1_descriptor_stack_path"]
    elif "s1_stack_path" in inv.columns:
        inv = inv[inv["s1_stack_path"].astype(str).ne("")].copy()
        inv["date"] = pd.to_datetime(inv.get("date", ""), errors="coerce")
        inv["scene"] = inv.get("scene", "")
        inv["descriptor_stack"] = inv["s1_stack_path"]
    elif "descriptor_stack" not in inv.columns:
        return pd.DataFrame()
    if "date" in inv.columns:
        inv["date"] = pd.to_datetime(inv["date"], errors="coerce")
    return inv.dropna(subset=["date"])


def _resolve_s1_object_descriptor_names(match: pd.Series, stack: dict) -> List[str]:
    cfg = S1_DESCRIPTOR_NAMES
    if isinstance(cfg, str) and cfg.lower().strip() == "auto":
        found = []
        txt = str(match.get("descriptor_bands_available", ""))
        for name in [x.strip() for x in txt.split(",") if x.strip()]:
            if name not in found and name not in S1_DESCRIPTOR_EXCLUDE_FROM_OBJECT_STATS:
                found.append(name)
        if not found:
            for name, arr in stack.items():
                if name in S1_DESCRIPTOR_EXCLUDE_FROM_OBJECT_STATS:
                    continue
                if arr is not None and np.isfinite(arr).any():
                    found.append(name)
        return found
    return [str(x) for x in cfg if str(x) not in S1_DESCRIPTOR_EXCLUDE_FROM_OBJECT_STATS]


def load_s1_stack_for_date(date: str, inventory: pd.DataFrame, ref_profile: dict):
    if inventory is None or inventory.empty or read_float32_stack is None:
        return {}, None
    match = match_nearest_scene_by_date(pd.Timestamp(date), inventory, max_days=S1_MAX_MATCH_DAYS)
    if match is None:
        return {}, None
    stack_path = Path(str(match.get("s1_stack_path", match.get("descriptor_stack", ""))))
    if not stack_path.exists():
        return {}, None
    if isinstance(S1_DESCRIPTOR_NAMES, str) and S1_DESCRIPTOR_NAMES.lower().strip() == "auto":
        stack_all, prof = read_float32_stack(stack_path)
        names = _resolve_s1_object_descriptor_names(match, stack_all)
        stack = {name: stack_all[name] for name in names if name in stack_all}
    else:
        stack, prof = read_float32_stack(stack_path, requested_band_names=S1_DESCRIPTOR_NAMES)
    if not same_grid(ref_profile, prof):
        raise ValueError(f"S1 descriptor stack not aligned to semantic grid: {stack_path}")
    match["matched_offset_days"] = int((pd.Timestamp(match["date"]) - pd.Timestamp(date)).days)
    match["s1_descriptors_loaded"] = ",".join(stack.keys())
    return stack, match

# =============================================================================
# Rule pack
# =============================================================================
RULES = [
    {
        "rule_id": "R1_Reject_NoCoreSmallWeak",
        "category": "reject",
        "label": "Reject small coreless transient object",
        "antecedent": "coreAreaHa = 0 and rawAreaHa < 0.50 and rawPersistenceNextFrac < 0.20",
        "consequence": "RejectNoise",
    },
    {
        "rule_id": "R2_Reject_NoCoreTransient",
        "category": "reject",
        "label": "Reject coreless transient fringe-only object",
        "antecedent": "coreAreaHa = 0 and fringeRatio >= 0.99 and rawPersistenceNextFrac < 0.35",
        "consequence": "RejectNoise",
    },
    {
        "rule_id": "R3_Reject_WeakCoreFragment",
        "category": "reject",
        "label": "Reject weak fragmented object with tiny core",
        "antecedent": "coreAreaHa > 0 and coreRatio < 0.15 and rawAreaHa < 0.75 and meanVoteDropRaw > -2.20",
        "consequence": "RejectNoise",
    },
    {
        "rule_id": "R4_Reject_VeryFringyLowPersistence",
        "category": "reject",
        "label": "Reject very fringy low-persistence object",
        "antecedent": "fringeRatio > 0.75 and rawPersistenceNextFrac < 0.20 and coreRatio < 0.25",
        "consequence": "RejectNoise",
    },
    {
        "rule_id": "R5_Accept_StrongCore",
        "category": "accept_core",
        "label": "Accept strong spectrally coherent core",
        "antecedent": "coreAreaHa >= 0.10 and meanVoteDropCore <= -2.50",
        "consequence": "AcceptCore",
    },
    {
        "rule_id": "R6_Accept_PersistentCore",
        "category": "accept_core",
        "label": "Accept persistent core",
        "antecedent": "coreAreaHa > 0 and corePersistenceNextFrac >= 0.50",
        "consequence": "AcceptCore",
    },
    {
        "rule_id": "R7_Accept_GroupCoherentCore",
        "category": "accept_core",
        "label": "Accept group-coherent compact core",
        "antecedent": "coreAreaHa > 0 and dominantGroupFrac >= 0.80 and coreRatio >= 0.40 and compactness >= 0.04",
        "consequence": "AcceptCore",
    },
    {
        "rule_id": "R8_Keep_PersistentFringe",
        "category": "accept_fringe",
        "label": "Keep persistent fringe contiguous to a core",
        "antecedent": "fringeAreaHa > 0 and fringePersistenceNextFrac >= 0.40 and coreAreaHa > 0",
        "consequence": "AcceptFringe",
    },
    {
        "rule_id": "R9_Keep_CoherentSmallFringe",
        "category": "accept_fringe",
        "label": "Keep small coherent fringe",
        "antecedent": "fringeAreaHa > 0 and fringeRatio <= 0.35 and dominantGroupFrac >= 0.80 and rawPersistenceNextFrac >= 0.20",
        "consequence": "AcceptFringe",
    },
    {
        "rule_id": "R10_Review_MultiGroupExpansion",
        "category": "review",
        "label": "Review multi-group expansion",
        "antecedent": "nGroups >= 3 and dominantGroupFrac < 0.60",
        "consequence": "NeedsReview",
    },
    {
        "rule_id": "R11_Review_LargeFringe",
        "category": "review",
        "label": "Review object dominated by uncertain fringe",
        "antecedent": "fringeRatio > 0.60 and coreAreaHa > 0 and fringePersistenceNextFrac < 0.40",
        "consequence": "NeedsReview",
    },
    {
        "rule_id": "R12_Review_LargeWeakCore",
        "category": "review",
        "label": "Review large object with weak core",
        "antecedent": "rawAreaHa >= 2.00 and coreRatio < 0.35",
        "consequence": "NeedsReview",
    },
]


def evaluate_rules(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Missing persistence values should not fire persistence-dependent rules.
    raw_p = out.raw_persistence_next_frac.fillna(-1.0)
    core_p = out.core_persistence_next_frac.fillna(-1.0)
    fringe_p = out.fringe_persistence_next_frac.fillna(-1.0)

    out["R1_Reject_NoCoreSmallWeak"] = (out.core_area_ha == 0) & (out.raw_area_ha < 0.50) & (raw_p < 0.20) & (raw_p >= 0)
    out["R2_Reject_NoCoreTransient"] = (out.core_area_ha == 0) & (out.fringe_ratio >= 0.99) & (raw_p < 0.35) & (raw_p >= 0)
    out["R3_Reject_WeakCoreFragment"] = (out.core_area_ha > 0) & (out.core_ratio < 0.15) & (out.raw_area_ha < 0.75) & (out.mean_vote_drop_raw > -2.20)
    out["R4_Reject_VeryFringyLowPersistence"] = (out.fringe_ratio > 0.75) & (raw_p < 0.20) & (raw_p >= 0) & (out.core_ratio < 0.25)

    out["R5_Accept_StrongCore"] = (out.core_area_ha >= 0.10) & (out.mean_vote_drop_core <= -2.50)
    out["R6_Accept_PersistentCore"] = (out.core_area_ha > 0) & (core_p >= 0.50)
    out["R7_Accept_GroupCoherentCore"] = (out.core_area_ha > 0) & (out.dominant_group_frac >= 0.80) & (out.core_ratio >= 0.40) & (out.compactness >= 0.04)

    out["R8_Keep_PersistentFringe"] = (out.fringe_area_ha > 0) & (fringe_p >= 0.40) & (out.core_area_ha > 0)
    out["R9_Keep_CoherentSmallFringe"] = (out.fringe_area_ha > 0) & (out.fringe_ratio <= 0.35) & (out.dominant_group_frac >= 0.80) & (raw_p >= 0.20)

    out["R10_Review_MultiGroupExpansion"] = (out.n_groups >= 3) & (out.dominant_group_frac < 0.60)
    out["R11_Review_LargeFringe"] = (out.fringe_ratio > 0.60) & (out.core_area_ha > 0) & ((fringe_p < 0.40) | (fringe_p < 0))
    out["R12_Review_LargeWeakCore"] = (out.raw_area_ha >= 2.00) & (out.core_ratio < 0.35)

    reject_cols = ["R1_Reject_NoCoreSmallWeak", "R2_Reject_NoCoreTransient", "R3_Reject_WeakCoreFragment", "R4_Reject_VeryFringyLowPersistence"]
    accept_core_cols = ["R5_Accept_StrongCore", "R6_Accept_PersistentCore", "R7_Accept_GroupCoherentCore"]
    accept_fringe_cols = ["R8_Keep_PersistentFringe", "R9_Keep_CoherentSmallFringe"]
    review_cols = ["R10_Review_MultiGroupExpansion", "R11_Review_LargeFringe", "R12_Review_LargeWeakCore"]

    out["reject_object"] = out[reject_cols].any(axis=1)
    out["accept_core"] = out[accept_core_cols].any(axis=1) & (~out["reject_object"])
    out["accept_fringe"] = out[accept_fringe_cols].any(axis=1) & out["accept_core"] & (~out["R11_Review_LargeFringe"])
    out["needs_review"] = out[review_cols].any(axis=1) | ((~out["reject_object"]) & (~out["accept_core"]))

    def final_status(row) -> str:
        if row["reject_object"]:
            return "RejectNoise"
        if row["accept_core"] and row["accept_fringe"] and not row["needs_review"]:
            return "AcceptCorePlusFringe"
        if row["accept_core"] and row["needs_review"]:
            return "AcceptCoreReviewFringe"
        if row["accept_core"]:
            return "AcceptCoreOnly"
        return "ReviewObject"

    out["final_status"] = out.apply(final_status, axis=1)
    out["retained_area_ha"] = np.where(
        out["final_status"] == "AcceptCorePlusFringe",
        out["raw_area_ha"],
        np.where(
            out["final_status"].isin(["AcceptCoreOnly", "AcceptCoreReviewFringe"]),
            out["core_area_ha"],
            0.0,
        ),
    )
    return out

# =============================================================================
# RDF writers
# =============================================================================
def write_rule_catalog_csv(path: Path) -> None:
    pd.DataFrame(RULES).to_csv(path, index=False)


def write_rule_catalog_ttl(path: Path) -> None:
    lines = [
        f"@prefix fd: <{NS_FD}> .",
        f"@prefix rdfs: <{NS_RDFS}> .",
        f"@prefix xsd: <{NS_XSD}> .",
        "",
        "fd:Rule a rdfs:Class .",
        "fd:RejectNoise a rdfs:Resource .",
        "fd:AcceptCore a rdfs:Resource .",
        "fd:AcceptFringe a rdfs:Resource .",
        "fd:NeedsReview a rdfs:Resource .",
        "",
    ]
    for rule in RULES:
        rid = rule["rule_id"]
        lines.extend([
            f"fd:{rid} a fd:Rule ;",
            f'    rdfs:label "{turtle_string(rule["label"])}" ;',
            f'    fd:ruleCategory "{turtle_string(rule["category"])}" ;',
            f'    fd:antecedentText "{turtle_string(rule["antecedent"])}" ;',
            f'    fd:consequenceText "{turtle_string(rule["consequence"])}" .',
            "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_graphdb_rule_updates(path: Path) -> None:
    # These updates assume object features have been loaded into GraphDB as RDF literals.
    # The final precedence resolution is still performed in Python in this script.
    def q(rule_id: str, where_body: str, consequence: str) -> str:
        return f"""# {rule_id}
PREFIX fd: <{NS_FD}>
PREFIX xsd: <{NS_XSD}>
INSERT {{
  ?o fd:triggeredRule fd:{rule_id} ;
     fd:inferredAction fd:{consequence} .
}}
WHERE {{
{where_body}
}};

"""
    parts = []
    parts.append(q("R1_Reject_NoCoreSmallWeak", """  ?o a fd:LossCandidateObject ;
     fd:coreAreaHa ?core ;
     fd:rawAreaHa ?raw ;
     fd:rawPersistenceNextFrac ?p .
  FILTER(?core = 0 && ?raw < 0.50 && ?p < 0.20)""", "RejectNoise"))
    parts.append(q("R2_Reject_NoCoreTransient", """  ?o a fd:LossCandidateObject ;
     fd:coreAreaHa ?core ;
     fd:fringeRatio ?fr ;
     fd:rawPersistenceNextFrac ?p .
  FILTER(?core = 0 && ?fr >= 0.99 && ?p < 0.35)""", "RejectNoise"))
    parts.append(q("R3_Reject_WeakCoreFragment", """  ?o a fd:LossCandidateObject ;
     fd:coreAreaHa ?core ;
     fd:coreRatio ?cr ;
     fd:rawAreaHa ?raw ;
     fd:meanVoteDropRaw ?vd .
  FILTER(?core > 0 && ?cr < 0.15 && ?raw < 0.75 && ?vd > -2.20)""", "RejectNoise"))
    parts.append(q("R4_Reject_VeryFringyLowPersistence", """  ?o a fd:LossCandidateObject ;
     fd:fringeRatio ?fr ;
     fd:rawPersistenceNextFrac ?p ;
     fd:coreRatio ?cr .
  FILTER(?fr > 0.75 && ?p < 0.20 && ?cr < 0.25)""", "RejectNoise"))
    parts.append(q("R5_Accept_StrongCore", """  ?o a fd:LossCandidateObject ;
     fd:coreAreaHa ?core ;
     fd:meanVoteDropCore ?vd .
  FILTER(?core >= 0.10 && ?vd <= -2.50)""", "AcceptCore"))
    parts.append(q("R6_Accept_PersistentCore", """  ?o a fd:LossCandidateObject ;
     fd:coreAreaHa ?core ;
     fd:corePersistenceNextFrac ?p .
  FILTER(?core > 0 && ?p >= 0.50)""", "AcceptCore"))
    parts.append(q("R7_Accept_GroupCoherentCore", """  ?o a fd:LossCandidateObject ;
     fd:coreAreaHa ?core ;
     fd:dominantGroupFrac ?g ;
     fd:coreRatio ?cr ;
     fd:compactness ?c .
  FILTER(?core > 0 && ?g >= 0.80 && ?cr >= 0.40 && ?c >= 0.04)""", "AcceptCore"))
    parts.append(q("R8_Keep_PersistentFringe", """  ?o a fd:LossCandidateObject ;
     fd:fringeAreaHa ?fr ;
     fd:fringePersistenceNextFrac ?p ;
     fd:coreAreaHa ?core .
  FILTER(?fr > 0 && ?p >= 0.40 && ?core > 0)""", "AcceptFringe"))
    parts.append(q("R9_Keep_CoherentSmallFringe", """  ?o a fd:LossCandidateObject ;
     fd:fringeAreaHa ?fr ;
     fd:fringeRatio ?frR ;
     fd:dominantGroupFrac ?g ;
     fd:rawPersistenceNextFrac ?p .
  FILTER(?fr > 0 && ?frR <= 0.35 && ?g >= 0.80 && ?p >= 0.20)""", "AcceptFringe"))
    parts.append(q("R10_Review_MultiGroupExpansion", """  ?o a fd:LossCandidateObject ;
     fd:nGroups ?n ;
     fd:dominantGroupFrac ?g .
  FILTER(?n >= 3 && ?g < 0.60)""", "NeedsReview"))
    parts.append(q("R11_Review_LargeFringe", """  ?o a fd:LossCandidateObject ;
     fd:fringeRatio ?frR ;
     fd:coreAreaHa ?core ;
     fd:fringePersistenceNextFrac ?p .
  FILTER(?frR > 0.60 && ?core > 0 && ?p < 0.40)""", "NeedsReview"))
    parts.append(q("R12_Review_LargeWeakCore", """  ?o a fd:LossCandidateObject ;
     fd:rawAreaHa ?raw ;
     fd:coreRatio ?cr .
  FILTER(?raw >= 2.00 && ?cr < 0.35)""", "NeedsReview"))
    path.write_text("\n".join(parts), encoding="utf-8")


def wkt_literal(geom, profile: dict) -> Optional[str]:
    if geom is None or geom.is_empty:
        return None
    crs_prefix = crs_wkt_uri_from_profile(profile)
    return f'"{crs_prefix}{geom.wkt}"^^geo:wktLiteral'


def write_objects_ttl(path: Path, df: pd.DataFrame, object_geoms: Dict[str, Dict[str, object]], profile: dict) -> None:
    lines = [
        f"@prefix fd: <{NS_FD}> .",
        f"@prefix geo: <{NS_GEO}> .",
        f"@prefix xsd: <{NS_XSD}> .",
        f"@prefix rdf: <{NS_RDF}> .",
        f"@prefix rdfs: <{NS_RDFS}> .",
        "",
    ]

    for status in ["RejectNoise", "AcceptCoreOnly", "AcceptCorePlusFringe", "AcceptCoreReviewFringe", "ReviewObject"]:
        lines.append(f"fd:{status} a rdfs:Resource .")
    lines.append("")

    for interval, sub in df.groupby("interval_id"):
        pre_date = sub["pre_date"].iloc[0]
        post_date = sub["post_date"].iloc[0]
        lines.extend([
            f"fd:interval_{interval} a fd:ObservationInterval ;",
            f"    fd:fromDate {typed_date(pre_date)} ;",
            f"    fd:toDate {typed_date(post_date)} .",
            "",
        ])

    for _, row in df.sort_values(["interval_id", "object_id"]).iterrows():
        uid = str(row["object_uid"])
        subj = f"fd:{uid}"
        props = [
            f"{subj} a fd:LossCandidateObject ;",
            f"    fd:inInterval fd:interval_{row['interval_id']} ;",
            f"    fd:objectId {typed_int(int(row['object_id']))} ;",
            f'    fd:intervalLabel "{turtle_string(row["interval_label"])}" ;',
            f"    fd:rawPixelCount {typed_int(int(row['raw_px']))} ;",
            f"    fd:corePixelCount {typed_int(int(row['core_px']))} ;",
            f"    fd:fringePixelCount {typed_int(int(row['fringe_px']))} ;",
        ]
        fixed_numeric_props = [
            "raw_area_ha", "core_area_ha", "fringe_area_ha", "core_ratio", "fringe_ratio",
            "mean_vote_drop_raw", "mean_vote_drop_core", "raw_persistence_next_frac",
            "core_persistence_next_frac", "fringe_persistence_next_frac", "raw_expansion_next_frac",
            "core_expansion_next_frac", "fringe_expansion_next_frac", "raw_support_next_frac",
            "core_support_next_frac", "fringe_support_next_frac", "dominant_group_frac", "compactness",
            "retained_area_ha",
        ]
        dynamic_numeric_props = [
            c for c in df.columns
            if c.startswith(("s1_", "raw_s1_", "core_s1_", "fringe_s1_", "raw_anomaly_", "core_anomaly_", "fringe_anomaly_", "raw_confidence_", "core_confidence_", "fringe_confidence_"))
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        for prop in fixed_numeric_props + sorted(dynamic_numeric_props):
            if prop not in row.index:
                continue
            lit = decimal_prop("".join([part.capitalize() if i else part for i, part in enumerate(prop.split("_"))]), row[prop])
            if lit is not None:
                props.append(lit)
        props.extend([
            f'    fd:dominantGroup "{turtle_string(row["dominant_group"])}" ;',
            f"    fd:dominantGroupId {typed_int(int(row['dominant_group_id']))} ;",
            f"    fd:nGroups {typed_int(int(row['n_groups']))} ;",
            f"    fd:touchesSceneBorder {typed_bool(bool(row['touches_scene_border']))} ;",
            f'    fd:transitionSource "{turtle_string(row.get("transition_source", "unknown"))}" ;',
            f'    fd:s1PreScene "{turtle_string(row.get("s1_pre_scene", ""))}" ;',
            f'    fd:s1PostScene "{turtle_string(row.get("s1_post_scene", ""))}" ;',
            f"    fd:acceptCore {typed_bool(bool(row['accept_core']))} ;",
            f"    fd:acceptFringe {typed_bool(bool(row['accept_fringe']))} ;",
            f"    fd:needsReview {typed_bool(bool(row['needs_review']))} ;",
            f"    fd:finalStatus fd:{row['final_status']} ;",
        ])
        triggered = [c for c in df.columns if c.startswith("R") and bool(row[c])]
        if triggered:
            props.append(f"    fd:triggeredRule {', '.join([f'fd:{r}' for r in triggered])} ;")
        props.append(f"    fd:hasRawGeometry fd:{uid}_rawGeom ;")
        if object_geoms[uid]["core"] is not None:
            props.append(f"    fd:hasCoreGeometry fd:{uid}_coreGeom ;")
        if object_geoms[uid]["fringe"] is not None:
            props.append(f"    fd:hasFringeGeometry fd:{uid}_fringeGeom ;")
        if object_geoms[uid]["feasible"] is not None:
            props.append(f"    fd:hasFeasibleGeometry fd:{uid}_feasibleGeom ;")
        props[-1] = props[-1].rstrip(" ;") + " ."
        lines.extend(props)
        lines.append("")

        for gname in ["raw", "core", "fringe", "feasible"]:
            geom = object_geoms[uid].get(gname)
            if geom is None:
                continue
            lit = wkt_literal(geom, profile)
            if lit is None:
                continue
            lines.extend([
                f"fd:{uid}_{gname}Geom a geo:Geometry ;",
                f"    geo:asWKT {lit} .",
                "",
            ])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_inferred_actions_ttl(path: Path, df: pd.DataFrame) -> None:
    lines = [f"@prefix fd: <{NS_FD}> .", f"@prefix xsd: <{NS_XSD}> .", ""]
    for _, row in df.sort_values(["interval_id", "object_id"]).iterrows():
        subj = f"fd:{row['object_uid']}"
        lines.extend([
            f"{subj} fd:finalStatus fd:{row['final_status']} ;",
            f"    fd:acceptCore {typed_bool(bool(row['accept_core']))} ;",
            f"    fd:acceptFringe {typed_bool(bool(row['accept_fringe']))} ;",
            f"    fd:needsReview {typed_bool(bool(row['needs_review']))} .",
            "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# Raster export helpers for raw/core/fringe/status products
# =============================================================================
STATUS_CODES = {
    "RejectNoise": 1,
    "AcceptCoreOnly": 2,
    "AcceptCorePlusFringe": 3,
    "AcceptCoreReviewFringe": 4,
    "ReviewObject": 5,
}

STATUS_LABELS = {
    0: "no_raw_candidate",
    1: "RejectNoise",
    2: "AcceptCoreOnly",
    3: "AcceptCorePlusFringe",
    4: "AcceptCoreReviewFringe",
    5: "ReviewObject",
    255: "nodata_or_not_comparable",
}

CORE_FRINGE_LABELS = {
    0: "no_raw_candidate",
    1: "fringe",
    2: "strict_core",
    255: "nodata_or_not_comparable",
}

TRANSITION_CLASS_LABELS = {
    0: "stable_drop",
    1: "stable_keep",
    2: "raw_loss_keep_to_drop",
    3: "raw_reentry_drop_to_keep",
    255: "nodata_or_not_comparable",
}


def write_raster(path: Path, arr: np.ndarray, profile: dict, dtype, nodata=None) -> None:
    """Write a single-band raster preserving grid/CRS from profile."""
    path.parent.mkdir(parents=True, exist_ok=True)
    p = profile.copy()
    p.update(count=1, dtype=dtype, compress="deflate")
    if nodata is not None:
        p.update(nodata=nodata)
    with rasterio.open(path, "w", **p) as dst:
        dst.write(arr.astype(dtype), 1)


def write_u8_mask(path: Path, mask: np.ndarray, domain: np.ndarray, profile: dict) -> None:
    """Write a 0/1/255 mask: 1=True, 0=False inside domain, 255=outside domain."""
    out = np.full(mask.shape, 255, dtype=np.uint8)
    out[domain] = 0
    out[mask] = 1
    write_raster(path, out, profile, rasterio.uint8, nodata=255)


def write_lookup_csv(path: Path, mapping: Dict[int, str]) -> None:
    pd.DataFrame(
        [{"code": int(code), "label": str(label)} for code, label in sorted(mapping.items())]
    ).to_csv(path, index=False)


def write_interval_rasters(
    out_dir: Path,
    profile: dict,
    tr: dict,
    object_labels: np.ndarray,
    feat_df: pd.DataFrame,
) -> None:
    """
    Export raster layers for GIS visualization of semantic transitions,
    raw/core/fringe decomposition, and rule-based pruning decisions.

    object_labels:
        int32 raster where each retained raw loss object has a local object ID.
    tr["loss_core"]:
        strict vote-gated loss core used inside each raw object.
    feat_df:
        evaluated object table containing final_status for each object.
    """
    raster_dir = out_dir / "rasters"
    raster_dir.mkdir(parents=True, exist_ok=True)

    common = tr["common"]
    loss_core = tr["loss_core"]
    loss_raw = tr["loss_raw"]
    gain_raw = tr["gain_raw"]
    stable_keep_raw = tr["stable_keep_raw"]
    stable_drop_raw = tr["stable_drop_raw"]

    # -------------------------------------------------------------------------
    # Semantic transition class raster, for context.
    # -------------------------------------------------------------------------
    transition_class = np.full(object_labels.shape, 255, dtype=np.uint8)
    transition_class[common] = 255  # remain nodata unless a valid transition class is assigned
    transition_class[stable_drop_raw] = 0
    transition_class[stable_keep_raw] = 1
    transition_class[loss_raw] = 2
    transition_class[gain_raw] = 3
    write_raster(raster_dir / "semantic_transition_class.tif", transition_class, profile, rasterio.uint8, nodata=255)
    write_lookup_csv(raster_dir / "semantic_transition_class_lookup.csv", TRANSITION_CLASS_LABELS)

    # Optional binary semantic masks for simple overlay display.
    write_u8_mask(raster_dir / "stable_drop_mask.tif", stable_drop_raw, common, profile)
    write_u8_mask(raster_dir / "stable_keep_mask.tif", stable_keep_raw, common, profile)
    write_u8_mask(raster_dir / "raw_loss_keep_to_drop_mask.tif", loss_raw, common, profile)
    write_u8_mask(raster_dir / "raw_reentry_drop_to_keep_mask.tif", gain_raw, common, profile)

    # -------------------------------------------------------------------------
    # Raw/core/fringe object-ID rasters.
    # -------------------------------------------------------------------------
    raw_id = object_labels.astype(np.int32)
    core_id = np.where((object_labels > 0) & loss_core, object_labels, 0).astype(np.int32)
    fringe_id = np.where((object_labels > 0) & (~loss_core), object_labels, 0).astype(np.int32)

    # Class map: 1 = fringe, 2 = strict core; 0 = no raw candidate inside common domain.
    core_fringe_class = np.full(object_labels.shape, 255, dtype=np.uint8)
    core_fringe_class[common] = 0
    core_fringe_class[fringe_id > 0] = 1
    core_fringe_class[core_id > 0] = 2

    # -------------------------------------------------------------------------
    # Final status map and derived object-ID rasters.
    # -------------------------------------------------------------------------
    status_map = np.full(object_labels.shape, 255, dtype=np.uint8)
    status_map[common] = 0

    retained_id = np.zeros_like(object_labels, dtype=np.int32)
    accepted_core_id = np.zeros_like(object_labels, dtype=np.int32)
    review_id = np.zeros_like(object_labels, dtype=np.int32)
    rejected_id = np.zeros_like(object_labels, dtype=np.int32)

    if not feat_df.empty and "final_status" in feat_df.columns:
        for _, row in feat_df.iterrows():
            oid = int(row["object_id"])
            status = str(row["final_status"])

            raw_obj = object_labels == oid
            core_obj = raw_obj & loss_core
            fringe_obj = raw_obj & (~loss_core)

            status_map[raw_obj] = STATUS_CODES.get(status, 0)

            if status == "AcceptCorePlusFringe":
                retained_id[raw_obj] = oid
                accepted_core_id[core_obj] = oid
            elif status == "AcceptCoreOnly":
                retained_id[core_obj] = oid
                accepted_core_id[core_obj] = oid
            elif status == "AcceptCoreReviewFringe":
                retained_id[core_obj] = oid
                accepted_core_id[core_obj] = oid
                review_id[fringe_obj] = oid
            elif status == "ReviewObject":
                review_id[raw_obj] = oid
            elif status == "RejectNoise":
                rejected_id[raw_obj] = oid

    # Object-ID rasters. Zero is nodata/no object.
    write_raster(raster_dir / "raw_loss_object_id.tif", raw_id, profile, rasterio.int32, nodata=0)
    write_raster(raster_dir / "strict_core_object_id.tif", core_id, profile, rasterio.int32, nodata=0)
    write_raster(raster_dir / "fringe_object_id.tif", fringe_id, profile, rasterio.int32, nodata=0)
    write_raster(raster_dir / "retained_feasible_object_id.tif", retained_id, profile, rasterio.int32, nodata=0)
    write_raster(raster_dir / "accepted_core_object_id.tif", accepted_core_id, profile, rasterio.int32, nodata=0)
    write_raster(raster_dir / "review_object_or_fringe_id.tif", review_id, profile, rasterio.int32, nodata=0)
    write_raster(raster_dir / "rejected_object_id.tif", rejected_id, profile, rasterio.int32, nodata=0)

    # Class/status rasters.
    write_raster(raster_dir / "raw_core_fringe_class.tif", core_fringe_class, profile, rasterio.uint8, nodata=255)
    write_lookup_csv(raster_dir / "raw_core_fringe_class_lookup.csv", CORE_FRINGE_LABELS)

    write_raster(raster_dir / "final_status_code.tif", status_map, profile, rasterio.uint8, nodata=255)
    write_lookup_csv(raster_dir / "final_status_code_lookup.csv", STATUS_LABELS)

    # Binary masks for quick GIS overlay.
    write_u8_mask(raster_dir / "raw_loss_candidate_mask.tif", raw_id > 0, common, profile)
    write_u8_mask(raster_dir / "strict_core_candidate_mask.tif", core_id > 0, common, profile)
    write_u8_mask(raster_dir / "fringe_candidate_mask.tif", fringe_id > 0, common, profile)
    write_u8_mask(raster_dir / "retained_feasible_mask.tif", retained_id > 0, common, profile)
    write_u8_mask(raster_dir / "accepted_core_mask.tif", accepted_core_id > 0, common, profile)
    write_u8_mask(raster_dir / "review_mask.tif", review_id > 0, common, profile)
    write_u8_mask(raster_dir / "rejected_mask.tif", rejected_id > 0, common, profile)

# =============================================================================
# Semantic-mask and transition helpers
# =============================================================================
def ensure_semantic_masks(date: str, joinfix_cache: dict) -> Path:
    date_dir = SEMANTIC_MASK_ROOT / date
    required = [date_dir / KEEP_MASK_NAME, date_dir / VOTE_MASK_NAME, date_dir / GROUP_MASK_NAME]
    if USE_EXISTING_MASKS and all(p.exists() for p in required):
        return date_dir

    if not GENERATE_MISSING_MASKS:
        raise FileNotFoundError(f"Missing semantic masks for {date}: {date_dir}")

    if "module" not in joinfix_cache:
        joinfix = load_module(JOINFIX_SCRIPT, "joinfix_scene_filtered")
        joinfix.OUTPUT_DIR = SEMANTIC_MASK_ROOT
        joinfix_cache["module"] = joinfix
        joinfix_cache["env"] = joinfix.load_envelopes()
        joinfix_cache["polygons"] = joinfix.load_grouped_subparcels_all()

    zip_path = find_zip_for_date(date)
    joinfix_cache["module"].process_scene(zip_path, joinfix_cache["env"], joinfix_cache["polygons"])
    if not all(p.exists() for p in required):
        raise FileNotFoundError(f"Semantic masks for {date} were not created as expected in {date_dir}")
    return date_dir


def load_group_lut(date_dir: Path) -> Dict[int, str]:
    csv_path = date_dir / GROUP_SUMMARY_NAME
    if not csv_path.exists():
        return {}
    gsum = pd.read_csv(csv_path)
    if "group_id" not in gsum.columns or "group_code" not in gsum.columns:
        return {}
    return {int(r.group_id): str(r.group_code) for r in gsum[["group_id", "group_code"]].drop_duplicates().itertuples(index=False)}




def read_algorithm3_transition_masks(pre_date: str, post_date: str, ref_profile: dict):
    if not USE_ALGORITHM3_TRANSITION_RASTERS:
        return None
    pair_dir = CHANGE_DETECTION_ROOT / interval_label(pre_date, post_date)
    required = [
        pair_dir / RAW_LOSS_MASK_NAME,
        pair_dir / STRICT_LOSS_MASK_NAME,
        pair_dir / STABLE_DROP_PAIR_MASK_NAME,
        pair_dir / STABLE_KEEP_PAIR_MASK_NAME,
    ]
    if not all(p.exists() for p in required):
        return None
    raw_loss, p0 = read_mask(pair_dir / RAW_LOSS_MASK_NAME)
    strict_loss, p1 = read_mask(pair_dir / STRICT_LOSS_MASK_NAME)
    stable_drop, p2 = read_mask(pair_dir / STABLE_DROP_PAIR_MASK_NAME)
    stable_keep, p3 = read_mask(pair_dir / STABLE_KEEP_PAIR_MASK_NAME)
    for prof in (p0, p1, p2, p3):
        if not same_grid(ref_profile, prof):
            raise ValueError(f"Algorithm #3 rasters are not aligned for {pre_date}->{post_date}")
    raw_reentry_path = pair_dir / RAW_REENTRY_MASK_NAME
    if raw_reentry_path.exists():
        raw_reentry, p4 = read_mask(raw_reentry_path)
        if not same_grid(ref_profile, p4):
            raise ValueError(f"Algorithm #3 raw re-entry raster is not aligned for {pre_date}->{post_date}")
    else:
        raw_reentry = np.full(raw_loss.shape, 255, dtype=np.uint8)
    return {
        "pair_dir": pair_dir,
        "common": raw_loss != 255,
        "loss_raw": raw_loss == 1,
        "loss_core": strict_loss == 1,
        "stable_drop_raw": stable_drop == 1,
        "stable_keep_raw": stable_keep == 1,
        "gain_raw": raw_reentry == 1,
    }

def compute_transition(pre_date: str, post_date: str, joinfix_cache: dict, s1_inventory: pd.DataFrame | None = None) -> dict:
    pre_dir = ensure_semantic_masks(pre_date, joinfix_cache)
    post_dir = ensure_semantic_masks(post_date, joinfix_cache)

    keep_pre, profile_pre = read_mask(pre_dir / KEEP_MASK_NAME)
    keep_post, profile_post = read_mask(post_dir / KEEP_MASK_NAME)
    vote_pre, _ = read_mask(pre_dir / VOTE_MASK_NAME)
    vote_post, _ = read_mask(post_dir / VOTE_MASK_NAME)
    group_pre, _ = read_mask(pre_dir / GROUP_MASK_NAME)
    group_post, _ = read_mask(post_dir / GROUP_MASK_NAME)

    if not same_grid(profile_pre, profile_post):
        raise ValueError(f"Pre/post semantic masks are not on the same grid: {pre_date} -> {post_date}")

    group_use = np.where(group_post > 0, group_post, group_pre)
    common = (keep_pre != 255) & (keep_post != 255) & (group_use > 0)

    good_votes = common & (vote_pre != 255) & (vote_post != 255)
    vp = vote_pre.astype(np.int16)
    vq = vote_post.astype(np.int16)
    vote_delta = np.full(common.shape, -32768, dtype=np.int16)
    vote_delta[good_votes] = vq[good_votes] - vp[good_votes]

    alg3 = read_algorithm3_transition_masks(pre_date, post_date, profile_pre)
    if alg3 is not None:
        common = alg3["common"] & (group_use > 0)
        stable_keep_raw = alg3["stable_keep_raw"] & common
        stable_drop_raw = alg3["stable_drop_raw"] & common
        loss_raw = alg3["loss_raw"] & common
        gain_raw = alg3["gain_raw"] & common
        loss_core = alg3["loss_core"] & common
        transition_source = "algorithm3_rasters"
    else:
        stable_keep_raw = common & (keep_pre == 1) & (keep_post == 1)
        stable_drop_raw = common & (keep_pre == 0) & (keep_post == 0)
        loss_raw = common & (keep_pre == 1) & (keep_post == 0)
        gain_raw = common & (keep_pre == 0) & (keep_post == 1)
        loss_gate = good_votes & (vp >= MIN_PRE_VOTES_FOR_LOSS) & (vq <= MAX_POST_VOTES_FOR_LOSS) & (vote_delta <= -MIN_VOTE_DROP_FOR_LOSS)
        loss_core = loss_raw & loss_gate
        if ERODE_CHANGE_MASKS:
            loss_core = binary_erode(loss_core, ERODE_ITERATIONS, ERODE_CONNECTIVITY)
        loss_core = filter_small_components(loss_core, MIN_CLASS_PATCH_PIXELS, CONNECTIVITY)
        transition_source = "computed_in_algorithm4"

    anomaly_pre, anomaly_pre_available = optional_read_float(pre_dir / ANOMALY_MASK_NAME, profile_pre)
    anomaly_post, anomaly_post_available = optional_read_float(post_dir / ANOMALY_MASK_NAME, profile_pre)
    confidence_pre, confidence_pre_available = optional_read_u8(pre_dir / ENVELOPE_CONFIDENCE_MASK_NAME, profile_pre)
    confidence_post, confidence_post_available = optional_read_u8(post_dir / ENVELOPE_CONFIDENCE_MASK_NAME, profile_pre)
    s1_pre, s1_pre_meta = load_s1_stack_for_date(pre_date, s1_inventory if s1_inventory is not None else pd.DataFrame(), profile_pre)
    s1_post, s1_post_meta = load_s1_stack_for_date(post_date, s1_inventory if s1_inventory is not None else pd.DataFrame(), profile_pre)

    group_lut = load_group_lut(pre_dir)
    group_lut.update(load_group_lut(post_dir))

    return {
        "pre_date": pre_date,
        "post_date": post_date,
        "pre_dir": pre_dir,
        "post_dir": post_dir,
        "profile": profile_pre,
        "common": common,
        "stable_keep_raw": stable_keep_raw,
        "stable_drop_raw": stable_drop_raw,
        "loss_raw": loss_raw,
        "gain_raw": gain_raw,
        "loss_core": loss_core,
        "vote_delta": vote_delta,
        "group_use": group_use,
        "group_lut": group_lut,
        "transition_source": transition_source,
        "anomaly_pre": anomaly_pre,
        "anomaly_post": anomaly_post,
        "anomaly_pre_available": anomaly_pre_available,
        "anomaly_post_available": anomaly_post_available,
        "confidence_pre": confidence_pre,
        "confidence_post": confidence_post,
        "confidence_pre_available": confidence_pre_available,
        "confidence_post_available": confidence_post_available,
        "s1_pre": s1_pre,
        "s1_post": s1_post,
        "s1_pre_meta": s1_pre_meta,
        "s1_post_meta": s1_post_meta,
    }

# =============================================================================
# Interval processing
# =============================================================================
def fraction_overlap(obj_mask: np.ndarray, support_mask: Optional[np.ndarray]) -> float:
    denom = int(obj_mask.sum())
    if denom == 0:
        return np.nan
    if support_mask is None:
        return np.nan
    return float(np.sum(obj_mask & support_mask) / denom)


def object_descriptor_features(prefix: str, mask: np.ndarray, tr: dict) -> dict:
    out = {}
    if not np.any(mask):
        return out
    if tr.get("anomaly_pre_available") and tr.get("anomaly_post_available"):
        an_pre = tr["anomaly_pre"]
        an_post = tr["anomaly_post"]
        out[f"{prefix}_anomaly_pre_mean"] = safe_mean(an_pre, mask)
        out[f"{prefix}_anomaly_post_mean"] = safe_mean(an_post, mask)
        out[f"{prefix}_anomaly_delta_mean"] = safe_mean(an_post - an_pre, mask)
        out[f"{prefix}_anomaly_valid_frac"] = finite_fraction(an_post - an_pre, mask)
    if tr.get("confidence_pre_available") and tr.get("confidence_post_available"):
        worst = np.maximum(tr["confidence_pre"].astype(np.uint8), tr["confidence_post"].astype(np.uint8))
        denom = float(mask.sum())
        for code, label in [(1, "high"), (2, "medium"), (3, "low"), (4, "missing")]:
            out[f"{prefix}_confidence_{label}_frac"] = float(np.sum(mask & (worst == code)) / denom) if denom else np.nan
    s1_pre = tr.get("s1_pre") or {}
    s1_post = tr.get("s1_post") or {}
    if s1_pre and s1_post:
        for name in sorted(set(s1_pre.keys()) & set(s1_post.keys())):
            if name in S1_DESCRIPTOR_EXCLUDE_FROM_OBJECT_STATS:
                continue
            safe_name = name.lower()
            delta = s1_post[name] - s1_pre[name]
            out[f"{prefix}_{safe_name}_pre_mean"] = safe_mean(s1_pre[name], mask)
            out[f"{prefix}_{safe_name}_post_mean"] = safe_mean(s1_post[name], mask)
            out[f"{prefix}_{safe_name}_delta_mean"] = safe_mean(delta, mask)
            out[f"{prefix}_{safe_name}_valid_frac"] = finite_fraction(delta, mask)
    return out


def process_interval(idx: int, transitions: List[dict]) -> Tuple[pd.DataFrame, Dict[str, Dict[str, object]], dict]:
    tr = transitions[idx]
    pre_date = tr["pre_date"]
    post_date = tr["post_date"]
    iid = interval_id(pre_date, post_date)
    ilabel = interval_label(pre_date, post_date)
    out_dir = OUTPUT_ROOT / ilabel
    out_dir.mkdir(parents=True, exist_ok=True)

    profile = tr["profile"]
    transformer = transformer_from_profile(profile, PROJECTED_CRS)
    pixel_area_ha = compute_pixel_area_ha_grid(profile)

    raw_labels = label_components(tr["loss_raw"], CONNECTIVITY)
    ids, counts = np.unique(raw_labels[raw_labels > 0], return_counts=True)
    keep_ids = set(ids[counts >= MIN_RAW_OBJECT_PIXELS].tolist())
    raw_kept = np.isin(raw_labels, list(keep_ids)) if keep_ids else np.zeros_like(raw_labels, dtype=bool)
    raw_labels = np.where(raw_kept, raw_labels, 0)
    old_ids = [int(u) for u in np.unique(raw_labels) if u > 0]
    remap = {u: i + 1 for i, u in enumerate(old_ids)}
    object_labels = np.zeros_like(raw_labels, dtype=np.int32)
    for old, new in remap.items():
        object_labels[raw_labels == old] = new

    # Next interval support masks.
    next_stable_drop = None
    next_loss = None
    next_support = None
    if idx + 1 < len(transitions):
        next_tr = transitions[idx + 1]
        next_stable_drop = next_tr["stable_drop_raw"]
        next_loss = next_tr["loss_core"]
        next_support = next_stable_drop | next_loss

    rows = []
    object_geoms: Dict[str, Dict[str, object]] = {}
    vd = tr["vote_delta"].astype(np.float32)
    vd = np.where(vd == -32768, np.nan, vd)
    group_use = tr["group_use"]
    group_lut = tr["group_lut"]

    for obj_id in range(1, len(old_ids) + 1):
        raw_obj = object_labels == obj_id
        core_obj = raw_obj & tr["loss_core"]
        fringe_obj = raw_obj & (~core_obj)

        raw_geom = mask_to_geometry(raw_obj, profile["transform"])
        core_geom = mask_to_geometry(core_obj, profile["transform"])
        fringe_geom = mask_to_geometry(fringe_obj, profile["transform"])

        raw_px = int(raw_obj.sum())
        core_px = int(core_obj.sum())
        fringe_px = int(fringe_obj.sum())
        raw_area_ha = float(pixel_area_ha[raw_obj].sum())
        core_area_ha = float(pixel_area_ha[core_obj].sum())
        fringe_area_ha = float(pixel_area_ha[fringe_obj].sum())
        compactness = geom_compactness(raw_geom, transformer)

        mean_vote_drop_raw = float(np.nanmean(vd[raw_obj])) if np.any(np.isfinite(vd[raw_obj])) else np.nan
        mean_vote_drop_core = float(np.nanmean(vd[core_obj])) if core_px and np.any(np.isfinite(vd[core_obj])) else np.nan

        gids = group_use[raw_obj]
        gids = gids[gids > 0]
        if len(gids):
            vc = pd.Series(gids).value_counts()
            dominant_group_id = int(vc.index[0])
            dominant_group = group_lut.get(dominant_group_id, str(dominant_group_id))
            dominant_group_frac = float(vc.iloc[0] / vc.sum())
            n_groups = int(len(vc))
        else:
            dominant_group_id = 0
            dominant_group = "UNKNOWN"
            dominant_group_frac = np.nan
            n_groups = 0

        rr, cc = np.where(raw_obj)
        touches_scene_border = bool((rr == 0).any() or (rr == raw_obj.shape[0] - 1).any() or (cc == 0).any() or (cc == raw_obj.shape[1] - 1).any()) if raw_px else False

        uid = f"lossObj_{iid}_{obj_id:04d}"
        extra_features = {}
        extra_features.update(object_descriptor_features("raw", raw_obj, tr))
        extra_features.update(object_descriptor_features("core", core_obj, tr))
        extra_features.update(object_descriptor_features("fringe", fringe_obj, tr))
        s1_pre_meta = tr.get("s1_pre_meta") or {}
        s1_post_meta = tr.get("s1_post_meta") or {}
        rows.append({
            "interval_id": iid,
            "interval_label": ilabel,
            "pre_date": pre_date,
            "post_date": post_date,
            "object_uid": uid,
            "object_id": obj_id,
            "raw_px": raw_px,
            "core_px": core_px,
            "fringe_px": fringe_px,
            "raw_area_ha": raw_area_ha,
            "core_area_ha": core_area_ha,
            "fringe_area_ha": fringe_area_ha,
            "core_ratio": (core_px / raw_px) if raw_px else np.nan,
            "fringe_ratio": (fringe_px / raw_px) if raw_px else np.nan,
            "mean_vote_drop_raw": mean_vote_drop_raw,
            "mean_vote_drop_core": mean_vote_drop_core,
            "raw_persistence_next_frac": fraction_overlap(raw_obj, next_stable_drop),
            "core_persistence_next_frac": fraction_overlap(core_obj, next_stable_drop),
            "fringe_persistence_next_frac": fraction_overlap(fringe_obj, next_stable_drop),
            "raw_expansion_next_frac": fraction_overlap(raw_obj, next_loss),
            "core_expansion_next_frac": fraction_overlap(core_obj, next_loss),
            "fringe_expansion_next_frac": fraction_overlap(fringe_obj, next_loss),
            "raw_support_next_frac": fraction_overlap(raw_obj, next_support),
            "core_support_next_frac": fraction_overlap(core_obj, next_support),
            "fringe_support_next_frac": fraction_overlap(fringe_obj, next_support),
            "dominant_group_id": dominant_group_id,
            "dominant_group": dominant_group,
            "dominant_group_frac": dominant_group_frac,
            "n_groups": n_groups,
            "compactness": compactness,
            "touches_scene_border": touches_scene_border,
            "transition_source": tr.get("transition_source", "unknown"),
            "s1_pre_scene": str(s1_pre_meta.get("scene", "")),
            "s1_post_scene": str(s1_post_meta.get("scene", "")),
            "s1_pre_offset_days": s1_pre_meta.get("matched_offset_days", np.nan),
            "s1_post_offset_days": s1_post_meta.get("matched_offset_days", np.nan),
            **extra_features,
        })
        object_geoms[uid] = {"raw": raw_geom, "core": core_geom, "fringe": fringe_geom, "feasible": None}

    feat_df = pd.DataFrame(rows)
    if feat_df.empty:
        # Keep columns stable if no objects are present.
        feat_df = pd.DataFrame(columns=[
            "interval_id", "interval_label", "pre_date", "post_date", "object_uid", "object_id",
            "raw_px", "core_px", "fringe_px", "raw_area_ha", "core_area_ha", "fringe_area_ha",
            "core_ratio", "fringe_ratio", "mean_vote_drop_raw", "mean_vote_drop_core",
            "raw_persistence_next_frac", "core_persistence_next_frac", "fringe_persistence_next_frac",
            "raw_expansion_next_frac", "core_expansion_next_frac", "fringe_expansion_next_frac",
            "raw_support_next_frac", "core_support_next_frac", "fringe_support_next_frac",
            "dominant_group_id", "dominant_group", "dominant_group_frac", "n_groups",
            "compactness", "touches_scene_border", "transition_source",
            "s1_pre_scene", "s1_post_scene", "s1_pre_offset_days", "s1_post_offset_days",
        ])
    else:
        feat_df = evaluate_rules(feat_df)
        for _, row in feat_df.iterrows():
            uid = str(row["object_uid"])
            if row["final_status"] == "AcceptCorePlusFringe":
                object_geoms[uid]["feasible"] = object_geoms[uid]["raw"]
            elif row["final_status"] in ("AcceptCoreOnly", "AcceptCoreReviewFringe"):
                object_geoms[uid]["feasible"] = object_geoms[uid]["core"]
            else:
                object_geoms[uid]["feasible"] = None

    # Write raster products for GIS visualization of semantic transitions,
    # raw/core/fringe decomposition, and rule-based pruning decisions.
    write_interval_rasters(
        out_dir=out_dir,
        profile=profile,
        tr=tr,
        object_labels=object_labels,
        feat_df=feat_df,
    )

    # Write interval tabular/RDF outputs.
    feat_df.to_csv(out_dir / "object_features.csv", index=False)
    feat_df.to_csv(out_dir / "rule_test_results.csv", index=False)
    if not feat_df.empty:
        write_objects_ttl(out_dir / "loss_objects_rdf.ttl", feat_df, object_geoms, profile)
        write_inferred_actions_ttl(out_dir / "inferred_actions.ttl", feat_df)
    else:
        (out_dir / "loss_objects_rdf.ttl").write_text("", encoding="utf-8")
        (out_dir / "inferred_actions.ttl").write_text("", encoding="utf-8")

    area = pixel_area_ha
    summary = {
        "interval_id": iid,
        "interval_label": ilabel,
        "pre_date": pre_date,
        "post_date": post_date,
        "common_px": int(tr["common"].sum()),
        "common_area_ha": float(area[tr["common"]].sum()),
        "raw_loss_px": int(tr["loss_raw"].sum()),
        "raw_loss_area_ha": float(area[tr["loss_raw"]].sum()),
        "strict_core_loss_px": int(tr["loss_core"].sum()),
        "strict_core_loss_area_ha": float(area[tr["loss_core"]].sum()),
        "raw_loss_objects_ge_25px": int(feat_df.shape[0]),
        "raw_object_area_ha_total": float(feat_df["raw_area_ha"].sum()) if "raw_area_ha" in feat_df else 0.0,
        "core_area_ha_total": float(feat_df["core_area_ha"].sum()) if "core_area_ha" in feat_df else 0.0,
        "retained_area_ha_total": float(feat_df["retained_area_ha"].sum()) if "retained_area_ha" in feat_df else 0.0,
        "has_next_interval": bool(idx + 1 < len(transitions)),
        "transition_source": tr.get("transition_source", "unknown"),
        "s1_pre_scene": str((tr.get("s1_pre_meta") or {}).get("scene", "")),
        "s1_post_scene": str((tr.get("s1_post_meta") or {}).get("scene", "")),
        "s1_pre_offset_days": (tr.get("s1_pre_meta") or {}).get("matched_offset_days", np.nan),
        "s1_post_offset_days": (tr.get("s1_post_meta") or {}).get("matched_offset_days", np.nan),
    }
    if not feat_df.empty and "final_status" in feat_df:
        for status, n in feat_df["final_status"].value_counts().items():
            summary[f"status_{status}"] = int(n)
    summary_lines = [
        "Annual RDF export and GraphDB-style pruning test",
        "================================================",
        "",
        f"Interval: {pre_date} -> {post_date}",
        f"Comparable pixels: {summary['common_px']}",
        f"Raw keep->drop pixels: {summary['raw_loss_px']}",
        f"Strict-core loss pixels: {summary['strict_core_loss_px']}",
        f"Raw loss objects >= {MIN_RAW_OBJECT_PIXELS} px: {summary['raw_loss_objects_ge_25px']}",
        "",
        "Final object status counts:",
    ]
    if not feat_df.empty and "final_status" in feat_df:
        for status, n in feat_df["final_status"].value_counts().items():
            summary_lines.append(f"- {status}: {int(n)}")
    else:
        summary_lines.append("- no objects")
    summary_lines.extend([
        "",
        "Area summary (ha):",
        f"- raw_area_ha_total: {summary['raw_object_area_ha_total']:.3f}",
        f"- core_area_ha_total: {summary['core_area_ha_total']:.3f}",
        f"- retained_area_ha_total: {summary['retained_area_ha_total']:.3f}",
        "",
        "Interpretation:",
        "- AcceptCorePlusFringe: keep the raw object as the feasible limit",
        "- AcceptCoreOnly: keep only the core geometry",
        "- AcceptCoreReviewFringe: keep the core, review the fringe",
        "- RejectNoise / ReviewObject: do not accept automatically",
        "",
        "Persistence feature definition:",
        "- persistence_next_frac = overlap with next-interval stable_drop",
        "- expansion_next_frac = overlap with next-interval strict loss",
        "- support_next_frac = overlap with next-interval stable_drop OR strict loss",
        "",
        "Raster outputs:",
        "- rasters/semantic_transition_class.tif: stable_drop/stable_keep/raw_loss/raw_reentry context",
        "- rasters/raw_core_fringe_class.tif: fringe vs strict-core decomposition",
        "- rasters/final_status_code.tif: rule outcome by raw object",
        "- rasters/retained_feasible_object_id.tif: automatically retained geometry",
        "- rasters/review_object_or_fringe_id.tif: areas routed to review",
    ])
    (out_dir / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    return feat_df, object_geoms, summary

# =============================================================================
# Main
# =============================================================================
def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    joinfix_cache: dict = {}

    s1_inventory = load_s1_inventory()

    # Ensure all annual anchor masks exist first.
    for d in ANNUAL_ANCHOR_DATES:
        ensure_semantic_masks(d, joinfix_cache)

    # Compute all transitions once, so next-interval support is available.
    transitions = []
    for pre_date, post_date in zip(ANNUAL_ANCHOR_DATES[:-1], ANNUAL_ANCHOR_DATES[1:]):
        transitions.append(compute_transition(pre_date, post_date, joinfix_cache, s1_inventory=s1_inventory))

    all_features: List[pd.DataFrame] = []
    all_geoms: Dict[str, Dict[str, object]] = {}
    summaries: List[dict] = []
    profile_for_rdf = transitions[0]["profile"] if transitions else None

    for idx in range(len(transitions)):
        feat_df, geoms, summary = process_interval(idx, transitions)
        all_features.append(feat_df)
        all_geoms.update(geoms)
        summaries.append(summary)

    all_df = pd.concat(all_features, ignore_index=True) if all_features else pd.DataFrame()
    all_df.to_csv(OUTPUT_ROOT / "object_features_all_intervals.csv", index=False)
    all_df.to_csv(OUTPUT_ROOT / "rule_test_results_all_intervals.csv", index=False)
    pd.DataFrame(summaries).to_csv(OUTPUT_ROOT / "annual_interval_summary.csv", index=False)

    write_rule_catalog_csv(OUTPUT_ROOT / "graphdb_pruning_rule_catalog.csv")
    write_rule_catalog_ttl(OUTPUT_ROOT / "graphdb_pruning_rules_v1.ttl")
    write_graphdb_rule_updates(OUTPUT_ROOT / "graphdb_pruning_rules_v1.rq")
    if profile_for_rdf is not None and not all_df.empty:
        write_objects_ttl(OUTPUT_ROOT / "loss_objects_all_intervals_rdf.ttl", all_df, all_geoms, profile_for_rdf)
        write_inferred_actions_ttl(OUTPUT_ROOT / "inferred_actions_all_intervals.ttl", all_df)
    else:
        (OUTPUT_ROOT / "loss_objects_all_intervals_rdf.ttl").write_text("", encoding="utf-8")
        (OUTPUT_ROOT / "inferred_actions_all_intervals.ttl").write_text("", encoding="utf-8")

    lines = [
        "Annual RDF export and GraphDB-style pruning summary",
        "===================================================",
        "",
        f"Semantic mask root: {SEMANTIC_MASK_ROOT}",
        f"Annual anchors: {', '.join(ANNUAL_ANCHOR_DATES)}",
        f"Intervals processed: {len(transitions)}",
        f"Objects exported: {len(all_df)}",
        "",
    ]
    if not all_df.empty and "final_status" in all_df.columns:
        lines.append("Final status counts across all intervals:")
        for status, n in all_df["final_status"].value_counts().items():
            lines.append(f"- {status}: {int(n)}")
        lines.extend([
            "",
            "Area totals across all intervals (ha):",
            f"- raw area total: {all_df['raw_area_ha'].sum():.3f}",
            f"- core area total: {all_df['core_area_ha'].sum():.3f}",
            f"- retained area total: {all_df['retained_area_ha'].sum():.3f}",
            "",
            "Per-interval raster products are written under each interval folder in rasters/.",
        ])
    (OUTPUT_ROOT / "summary_all_intervals.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"Done. Outputs written to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
