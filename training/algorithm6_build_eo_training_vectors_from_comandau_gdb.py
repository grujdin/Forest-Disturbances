"""
Algorithm #6. Build EO training vectors from the labelled Comandau FileGDB.

Purpose
-------
This script reads interval-specific candidate disturbance polygons from a
FileGDB, normalizes the currently assigned labels to the RF target classes
HV / WT / IN, and builds object-level EO training vectors:

    x = [x_EO, x_delta, x_shape, x_topo, m]

The script is designed for the Comandau.gdb structure, where candidate objects
are stored in interval layers such as:

    Comandau_20170804_20180814_060_060
    Comandau_2019_08_14_2020_07_09_062_062

The DEM/topographic extraction remains inside Algorithm #6 for training-vector
construction. It can read a zipped GeoTIFF DEM or a direct GeoTIFF path. A DEM
stored only as a FileGDB raster may not be readable by open-source GDAL/Rasterio;
for automation, keep an exported DEM GeoTIFF or ZIP copy.

Outputs
-------
- eo_rf_training_vectors_comandau.csv
- eo_rf_training_vectors_comandau.gpkg
- gdb_layer_inventory.csv
- gdb_label_summary.csv
- training_label_summary.csv
- eo_training_feature_dictionary.csv
- algorithm6_config.json

Optional:
- a new FileGDB layer named eo_rf_training_vectors_algorithm6, if the local GDAL
  build supports writing to OpenFileGDB and the GDB is not locked.

Hardcoded by design, but with clear path parameters at the top.
"""
from __future__ import annotations

import json
import math
import re
import shutil
import zipfile
from datetime import datetime
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
from shapely.geometry import mapping
from shapely.ops import transform as shapely_transform
from pyproj import CRS, Transformer

try:
    import geopandas as gpd
except Exception as exc:  # pragma: no cover
    raise RuntimeError("geopandas is required for Algorithm #6") from exc

try:
    import fiona
except Exception as exc:  # pragma: no cover
    raise RuntimeError("fiona is required for reading FileGDB layers") from exc

# =============================================================================
# HARD-CODED CONFIG
# =============================================================================

# Correct Comandau GDB source.
TRAINING_GDB = Path("D:/Forest_Disturbance/Databases/Comandau.gdb")
TRAINING_GDB_ZIP_FALLBACK = Path("D:/Forest_Disturbance/Databases/Comandau.gdb.zip")

# Output folder.
OUTPUT_ROOT = Path("D:/Forest_Disturbance/outputs/rf_training_vectors_comandau_algorithm6")

# Candidate layers are interval-specific Comandau layers. FMU and no-forest
# diagnostic layers are excluded by default.
TRAINING_LAYER_PREFIX = "Comandau_"
EXCLUDE_LAYER_NAME_CONTAINS = ["NoForestArea", "FMU_"]
INCLUDE_UNLABELLED_OBJECTS = False
INCLUDE_NA_OBJECTS = False

# Label handling.
LABEL_FIELD_CANDIDATES = [
    "rf_label",
    "RF_LABEL",
    "label",
    "Label",
    "class",
    "Class",
    "disturbance_code",
    "disturbance_type",
    "disturbance_type_raw",
]
RF_CLASSES = ["HV", "WT", "IN"]
EXCLUDED_LABELS = {"", "NA", "N/A", "NONE", "NULL", "UNASSIGNED", "NO_DATA", "MIXT", "MIXED"}

# Text normalization map. Extend this if your final GDB uses other terms.
LABEL_TEXT_MAP = {
    "HV": "HV",
    "HARVEST": "HV",
    "HARVESTING": "HV",
    "LOGGING": "HV",
    "RARITURA": "HV",
    "RARITURI": "HV",
    "RĂRITURĂ": "HV",
    "RĂRITURI": "HV",
    "PROBABLYHV": "HV",
    "PROBABLY_HV": "HV",
    "POSIBBLYRARITURA": "HV",
    "POSSIBLYRARITURA": "HV",
    "WT": "WT",
    "WINDTHROW": "WT",
    "WIND_THROW": "WT",
    "DOBORATURI": "WT",
    "DOBORÂTURI": "WT",
    "IN": "IN",
    "IPIDE": "IN",
    "INSECT": "IN",
    "INSECTS": "IN",
    "BARKBEETLE": "IN",
    "BARK_BEETLE": "IN",
    "BARK-BEETLE": "IN",
}

# Sentinel-2 cache. The script will continue with missingness flags if this
# folder does not exist yet.
S2_CACHE_ROOT = Path("D:/Forest_Disturbance/outputs/comandau_phase3_preprocessing_cache")
S2_INDEX_BANDS = ["NDVI", "NDMI", "NBR", "NDRE", "NDSI"]
S2_MAX_MATCH_DAYS = 60
S2_NODATA_VALUES = {-9999.0, -32768.0}

# Sentinel-1 descriptor cache. Optional.
S1_CACHE_ROOT = Path("D:/Forest_Disturbance/outputs/comandau_phase5_sentinel1_descriptor_cache")
S1_MAX_MATCH_DAYS = 60
S1_DESCRIPTOR_EXCLUDE = {"S1_VALID"}
S1_NODATA_VALUES = {-9999.0, -32768.0}

# Optional semantic-mask folder. If unavailable, semantic features are skipped.
SEMANTIC_MASK_ROOT = Path("D:/Forest_Disturbance/outputs/comandau_semantic_masks_s1s2_blacklist_and_caution_eligible")
SEMANTIC_MAX_MATCH_DAYS = 0  # date-folder match by default

# DEM source. Use either a direct GeoTIFF or a zipped GeoTIFF. If you store the
# DEM in the GDB for GIS convenience, also export a GeoTIFF/ZIP copy for the
# automated script.
DEM_TIF = Path("D:/Forest_Disturbance/imagery_zip/Comandau_DEM_Copernicus_30.tif")
DEM_ZIP = Path("D:/Forest_Disturbance/imagery_zip/Comandau_DEM_Copernicus_30.zip")
# Fallback if you temporarily use the SdV DEM name/path for testing.
DEM_ZIP_FALLBACK = Path("D:/Forest_Disturbance/imagery_zip/SdV_DEM_Copernicus_30.zip")

PROJECTED_CRS = "EPSG:32634"
DEM_PROJECTED_RESOLUTION_M = 30.0
FORCE_REBUILD_TOPO_CACHE = False
ZONAL_ALL_TOUCHED = False

# Predictor statistics.
CONTINUOUS_STATS = ["mean", "median", "std", "q10", "q90", "min", "max"]
MIN_VALID_PIXELS_FOR_STATS = 1

# Write outputs.
WRITE_GPKG = True
WRITE_BACK_TO_GDB = True
WRITE_ALL_CANDIDATE_OBJECTS = True
OUTPUT_GDB_LAYER_NAME = "eo_rf_training_vectors_algorithm6"

DATE_RE = re.compile(r"(20\d{2})[_-]?(\d{2})[_-]?(\d{2})")

# =============================================================================
# Logging and generic helpers
# =============================================================================

def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def norm_text(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return str(x).strip()


def norm_label_key(x) -> str:
    s = norm_text(x).upper()
    s = s.replace(" ", "").replace("-", "_")
    # remove Romanian diacritics in a simple deterministic way
    repl = {
        "Ă": "A", "Â": "A", "Î": "I", "Ș": "S", "Ş": "S", "Ț": "T", "Ţ": "T",
        "ă": "A", "â": "A", "î": "I", "ș": "S", "ş": "S", "ț": "T", "ţ": "T",
    }
    for a, b in repl.items():
        s = s.replace(a, b)
    return s


def normalize_rf_label(raw_value) -> Tuple[str, str]:
    """Return (label, status). status is mapped/excluded/unassigned/unknown."""
    raw = norm_text(raw_value)
    key = norm_label_key(raw)
    if key in EXCLUDED_LABELS:
        if key == "":
            return "", "unassigned"
        return "", "excluded"
    if key in LABEL_TEXT_MAP:
        return LABEL_TEXT_MAP[key], "mapped"
    # Direct fallback for clean class codes.
    if key in RF_CLASSES:
        return key, "mapped"
    return "", "unknown_label"


def find_first_existing_path(paths: Sequence[Path]) -> Optional[Path]:
    for p in paths:
        if p and p.exists():
            return p
    return None


def resolve_gdb_path() -> Path:
    if TRAINING_GDB.exists() and TRAINING_GDB.is_dir():
        return TRAINING_GDB
    if TRAINING_GDB_ZIP_FALLBACK.exists():
        extract_root = OUTPUT_ROOT / "_extracted_gdb"
        gdb_dir = extract_root / TRAINING_GDB_ZIP_FALLBACK.stem
        if not gdb_dir.exists():
            log(f"Extracting GDB ZIP fallback to {extract_root}")
            if extract_root.exists():
                shutil.rmtree(extract_root)
            extract_root.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(TRAINING_GDB_ZIP_FALLBACK, "r") as zf:
                zf.extractall(extract_root)
        if gdb_dir.exists():
            return gdb_dir
    raise FileNotFoundError(
        f"Could not find Comandau GDB folder {TRAINING_GDB} or ZIP fallback {TRAINING_GDB_ZIP_FALLBACK}"
    )


def parse_dates_from_layer_name(layer_name: str) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    matches = DATE_RE.findall(layer_name)
    dates = []
    for y, m, d in matches:
        try:
            dates.append(pd.Timestamp(year=int(y), month=int(m), day=int(d)))
        except Exception:
            pass
    if len(dates) >= 2:
        return dates[0], dates[1]
    return None, None


def is_interval_training_layer(layer_name: str) -> bool:
    if not layer_name.startswith(TRAINING_LAYER_PREFIX):
        return False
    if any(token in layer_name for token in EXCLUDE_LAYER_NAME_CONTAINS):
        return False
    pre, post = parse_dates_from_layer_name(layer_name)
    return pre is not None and post is not None


def safe_float(x, default=np.nan) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def interval_name(pre: pd.Timestamp, post: pd.Timestamp) -> str:
    return f"{pre.date().isoformat()}_to_{post.date().isoformat()}"

# =============================================================================
# GDB inspection and training-object loading
# =============================================================================

def inspect_gdb_layers(gdb_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    layer_rows: List[dict] = []
    label_rows: List[dict] = []
    layers = fiona.listlayers(str(gdb_path))
    for layer in layers:
        try:
            with fiona.open(str(gdb_path), layer=layer) as src:
                props = list(src.schema.get("properties", {}).keys())
                label_fields = [f for f in props if f in LABEL_FIELD_CANDIDATES]
                grid_counts: Dict[str, int] = {}
                label_counts: Dict[Tuple[str, str], dict] = {}
                areas: List[float] = []
                for feat in src:
                    p = dict(feat["properties"])
                    if "gridcode" in p:
                        k = str(p.get("gridcode"))
                        grid_counts[k] = grid_counts.get(k, 0) + 1
                    a = safe_float(p.get("Shape_Area"), np.nan)
                    if math.isfinite(a):
                        areas.append(a)
                    for lf in label_fields:
                        v = norm_text(p.get(lf))
                        key = (lf, v)
                        if key not in label_counts:
                            label_counts[key] = {"count": 0, "area_sum": 0.0}
                        label_counts[key]["count"] += 1
                        if math.isfinite(a):
                            label_counts[key]["area_sum"] += a
                pre, post = parse_dates_from_layer_name(layer)
                layer_rows.append({
                    "layer": layer,
                    "is_interval_training_layer": is_interval_training_layer(layer),
                    "pre_date": pre.date().isoformat() if pre is not None else "",
                    "post_date": post.date().isoformat() if post is not None else "",
                    "feature_count": len(src),
                    "geometry_type": src.schema.get("geometry"),
                    "crs": str(src.crs),
                    "n_fields": len(props),
                    "fields": ";".join(props),
                    "label_fields_detected": ";".join(label_fields),
                    "gridcode_counts": json.dumps(grid_counts, sort_keys=True),
                    "area_m2_min": float(np.min(areas)) if areas else np.nan,
                    "area_m2_median": float(np.median(areas)) if areas else np.nan,
                    "area_m2_max": float(np.max(areas)) if areas else np.nan,
                    "area_m2_sum": float(np.sum(areas)) if areas else np.nan,
                })
                for (lf, v), d in label_counts.items():
                    rf_label, label_status = normalize_rf_label(v)
                    label_rows.append({
                        "layer": layer,
                        "field": lf,
                        "raw_value": v,
                        "rf_label": rf_label,
                        "label_status": label_status,
                        "count": d["count"],
                        "area_m2_sum": d["area_sum"],
                        "area_ha_sum": d["area_sum"] / 10000.0,
                    })
        except Exception as exc:
            layer_rows.append({"layer": layer, "error": repr(exc)})
    return pd.DataFrame(layer_rows), pd.DataFrame(label_rows)


def choose_label_value(row: pd.Series) -> Tuple[str, str, str]:
    for field in LABEL_FIELD_CANDIDATES:
        if field in row.index:
            raw = row.get(field)
            if norm_text(raw) != "":
                rf_label, status = normalize_rf_label(raw)
                return norm_text(raw), rf_label, status
    return "", "", "unassigned"


def load_training_objects(gdb_path: Path) -> gpd.GeoDataFrame:
    layers = [l for l in fiona.listlayers(str(gdb_path)) if is_interval_training_layer(l)]
    if not layers:
        raise RuntimeError("No interval training layers found in the Comandau GDB.")
    all_gdfs: List[gpd.GeoDataFrame] = []
    for layer in layers:
        pre, post = parse_dates_from_layer_name(layer)
        log(f"Reading layer {layer}")
        gdf = gpd.read_file(str(gdb_path), layer=layer)
        if gdf.empty:
            continue
        if gdf.crs is None:
            raise ValueError(f"Layer {layer} has no CRS")
        raw_labels, rf_labels, label_status = [], [], []
        for _, row in gdf.drop(columns="geometry").iterrows():
            raw, lab, stat = choose_label_value(row)
            raw_labels.append(raw)
            rf_labels.append(lab)
            label_status.append(stat)
        gdf = gdf.copy()
        gdf["source_layer"] = layer
        gdf["source_id"] = gdf["Id"] if "Id" in gdf.columns else np.arange(1, len(gdf) + 1)
        gdf["training_object_id"] = [f"{layer}__{int(i) if pd.notna(i) else j}" for j, i in enumerate(gdf["source_id"], start=1)]
        gdf["pre_date"] = pre.date().isoformat()
        gdf["post_date"] = post.date().isoformat()
        gdf["interval"] = interval_name(pre, post)
        gdf["days_between"] = int((post - pre).days)
        gdf["label_raw"] = raw_labels
        gdf["rf_label"] = rf_labels
        gdf["label_status"] = label_status
        all_gdfs.append(gdf)
    out = gpd.GeoDataFrame(pd.concat(all_gdfs, ignore_index=True), crs=all_gdfs[0].crs)
    return out

# =============================================================================
# Shape predictors
# =============================================================================

def _count_parts(geom) -> int:
    if geom is None or geom.is_empty:
        return 0
    if geom.geom_type.startswith("Multi"):
        return len(list(geom.geoms))
    return 1


def _count_holes(geom) -> int:
    if geom is None or geom.is_empty:
        return 0
    if geom.geom_type == "Polygon":
        return len(geom.interiors)
    if geom.geom_type == "MultiPolygon":
        return sum(len(poly.interiors) for poly in geom.geoms)
    return 0


def _mrr_metrics(geom) -> Tuple[float, float, float, float]:
    """Return long side, short side, elongation, orientation_deg."""
    if geom is None or geom.is_empty:
        return np.nan, np.nan, np.nan, np.nan
    rect = geom.minimum_rotated_rectangle
    if rect.is_empty or rect.geom_type != "Polygon":
        return np.nan, np.nan, np.nan, np.nan
    coords = list(rect.exterior.coords)
    if len(coords) < 5:
        return np.nan, np.nan, np.nan, np.nan
    edges = []
    for i in range(4):
        x1, y1 = coords[i]
        x2, y2 = coords[i + 1]
        length = math.hypot(x2 - x1, y2 - y1)
        angle = (math.degrees(math.atan2(y2 - y1, x2 - x1)) + 180.0) % 180.0
        edges.append((length, angle))
    edges_sorted = sorted(edges, key=lambda x: x[0], reverse=True)
    long_len, orientation = edges_sorted[0]
    short_len = edges_sorted[-1][0]
    elongation = long_len / short_len if short_len > 0 else np.nan
    return long_len, short_len, elongation, orientation


def add_shape_predictors(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdfp = gdf.to_crs(PROJECTED_CRS).copy()
    rows = []
    for geom in gdfp.geometry:
        area = float(geom.area) if geom and not geom.is_empty else np.nan
        perim = float(geom.length) if geom and not geom.is_empty else np.nan
        hull = geom.convex_hull if geom and not geom.is_empty else None
        hull_area = float(hull.area) if hull and not hull.is_empty else np.nan
        hull_perim = float(hull.length) if hull and not hull.is_empty else np.nan
        compactness = (4.0 * math.pi * area / (perim ** 2)) if perim and perim > 0 and area >= 0 else np.nan
        equiv_diam = math.sqrt(4.0 * area / math.pi) if area >= 0 else np.nan
        solidity = area / hull_area if hull_area and hull_area > 0 else np.nan
        convexity = hull_perim / perim if perim and perim > 0 and hull_perim >= 0 else np.nan
        edge_complexity = perim / (2.0 * math.sqrt(math.pi * area)) if area and area > 0 else np.nan
        minx, miny, maxx, maxy = geom.bounds if geom and not geom.is_empty else (np.nan, np.nan, np.nan, np.nan)
        bbox_w = maxx - minx if math.isfinite(maxx) else np.nan
        bbox_h = maxy - miny if math.isfinite(maxy) else np.nan
        bbox_aspect = max(bbox_w, bbox_h) / min(bbox_w, bbox_h) if bbox_w > 0 and bbox_h > 0 else np.nan
        long_m, short_m, elong, orient = _mrr_metrics(geom)
        centroid = geom.centroid if geom and not geom.is_empty else None
        rows.append({
            "shape_area_m2": area,
            "shape_area_ha": area / 10000.0 if math.isfinite(area) else np.nan,
            "shape_perimeter_m": perim,
            "shape_compactness": compactness,
            "shape_equiv_diameter_m": equiv_diam,
            "shape_perimeter_area_ratio": perim / area if area and area > 0 else np.nan,
            "shape_hull_area_m2": hull_area,
            "shape_hull_perimeter_m": hull_perim,
            "shape_solidity": solidity,
            "shape_convexity": convexity,
            "shape_edge_complexity": edge_complexity,
            "shape_mrr_long_m": long_m,
            "shape_mrr_short_m": short_m,
            "shape_elongation": elong,
            "shape_orientation_deg": orient,
            "shape_bbox_width_m": bbox_w,
            "shape_bbox_height_m": bbox_h,
            "shape_bbox_aspect_ratio": bbox_aspect,
            "shape_centroid_x": centroid.x if centroid else np.nan,
            "shape_centroid_y": centroid.y if centroid else np.nan,
            "shape_n_parts": _count_parts(geom),
            "shape_n_holes": _count_holes(geom),
        })
    feat = pd.DataFrame(rows)
    for col in feat.columns:
        gdfp[col] = feat[col].values
    return gdfp

# =============================================================================
# Raster sampling helpers
# =============================================================================

def transform_geometry_to_crs(geom, src_crs, dst_crs):
    if geom is None or geom.is_empty:
        return geom
    if src_crs is None or dst_crs is None or str(src_crs) == str(dst_crs):
        return geom
    transformer = Transformer.from_crs(CRS.from_user_input(src_crs), CRS.from_user_input(dst_crs), always_xy=True)
    return shapely_transform(transformer.transform, geom)


def bound_window(win: rasterio.windows.Window, width: int, height: int) -> Optional[rasterio.windows.Window]:
    full = rasterio.windows.Window(0, 0, width, height)
    try:
        w = rasterio.windows.intersection(win, full)
    except Exception:
        return None
    if w.width <= 0 or w.height <= 0:
        return None
    return w.round_offsets().round_lengths()


def compute_vals_stats(vals: np.ndarray) -> Dict[str, float]:
    vals = vals[np.isfinite(vals)]
    if vals.size < MIN_VALID_PIXELS_FOR_STATS:
        return {s: np.nan for s in CONTINUOUS_STATS} | {"valid_px": int(vals.size)}
    return {
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "std": float(np.std(vals)),
        "q10": float(np.quantile(vals, 0.10)),
        "q90": float(np.quantile(vals, 0.90)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "valid_px": int(vals.size),
    }


def zonal_stats_dataset(ds: rasterio.io.DatasetReader, geom, geom_crs, band_names: Sequence[str], prefix: str, nodata_values: Iterable[float]) -> Dict[str, float]:
    result: Dict[str, float] = {}
    if geom is None or geom.is_empty:
        for b in band_names:
            for s in CONTINUOUS_STATS:
                result[f"{prefix}{b.lower()}_{s}"] = np.nan
            result[f"m_{prefix}{b.lower()}_valid_px"] = 0
        return result
    geom_r = transform_geometry_to_crs(geom, geom_crs, ds.crs)
    if geom_r is None or geom_r.is_empty:
        return result
    try:
        win = rasterio.windows.from_bounds(*geom_r.bounds, transform=ds.transform)
        win = bound_window(win, ds.width, ds.height)
    except Exception:
        win = None
    if win is None:
        for b in band_names:
            for s in CONTINUOUS_STATS:
                result[f"{prefix}{b.lower()}_{s}"] = np.nan
            result[f"m_{prefix}{b.lower()}_valid_px"] = 0
        return result
    out_shape = (int(win.height), int(win.width))
    if out_shape[0] <= 0 or out_shape[1] <= 0:
        return result
    wtransform = rasterio.windows.transform(win, ds.transform)
    try:
        geom_mask = rasterio.features.geometry_mask(
            [mapping(geom_r)], out_shape=out_shape, transform=wtransform, invert=True, all_touched=ZONAL_ALL_TOUCHED
        )
    except Exception:
        geom_mask = np.zeros(out_shape, dtype=bool)
    geom_px = int(geom_mask.sum())
    for band_i, band_name in enumerate(band_names, start=1):
        if band_i > ds.count:
            vals = np.array([], dtype=np.float32)
        else:
            arr = ds.read(band_i, window=win).astype(np.float32)
            valid = geom_mask & np.isfinite(arr)
            nodata = ds.nodata
            if nodata is not None:
                valid &= arr != float(nodata)
            for nd in nodata_values:
                valid &= arr != float(nd)
            vals = arr[valid]
        stats = compute_vals_stats(vals)
        for s in CONTINUOUS_STATS:
            result[f"{prefix}{band_name.lower()}_{s}"] = stats[s]
        result[f"m_{prefix}{band_name.lower()}_valid_px"] = stats["valid_px"]
        result[f"m_{prefix}{band_name.lower()}_valid_frac"] = float(stats["valid_px"] / geom_px) if geom_px else np.nan
    return result


def read_inventory(cache_root: Path, inventory_name: str) -> pd.DataFrame:
    path = cache_root / inventory_name
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date_ts"] = pd.to_datetime(df["date"], errors="coerce")
    elif "scene_date" in df.columns:
        df["date_ts"] = pd.to_datetime(df["scene_date"], errors="coerce")
    else:
        df["date_ts"] = pd.NaT
    return df


def nearest_inventory_record(inv: pd.DataFrame, target_date: str, path_cols: Sequence[str], max_days: int) -> Tuple[Optional[pd.Series], int]:
    if inv.empty or "date_ts" not in inv.columns:
        return None, 999999
    target = pd.Timestamp(target_date)
    temp = inv.copy()
    temp["_abs_days"] = (temp["date_ts"] - target).abs().dt.days
    temp = temp.sort_values(["_abs_days"])
    for _, row in temp.iterrows():
        if pd.isna(row["_abs_days"]) or int(row["_abs_days"]) > max_days:
            continue
        for pc in path_cols:
            p = norm_text(row.get(pc)) if pc in row.index else ""
            if p and Path(p).exists():
                return row, int(row["_abs_days"])
    return None, 999999


def sample_stack_features_for_rows(
    gdf: gpd.GeoDataFrame,
    inventory: pd.DataFrame,
    date_col: str,
    path_cols: Sequence[str],
    band_names: Sequence[str],
    prefix: str,
    max_days: int,
    nodata_values: Iterable[float],
) -> pd.DataFrame:
    rows: List[dict] = []
    # simple dataset cache by path
    ds_cache: Dict[str, rasterio.io.DatasetReader] = {}
    try:
        for _, row in gdf.iterrows():
            target_date = row[date_col]
            rec, off_days = nearest_inventory_record(inventory, target_date, path_cols, max_days)
            out: Dict[str, object] = {f"m_{prefix}scene_available": int(rec is not None), f"m_{prefix}offset_days": off_days if rec is not None else np.nan}
            if rec is None:
                rows.append(out)
                continue
            stack_path = ""
            for pc in path_cols:
                val = norm_text(rec.get(pc)) if pc in rec.index else ""
                if val and Path(val).exists():
                    stack_path = val
                    break
            out[f"m_{prefix}scene"] = norm_text(rec.get("scene")) if "scene" in rec.index else Path(stack_path).name
            out[f"m_{prefix}date"] = str(rec.get("date"))[:10] if "date" in rec.index else str(rec.get("date_ts"))[:10]
            if stack_path:
                if stack_path not in ds_cache:
                    ds_cache[stack_path] = rasterio.open(stack_path)
                stats = zonal_stats_dataset(ds_cache[stack_path], row.geometry, gdf.crs, band_names, prefix, nodata_values)
                out.update(stats)
            rows.append(out)
    finally:
        for ds in ds_cache.values():
            ds.close()
    return pd.DataFrame(rows, index=gdf.index)

# =============================================================================
# DEM/topography cache inside Algorithm #6
# =============================================================================

def first_tif_in_zip(zip_path: Path) -> Optional[str]:
    if not zip_path.exists():
        return None
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = [n for n in zf.namelist() if n.lower().endswith((".tif", ".tiff"))]
    return names[0] if names else None


def dem_vsi_path() -> Optional[str]:
    if DEM_TIF.exists():
        return str(DEM_TIF)
    dem_zip = find_first_existing_path([DEM_ZIP, DEM_ZIP_FALLBACK])
    if dem_zip is None:
        return None
    inner = first_tif_in_zip(dem_zip)
    if inner is None:
        return None
    return f"/vsizip/{dem_zip.as_posix()}/{inner}"


def shift_nan(arr: np.ndarray, dr: int, dc: int) -> np.ndarray:
    out = np.full(arr.shape, np.nan, dtype=np.float32)
    rs_src = slice(max(0, -dr), arr.shape[0] - max(0, dr))
    cs_src = slice(max(0, -dc), arr.shape[1] - max(0, dc))
    rs_dst = slice(max(0, dr), arr.shape[0] - max(0, -dr))
    cs_dst = slice(max(0, dc), arr.shape[1] - max(0, -dc))
    out[rs_dst, cs_dst] = arr[rs_src, cs_src]
    return out


def nanmean_filter3(arr: np.ndarray) -> np.ndarray:
    stack = np.stack([shift_nan(arr, dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)], axis=0)
    finite = np.isfinite(stack)
    count = finite.sum(axis=0).astype(np.float32)
    total = np.where(finite, stack, 0.0).sum(axis=0, dtype=np.float32)
    out = np.full(arr.shape, np.nan, dtype=np.float32)
    np.divide(total, count, out=out, where=count > 0)
    return out


def nanstd_filter3(arr: np.ndarray) -> np.ndarray:
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


def write_float_raster(path: Path, arr: np.ndarray, profile: dict, nodata: float = -9999.0) -> None:
    profile = profile.copy()
    profile.update(dtype=rasterio.float32, count=1, nodata=nodata, compress="deflate")
    out = np.where(np.isfinite(arr), arr, nodata).astype(np.float32)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out, 1)


def build_topography_cache(target_gdf: gpd.GeoDataFrame) -> Tuple[Dict[str, Path], pd.DataFrame]:
    topo_root = OUTPUT_ROOT / "topography_cache"
    topo_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "elev_m": topo_root / "topo_elev_m.tif",
        "slope_deg": topo_root / "topo_slope_deg.tif",
        "aspect_deg": topo_root / "topo_aspect_deg.tif",
        "northness": topo_root / "topo_northness.tif",
        "eastness": topo_root / "topo_eastness.tif",
        "d30": topo_root / "topo_d30.tif",
        "d120": topo_root / "topo_d120.tif",
        "tpi3": topo_root / "topo_tpi3.tif",
        "roughness3": topo_root / "topo_roughness3.tif",
    }
    if not FORCE_REBUILD_TOPO_CACHE and all(p.exists() for p in paths.values()):
        return paths, pd.DataFrame([{"cache_reused": True}])

    dem_path = dem_vsi_path()
    if dem_path is None:
        log("DEM not found; topographic predictors will be skipped")
        return {}, pd.DataFrame([{"dem_found": False}])

    log(f"Building DEM/topography cache from {dem_path}")
    with rasterio.open(dem_path) as src:
        dem_band = 1
        dst_crs = CRS.from_user_input(PROJECTED_CRS)
        # Reproject full DEM to projected CRS at 30 m. The DEM extent is small enough for operational use.
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds, resolution=DEM_PROJECTED_RESOLUTION_M
        )
        profile = src.profile.copy()
        profile.update(crs=dst_crs, transform=transform, width=width, height=height, count=1, dtype=rasterio.float32, nodata=-9999.0)
        dem = np.full((height, width), np.nan, dtype=np.float32)
        src_nodata = src.nodata
        tmp = np.full((height, width), -9999.0, dtype=np.float32)
        reproject(
            source=rasterio.band(src, dem_band),
            destination=tmp,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src_nodata,
            dst_transform=transform,
            dst_crs=dst_crs,
            dst_nodata=-9999.0,
            resampling=Resampling.bilinear,
        )
        dem = np.where(tmp == -9999.0, np.nan, tmp).astype(np.float32)

    # derivatives
    xres = abs(profile["transform"].a)
    yres = abs(profile["transform"].e)
    dzdy, dzdx = np.gradient(dem.astype(np.float32), yres, xres)
    slope_rad = np.arctan(np.sqrt(dzdx ** 2 + dzdy ** 2))
    slope_deg = np.degrees(slope_rad).astype(np.float32)
    aspect = (np.degrees(np.arctan2(dzdx, -dzdy)) + 360.0) % 360.0
    aspect = aspect.astype(np.float32)
    northness = np.cos(np.deg2rad(aspect)).astype(np.float32)
    eastness = np.sin(np.deg2rad(aspect)).astype(np.float32)
    d30 = np.cos(np.deg2rad(aspect - 30.0)).astype(np.float32)
    d120 = np.cos(np.deg2rad(aspect - 120.0)).astype(np.float32)
    local_mean = nanmean_filter3(dem)
    tpi3 = (dem - local_mean).astype(np.float32)
    roughness3 = nanstd_filter3(dem)

    arrays = {
        "elev_m": dem,
        "slope_deg": slope_deg,
        "aspect_deg": aspect,
        "northness": northness,
        "eastness": eastness,
        "d30": d30,
        "d120": d120,
        "tpi3": tpi3,
        "roughness3": roughness3,
    }
    for name, arr in arrays.items():
        write_float_raster(paths[name], arr, profile)

    diag = pd.DataFrame([{
        "dem_found": True,
        "dem_source": dem_path,
        "target_crs": PROJECTED_CRS,
        "resolution_m": DEM_PROJECTED_RESOLUTION_M,
        "width": width,
        "height": height,
        "valid_elevation_px": int(np.isfinite(dem).sum()),
        "valid_elevation_frac": float(np.isfinite(dem).mean()),
        "all_nodata_3x3_neighbourhood_px": int((~np.isfinite(local_mean)).sum()),
        "all_nodata_3x3_neighbourhood_frac": float((~np.isfinite(local_mean)).mean()),
    }])
    return paths, diag


def sample_topography(gdf: gpd.GeoDataFrame, topo_paths: Dict[str, Path]) -> pd.DataFrame:
    if not topo_paths:
        return pd.DataFrame(index=gdf.index)
    out_rows = [dict() for _ in range(len(gdf))]
    for topo_name, path in topo_paths.items():
        if not path.exists():
            continue
        with rasterio.open(path) as ds:
            for pos, (_, row) in enumerate(gdf.iterrows()):
                stats = zonal_stats_dataset(ds, row.geometry, gdf.crs, [topo_name], prefix="topo_", nodata_values={-9999.0})
                out_rows[pos].update(stats)
    return pd.DataFrame(out_rows, index=gdf.index)

# =============================================================================
# Feature construction
# =============================================================================

def load_s2_inventory() -> pd.DataFrame:
    inv = read_inventory(S2_CACHE_ROOT, "scene_inventory.csv")
    if inv.empty:
        log(f"S2 cache inventory not found at {S2_CACHE_ROOT}; S2 predictors will be marked missing")
    return inv


def load_s1_inventory_and_bands() -> Tuple[pd.DataFrame, List[str]]:
    inv = read_inventory(S1_CACHE_ROOT, "s1_scene_inventory.csv")
    if inv.empty:
        log(f"S1 cache inventory not found at {S1_CACHE_ROOT}; S1 predictors will be marked missing")
        return inv, []
    # Use available descriptors from inventory if possible.
    names: List[str] = []
    if "descriptor_bands_available" in inv.columns:
        for s in inv["descriptor_bands_available"].fillna(""):
            for n in str(s).split(","):
                n = n.strip()
                if n and n not in S1_DESCRIPTOR_EXCLUDE and n not in names:
                    names.append(n)
    if not names:
        # Fallback to common v4 descriptor names.
        names = [
            "S1_VH_DB", "S1_VV_DB", "S1_VH_LINEAR", "S1_VV_LINEAR",
            "S1_VV_MINUS_VH_DB", "S1_VV_DIV_VH_LINEAR",
            "S1_VH_LOCAL_MEAN3", "S1_VH_LOCAL_STD3", "S1_VV_LOCAL_MEAN3", "S1_VV_LOCAL_STD3",
        ]
    return inv, names


def add_delta_features(df: pd.DataFrame, band_names: Sequence[str], pre_prefix: str, post_prefix: str, delta_prefix: str) -> pd.DataFrame:
    out = df.copy()
    for b in band_names:
        bkey = b.lower()
        for stat in ["mean", "median", "q10", "q90"]:
            pre = f"{pre_prefix}{bkey}_{stat}"
            post = f"{post_prefix}{bkey}_{stat}"
            if pre in out.columns and post in out.columns:
                delta_col = f"{delta_prefix}{bkey}_{stat}"
                out[delta_col] = out[post] - out[pre]
                out[f"{delta_col}_per_year"] = out[delta_col] / (out["days_between"].replace(0, np.nan) / 365.25)
    return out


def add_semantic_features(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    # Optional, intentionally lightweight. Uses exact date folders if present.
    if not SEMANTIC_MASK_ROOT.exists():
        return pd.DataFrame(index=gdf.index)
    raster_specs = [
        ("semantic_vote_count.tif", "vote_count", "uint"),
        ("semantic_anomaly_score.tif", "anomaly", "float"),
        ("semantic_envelope_confidence.tif", "confidence", "uint"),
        ("semantic_keep_mask.tif", "keep_mask", "uint"),
    ]
    rows = [dict() for _ in range(len(gdf))]
    for role, date_col in [("pre", "pre_date"), ("post", "post_date")]:
        for pos, (_, row) in enumerate(gdf.iterrows()):
            scene_dir = SEMANTIC_MASK_ROOT / str(row[date_col])
            rows[pos][f"m_semantic_{role}_available"] = int(scene_dir.exists())
            if not scene_dir.exists():
                continue
            for fname, name, kind in raster_specs:
                path = scene_dir / fname
                if not path.exists():
                    continue
                with rasterio.open(path) as ds:
                    stats = zonal_stats_dataset(ds, row.geometry, gdf.crs, [name], prefix=f"eo_semantic_{role}_", nodata_values={255.0, -9999.0, -32768.0})
                    rows[pos].update(stats)
    return pd.DataFrame(rows, index=gdf.index)


def build_feature_dictionary(columns: Sequence[str]) -> pd.DataFrame:
    rows = []
    for c in columns:
        if c in {"geometry"}:
            continue
        if c.startswith("eo_s2_"):
            block = "x_EO / Sentinel-2"
        elif c.startswith("delta_s2_"):
            block = "x_delta / Sentinel-2"
        elif c.startswith("eo_s1_"):
            block = "x_EO / Sentinel-1"
        elif c.startswith("delta_s1_"):
            block = "x_delta / Sentinel-1"
        elif c.startswith("shape_"):
            block = "x_shape"
        elif c.startswith("topo_"):
            block = "x_topo"
        elif c.startswith("m_") or c.startswith("support_"):
            block = "m / support and missingness"
        elif c in {"rf_label", "label_raw", "label_status"}:
            block = "target label"
        else:
            block = "metadata"
        rows.append({"feature": c, "block": block})
    return pd.DataFrame(rows)

# =============================================================================
# Main
# =============================================================================

def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    log("Starting Algorithm #6: EO training vectors from Comandau.gdb")
    gdb_path = resolve_gdb_path()
    log(f"Using GDB: {gdb_path}")

    layer_inv, label_summary = inspect_gdb_layers(gdb_path)
    layer_inv.to_csv(OUTPUT_ROOT / "gdb_layer_inventory.csv", index=False)
    label_summary.to_csv(OUTPUT_ROOT / "gdb_label_summary.csv", index=False)

    all_obj = load_training_objects(gdb_path)
    all_obj_proj = add_shape_predictors(all_obj)
    all_obj_proj.to_file(OUTPUT_ROOT / "all_candidate_objects_from_gdb.gpkg", layer="all_candidate_objects", driver="GPKG")

    # Select training objects.
    valid = all_obj_proj["rf_label"].isin(RF_CLASSES)
    if not INCLUDE_UNLABELLED_OBJECTS:
        valid &= all_obj_proj["label_status"].ne("unassigned")
    if not INCLUDE_NA_OBJECTS:
        valid &= all_obj_proj["label_status"].ne("excluded")
    train_gdf = all_obj_proj[valid].copy()
    log(f"Training-labelled objects selected: {len(train_gdf)} / {len(all_obj_proj)}")

    if train_gdf.empty:
        raise RuntimeError("No labelled HV/WT/IN training objects found. Continue labelling the GDB and rerun Algorithm #6.")

    # Topography.
    topo_paths, topo_diag = build_topography_cache(train_gdf)
    topo_diag.to_csv(OUTPUT_ROOT / "topography_cache_diagnostics.csv", index=False)
    topo_df = sample_topography(train_gdf, topo_paths)

    # Sentinel-2.
    s2_inv = load_s2_inventory()
    s2_pre = sample_stack_features_for_rows(
        train_gdf, s2_inv, "pre_date", ["index_stack_path"], S2_INDEX_BANDS,
        prefix="eo_s2_pre_", max_days=S2_MAX_MATCH_DAYS, nodata_values=S2_NODATA_VALUES,
    ) if not s2_inv.empty else pd.DataFrame(index=train_gdf.index)
    s2_post = sample_stack_features_for_rows(
        train_gdf, s2_inv, "post_date", ["index_stack_path"], S2_INDEX_BANDS,
        prefix="eo_s2_post_", max_days=S2_MAX_MATCH_DAYS, nodata_values=S2_NODATA_VALUES,
    ) if not s2_inv.empty else pd.DataFrame(index=train_gdf.index)

    # Sentinel-1.
    s1_inv, s1_bands = load_s1_inventory_and_bands()
    s1_pre = sample_stack_features_for_rows(
        train_gdf, s1_inv, "pre_date", ["s1_available_stack_path", "s1_descriptor_stack_path"], s1_bands,
        prefix="eo_s1_pre_", max_days=S1_MAX_MATCH_DAYS, nodata_values=S1_NODATA_VALUES,
    ) if not s1_inv.empty and s1_bands else pd.DataFrame(index=train_gdf.index)
    s1_post = sample_stack_features_for_rows(
        train_gdf, s1_inv, "post_date", ["s1_available_stack_path", "s1_descriptor_stack_path"], s1_bands,
        prefix="eo_s1_post_", max_days=S1_MAX_MATCH_DAYS, nodata_values=S1_NODATA_VALUES,
    ) if not s1_inv.empty and s1_bands else pd.DataFrame(index=train_gdf.index)

    # Semantic optional.
    semantic_df = add_semantic_features(train_gdf)

    base_cols = [
        "training_object_id", "source_layer", "source_id", "pre_date", "post_date", "interval", "days_between",
        "label_raw", "rf_label", "label_status", "gridcode" if "gridcode" in train_gdf.columns else None,
        "Shape_Length" if "Shape_Length" in train_gdf.columns else None,
        "Shape_Area" if "Shape_Area" in train_gdf.columns else None,
    ]
    base_cols = [c for c in base_cols if c is not None and c in train_gdf.columns]
    shape_cols = [c for c in train_gdf.columns if c.startswith("shape_")]
    feat_df = pd.concat(
        [
            train_gdf[base_cols + shape_cols].reset_index(drop=True),
            topo_df.reset_index(drop=True),
            s2_pre.reset_index(drop=True),
            s2_post.reset_index(drop=True),
            s1_pre.reset_index(drop=True),
            s1_post.reset_index(drop=True),
            semantic_df.reset_index(drop=True),
        ],
        axis=1,
    )
    feat_df = add_delta_features(feat_df, S2_INDEX_BANDS, "eo_s2_pre_", "eo_s2_post_", "delta_s2_")
    if s1_bands:
        feat_df = add_delta_features(feat_df, s1_bands, "eo_s1_pre_", "eo_s1_post_", "delta_s1_")

    # Combine geometry for vector outputs.
    out_gdf = train_gdf[["geometry"]].reset_index(drop=True).join(feat_df)
    out_gdf = gpd.GeoDataFrame(out_gdf, geometry="geometry", crs=PROJECTED_CRS)

    # Summaries.
    label_counts = out_gdf.groupby("rf_label", as_index=False).agg(
        n_objects=("training_object_id", "count"),
        area_ha_sum=("shape_area_ha", "sum"),
        area_ha_median=("shape_area_ha", "median"),
        area_ha_min=("shape_area_ha", "min"),
        area_ha_max=("shape_area_ha", "max"),
    )
    label_counts.to_csv(OUTPUT_ROOT / "training_label_summary.csv", index=False)

    csv_path = OUTPUT_ROOT / "eo_rf_training_vectors_comandau.csv"
    out_gdf.drop(columns="geometry").to_csv(csv_path, index=False)
    log(f"Wrote training-vector CSV: {csv_path}")

    if WRITE_GPKG:
        gpkg_path = OUTPUT_ROOT / "eo_rf_training_vectors_comandau.gpkg"
        out_gdf.to_file(gpkg_path, layer="eo_rf_training_vectors", driver="GPKG")
        log(f"Wrote training-vector GeoPackage: {gpkg_path}")

    if WRITE_BACK_TO_GDB:
        try:
            # OpenFileGDB write support is GDAL-version dependent. If it fails,
            # the CSV/GPKG outputs remain the authoritative products.
            out_gdf.to_file(str(gdb_path), layer=OUTPUT_GDB_LAYER_NAME, driver="OpenFileGDB")
            log(f"Wrote training-vector layer back to GDB: {OUTPUT_GDB_LAYER_NAME}")
        except Exception as exc:
            (OUTPUT_ROOT / "write_back_to_gdb_warning.txt").write_text(
                f"Could not write training vectors back to the FileGDB. This is usually caused by missing OpenFileGDB write support or a locked GDB.\n\n{repr(exc)}\n",
                encoding="utf-8",
            )
            log("Could not write back to FileGDB; wrote warning file and kept CSV/GPKG outputs")

    feature_dict = build_feature_dictionary(out_gdf.drop(columns="geometry").columns)
    feature_dict.to_csv(OUTPUT_ROOT / "eo_training_feature_dictionary.csv", index=False)
    feature_dict.groupby("block", as_index=False).size().rename(columns={"size": "n_features"}).to_csv(
        OUTPUT_ROOT / "eo_training_feature_block_summary.csv", index=False
    )

    config = {
        "training_gdb": str(TRAINING_GDB),
        "training_gdb_zip_fallback": str(TRAINING_GDB_ZIP_FALLBACK),
        "output_root": str(OUTPUT_ROOT),
        "s2_cache_root": str(S2_CACHE_ROOT),
        "s1_cache_root": str(S1_CACHE_ROOT),
        "semantic_mask_root": str(SEMANTIC_MASK_ROOT),
        "dem_tif": str(DEM_TIF),
        "dem_zip": str(DEM_ZIP),
        "projected_crs": PROJECTED_CRS,
        "rf_classes": RF_CLASSES,
        "label_field_candidates": LABEL_FIELD_CANDIDATES,
        "include_unlabelled_objects": INCLUDE_UNLABELLED_OBJECTS,
        "include_na_objects": INCLUDE_NA_OBJECTS,
        "n_training_objects": int(len(out_gdf)),
    }
    (OUTPUT_ROOT / "algorithm6_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    log(f"Done. Training vectors in {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
