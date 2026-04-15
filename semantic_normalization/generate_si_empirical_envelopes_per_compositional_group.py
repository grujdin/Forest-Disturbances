
"""
Derive empirical healthy-forest spectral-index envelopes from EO Browser ZIP exports,
stratified by analytical composition group and DOY bin.

Enhanced version:
- no aliases; uses workbook group names as-is
- timestamped progress output
- no plots by default
- computes per-scene quality indicators
- writes keep/caution/blacklist recommendations
- can optionally skip blacklisted scenes automatically before envelope aggregation
"""
from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple
import time
from datetime import datetime

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.warp import Resampling, reproject

# =============================================================================
# HARD-CODED CONFIG
# =============================================================================

ZIP_GLOB = "D:/Forest_Disturbance/imagery_zip/Stana_de_Vale_BH/SdV_*.zip"
ZIP_FILES = sorted(Path("D:/Forest_Disturbance/imagery_zip/Stana_de_Vale_BH").glob("SdV_*.zip"))

FMU_GEOJSON = Path("D:/Forest_Disturbance/tables/SdV_FMU.geojson")
GROUP_WORKBOOK = Path("D:/Forest_Disturbance/tables/sdv_compos_groups_loss_causes_reference.xlsx")
GROUP_SHEET = "LOSS_CAUSES"

FMU_JOIN_FIELD = "ua"
WORKBOOK_JOIN_FIELD = "ua"
GROUP_CODE_FIELD = "COMPOZ_TYPE_CODE"
GROUP_LABEL_FIELD = "COMPOZ_TYPE_LABEL"
TOTAL_LOSS_FIELD = "Total loss"
AREA_HA_FIELD = "ha"
YEAR_LOSS_FIELDS = ["2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025", "2026"]

EXCLUDE_GROUP_CODES = {"NO_DATA"}

USE_ONLY_STABLE_SUBPARCELS = True
STABLE_TOTAL_LOSS_FRAC_MAX = 0.005
STABLE_MAX_YEAR_LOSS_FRAC_MAX = 0.003

DOY_BIN_WIDTH = 30
ENV_Q_LOW = 0.10
ENV_Q_HIGH = 0.90
USE_INDICES = ["NDVI", "NDMI", "NBR", "NDRE"]

USE_SCL_IF_AVAILABLE = True
SCL_EXCLUDE_CLASSES = {0, 1, 2, 3, 8, 9, 10, 11}
ALLOW_UNKNOWN_SCL_COLORS = False
MIN_PIXELS_PER_GROUP_DATE = 50
MIN_DATES_PER_BIN_FOR_PLOT = 1

# Scene-quality controls
COMPUTE_SCENE_QUALITY = True
SCENE_FILTER_MODE = "blacklist_and_caution" # Options: "none", "blacklist", "blacklist_and_caution"

# Recommendation thresholds
BLACKLIST_CLEAR_FRAC_MAX = 0.10
BLACKLIST_CLOUD_FRAC_MIN = 0.80
BLACKLIST_NDSI_MED_MIN = 0.20
BLACKLIST_VEG_FRAC_ZERO = 0.0

CAUTION_CLEAR_FRAC_MAX = 0.60
CAUTION_VEG_FRAC_MAX = 0.20
CAUTION_SNOW_FRAC_MIN = 0.10
CAUTION_SHADOW_FRAC_MIN = 0.30
CAUTION_SCL7_FRAC_MIN = 0.10
CAUTION_NDSI_MED_MIN = 0.00

OUTPUT_DIR = Path("D:/Forest_Disturbance/outputs/sdv_si_empirical_envelopes_per_compositional_group")
PLOTS_DIRNAME = "plots"
WRITE_INTERMEDIATE = True
VERBOSE = True
MAKE_PLOTS = True

# =============================================================================
# Helpers
# =============================================================================

DATE_RE = re.compile(r"(20\d{2})-(\d{2})-(\d{2})")
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

def log(msg: str) -> None:
    if VERBOSE:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def _scl_color_u16(rgb: Tuple[int, int, int]) -> Tuple[int, int, int]:
    return tuple(int(round(c * 65535.0 / 255.0)) for c in rgb)

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

def doy_bin_center(doy: int, width: int) -> int:
    start = ((doy - 1) // width) * width + 1
    end = min(start + width - 1, 365)
    return int(round((start + end) / 2))

def doy_bin_label(doy: int, width: int) -> str:
    start = ((doy - 1) // width) * width + 1
    end = min(start + width - 1, 365)
    return f"{start:03d}-{end:03d}"

def parse_date_from_name(name: str) -> pd.Timestamp:
    m = DATE_RE.search(name)
    if not m:
        raise ValueError(f"Could not parse date from: {name}")
    y, mm, dd = map(int, m.groups())
    return pd.Timestamp(year=y, month=mm, day=dd)

def zip_members_by_tag(zip_path: Path) -> Dict[str, str]:
    need = {
        "B03": "B03_(Raw).tiff",
        "B04": "B04_(Raw).tiff",
        "B05": "B05_(Raw).tiff",
        "B08": "B08_(Raw).tiff",
        "B8A": "B8A_(Raw).tiff",
        "B11": "B11_(Raw).tiff",
        "B12": "B12_(Raw).tiff",
    }
    out: Dict[str, str] = {}
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        lower_names = {n.lower(): n for n in names}
        for tag, pattern in need.items():
            pat = pattern.lower()
            matches = [orig for low, orig in lower_names.items() if pat in low]
            if not matches:
                raise FileNotFoundError(f"Could not find {pattern} in {zip_path.name}")
            out[tag] = matches[0]
        scl_matches = [orig for low, orig in lower_names.items() if "scene_classification_map" in low or "_scl" in low or "scl" in low]
        out["SCL"] = scl_matches[0] if scl_matches else None
        out["DATE_STRING"] = out["B04"]
    return out

def vsizip_path(zip_path: Path, inner_member: str) -> str:
    return f"/vsizip/{zip_path}/{inner_member}"

def read_single_band(zip_path: Path, inner_member: str):
    path = vsizip_path(zip_path, inner_member)
    with rasterio.open(path) as ds:
        arr = ds.read(1)
        profile = ds.profile.copy()
        valid = ds.read_masks(1) > 0
    return arr, profile, valid

def align_multiband_to(ref_profile: dict, src_path: str, band_indexes: List[int]) -> np.ndarray:
    with rasterio.open(src_path) as src:
        out = np.zeros((len(band_indexes), ref_profile["height"], ref_profile["width"]), dtype=np.float32)
        for out_i, band_i in enumerate(band_indexes):
            src_arr = src.read(band_i)
            reproject(
                source=src_arr.astype(np.float32),
                destination=out[out_i],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_profile["transform"],
                dst_crs=ref_profile["crs"],
                resampling=Resampling.nearest,
            )
    return out

def same_grid(profile_a: dict, profile_b: dict) -> bool:
    return (
        profile_a.get("height") == profile_b.get("height")
        and profile_a.get("width") == profile_b.get("width")
        and profile_a.get("transform") == profile_b.get("transform")
        and str(profile_a.get("crs")) == str(profile_b.get("crs"))
    )

def _decode_rendered_scl_classes(rgb: np.ndarray) -> np.ndarray:
    vmax = float(np.nanmax(rgb)) if rgb.size else 0.0
    use_u16 = vmax > 255.5
    base_palette = SCL_COLORS_U16 if use_u16 else SCL_COLORS_U8
    palette = {cid: np.array(base_palette[name], dtype=np.int32) for name, cid in SCL_CLASS_BY_NAME.items()}
    scl = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    rgb_i32 = rgb.astype(np.int32)
    for cid, color in palette.items():
        match = np.all(np.abs(rgb_i32 - color) <= 2, axis=2)
        scl[match] = cid
    return scl

def load_scl_classes(zip_path: Path, scl_member: str, ref_profile: dict) -> np.ndarray:
    if scl_member is None:
        return np.full((ref_profile["height"], ref_profile["width"]), 255, dtype=np.uint8)
    scl_path = vsizip_path(zip_path, scl_member)
    with rasterio.open(scl_path) as ds:
        if ds.count == 1:
            arr = ds.read(1)
            if not same_grid(ds.profile, ref_profile):
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
        rgb = align_multiband_to(ref_profile, scl_path, [1, 2, 3]).transpose(1, 2, 0)
        return _decode_rendered_scl_classes(rgb)

def build_clear_mask(data_mask: np.ndarray, scl_classes: np.ndarray) -> np.ndarray:
    invalid = np.isin(scl_classes, list(SCL_EXCLUDE_CLASSES))
    if not ALLOW_UNKNOWN_SCL_COLORS:
        invalid |= (scl_classes == 255)
    return data_mask & (~invalid)

def safe_norm_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    denom = a + b
    out = np.full(a.shape, np.nan, dtype=np.float32)
    valid = denom != 0
    out[valid] = (a[valid] - b[valid]) / denom[valid]
    return out

def compute_indices(bands: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out = {}
    if "NDVI" in USE_INDICES:
        out["NDVI"] = safe_norm_diff(bands["B08"], bands["B04"])
    if "NDMI" in USE_INDICES:
        out["NDMI"] = safe_norm_diff(bands["B08"], bands["B11"])
    if "NBR" in USE_INDICES:
        out["NBR"] = safe_norm_diff(bands["B08"], bands["B12"])
    if "NDRE" in USE_INDICES:
        out["NDRE"] = safe_norm_diff(bands["B8A"], bands["B05"])
    return out

def compute_ndsi(bands: Dict[str, np.ndarray]) -> np.ndarray:
    return safe_norm_diff(bands["B03"], bands["B11"])

def resolve_group_sheet(workbook_path: Path, preferred_sheet: str) -> str:
    xls = pd.ExcelFile(workbook_path)
    if preferred_sheet in xls.sheet_names:
        return preferred_sheet
    required = {WORKBOOK_JOIN_FIELD, GROUP_CODE_FIELD, GROUP_LABEL_FIELD, TOTAL_LOSS_FIELD, AREA_HA_FIELD}
    for sheet in xls.sheet_names:
        try:
            cols = set(pd.read_excel(workbook_path, sheet_name=sheet, nrows=0).columns)
        except Exception:
            continue
        if required.issubset(cols):
            return sheet
    raise ValueError(
        f"Could not find a workbook sheet containing the required columns {sorted(required)}. "
        f"Available sheets: {xls.sheet_names}"
    )

def load_grouped_subparcels() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(FMU_GEOJSON)
    sheet = resolve_group_sheet(GROUP_WORKBOOK, GROUP_SHEET)
    attrs = pd.read_excel(GROUP_WORKBOOK, sheet_name=sheet)
    keep_cols = [WORKBOOK_JOIN_FIELD, GROUP_CODE_FIELD, GROUP_LABEL_FIELD, TOTAL_LOSS_FIELD, AREA_HA_FIELD] + YEAR_LOSS_FIELDS
    missing = [c for c in keep_cols if c not in attrs.columns]
    if missing:
        raise ValueError(f"Workbook sheet '{sheet}' is missing required columns: {missing}")
    attrs = attrs[keep_cols].copy()
    attrs[WORKBOOK_JOIN_FIELD] = attrs[WORKBOOK_JOIN_FIELD].astype(str)
    attrs = attrs.drop_duplicates(subset=[WORKBOOK_JOIN_FIELD])
    gdf[FMU_JOIN_FIELD] = gdf[FMU_JOIN_FIELD].astype(str)
    merged = gdf.merge(attrs, left_on=FMU_JOIN_FIELD, right_on=WORKBOOK_JOIN_FIELD, how="left")

    area_m2 = pd.to_numeric(merged[AREA_HA_FIELD], errors="coerce").fillna(0) * 10000.0
    total_loss = pd.to_numeric(merged[TOTAL_LOSS_FIELD], errors="coerce").fillna(0)
    year_loss = merged[YEAR_LOSS_FIELDS].apply(pd.to_numeric, errors="coerce").fillna(0)
    max_year_loss = year_loss.max(axis=1)

    with np.errstate(divide='ignore', invalid='ignore'):
        merged['stable_total_loss_frac'] = np.where(area_m2 > 0, total_loss / area_m2, np.nan)
        merged['stable_max_year_loss_frac'] = np.where(area_m2 > 0, max_year_loss / area_m2, np.nan)
        merged['stable_total_loss_pct'] = merged['stable_total_loss_frac'] * 100.0
        merged['stable_max_year_loss_pct'] = merged['stable_max_year_loss_frac'] * 100.0

    if USE_ONLY_STABLE_SUBPARCELS:
        stable = (
            (area_m2 > 0)
            & (merged['stable_total_loss_frac'].fillna(np.inf) <= STABLE_TOTAL_LOSS_FRAC_MAX)
            & (merged['stable_max_year_loss_frac'].fillna(np.inf) <= STABLE_MAX_YEAR_LOSS_FRAC_MAX)
        )
    else:
        stable = pd.Series(True, index=merged.index)

    good_group = merged[GROUP_CODE_FIELD].notna() & (~merged[GROUP_CODE_FIELD].isin(EXCLUDE_GROUP_CODES))
    merged = merged.loc[stable & good_group].copy()
    merged = merged.loc[~merged.geometry.is_empty & merged.geometry.notna()].copy()
    return merged

def rasterize_groups(group_gdf: gpd.GeoDataFrame, ref_profile: dict):
    if str(group_gdf.crs) != str(ref_profile['crs']):
        group_gdf = group_gdf.to_crs(ref_profile['crs'])
    group_table = group_gdf[[GROUP_CODE_FIELD, GROUP_LABEL_FIELD]].drop_duplicates().sort_values([GROUP_CODE_FIELD]).reset_index(drop=True).copy()
    group_table['group_id'] = np.arange(1, len(group_table) + 1, dtype=np.int32)
    code_to_id = dict(zip(group_table[GROUP_CODE_FIELD], group_table['group_id']))
    shapes = [
        (geom, int(code_to_id[code]))
        for geom, code in zip(group_gdf.geometry, group_gdf[GROUP_CODE_FIELD])
        if geom is not None and not geom.is_empty
    ]
    group_raster = rasterize(
        shapes=shapes,
        out_shape=(ref_profile['height'], ref_profile['width']),
        transform=ref_profile['transform'],
        fill=0,
        all_touched=False,
        dtype='int32',
    )
    return group_raster, group_table

def summarize_support(group_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    return (
        group_gdf.groupby([GROUP_CODE_FIELD, GROUP_LABEL_FIELD], dropna=False)
        .agg(
            n_subparcels=(FMU_JOIN_FIELD, 'count'),
            total_area_ha=(AREA_HA_FIELD, 'sum'),
            median_total_loss_pct=('stable_total_loss_pct', 'median'),
            max_total_loss_pct=('stable_total_loss_pct', 'max'),
            max_single_year_loss_pct=('stable_max_year_loss_pct', 'max'),
        )
        .reset_index()
        .rename(columns={GROUP_CODE_FIELD: 'group_code', GROUP_LABEL_FIELD: 'group_label'})
    )

def scene_quality_recommendation(row: pd.Series) -> Tuple[str, str]:
    reasons = []
    clear_frac = float(row.get("clear_frac", np.nan))
    veg_frac = float(row.get("veg_frac", np.nan))
    cloud_frac = float(row.get("cloud_frac", np.nan))
    shadow_frac = float(row.get("shadow_frac", np.nan))
    snow_frac = float(row.get("snow_frac", np.nan))
    scl7_frac = float(row.get("scl7_frac", np.nan))
    ndsi_med = float(row.get("ndsi_med_clear", np.nan))
    clear_px = int(row.get("clear_support_px", 0))

    if clear_px == 0 or not np.isfinite(row.get("ndvi_med_clear", np.nan)) or clear_frac <= BLACKLIST_CLEAR_FRAC_MAX:
        reasons.append("too_little_clear_support")
    if cloud_frac >= BLACKLIST_CLOUD_FRAC_MIN:
        reasons.append("cloud_dominated")
    if veg_frac <= BLACKLIST_VEG_FRAC_ZERO and np.isfinite(ndsi_med) and ndsi_med >= BLACKLIST_NDSI_MED_MIN:
        reasons.append("no_vegetation_and_snow_like_clear_pixels")

    if reasons:
        return "blacklist", ";".join(reasons)

    caution = []
    if clear_frac <= CAUTION_CLEAR_FRAC_MAX:
        caution.append("low_clear_support")
    if veg_frac <= CAUTION_VEG_FRAC_MAX:
        caution.append("low_vegetation_support")
    if snow_frac >= CAUTION_SNOW_FRAC_MIN:
        caution.append("snow_present")
    if shadow_frac >= CAUTION_SHADOW_FRAC_MIN:
        caution.append("high_shadow_fraction")
    if scl7_frac >= CAUTION_SCL7_FRAC_MIN:
        caution.append("high_unclassified_fraction")
    if np.isfinite(ndsi_med) and ndsi_med >= CAUTION_NDSI_MED_MIN:
        caution.append("positive_ndsi_clear_pixels")

    if caution:
        return "caution", ";".join(caution)

    return "keep", "no_major_scene_quality_issue"

def write_scene_quality_csv(rows: List[dict], outdir: Path) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    qdf = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    qdf.to_csv(outdir / "scene_quality_summary.csv", index=False)

    recs = []
    for _, row in qdf.iterrows():
        rec, why = scene_quality_recommendation(row)
        recs.append({
            **row.to_dict(),
            "recommendation": rec,
            "recommendation_reason": why,
        })
    rdf = pd.DataFrame(recs)
    rdf.to_csv(outdir / "scene_blacklist_recommendation.csv", index=False)
    return rdf

def main():
    t0 = time.time()
    log("Starting empirical envelope extraction")
    if not ZIP_FILES:
        raise RuntimeError(f'No ZIP files found with pattern: {ZIP_GLOB}')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log(f"Found {len(ZIP_FILES)} ZIP files")
    plots_dir = OUTPUT_DIR / PLOTS_DIRNAME
    plots_dir.mkdir(parents=True, exist_ok=True)

    first_members = zip_members_by_tag(ZIP_FILES[0])
    _, ref_profile, _ = read_single_band(ZIP_FILES[0], first_members['B04'])

    group_gdf = load_grouped_subparcels()
    if len(group_gdf) == 0:
        raise RuntimeError('No grouped subparcels remained after the healthy-support filter.')
    group_gdf.to_file(OUTPUT_DIR / 'stable_subparcels_used.geojson', driver='GeoJSON')
    support_df = summarize_support(group_gdf)
    support_df.to_csv(OUTPUT_DIR / 'stable_support_by_group.csv', index=False)
    log(f'Using {len(group_gdf)} stable support subparcels across {support_df.shape[0]} groups')

    support_summary = pd.DataFrame([{
        'zip_count_input': len(ZIP_FILES),
        'support_rule': 'relative loss thresholds',
        'group_sheet_used': resolve_group_sheet(GROUP_WORKBOOK, GROUP_SHEET),
        'stable_total_loss_frac_max': STABLE_TOTAL_LOSS_FRAC_MAX,
        'stable_total_loss_pct_max': STABLE_TOTAL_LOSS_FRAC_MAX * 100.0,
        'stable_max_year_loss_frac_max': STABLE_MAX_YEAR_LOSS_FRAC_MAX,
        'stable_max_year_loss_pct_max': STABLE_MAX_YEAR_LOSS_FRAC_MAX * 100.0,
        'n_support_subparcels': len(group_gdf),
        'support_area_ha': float(pd.to_numeric(group_gdf[AREA_HA_FIELD], errors='coerce').fillna(0).sum()),
        'scene_filter_mode': SCENE_FILTER_MODE,
    }])
    support_summary.to_csv(OUTPUT_DIR / 'support_rule_summary.csv', index=False)

    group_raster, group_table = rasterize_groups(group_gdf, ref_profile)
    support_domain = group_raster > 0

    # Pass 1: scene-quality analysis
    quality_rows = []
    scan_records = []
    for i_scene, zip_path in enumerate(ZIP_FILES, start=1):
        log(f"[Q {i_scene}/{len(ZIP_FILES)}] Scoring {zip_path.name}")
        members = zip_members_by_tag(zip_path)
        date = parse_date_from_name(members['DATE_STRING'])
        doy = int(date.dayofyear)

        bands = {}
        data_masks = []
        profile_ref = None
        for tag in ['B03', 'B04', 'B05', 'B08', 'B8A', 'B11', 'B12']:
            arr, profile, valid = read_single_band(zip_path, members[tag])
            if profile_ref is None:
                profile_ref = profile
            bands[tag] = arr.astype(np.float32)
            data_masks.append(valid)
        if not same_grid(profile_ref, ref_profile):
            raise RuntimeError(f'{zip_path.name} is not on the same grid as the reference ZIP.')

        data_mask = np.logical_and.reduce(data_masks)
        if USE_SCL_IF_AVAILABLE and members.get('SCL') is not None:
            scl_classes = load_scl_classes(zip_path, members['SCL'], ref_profile)
            clear_mask = build_clear_mask(data_mask, scl_classes)
        else:
            scl_classes = np.full(data_mask.shape, 255, dtype=np.uint8)
            clear_mask = data_mask.copy()

        m = support_domain
        n_support_px = int(m.sum())
        if n_support_px == 0:
            raise RuntimeError("No support-domain pixels available after rasterization.")

        support_clear = m & clear_mask
        ndvi = safe_norm_diff(bands["B08"], bands["B04"])
        ndre = safe_norm_diff(bands["B8A"], bands["B05"])
        ndmi = safe_norm_diff(bands["B08"], bands["B11"])
        nbr = safe_norm_diff(bands["B08"], bands["B12"])
        ndsi = compute_ndsi(bands)

        def med_on(mask, arr):
            vals = arr[mask]
            vals = vals[np.isfinite(vals)]
            return float(np.median(vals)) if vals.size else np.nan

        quality_rows.append({
            "scene": zip_path.name,
            "date": date.date().isoformat(),
            "doy": doy,
            "support_px": n_support_px,
            "clear_support_px": int(support_clear.sum()),
            "clear_frac": float(support_clear.sum() / n_support_px),
            "veg_frac": float(np.mean(scl_classes[m] == 4)),
            "snow_frac": float(np.mean(scl_classes[m] == 11)),
            "cloud_frac": float(np.mean(np.isin(scl_classes[m], [8, 9, 10]))),
            "shadow_frac": float(np.mean(np.isin(scl_classes[m], [2, 3]))),
            "scl7_frac": float(np.mean(scl_classes[m] == 7)),
            "ndvi_med_clear": med_on(support_clear, ndvi),
            "ndre_med_clear": med_on(support_clear, ndre),
            "ndmi_med_clear": med_on(support_clear, ndmi),
            "nbr_med_clear": med_on(support_clear, nbr),
            "ndsi_med_clear": med_on(support_clear, ndsi),
        })
        scan_records.append((zip_path, members, bands, data_mask, clear_mask))

    rec_df = write_scene_quality_csv(quality_rows, OUTPUT_DIR)
    if COMPUTE_SCENE_QUALITY and not rec_df.empty:
        n_black = int((rec_df["recommendation"] == "blacklist").sum())
        n_caut = int((rec_df["recommendation"] == "caution").sum())
        log(f"Scene-quality scan done: {n_black} blacklist, {n_caut} caution, {len(rec_df)-n_black-n_caut} keep")

    if not rec_df.empty and SCENE_FILTER_MODE != "none":
        if SCENE_FILTER_MODE == "blacklist":
            skip_labels = {"blacklist"}
        elif SCENE_FILTER_MODE == "blacklist_and_caution":
            skip_labels = {"blacklist", "caution"}
        else:
            raise ValueError(
                f"Unsupported SCENE_FILTER_MODE={SCENE_FILTER_MODE!r}. "
                "Use 'none', 'blacklist', or 'blacklist_and_caution'."
            )

        skip_scenes = set(
            rec_df.loc[rec_df["recommendation"].isin(skip_labels), "scene"].astype(str)
        )
        scan_records = [rec for rec in scan_records if rec[0].name not in skip_scenes]
        log(
            f"After scene filtering mode '{SCENE_FILTER_MODE}' "
            f"({', '.join(sorted(skip_labels))}): {len(scan_records)} scenes remain"
        )

    date_rows = []
    for i_scene, (zip_path, members, bands, data_mask, clear_mask) in enumerate(scan_records, start=1):
        t_scene = time.time()
        log(f"[E {i_scene}/{len(scan_records)}] Processing {zip_path.name}")
        date = parse_date_from_name(members['DATE_STRING'])
        doy = int(date.dayofyear)
        bin_label = doy_bin_label(doy, DOY_BIN_WIDTH)
        bin_center = doy_bin_center(doy, DOY_BIN_WIDTH)

        valid_mask = clear_mask & support_domain
        idx = compute_indices(bands)
        for group_id, group_code, group_label in group_table[['group_id', GROUP_CODE_FIELD, GROUP_LABEL_FIELD]].itertuples(index=False):
            group_mask = valid_mask & (group_raster == group_id)
            n_group_pixels = int(group_mask.sum())
            if n_group_pixels < MIN_PIXELS_PER_GROUP_DATE:
                continue
            for index_name, arr in idx.items():
                vals = arr[group_mask]
                vals = vals[np.isfinite(vals)]
                if vals.size < MIN_PIXELS_PER_GROUP_DATE:
                    continue
                date_rows.append({
                    'zip_file': zip_path.name,
                    'date': date.date().isoformat(),
                    'year': int(date.year),
                    'month': int(date.month),
                    'day': int(date.day),
                    'doy': doy,
                    'doy_bin_label': bin_label,
                    'doy_bin_center': bin_center,
                    'group_id': int(group_id),
                    'group_code': group_code,
                    'group_label': group_label,
                    'index': index_name,
                    'n_valid_pixels': int(vals.size),
                    'date_min': float(np.min(vals)),
                    'date_q10': float(np.quantile(vals, 0.10)),
                    'date_median': float(np.median(vals)),
                    'date_q90': float(np.quantile(vals, 0.90)),
                    'date_max': float(np.max(vals)),
                    'date_mean': float(np.mean(vals)),
                })

        log(f"[E {i_scene}/{len(scan_records)}] Finished {zip_path.name} in {time.time() - t_scene:.1f}s; cumulative rows = {len(date_rows)}")
        if WRITE_INTERMEDIATE and date_rows:
            pd.DataFrame(date_rows).to_csv(OUTPUT_DIR / 'date_group_index_medians_intermediate.csv', index=False)

    if not date_rows:
        raise RuntimeError('No date-level group summaries could be computed.')

    date_df = pd.DataFrame(date_rows).sort_values(['group_code', 'index', 'date']).reset_index(drop=True)
    date_df.to_csv(OUTPUT_DIR / 'date_group_index_medians.csv', index=False)

    env_rows = []
    group_cols = ['group_code', 'group_label', 'index', 'doy_bin_label', 'doy_bin_center']
    for keys, sub in date_df.groupby(group_cols, dropna=False):
        group_code, group_label, index_name, bin_label, bin_center = keys
        medians = sub['date_median'].to_numpy(dtype=float)
        q10s = sub['date_q10'].to_numpy(dtype=float)
        q90s = sub['date_q90'].to_numpy(dtype=float)
        mins = sub['date_min'].to_numpy(dtype=float)
        maxs = sub['date_max'].to_numpy(dtype=float)
        env_rows.append({
            'group_code': group_code,
            'group_label': group_label,
            'index': index_name,
            'doy_bin_label': bin_label,
            'doy_bin_center': int(bin_center),
            'n_dates': int(len(sub)),
            'support_pixels_total': int(sub['n_valid_pixels'].sum()),
            'env_q_low': float(np.quantile(medians, ENV_Q_LOW)),
            'env_q_med': float(np.quantile(medians, 0.50)),
            'env_q_high': float(np.quantile(medians, ENV_Q_HIGH)),
            'curve_min': float(np.min(medians)),
            'curve_med': float(np.median(medians)),
            'curve_max': float(np.max(medians)),
            'median_of_date_q10': float(np.median(q10s)),
            'median_of_date_q90': float(np.median(q90s)),
            'min_of_date_min': float(np.min(mins)),
            'max_of_date_max': float(np.max(maxs)),
        })
    env_df = pd.DataFrame(env_rows).sort_values(['group_code', 'index', 'doy_bin_center']).reset_index(drop=True)
    env_df.to_csv(OUTPUT_DIR / 'doy_bin_envelopes.csv', index=False)

    coverage_df = (
        date_df.groupby(['group_code', 'index'])
        .agg(n_dates=('date', 'nunique'), min_doy=('doy', 'min'), max_doy=('doy', 'max'))
        .reset_index()
        .sort_values(['group_code', 'index'])
    )
    coverage_df.to_csv(OUTPUT_DIR / 'group_index_date_coverage.csv', index=False)

    support_summary = support_summary.copy()
    support_summary['zip_count_used'] = len(scan_records)
    support_summary.to_csv(OUTPUT_DIR / 'support_rule_summary.csv', index=False)

    if MAKE_PLOTS:
        log('Finished envelope aggregation; starting plots')
        for (group_code, group_label, index_name), sub in env_df.groupby(['group_code', 'group_label', 'index'], dropna=False):
            sub = sub.sort_values('doy_bin_center')
            if len(sub) < MIN_DATES_PER_BIN_FOR_PLOT:
                continue
            x = sub['doy_bin_center'].to_numpy()
            env_low = sub['env_q_low'].to_numpy(dtype=float)
            env_med = sub['env_q_med'].to_numpy(dtype=float)
            env_high = sub['env_q_high'].to_numpy(dtype=float)
            cmin = sub['curve_min'].to_numpy(dtype=float)
            cmax = sub['curve_max'].to_numpy(dtype=float)
            fig, ax = plt.subplots(figsize=(8.6, 4.9))
            ax.fill_between(x, env_low, env_high, alpha=0.20, label=f'Envelope q{int(ENV_Q_LOW*100)}–q{int(ENV_Q_HIGH*100)}')
            ax.plot(x, env_med, linewidth=2.0, label='Envelope median (q50)')
            ax.plot(x, cmin, linestyle='--', linewidth=1.3, label='Min of date medians')
            ax.plot(x, cmax, linestyle='--', linewidth=1.3, label='Max of date medians')
            pts = date_df[(date_df['group_code'] == group_code) & (date_df['index'] == index_name)].copy()
            if not pts.empty:
                ax.scatter(pts['doy'], pts['date_median'], s=20, alpha=0.75, label='Date medians')
            ax.set_xlim(1, 365)
            ax.set_xticks([1, 60, 120, 180, 240, 300, 365])
            ax.set_xlabel('DOY')
            ax.set_ylabel(index_name)
            ax.set_title(f'{group_code} — {index_name}')
            ax.grid(True, alpha=0.25)
            ax.legend(loc='best', fontsize=8)
            fig.tight_layout()
            safe_name = f'{group_code}_{index_name}_envelope.png'.replace('/', '_')
            fig.savefig(plots_dir / safe_name, dpi=180, bbox_inches='tight')
            plt.close(fig)
    else:
        log('Finished envelope aggregation; skipping plot generation (MAKE_PLOTS=False)')

    log(f"Done in {time.time() - t0:.1f}s. Outputs in {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
