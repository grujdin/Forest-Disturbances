"""
Algorithm #0 / Phase 3. Shared Sentinel-2 preprocessing cache for the Stana de Vale workflow.

Purpose
-------
This script performs once the expensive preprocessing that Algorithms #1 and #2
previously repeated:
- scan EO Browser ZIP scenes;
- read raw Sentinel-2 bands;
- decode SCL and build clear masks;
- compute NDVI, NDMI, NBR, NDRE, and NDSI;
- assign FMU polygons to analytical composition groups;
- rasterize stable-support and all-FMU group rasters;
- write scene-quality summaries and keep/caution/blacklist recommendations.

Algorithms #1 and #2 Phase 3 read these cached products instead of reopening all
raw bands and recomputing indices/SCL masks.
"""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd

from sdv_shared import (
    CACHE_INDEX_BANDS,
    cache_scene_paths,
    parse_date_from_name,
    doy_bin_label,
    doy_bin_center,
    zip_members_by_tag as _shared_zip_members_by_tag,
    read_single_band,
    same_grid,
    load_scl_classes,
    build_clear_mask,
    compute_indices,
    compute_ndsi,
    rasterize_groups,
    resolve_group_sheet,
    norm_text,
    norm_ua_from_parts,
    write_uint8_mask,
    write_int16,
    write_float32_stack,
    write_json,
)

# =============================================================================
# HARD-CODED CONFIG
# =============================================================================

ZIP_GLOB = "D:/Forest_Disturbance/imagery_zip/Stana_de_Vale_S2/SdV_*.zip"
ZIP_FILES = sorted(Path("D:/Forest_Disturbance/imagery_zip/Stana_de_Vale_S2").glob("SdV_*.zip"))

FMU_GEOJSON = Path("D:/Forest_Disturbance/vector_data/SdV_FMU.geojson")
GROUP_WORKBOOK = Path("D:/Forest_Disturbance/tables/sdv_compos_groups_loss_causes_reference.xlsx")
GROUP_SHEET = "LOSS_CAUSES"

FMU_JOIN_FIELD = "ua"
WORKBOOK_JOIN_FIELD = "ua"
GROUP_CODE_FIELD = "COMPOZ_TYPE_CODE"
GROUP_LABEL_FIELD = "COMPOZ_TYPE_LABEL"
TOTAL_LOSS_FIELD = "Total loss"
AREA_HA_FIELD = "ha"
YEAR_LOSS_FIELDS = ["2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025", "2026"]

EXCLUDE_GROUP_CODES_STABLE = {"NO_DATA"}
EXCLUDE_GROUP_CODES_ALL = set()
USE_ALL_FOREST_FALLBACK_FOR_UNKNOWN = True
ALL_FOREST_FALLBACK_CODE = "ALL_FOREST"
ALL_FOREST_FALLBACK_LABEL = "All forest fallback"

USE_ONLY_STABLE_SUBPARCELS = True
STABLE_TOTAL_LOSS_FRAC_MAX = 0.005
STABLE_MAX_YEAR_LOSS_FRAC_MAX = 0.003

DOY_BIN_WIDTH = 30
INDEX_STACK_BANDS = ["NDVI", "NDMI", "NBR", "NDRE", "NDSI"]
S2_REQUIRED_BANDS = ("B03", "B04", "B05", "B08", "B8A", "B11", "B12")

USE_SCL_IF_AVAILABLE = True
SCL_EXCLUDE_CLASSES = {0, 1, 2, 3, 8, 9, 10, 11}
ALLOW_UNKNOWN_SCL_COLORS = False

# Scene-quality thresholds, kept identical to Algorithm #1.
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

CACHE_ROOT = Path("D:/Forest_Disturbance/outputs/sdv_phase3_preprocessing_cache")
FORCE_REBUILD_SCENE_CACHE = False
VERBOSE = True

# =============================================================================
# Helpers
# =============================================================================


def log(msg: str) -> None:
    if VERBOSE:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def zip_members_by_tag(zip_path: Path) -> Dict[str, str | None]:
    return _shared_zip_members_by_tag(
        zip_path,
        required_tags=S2_REQUIRED_BANDS,
        include_scl=True,
        include_date_string=True,
    )


def load_stable_subparcels() -> Tuple[gpd.GeoDataFrame, str, pd.DataFrame]:
    gdf = gpd.read_file(FMU_GEOJSON)
    required = {WORKBOOK_JOIN_FIELD, GROUP_CODE_FIELD, GROUP_LABEL_FIELD, TOTAL_LOSS_FIELD, AREA_HA_FIELD}
    sheet = resolve_group_sheet(GROUP_WORKBOOK, GROUP_SHEET, required)
    attrs_raw = pd.read_excel(GROUP_WORKBOOK, sheet_name=sheet)
    keep_cols = [WORKBOOK_JOIN_FIELD, GROUP_CODE_FIELD, GROUP_LABEL_FIELD, TOTAL_LOSS_FIELD, AREA_HA_FIELD] + YEAR_LOSS_FIELDS
    missing = [c for c in keep_cols if c not in attrs_raw.columns]
    if missing:
        raise ValueError(f"Workbook sheet {sheet!r} is missing required columns: {missing}")

    attrs = attrs_raw[keep_cols].copy()
    attrs[WORKBOOK_JOIN_FIELD] = attrs[WORKBOOK_JOIN_FIELD].astype(str)
    attrs = attrs.drop_duplicates(subset=[WORKBOOK_JOIN_FIELD])
    gdf = gdf.copy()
    gdf[FMU_JOIN_FIELD] = gdf[FMU_JOIN_FIELD].astype(str)
    merged = gdf.merge(attrs, left_on=FMU_JOIN_FIELD, right_on=WORKBOOK_JOIN_FIELD, how="left")

    area_m2 = pd.to_numeric(merged[AREA_HA_FIELD], errors="coerce").fillna(0) * 10000.0
    total_loss = pd.to_numeric(merged[TOTAL_LOSS_FIELD], errors="coerce").fillna(0)
    year_loss = merged[YEAR_LOSS_FIELDS].apply(pd.to_numeric, errors="coerce").fillna(0)
    max_year_loss = year_loss.max(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        merged["stable_total_loss_frac"] = np.where(area_m2 > 0, total_loss / area_m2, np.nan)
        merged["stable_max_year_loss_frac"] = np.where(area_m2 > 0, max_year_loss / area_m2, np.nan)
        merged["stable_total_loss_pct"] = merged["stable_total_loss_frac"] * 100.0
        merged["stable_max_year_loss_pct"] = merged["stable_max_year_loss_frac"] * 100.0

    if USE_ONLY_STABLE_SUBPARCELS:
        stable = (
            (area_m2 > 0)
            & (merged["stable_total_loss_frac"].fillna(np.inf) <= STABLE_TOTAL_LOSS_FRAC_MAX)
            & (merged["stable_max_year_loss_frac"].fillna(np.inf) <= STABLE_MAX_YEAR_LOSS_FRAC_MAX)
        )
    else:
        stable = pd.Series(True, index=merged.index)

    good_group = merged[GROUP_CODE_FIELD].notna() & (~merged[GROUP_CODE_FIELD].isin(EXCLUDE_GROUP_CODES_STABLE))
    merged = merged.loc[stable & good_group].copy()
    merged = merged.loc[~merged.geometry.is_empty & merged.geometry.notna()].copy()
    if merged.crs is None:
        raise ValueError("FMU GeoJSON has no CRS.")
    return merged, sheet, attrs_raw


def load_all_grouped_subparcels() -> gpd.GeoDataFrame:
    """Load all FMU polygons with Algorithm #2-compatible group assignment."""
    gdf = gpd.read_file(FMU_GEOJSON)
    attrs = pd.read_excel(GROUP_WORKBOOK, sheet_name=GROUP_SHEET)

    attrs = attrs.copy()
    attrs["ua_norm"] = [
        norm_ua_from_parts(p, s, u)
        for p, s, u in zip(attrs.get("parc", ""), attrs.get("subp", ""), attrs.get(WORKBOOK_JOIN_FIELD, ""))
    ]
    attrs["parc_norm"] = attrs.get("parc", "").astype(str).map(norm_text)

    gdf = gdf.copy()
    gdf["ua_norm"] = [
        norm_ua_from_parts(p, s, u)
        for p, s, u in zip(gdf.get("Parcela", ""), gdf.get("Subparcela", ""), gdf.get(FMU_JOIN_FIELD, ""))
    ]
    gdf["parc_norm"] = gdf.get("Parcela", "").astype(str).map(norm_text)

    attrs_ua = attrs.drop_duplicates("ua_norm")
    keep_cols = ["ua_norm", GROUP_CODE_FIELD, GROUP_LABEL_FIELD]
    out = gdf.merge(attrs_ua[keep_cols], on="ua_norm", how="left")

    unique_parc = attrs.groupby("parc_norm").filter(lambda d: len(d.dropna(subset=[GROUP_CODE_FIELD])) == 1)
    unique_parc = unique_parc.drop_duplicates("parc_norm")
    miss = out[GROUP_CODE_FIELD].isna() & out["parc_norm"].ne("")
    fill = out.loc[miss, ["parc_norm"]].merge(
        unique_parc[["parc_norm", GROUP_CODE_FIELD, GROUP_LABEL_FIELD]], on="parc_norm", how="left"
    )
    out.loc[miss, GROUP_CODE_FIELD] = fill[GROUP_CODE_FIELD].values
    out.loc[miss, GROUP_LABEL_FIELD] = fill[GROUP_LABEL_FIELD].values

    out[GROUP_CODE_FIELD] = out[GROUP_CODE_FIELD].fillna("NO_DATA")
    out[GROUP_LABEL_FIELD] = out[GROUP_LABEL_FIELD].fillna("No data")

    if USE_ALL_FOREST_FALLBACK_FOR_UNKNOWN:
        unk = out[GROUP_CODE_FIELD].eq("NO_DATA")
        out.loc[unk, GROUP_CODE_FIELD] = ALL_FOREST_FALLBACK_CODE
        out.loc[unk, GROUP_LABEL_FIELD] = ALL_FOREST_FALLBACK_LABEL

    out = out[~out[GROUP_CODE_FIELD].isin(EXCLUDE_GROUP_CODES_ALL)].copy()
    if out.crs is None:
        raise ValueError("FMU GeoJSON has no CRS.")
    return out


def summarize_support(group_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    return (
        group_gdf.groupby([GROUP_CODE_FIELD, GROUP_LABEL_FIELD], dropna=False)
        .agg(
            n_subparcels=(FMU_JOIN_FIELD, "count"),
            total_area_ha=(AREA_HA_FIELD, "sum"),
            median_total_loss_pct=("stable_total_loss_pct", "median"),
            max_total_loss_pct=("stable_total_loss_pct", "max"),
            max_single_year_loss_pct=("stable_max_year_loss_pct", "max"),
        )
        .reset_index()
        .rename(columns={GROUP_CODE_FIELD: "group_code", GROUP_LABEL_FIELD: "group_label"})
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


def med_on(mask: np.ndarray, arr: np.ndarray) -> float:
    vals = arr[mask]
    vals = vals[np.isfinite(vals)]
    return float(np.median(vals)) if vals.size else np.nan


def write_scene_quality_outputs(rows: List[dict]) -> pd.DataFrame:
    qdf = pd.DataFrame(rows).sort_values(["date", "scene"]).reset_index(drop=True)
    qdf.to_csv(CACHE_ROOT / "scene_quality_summary.csv", index=False)
    recs = []
    for _, row in qdf.iterrows():
        rec, why = scene_quality_recommendation(row)
        recs.append({**row.to_dict(), "recommendation": rec, "recommendation_reason": why})
    rdf = pd.DataFrame(recs)
    rdf.to_csv(CACHE_ROOT / "scene_blacklist_recommendation.csv", index=False)
    return rdf


def read_scene_bands(zip_path: Path, members: Dict[str, str | None], ref_profile: dict) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    bands = {}
    data_masks = []
    profile_ref = None
    for tag in S2_REQUIRED_BANDS:
        arr, profile, valid = read_single_band(zip_path, members[tag])
        if profile_ref is None:
            profile_ref = profile
        if not same_grid(profile, ref_profile):
            raise RuntimeError(f"{zip_path.name}: band {tag} is not on the reference grid.")
        bands[tag] = arr.astype(np.float32)
        data_masks.append(valid)
    return bands, np.logical_and.reduce(data_masks)


def main() -> None:
    t0 = time.time()
    if not ZIP_FILES:
        raise RuntimeError(f"No ZIP files found with pattern: {ZIP_GLOB}")

    for sub in ["index_stacks", "clear_masks", "scl_classes", "group_rasters"]:
        (CACHE_ROOT / sub).mkdir(parents=True, exist_ok=True)
    log(f"Starting Phase 3 shared preprocessing cache in {CACHE_ROOT}")
    log(f"Found {len(ZIP_FILES)} ZIP files")

    first_members = zip_members_by_tag(ZIP_FILES[0])
    _, ref_profile, _ = read_single_band(ZIP_FILES[0], first_members["B04"])

    stable_gdf, sheet_used, workbook_attrs_raw = load_stable_subparcels()
    stable_gdf.to_file(CACHE_ROOT / "stable_subparcels_used.geojson", driver="GeoJSON")
    stable_support_df = summarize_support(stable_gdf)
    stable_support_df.to_csv(CACHE_ROOT / "stable_support_by_group.csv", index=False)
    log(f"Stable support: {len(stable_gdf)} subparcels across {stable_support_df.shape[0]} groups")

    all_gdf = load_all_grouped_subparcels()
    all_gdf.to_file(CACHE_ROOT / "fmu_group_assignment_all.geojson", driver="GeoJSON")
    diag_cols = [c for c in [FMU_JOIN_FIELD, "Parcela", "Subparcela", "ua_norm", "parc_norm", GROUP_CODE_FIELD, GROUP_LABEL_FIELD] if c in all_gdf.columns]
    all_gdf[diag_cols].to_csv(CACHE_ROOT / "fmu_group_join_diagnostics.csv", index=False)

    stable_group_raster, stable_lut = rasterize_groups(
        stable_gdf, ref_profile, GROUP_CODE_FIELD, GROUP_LABEL_FIELD, all_touched=False, rename_lut_columns=True
    )
    all_group_raster, all_lut = rasterize_groups(
        all_gdf, ref_profile, GROUP_CODE_FIELD, GROUP_LABEL_FIELD, all_touched=False, rename_lut_columns=True
    )
    # Preferred Phase 3 cache names.
    stable_lut.to_csv(CACHE_ROOT / "group_lut_stable_support.csv", index=False)
    all_lut.to_csv(CACHE_ROOT / "group_lut_all_fmu.csv", index=False)
    write_int16(CACHE_ROOT / "stable_support_group_id.tif", stable_group_raster, ref_profile, nodata=0)
    write_int16(CACHE_ROOT / "all_fmu_group_id.tif", all_group_raster, ref_profile, nodata=0)

    # Backward-compatible aliases under group_rasters/ and shorter LUT names.
    stable_lut.to_csv(CACHE_ROOT / "group_lut_stable.csv", index=False)
    all_lut.to_csv(CACHE_ROOT / "group_lut_all.csv", index=False)
    write_int16(CACHE_ROOT / "group_rasters" / "stable_support_group_id.tif", stable_group_raster, ref_profile, nodata=0)
    write_int16(CACHE_ROOT / "group_rasters" / "all_fmu_group_id.tif", all_group_raster, ref_profile, nodata=0)
    support_domain = stable_group_raster > 0

    inventory_rows = []
    quality_rows = []
    for i_scene, zip_path in enumerate(ZIP_FILES, start=1):
        t_scene = time.time()
        log(f"[{i_scene}/{len(ZIP_FILES)}] Preprocessing {zip_path.name}")
        members = zip_members_by_tag(zip_path)
        date = parse_date_from_name(zip_path.name)
        paths = cache_scene_paths(CACHE_ROOT, zip_path.name)
        cache_complete = paths["index_stack"].exists() and paths["clear_mask"].exists() and paths["scl_class"].exists()

        try:
            if cache_complete and not FORCE_REBUILD_SCENE_CACHE:
                # Reuse existing cache files but still compute quality from cached rasters below is intentionally avoided.
                # Existing quality rows are regenerated only when the scene is rebuilt. For routine reruns, delete the
                # quality CSVs or set FORCE_REBUILD_SCENE_CACHE=True if quality thresholds/configuration changed.
                cache_status = "existing"
                from sdv_shared import read_float32_stack, read_uint8_raster
                idx, _ = read_float32_stack(paths["index_stack"], INDEX_STACK_BANDS)
                clear_arr, _ = read_uint8_raster(paths["clear_mask"])
                scl_classes, _ = read_uint8_raster(paths["scl_class"])
                clear_mask = clear_arr == 1
                scl_used = bool(members.get("SCL") is not None and USE_SCL_IF_AVAILABLE)
            else:
                bands, data_mask = read_scene_bands(zip_path, members, ref_profile)
                if USE_SCL_IF_AVAILABLE and members.get("SCL") is not None:
                    scl_classes = load_scl_classes(zip_path, members["SCL"], ref_profile)
                    clear_mask = build_clear_mask(data_mask, scl_classes, SCL_EXCLUDE_CLASSES, ALLOW_UNKNOWN_SCL_COLORS)
                    scl_used = True
                else:
                    scl_classes = np.full(data_mask.shape, 255, dtype=np.uint8)
                    clear_mask = data_mask.copy()
                    scl_used = False

                idx = compute_indices(bands, indices=["NDVI", "NDMI", "NBR", "NDRE"])
                idx["NDSI"] = compute_ndsi(bands)
                write_float32_stack(paths["index_stack"], idx, ref_profile, INDEX_STACK_BANDS)
                write_uint8_mask(paths["clear_mask"], clear_mask.astype(np.uint8), ref_profile, nodata=255)
                write_uint8_mask(paths["scl_class"], scl_classes.astype(np.uint8), ref_profile, nodata=255)
                cache_status = "rebuilt" if cache_complete else "created"

            m = support_domain
            n_support_px = int(m.sum())
            if n_support_px == 0:
                raise RuntimeError("No support-domain pixels available after rasterization.")
            support_clear = m & clear_mask
            quality_rows.append({
                "scene": zip_path.name,
                "date": date.date().isoformat(),
                "doy": int(date.dayofyear),
                "support_px": n_support_px,
                "clear_support_px": int(support_clear.sum()),
                "clear_frac": float(support_clear.sum() / n_support_px),
                "veg_frac": float(np.mean(scl_classes[m] == 4)),
                "snow_frac": float(np.mean(scl_classes[m] == 11)),
                "cloud_frac": float(np.mean(np.isin(scl_classes[m], [8, 9, 10]))),
                "shadow_frac": float(np.mean(np.isin(scl_classes[m], [2, 3]))),
                "scl7_frac": float(np.mean(scl_classes[m] == 7)),
                "ndvi_med_clear": med_on(support_clear, idx["NDVI"]),
                "ndre_med_clear": med_on(support_clear, idx["NDRE"]),
                "ndmi_med_clear": med_on(support_clear, idx["NDMI"]),
                "nbr_med_clear": med_on(support_clear, idx["NBR"]),
                "ndsi_med_clear": med_on(support_clear, idx["NDSI"]),
                "scl_used": bool(scl_used),
            })
            inventory_rows.append({
                "scene": zip_path.name,
                "zip_path": str(zip_path),
                "date": date.date().isoformat(),
                "year": int(date.year),
                "month": int(date.month),
                "day": int(date.day),
                "doy": int(date.dayofyear),
                "doy_bin_label": doy_bin_label(int(date.dayofyear), DOY_BIN_WIDTH),
                "doy_bin_center": doy_bin_center(int(date.dayofyear), DOY_BIN_WIDTH),
                "scl_available": bool(members.get("SCL") is not None),
                "scl_used": bool(scl_used),
                "index_stack_path": str(paths["index_stack"]),
                "clear_mask_path": str(paths["clear_mask"]),
                "scl_class_path": str(paths["scl_class"]),
                "cache_status": cache_status,
                "preprocess_error": "",
            })
            log(f"[{i_scene}/{len(ZIP_FILES)}] Finished {zip_path.name} in {time.time() - t_scene:.1f}s ({cache_status})")
        except Exception as exc:
            inventory_rows.append({
                "scene": zip_path.name,
                "zip_path": str(zip_path),
                "date": date.date().isoformat() if 'date' in locals() else "",
                "year": int(date.year) if 'date' in locals() else np.nan,
                "month": int(date.month) if 'date' in locals() else np.nan,
                "day": int(date.day) if 'date' in locals() else np.nan,
                "doy": int(date.dayofyear) if 'date' in locals() else np.nan,
                "doy_bin_label": doy_bin_label(int(date.dayofyear), DOY_BIN_WIDTH) if 'date' in locals() else "",
                "doy_bin_center": doy_bin_center(int(date.dayofyear), DOY_BIN_WIDTH) if 'date' in locals() else np.nan,
                "scl_available": False,
                "scl_used": False,
                "index_stack_path": str(paths["index_stack"]) if 'paths' in locals() else "",
                "clear_mask_path": str(paths["clear_mask"]) if 'paths' in locals() else "",
                "scl_class_path": str(paths["scl_class"]) if 'paths' in locals() else "",
                "cache_status": "error",
                "preprocess_error": str(exc),
            })
            log(f"ERROR for {zip_path.name}: {exc}")
            raise

    inventory_df = pd.DataFrame(inventory_rows).sort_values(["date", "scene"], na_position="last").reset_index(drop=True)
    inventory_df.to_csv(CACHE_ROOT / "scene_inventory.csv", index=False)
    rec_df = write_scene_quality_outputs(quality_rows)

    support_summary = pd.DataFrame([{
        "zip_count_input": len(ZIP_FILES),
        "support_rule": "relative loss thresholds",
        "group_sheet_used": sheet_used,
        "stable_total_loss_frac_max": STABLE_TOTAL_LOSS_FRAC_MAX,
        "stable_total_loss_pct_max": STABLE_TOTAL_LOSS_FRAC_MAX * 100.0,
        "stable_max_year_loss_frac_max": STABLE_MAX_YEAR_LOSS_FRAC_MAX,
        "stable_max_year_loss_pct_max": STABLE_MAX_YEAR_LOSS_FRAC_MAX * 100.0,
        "n_support_subparcels": len(stable_gdf),
        "support_area_ha": float(pd.to_numeric(stable_gdf[AREA_HA_FIELD], errors="coerce").fillna(0).sum()),
        "cache_root": str(CACHE_ROOT),
        "index_stack_bands": ",".join(INDEX_STACK_BANDS),
    }])
    support_summary.to_csv(CACHE_ROOT / "support_rule_summary.csv", index=False)

    write_json(CACHE_ROOT / "preprocessing_manifest.json", {
        "phase": 3,
        "purpose": "shared preprocessing cache for Algorithms #1 and #2",
        "zip_count_input": len(ZIP_FILES),
        "zip_count": len(ZIP_FILES),
        "cache_root": str(CACHE_ROOT),
        "doy_bin_width": DOY_BIN_WIDTH,
        "index_stack_bands": INDEX_STACK_BANDS,
        "scl_exclude_classes": sorted(SCL_EXCLUDE_CLASSES),
        "allow_unknown_scl_colors": ALLOW_UNKNOWN_SCL_COLORS,
        "stable_total_loss_frac_max": STABLE_TOTAL_LOSS_FRAC_MAX,
        "stable_max_year_loss_frac_max": STABLE_MAX_YEAR_LOSS_FRAC_MAX,
        "n_stable_subparcels": len(stable_gdf),
        "n_all_group_polygons": len(all_gdf),
        "n_scene_quality_keep": int((rec_df["recommendation"] == "keep").sum()),
        "n_scene_quality_caution": int((rec_df["recommendation"] == "caution").sum()),
        "n_scene_quality_blacklist": int((rec_df["recommendation"] == "blacklist").sum()),
    })

    n_black = int((rec_df["recommendation"] == "blacklist").sum())
    n_caut = int((rec_df["recommendation"] == "caution").sum())
    log(f"Scene-quality scan done: {n_black} blacklist, {n_caut} caution, {len(rec_df) - n_black - n_caut} keep")
    log(f"Done in {time.time() - t0:.1f}s. Cache outputs in {CACHE_ROOT}")


if __name__ == "__main__":
    main()
