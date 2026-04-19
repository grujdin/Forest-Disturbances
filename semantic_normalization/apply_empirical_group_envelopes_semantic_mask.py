"""
Apply group-specific empirical spectral envelopes to EO Browser ZIP scenes and build
semantic keep/drop masks.

Core idea
---------
A pixel is considered semantically compatible with healthy forest for its analytical
composition group and day-of-year (DOY) if enough spectral indices fall inside that
 group's empirical healthy envelope for the corresponding DOY bin.

Default operational rule
------------------------
- use group-specific envelopes derived from the user's local-run bundle
- use the three core indices: NDVI, NDMI, NBR
- keep a pixel if at least 2 of the 3 core indices are inside the envelope
- NDRE is computed and summarized, but not used in the default vote
- compare only pixels that are valid, clear, and fall inside a grouped FMU polygon

Outputs per scene
-----------------
- semantic_keep_mask.tif   : 1 keep, 0 drop, 255 nodata
- semantic_drop_mask.tif   : 1 drop, 0 keep, 255 nodata
- semantic_vote_count.tif  : number of core indices inside envelope (0..3), 255 nodata
- semantic_group_id.tif    : rasterized analytical group id, 0 nodata/outside grouped FMU
- group_scene_summary.csv  : per-group counts for the scene
- group_lookup_used.csv    : envelope rows used for the scene and any fallback info

Global outputs
--------------
- scene_summary.csv        : one summary row per processed scene
- README.txt              : concise description of the rule used

This script is hardcoded by design.
"""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.warp import Resampling, reproject

# =============================================================================
# HARD-CODED CONFIG
# =============================================================================

# Best available empirical envelopes: the user's local run with all available EO Browser ZIPs.
ENVELOPE_SOURCE = Path("D:/Forest_Disturbance/outputs/sdv_si_empirical_envelopes_per_compositional_group_blacklist_and_caution.zip")
ENVELOPE_CSV_IN_ZIP = "sdv_si_empirical_envelopes_per_compositional_group_blacklist_and_caution/doy_bin_envelopes.csv"

TARGET_ZIP_GLOB = "D:/Forest_Disturbance/imagery_zip/Stana_de_Vale_BH/SdV_*.zip"
TARGET_ZIPS = sorted(Path("D:/Forest_Disturbance/imagery_zip/Stana_de_Vale_BH").glob("SdV_*.zip"))

FMU_GEOJSON = Path("D:/Forest_Disturbance/vector_data/SdV_FMU.geojson")
GROUP_WORKBOOK = Path("D:/Forest_Disturbance/tables/sdv_compos_groups_loss_causes_reference.xlsx")
GROUP_SHEET = "LOSS_CAUSES"

FMU_JOIN_FIELD = "ua"
WORKBOOK_JOIN_FIELD = "ua"
GROUP_CODE_FIELD = "COMPOZ_TYPE_CODE"
GROUP_LABEL_FIELD = "COMPOZ_TYPE_LABEL"

EXCLUDE_GROUP_CODES = set()
USE_ALL_FOREST_FALLBACK_FOR_UNKNOWN = True
ALL_FOREST_FALLBACK_CODE = "ALL_FOREST"
ALL_FOREST_FALLBACK_LABEL = "All forest fallback"
RASTERIZE_ALL_TOUCHED = False

DOY_BIN_WIDTH = 30
CORE_INDICES = ["NDVI", "NDMI", "NBR"]
OPTIONAL_INDICES = ["NDRE"]
KEEP_VOTES_REQUIRED = 2

# Which envelope columns define the healthy range.
# Recommended operational choice: pixel-level healthy range estimated from the median of
# date-level q10/q90 values within the DOY bin. This is more robust than env_q_low/high
# when a bin is supported by only one or a few dates.
ENVELOPE_LOW_COL = "median_of_date_q10"
ENVELOPE_HIGH_COL = "median_of_date_q90"
USE_NEAREST_BIN_FALLBACK = True

USE_SCL_IF_AVAILABLE = True
SCL_EXCLUDE_CLASSES = {0, 1, 2, 3, 8, 9, 10, 11}
ALLOW_UNKNOWN_SCL_COLORS = False

WRITE_INDEX_INSIDE_MASKS = False
OUTPUT_DIR = Path("D:/Forest_Disturbance/outputs/semantic_masks_empirical_envelopes_blacklist_and_caution")

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


def parse_date_from_name(name: str) -> pd.Timestamp:
    m = DATE_RE.search(name)
    if not m:
        raise ValueError(f"Could not parse date from: {name}")
    y, mm, dd = map(int, m.groups())
    return pd.Timestamp(year=y, month=mm, day=dd)


def doy_bin_label(doy: int, width: int) -> str:
    start = ((doy - 1) // width) * width + 1
    end = min(start + width - 1, 365)
    return f"{start:03d}-{end:03d}"


def doy_bin_center_from_label(label: str) -> int:
    a, b = label.split("-")
    a = int(a)
    b = int(b)
    return int(round((a + b) / 2))


def _norm_text(x) -> str:
    s = "" if pd.isna(x) else str(x)
    s = s.strip().upper().replace(" ", "")
    if s in {"", "NONE", "NAN", "NULL"}:
        return ""
    return s


def _norm_ua_from_parts(parc, subp, ua) -> str:
    ua_n = _norm_text(ua)
    parc_n = _norm_text(parc)
    subp_n = _norm_text(subp)
    if ua_n and parc_n and ua_n.startswith(parc_n) and len(ua_n) > len(parc_n):
        return ua_n
    if ua_n and parc_n and subp_n and ua_n == subp_n:
        return f"{parc_n}{subp_n}"
    if not ua_n and parc_n and subp_n:
        return f"{parc_n}{subp_n}"
    if ua_n:
        return ua_n
    if parc_n and subp_n:
        return f"{parc_n}{subp_n}"
    if parc_n:
        return parc_n
    return ""


def zip_members_by_tag(zip_path: Path) -> Dict[str, str]:
    need = {
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
    out = {
        "NDVI": safe_norm_diff(bands["B08"], bands["B04"]),
        "NDMI": safe_norm_diff(bands["B08"], bands["B11"]),
        "NBR": safe_norm_diff(bands["B08"], bands["B12"]),
        "NDRE": safe_norm_diff(bands["B8A"], bands["B05"]),
    }
    return out


def load_grouped_subparcels_all() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(FMU_GEOJSON)
    attrs = pd.read_excel(GROUP_WORKBOOK, sheet_name=GROUP_SHEET)

    attrs = attrs.copy()
    attrs["ua_norm"] = [
        _norm_ua_from_parts(p, s, u)
        for p, s, u in zip(attrs.get("parc", ""), attrs.get("subp", ""), attrs.get(WORKBOOK_JOIN_FIELD, ""))
    ]
    attrs["parc_norm"] = attrs.get("parc", "").astype(str).map(_norm_text)

    gdf = gdf.copy()
    gdf["ua_norm"] = [
        _norm_ua_from_parts(p, s, u)
        for p, s, u in zip(gdf.get("Parcela", ""), gdf.get("Subparcela", ""), gdf.get(FMU_JOIN_FIELD, ""))
    ]
    gdf["parc_norm"] = gdf.get("Parcela", "").astype(str).map(_norm_text)

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

    out = out[~out[GROUP_CODE_FIELD].isin(EXCLUDE_GROUP_CODES)].copy()
    diag_cols = [c for c in [FMU_JOIN_FIELD, "Parcela", "Subparcela", "ua_norm", "parc_norm", GROUP_CODE_FIELD, GROUP_LABEL_FIELD] if c in out.columns]
    out[diag_cols].to_csv(OUTPUT_DIR / "fmu_group_join_diagnostics.csv", index=False)
    if out.crs is None:
        raise ValueError("FMU GeoJSON has no CRS.")
    return out


def rasterize_groups(gdf: gpd.GeoDataFrame, ref_profile: dict) -> Tuple[np.ndarray, pd.DataFrame]:
    if gdf.crs != ref_profile["crs"]:
        gdf = gdf.to_crs(ref_profile["crs"])
    groups = (
        gdf[[GROUP_CODE_FIELD, GROUP_LABEL_FIELD]]
        .drop_duplicates()
        .sort_values([GROUP_CODE_FIELD])
        .reset_index(drop=True)
        .copy()
    )
    groups["group_id"] = np.arange(1, len(groups) + 1, dtype=np.int32)
    lut = groups[["group_id", GROUP_CODE_FIELD, GROUP_LABEL_FIELD]].rename(
        columns={GROUP_CODE_FIELD: "group_code", GROUP_LABEL_FIELD: "group_label"}
    )
    gdf = gdf.merge(lut, left_on=[GROUP_CODE_FIELD, GROUP_LABEL_FIELD], right_on=["group_code", "group_label"], how="left")
    shapes = ((geom, int(gid)) for geom, gid in zip(gdf.geometry, gdf["group_id"]))
    out = rasterize(
        shapes=shapes,
        out_shape=(ref_profile["height"], ref_profile["width"]),
        transform=ref_profile["transform"],
        fill=0,
        dtype="int32",
        all_touched=RASTERIZE_ALL_TOUCHED,
    )
    return out, lut


def load_envelopes() -> pd.DataFrame:
    if ENVELOPE_SOURCE.suffix.lower() == ".zip":
        with zipfile.ZipFile(ENVELOPE_SOURCE, "r") as zf:
            with zf.open(ENVELOPE_CSV_IN_ZIP) as f:
                env = pd.read_csv(io.BytesIO(f.read()))
    else:
        env = pd.read_csv(ENVELOPE_SOURCE)
    env["doy_bin_center"] = env["doy_bin_label"].apply(doy_bin_center_from_label)
    if USE_ALL_FOREST_FALLBACK_FOR_UNKNOWN:
        num_cols = [c for c in env.columns if c not in {"group_code", "group_label", "index", "doy_bin_label"}]
        agg = {c: "median" for c in num_cols if pd.api.types.is_numeric_dtype(env[c])}
        af = env.groupby(["index", "doy_bin_label"], as_index=False).agg(agg)
        af["group_code"] = ALL_FOREST_FALLBACK_CODE
        af["group_label"] = ALL_FOREST_FALLBACK_LABEL
        if "doy_bin_center" not in af.columns:
            af["doy_bin_center"] = af["doy_bin_label"].apply(doy_bin_center_from_label)
        env = pd.concat([env, af[env.columns]], ignore_index=True)
    return env


def lookup_envelope_rows(env: pd.DataFrame, group_code: str, target_bin_label: str, indices: List[str]) -> Tuple[Dict[str, pd.Series], List[dict]]:
    out = {}
    logs = []
    group_env = env[env["group_code"] == group_code]
    target_center = doy_bin_center_from_label(target_bin_label)
    for idx in indices:
        subset = group_env[group_env["index"] == idx]
        exact = subset[subset["doy_bin_label"] == target_bin_label]
        if len(exact) == 1:
            row = exact.iloc[0]
            out[idx] = row
            logs.append({
                "group_code": group_code,
                "index": idx,
                "requested_bin": target_bin_label,
                "used_bin": row["doy_bin_label"],
                "fallback_used": False,
            })
            continue
        if USE_NEAREST_BIN_FALLBACK and len(subset) > 0:
            temp = subset.copy()
            temp["bin_distance"] = (temp["doy_bin_center"] - target_center).abs()
            row = temp.sort_values(["bin_distance", "n_dates", "support_pixels_total"], ascending=[True, False, False]).iloc[0]
            out[idx] = row
            logs.append({
                "group_code": group_code,
                "index": idx,
                "requested_bin": target_bin_label,
                "used_bin": row["doy_bin_label"],
                "fallback_used": True,
            })
        else:
            raise KeyError(f"No envelope found for group={group_code}, index={idx}, bin={target_bin_label}")
    return out, logs


def write_uint8_mask(path: Path, arr: np.ndarray, ref_profile: dict, nodata: int = 255) -> None:
    profile = ref_profile.copy()
    profile.update(dtype=rasterio.uint8, count=1, nodata=nodata, compress="deflate")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(np.uint8), 1)


def write_int16(path: Path, arr: np.ndarray, ref_profile: dict, nodata: int = -32768) -> None:
    profile = ref_profile.copy()
    profile.update(dtype=rasterio.int16, count=1, nodata=nodata, compress="deflate")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(np.int16), 1)


def process_scene(zip_path: Path, env: pd.DataFrame, polygons: gpd.GeoDataFrame) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    members = zip_members_by_tag(zip_path)
    scene_date = parse_date_from_name(zip_path.name)
    scene_doy = int(scene_date.dayofyear)
    bin_label = doy_bin_label(scene_doy, DOY_BIN_WIDTH)

    bands = {}
    ref_profile = None
    valid_mask = None
    for tag in ["B04", "B05", "B08", "B8A", "B11", "B12"]:
        arr, profile, valid = read_single_band(zip_path, members[tag])
        if ref_profile is None:
            ref_profile = profile
            valid_mask = valid.copy()
        else:
            if not same_grid(ref_profile, profile):
                raise ValueError(f"Band {tag} is not aligned with reference in {zip_path.name}")
            valid_mask &= valid
        bands[tag] = arr.astype(np.float32)

    scl_classes = load_scl_classes(zip_path, members["SCL"], ref_profile)
    clear_mask = build_clear_mask(valid_mask, scl_classes)

    group_id_raster, lut = rasterize_groups(polygons, ref_profile)
    group_lut = {int(r.group_id): {"group_code": r.group_code, "group_label": r.group_label} for r in lut.itertuples(index=False)}

    idx = compute_indices(bands)
    all_used_indices = CORE_INDICES + [x for x in OPTIONAL_INDICES if x not in CORE_INDICES]

    inside_by_index = {name: np.zeros(group_id_raster.shape, dtype=bool) for name in all_used_indices}
    vote_count = np.zeros(group_id_raster.shape, dtype=np.uint8)
    grouped_valid = clear_mask & (group_id_raster > 0)

    lookup_logs = []
    for gid, meta in group_lut.items():
        group_code = meta["group_code"]
        mask_g = grouped_valid & (group_id_raster == gid)
        if not np.any(mask_g):
            continue
        rows_by_index, logs = lookup_envelope_rows(env, group_code, bin_label, all_used_indices)
        lookup_logs.extend(logs)
        for name in all_used_indices:
            low = float(rows_by_index[name][ENVELOPE_LOW_COL])
            high = float(rows_by_index[name][ENVELOPE_HIGH_COL])
            arr = idx[name]
            ok = mask_g & np.isfinite(arr) & (arr >= low) & (arr <= high)
            inside_by_index[name][ok] = True

    for name in CORE_INDICES:
        vote_count = vote_count + inside_by_index[name].astype(np.uint8)

    finite_core = np.ones(group_id_raster.shape, dtype=bool)
    for name in CORE_INDICES:
        finite_core &= np.isfinite(idx[name])
    semantic_domain = grouped_valid & finite_core

    keep = semantic_domain & (vote_count >= KEEP_VOTES_REQUIRED)
    drop = semantic_domain & (~keep)

    out_scene_dir = OUTPUT_DIR / scene_date.strftime("%Y-%m-%d")
    out_scene_dir.mkdir(parents=True, exist_ok=True)

    keep_u8 = np.full(group_id_raster.shape, 255, dtype=np.uint8)
    keep_u8[semantic_domain] = 0
    keep_u8[keep] = 1
    write_uint8_mask(out_scene_dir / "semantic_keep_mask.tif", keep_u8, ref_profile)

    drop_u8 = np.full(group_id_raster.shape, 255, dtype=np.uint8)
    drop_u8[semantic_domain] = 0
    drop_u8[drop] = 1
    write_uint8_mask(out_scene_dir / "semantic_drop_mask.tif", drop_u8, ref_profile)

    votes_u8 = np.full(group_id_raster.shape, 255, dtype=np.uint8)
    votes_u8[semantic_domain] = vote_count[semantic_domain]
    write_uint8_mask(out_scene_dir / "semantic_vote_count.tif", votes_u8, ref_profile)

    group_out = group_id_raster.astype(np.int16)
    write_int16(out_scene_dir / "semantic_group_id.tif", group_out, ref_profile, nodata=-32768)

    if WRITE_INDEX_INSIDE_MASKS:
        for name in all_used_indices:
            arr = np.full(group_id_raster.shape, 255, dtype=np.uint8)
            arr[semantic_domain] = 0
            arr[inside_by_index[name] & semantic_domain] = 1
            write_uint8_mask(out_scene_dir / f"inside_{name.lower()}_envelope.tif", arr, ref_profile)

    # Scene/group summary
    group_rows = []
    for gid, meta in group_lut.items():
        mask_g = semantic_domain & (group_id_raster == gid)
        if not np.any(mask_g):
            continue
        row = {
            "scene": zip_path.name,
            "date": scene_date.date().isoformat(),
            "doy": scene_doy,
            "doy_bin_label": bin_label,
            "group_id": gid,
            "group_code": meta["group_code"],
            "group_label": meta["group_label"],
            "semantic_domain_px": int(mask_g.sum()),
            "keep_px": int((keep & (group_id_raster == gid)).sum()),
            "drop_px": int((drop & (group_id_raster == gid)).sum()),
            "keep_pct": float(100.0 * (keep & (group_id_raster == gid)).sum() / mask_g.sum()),
            "mean_vote_count": float(vote_count[mask_g].mean()),
        }
        for name in all_used_indices:
            row[f"inside_{name.lower()}_pct"] = float(100.0 * inside_by_index[name][mask_g].mean())
        group_rows.append(row)
    group_df = pd.DataFrame(group_rows)
    group_df.to_csv(out_scene_dir / "group_scene_summary.csv", index=False)

    lookup_df = pd.DataFrame(lookup_logs)
    # Attach numeric low/high for traceability.
    if not lookup_df.empty:
        enriched = []
        for rec in lookup_logs:
            row = env[(env["group_code"] == rec["group_code"]) & (env["index"] == rec["index"]) & (env["doy_bin_label"] == rec["used_bin"])].iloc[0]
            enriched.append({
                **rec,
                "env_q_low": row[ENVELOPE_LOW_COL],
                "env_q_high": row[ENVELOPE_HIGH_COL],
                "n_dates": row["n_dates"],
                "support_pixels_total": row["support_pixels_total"],
            })
        lookup_df = pd.DataFrame(enriched)
        lookup_df.to_csv(out_scene_dir / "group_lookup_used.csv", index=False)

    scene_summary = {
        "scene": zip_path.name,
        "date": scene_date.date().isoformat(),
        "doy": scene_doy,
        "doy_bin_label": bin_label,
        "n_grouped_valid_px": int(semantic_domain.sum()),
        "keep_px": int(keep.sum()),
        "drop_px": int(drop.sum()),
        "keep_pct": float(100.0 * keep.sum() / semantic_domain.sum()) if semantic_domain.sum() else np.nan,
        "drop_pct": float(100.0 * drop.sum() / semantic_domain.sum()) if semantic_domain.sum() else np.nan,
        "n_group_ids_used": int(group_df["group_id"].nunique()) if not group_df.empty else 0,
        "n_fallback_lookups": int(lookup_df["fallback_used"].sum()) if not lookup_df.empty else 0,
        "n_total_lookups": int(len(lookup_df)),
        "core_indices": ",".join(CORE_INDICES),
        "keep_votes_required": KEEP_VOTES_REQUIRED,
    }
    return scene_summary, group_df, lookup_df


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not TARGET_ZIPS:
        raise FileNotFoundError(f"No target ZIPs matched: {TARGET_ZIP_GLOB}")

    env = load_envelopes()
    polygons = load_grouped_subparcels_all()

    scene_rows = []
    group_rows_all = []
    lookup_rows_all = []

    for zp in TARGET_ZIPS:
        print(f"Processing {zp.name} ...")
        scene_summary, group_df, lookup_df = process_scene(zp, env, polygons)
        scene_rows.append(scene_summary)
        if not group_df.empty:
            group_rows_all.append(group_df)
        if not lookup_df.empty:
            tmp = lookup_df.copy()
            tmp.insert(0, "scene", zp.name)
            lookup_rows_all.append(tmp)

    pd.DataFrame(scene_rows).sort_values(["date", "scene"]).to_csv(OUTPUT_DIR / "scene_summary.csv", index=False)
    if group_rows_all:
        pd.concat(group_rows_all, ignore_index=True).to_csv(OUTPUT_DIR / "group_scene_summary_all.csv", index=False)
    if lookup_rows_all:
        pd.concat(lookup_rows_all, ignore_index=True).to_csv(OUTPUT_DIR / "lookup_trace_all.csv", index=False)

    readme = OUTPUT_DIR / "README.txt"
    readme.write_text(
        "Recommended semantic keep/drop masking using empirical group-specific envelopes.\n\n"
        f"Envelope source: {ENVELOPE_SOURCE.name}\n"
        f"Target ZIP count: {len(TARGET_ZIPS)}\n"
        f"Core indices: {', '.join(CORE_INDICES)}\n"
        f"Optional indices (reported only): {', '.join(OPTIONAL_INDICES) if OPTIONAL_INDICES else 'None'}\n"
        f"Keep rule: keep if at least {KEEP_VOTES_REQUIRED} of {len(CORE_INDICES)} core indices fall inside the healthy envelope\n"
        f"Envelope columns used: {ENVELOPE_LOW_COL}, {ENVELOPE_HIGH_COL}\n"
        f"DOY bin width: {DOY_BIN_WIDTH}\n"
        f"Fallback: {'nearest DOY bin' if USE_NEAREST_BIN_FALLBACK else 'none'}\n"
        "Valid domain: clear pixels inside grouped FMU polygons with finite core indices\n",
        encoding="utf-8",
    )
    print(f"Done. Outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()