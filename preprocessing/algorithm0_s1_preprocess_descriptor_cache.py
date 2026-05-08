"""
Algorithm #0-S1 / Phase 5 v4. Sentinel-1 descriptor cache for simple raw VV/VH ZIPs.

Purpose
-------
This script builds the Sentinel-1 descriptor cache used by Algorithms #1-#4 and
adds explicit diagnostics for EO Browser Sentinel-1 exports. It keeps the fixed
11-band descriptor stack required by the Phase 5 downstream scripts, but also
writes viewer-friendly products so that the user does not have to inspect a
multi-band stack containing several unavailable/nodata bands.

Main cached products
--------------------
- s1_scene_inventory.csv
- s1_descriptor_band_lookup.csv
- sentinel1_preprocessing_manifest.json
- s1_source_product_diagnostics.csv
- s1_cache_band_statistics.csv
- s1_flat_descriptor_warnings.csv
- s1_descriptor_stacks/<scene>__s1_descriptor_stack.tif
- s1_available_descriptor_stacks/<scene>__s1_available_descriptor_stack.tif
- s1_single_band_descriptors/<descriptor>/<scene>__<descriptor>.tif
- s1_display_quicklooks/<scene>__<descriptor>_quicklook.png
- s1_valid_masks/<scene>__s1_valid_mask.tif

Important interpretation note
-----------------------------
Preferred input is a simple Sentinel-1 IW-DV ZIP containing analytical raw
VV and VH layers. In the current EO Browser/Sentinel Hub format these raw layers
are two-band Float32 rasters: band 1 is backscatter and band 2 is dataMask. The
script uses band 1 as linear backscatter and derives dB internally.
"""
from __future__ import annotations

import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import rasterio

from sdv_shared import (
    S1_DESCRIPTOR_BANDS,
    build_s1_descriptor_stack,
    parse_date_from_name,
    read_int_raster,
    s1_cache_scene_paths,
    s1_members_by_tag,
    vsizip_path,
    write_float32_stack,
    write_json,
    write_uint8_mask,
)

# =============================================================================
# HARD-CODED CONFIG
# =============================================================================

S1_ZIP_ROOT = Path("D:/Forest_Disturbance/imagery_zip/Stana_de_Vale_S1")
S1_ZIP_GLOB = "*SdV*_S1*.zip"
S1_ZIP_FILES = sorted(S1_ZIP_ROOT.glob(S1_ZIP_GLOB))

# Existing S2/cache grid produced by Phase 3/4.
S2_CACHE_ROOT = Path("D:/Forest_Disturbance/outputs/sdv_phase3_preprocessing_cache")
REFERENCE_RASTER = S2_CACHE_ROOT / "all_fmu_group_id.tif"

S1_CACHE_ROOT = Path("D:/Forest_Disturbance/outputs/sdv_phase5_sentinel1_descriptor_cache")

# Force rebuild in v3 because new diagnostic/display products are added and old
# v2 stacks may already exist.
FORCE_REBUILD_S1_CACHE = True

# Raw Float32 VV/VH analytical products are preferred and kept as-is. Rendered
# integer products are normalized only for backward compatibility.
NORMALIZE_RENDERED_UINT = True
TREAT_ZERO_AS_NODATA = False

# Keep the fixed descriptor stack for downstream compatibility, but also write
# stacks/rasters containing only descriptors that are actually available.
WRITE_FIXED_DESCRIPTOR_STACK = True
WRITE_AVAILABLE_DESCRIPTOR_STACK = True
WRITE_SINGLE_BAND_DESCRIPTOR_RASTERS = True
WRITE_QUICKLOOK_PNGS = True
QUICKLOOK_PERCENTILES = (2.0, 98.0)
QUICKLOOK_DPI = 120

# Missing VV descriptors are expected when EO Browser exports only VH layers.
# They are retained as nodata in the fixed stack, but omitted from the available
# stack and listed in diagnostics.
ALWAYS_KEEP_DESCRIPTOR_NAMES = {"S1_VALID"}

VERBOSE = True

# =============================================================================
# Helpers
# =============================================================================


def log(msg: str) -> None:
    if VERBOSE:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _ensure_dirs() -> None:
    S1_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    for subdir in [
        "s1_descriptor_stacks",
        "s1_available_descriptor_stacks",
        "s1_single_band_descriptors",
        "s1_valid_masks",
        "s1_display_quicklooks",
    ]:
        (S1_CACHE_ROOT / subdir).mkdir(parents=True, exist_ok=True)


def _descriptor_has_data(arr: np.ndarray) -> bool:
    vals = arr[np.isfinite(arr)]
    return bool(vals.size > 0)


def _descriptor_stats(scene: str, descriptor: str, arr: np.ndarray) -> dict:
    vals = arr[np.isfinite(arr)]
    rec = {
        "scene": scene,
        "descriptor": descriptor,
        "valid_px": int(vals.size),
        "nodata_or_nan_px": int(arr.size - vals.size),
        "available": bool(vals.size > 0),
    }
    if vals.size:
        q = np.nanpercentile(vals.astype(np.float64), [0, 1, 2, 5, 10, 50, 90, 95, 98, 99, 100])
        rec.update({
            "min": float(q[0]),
            "p01": float(q[1]),
            "p02": float(q[2]),
            "p05": float(q[3]),
            "p10": float(q[4]),
            "median": float(q[5]),
            "p90": float(q[6]),
            "p95": float(q[7]),
            "p98": float(q[8]),
            "p99": float(q[9]),
            "max": float(q[10]),
            "mean": float(np.nanmean(vals)),
            "std": float(np.nanstd(vals)),
            "dynamic_range_p02_p98": float(q[8] - q[2]),
            "unique_values_sample_100k": int(len(np.unique(vals[: min(vals.size, 100000)]))),
            "flat_flag": bool(np.nanstd(vals) < 1e-6 or abs(q[8] - q[2]) < 1e-6),
        })
    else:
        rec.update({
            "min": np.nan,
            "p01": np.nan,
            "p02": np.nan,
            "p05": np.nan,
            "p10": np.nan,
            "median": np.nan,
            "p90": np.nan,
            "p95": np.nan,
            "p98": np.nan,
            "p99": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "dynamic_range_p02_p98": np.nan,
            "unique_values_sample_100k": 0,
            "flat_flag": True,
        })
    return rec


def _source_product_diagnostics(zip_path: Path, members: Dict[str, str | None]) -> List[dict]:
    rows: List[dict] = []
    for tag, member in members.items():
        if not member:
            rows.append({
                "scene": zip_path.name,
                "source_tag": tag,
                "member": "",
                "present": False,
                "product_type": "missing",
            })
            continue
        rec = {
            "scene": zip_path.name,
            "source_tag": tag,
            "member": member,
            "present": True,
        }
        try:
            with rasterio.open(vsizip_path(zip_path, member)) as ds:
                rec.update({
                    "count": int(ds.count),
                    "dtype_band1": ds.dtypes[0],
                    "nodata": ds.nodata,
                    "width": int(ds.width),
                    "height": int(ds.height),
                    "crs": str(ds.crs),
                })
                arr1 = ds.read(1).astype(np.float64)
                valid = ds.read_masks(1) > 0
                alpha_valid_px = np.nan
                rgb_equal = np.nan
                if ds.count >= 4:
                    alpha_valid_px = int((ds.read(4) > 0).sum())
                if ds.count >= 3:
                    try:
                        a2 = ds.read(2)
                        a3 = ds.read(3)
                        rgb_equal = bool(np.array_equal(ds.read(1), a2) and np.array_equal(ds.read(1), a3))
                    except Exception:
                        rgb_equal = np.nan
                vals = arr1[valid]
                q = np.nanpercentile(vals, [0, 1, 50, 99, 100]) if vals.size else [np.nan] * 5
                product_type = "unknown"
                band2_datamask = False
                if ds.count >= 2:
                    try:
                        b2 = ds.read(2)
                        uniq = np.unique(b2[: min(b2.shape[0], 256), : min(b2.shape[1], 256)])
                        band2_datamask = bool(len(uniq) <= 4 and set(float(x) for x in uniq.tolist()).issubset({0.0, 1.0, 255.0}))
                    except Exception:
                        band2_datamask = False
                if ds.count == 2 and np.issubdtype(np.dtype(ds.dtypes[0]), np.floating) and band2_datamask:
                    product_type = "raw_float_backscatter_with_band2_datamask"
                elif ds.count >= 4 and np.issubdtype(np.dtype(ds.dtypes[0]), np.integer):
                    product_type = "rendered_rgba_uint"
                elif ds.count == 1 and np.issubdtype(np.dtype(ds.dtypes[0]), np.floating):
                    product_type = "single_band_float_physical_candidate"
                elif ds.count == 1:
                    product_type = "single_band_nonfloat"
                rec.update({
                    "product_type": product_type,
                    "band2_looks_like_datamask": band2_datamask,
                    "rgb_bands_equal": rgb_equal,
                    "alpha_valid_px": alpha_valid_px,
                    "raw_band1_min": float(q[0]),
                    "raw_band1_p01": float(q[1]),
                    "raw_band1_median": float(q[2]),
                    "raw_band1_p99": float(q[3]),
                    "raw_band1_max": float(q[4]),
                    "raw_values_are_display_scaled_warning": bool(product_type == "rendered_rgba_uint"),
                })
        except Exception as exc:
            rec.update({"product_type": "read_error", "read_error": str(exc)})
        rows.append(rec)
    return rows


def _mask_local_descriptors_to_source_valid(stack: Dict[str, np.ndarray]) -> None:
    """Avoid valid-looking 3x3 local statistics outside the source-pol valid domain."""
    if "S1_VH_DB" in stack:
        vh_valid = np.isfinite(stack["S1_VH_DB"])
        for name in ["S1_VH_LOCAL_MEAN3", "S1_VH_LOCAL_STD3"]:
            if name in stack:
                stack[name] = np.where(vh_valid, stack[name], np.nan).astype(np.float32)
    if "S1_VV_DB" in stack:
        vv_valid = np.isfinite(stack["S1_VV_DB"])
        for name in ["S1_VV_LOCAL_MEAN3", "S1_VV_LOCAL_STD3"]:
            if name in stack:
                stack[name] = np.where(vv_valid, stack[name], np.nan).astype(np.float32)


def _available_descriptor_names(stack: Dict[str, np.ndarray]) -> List[str]:
    out = []
    for name in S1_DESCRIPTOR_BANDS:
        arr = stack.get(name)
        if arr is None:
            continue
        if name in ALWAYS_KEEP_DESCRIPTOR_NAMES or _descriptor_has_data(arr):
            out.append(name)
    return out


def _write_single_band_descriptor_rasters(scene: str, stack: Dict[str, np.ndarray], ref_profile: dict, names: List[str]) -> Dict[str, str]:
    paths = {}
    if not WRITE_SINGLE_BAND_DESCRIPTOR_RASTERS:
        return paths
    scene_key = Path(scene).stem
    for name in names:
        if name == "S1_VALID":
            continue
        out_dir = S1_CACHE_ROOT / "s1_single_band_descriptors" / name
        out_path = out_dir / f"{scene_key}__{name}.tif"
        write_float32_stack(out_path, {name: stack[name]}, ref_profile, [name])
        paths[name] = str(out_path)
    return paths


def _write_quicklook(scene: str, descriptor: str, arr: np.ndarray) -> str:
    if not WRITE_QUICKLOOK_PNGS or descriptor == "S1_VALID":
        return ""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return ""
    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        return ""
    lo, hi = np.nanpercentile(vals, QUICKLOOK_PERCENTILES)
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
        lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    if abs(hi - lo) < 1e-12:
        img = np.zeros(arr.shape, dtype=np.float32)
    else:
        img = np.clip((arr - lo) / (hi - lo), 0, 1).astype(np.float32)
    img[~np.isfinite(img)] = 0
    scene_key = Path(scene).stem
    out_path = S1_CACHE_ROOT / "s1_display_quicklooks" / f"{scene_key}__{descriptor}_quicklook.png"
    plt.figure(figsize=(8, 6))
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.title(f"{scene_key} — {descriptor} [{QUICKLOOK_PERCENTILES[0]:.0f}-{QUICKLOOK_PERCENTILES[1]:.0f}% stretch]")
    plt.tight_layout()
    plt.savefig(out_path, dpi=QUICKLOOK_DPI, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    return str(out_path)


def main() -> None:
    t0 = time.time()
    _ensure_dirs()

    if not REFERENCE_RASTER.exists():
        raise FileNotFoundError(
            f"Reference raster is missing: {REFERENCE_RASTER}\n"
            "Run the Phase 3/4 Sentinel-2 preprocessing cache before the Sentinel-1 cache."
        )
    _, ref_profile = read_int_raster(REFERENCE_RASTER)

    if not S1_ZIP_FILES:
        raise FileNotFoundError(f"No Sentinel-1 ZIP files matched: {S1_ZIP_ROOT / S1_ZIP_GLOB}")

    log("Starting Sentinel-1 descriptor preprocessing for raw VV/VH ZIPs")
    log(f"Found {len(S1_ZIP_FILES)} Sentinel-1 ZIP files")
    log(f"Reference grid: {REFERENCE_RASTER}")

    rows = []
    stat_rows = []
    source_rows = []
    quicklook_rows = []

    for i, zip_path in enumerate(S1_ZIP_FILES, start=1):
        t_scene = time.time()
        scene = zip_path.name
        date = parse_date_from_name(scene)
        paths = s1_cache_scene_paths(S1_CACHE_ROOT, scene)
        available_stack_path = S1_CACHE_ROOT / "s1_available_descriptor_stacks" / f"{Path(scene).stem}__s1_available_descriptor_stack.tif"
        members = s1_members_by_tag(zip_path)

        try:
            cache_complete = paths["s1_stack"].exists() and paths["s1_valid_mask"].exists() and available_stack_path.exists()
            if cache_complete and not FORCE_REBUILD_S1_CACHE:
                cache_status = "existing"
                # Statistics are still recomputed from the existing stack for diagnostics.
                stack = {}
                import rasterio as _rio
                with _rio.open(paths["s1_stack"]) as ds:
                    names = [d if d else f"band_{j}" for j, d in enumerate(ds.descriptions, start=1)]
                    for band_i, name in enumerate(names, start=1):
                        arr = ds.read(band_i).astype(np.float32)
                        nd = ds.nodata
                        if nd is not None:
                            arr[np.isclose(arr, float(nd))] = np.nan
                        stack[name] = arr
                valid_any = np.isfinite(stack.get("S1_VH_DB", np.full((ref_profile['height'], ref_profile['width']), np.nan)))
            else:
                log(f"[{i}/{len(S1_ZIP_FILES)}] Building Sentinel-1 descriptors for {scene}")
                stack, valid_any, members = build_s1_descriptor_stack(
                    zip_path,
                    ref_profile,
                    normalize_rendered_uint=NORMALIZE_RENDERED_UINT,
                    treat_zero_as_nodata=TREAT_ZERO_AS_NODATA,
                )
                _mask_local_descriptors_to_source_valid(stack)
                if WRITE_FIXED_DESCRIPTOR_STACK:
                    write_float32_stack(paths["s1_stack"], stack, ref_profile, S1_DESCRIPTOR_BANDS)
                write_uint8_mask(paths["s1_valid_mask"], valid_any.astype(np.uint8), ref_profile, nodata=255)
                cache_status = "rebuilt" if cache_complete else "created"

            available_names = _available_descriptor_names(stack)
            if WRITE_AVAILABLE_DESCRIPTOR_STACK:
                write_float32_stack(available_stack_path, stack, ref_profile, available_names)
            single_paths = _write_single_band_descriptor_rasters(scene, stack, ref_profile, available_names)

            source_rows.extend(_source_product_diagnostics(zip_path, members))
            for name in S1_DESCRIPTOR_BANDS:
                arr = stack.get(name)
                if arr is None:
                    arr = np.full((int(ref_profile["height"]), int(ref_profile["width"])), np.nan, dtype=np.float32)
                rec = _descriptor_stats(scene, name, arr)
                rec["in_available_stack"] = name in available_names
                stat_rows.append(rec)
                ql = _write_quicklook(scene, name, arr) if name in available_names else ""
                if ql:
                    quicklook_rows.append({"scene": scene, "descriptor": name, "quicklook_path": ql})

            rows.append({
                "scene": scene,
                "zip_path": str(zip_path),
                "date": date.date().isoformat(),
                "year": int(date.year),
                "month": int(date.month),
                "day": int(date.day),
                "doy": int(date.dayofyear),
                "s1_stack_path": str(paths["s1_stack"]),
                "s1_available_stack_path": str(available_stack_path),
                "s1_valid_mask_path": str(paths["s1_valid_mask"]),
                "vh_db_member": members.get("VH_DB") or "",
                "vv_db_member": members.get("VV_DB") or "",
                "vh_linear_member": members.get("VH_LINEAR") or "",
                "vv_linear_member": members.get("VV_LINEAR") or "",
                "has_vh_db": bool(members.get("VH_DB")),
                "has_vv_db": bool(members.get("VV_DB")),
                "has_vh_linear": bool(members.get("VH_LINEAR")),
                "has_vv_linear": bool(members.get("VV_LINEAR")),
                "descriptor_bands_fixed": ",".join(S1_DESCRIPTOR_BANDS),
                "descriptor_bands_available": ",".join(available_names),
                "single_band_descriptor_paths_json": str(single_paths),
                "cache_status": cache_status,
                "preprocess_error": "",
            })
            log(f"[{i}/{len(S1_ZIP_FILES)}] Finished {scene} in {time.time() - t_scene:.1f}s ({cache_status}); available descriptors: {', '.join(available_names)}")
        except Exception as exc:
            rows.append({
                "scene": scene,
                "zip_path": str(zip_path),
                "date": date.date().isoformat(),
                "year": int(date.year),
                "month": int(date.month),
                "day": int(date.day),
                "doy": int(date.dayofyear),
                "s1_stack_path": str(paths["s1_stack"]),
                "s1_available_stack_path": str(available_stack_path),
                "s1_valid_mask_path": str(paths["s1_valid_mask"]),
                "cache_status": "error",
                "preprocess_error": str(exc),
            })
            pd.DataFrame(rows).to_csv(S1_CACHE_ROOT / "s1_scene_inventory.csv", index=False)
            raise

    inv = pd.DataFrame(rows).sort_values(["date", "scene"]).reset_index(drop=True)
    inv.to_csv(S1_CACHE_ROOT / "s1_scene_inventory.csv", index=False)
    pd.DataFrame({
        "band_number": list(range(1, len(S1_DESCRIPTOR_BANDS) + 1)),
        "descriptor": list(S1_DESCRIPTOR_BANDS),
        "fixed_stack_role": ["fixed_downstream_compatibility"] * len(S1_DESCRIPTOR_BANDS),
    }).to_csv(S1_CACHE_ROOT / "s1_descriptor_band_lookup.csv", index=False)

    stat_df = pd.DataFrame(stat_rows)
    stat_df.to_csv(S1_CACHE_ROOT / "s1_cache_band_statistics.csv", index=False)
    source_df = pd.DataFrame(source_rows)
    source_df.to_csv(S1_CACHE_ROOT / "s1_source_product_diagnostics.csv", index=False)
    pd.DataFrame(quicklook_rows).to_csv(S1_CACHE_ROOT / "s1_quicklook_inventory.csv", index=False)

    warning_df = stat_df[(stat_df["flat_flag"] == True) | (stat_df["available"] == False)].copy()
    warning_df.to_csv(S1_CACHE_ROOT / "s1_flat_descriptor_warnings.csv", index=False)

    write_json(S1_CACHE_ROOT / "sentinel1_preprocessing_manifest.json", {
        "phase": "5_v4",
        "purpose": "Sentinel-1 descriptor cache aligned to the Stana de Vale Sentinel-2/FMU grid; preferred input is simple raw IW-DV VV/VH ZIPs",
        "s1_zip_root": str(S1_ZIP_ROOT),
        "s1_zip_glob": S1_ZIP_GLOB,
        "s1_zip_count": len(S1_ZIP_FILES),
        "s1_cache_root": str(S1_CACHE_ROOT),
        "reference_raster": str(REFERENCE_RASTER),
        "fixed_descriptor_bands": list(S1_DESCRIPTOR_BANDS),
        "normalize_rendered_uint": NORMALIZE_RENDERED_UINT,
        "treat_zero_as_nodata": TREAT_ZERO_AS_NODATA,
        "write_available_descriptor_stack": WRITE_AVAILABLE_DESCRIPTOR_STACK,
        "write_single_band_descriptor_rasters": WRITE_SINGLE_BAND_DESCRIPTOR_RASTERS,
        "write_quicklook_pngs": WRITE_QUICKLOOK_PNGS,
        "n_scenes_with_vh_db": int(inv.get("has_vh_db", pd.Series(dtype=bool)).fillna(False).sum()),
        "n_scenes_with_vv_db": int(inv.get("has_vv_db", pd.Series(dtype=bool)).fillna(False).sum()),
        "n_scenes_with_vh_linear": int(inv.get("has_vh_linear", pd.Series(dtype=bool)).fillna(False).sum()),
        "n_scenes_with_vv_linear": int(inv.get("has_vv_linear", pd.Series(dtype=bool)).fillna(False).sum()),
        "important_warning": "If source_product_diagnostics.product_type is rendered_rgba_uint, values are display-scaled 0..1 after normalization and not physical dB/linear gamma0.",
    })

    log(f"Done in {time.time() - t0:.1f}s. Sentinel-1 cache outputs in {S1_CACHE_ROOT}")


if __name__ == "__main__":
    main()
