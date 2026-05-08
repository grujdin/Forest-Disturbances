"""
Shared utilities for the Stana de Vale Sentinel-2 forest-disturbance workflow.

Phase 2 purpose
---------------
This module centralizes low-level routines used by Algorithm #1 and Algorithm #2:
scene/date handling, EO Browser ZIP member discovery, raster reading/alignment,
SCL decoding, clear-mask construction, normalized-difference index computation,
raster writing, FMU group rasterization, and output ZIP creation.

The functions are intentionally parameterized so that each algorithm can keep its
own hard-coded configuration while using the same implementation.
"""
from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.warp import Resampling, reproject

DATE_RE = re.compile(r"(20\d{2})-(\d{2})-(\d{2})")

S2_RAW_MEMBER_PATTERNS = {
    "B03": "B03_(Raw).tiff",
    "B04": "B04_(Raw).tiff",
    "B05": "B05_(Raw).tiff",
    "B08": "B08_(Raw).tiff",
    "B8A": "B8A_(Raw).tiff",
    "B11": "B11_(Raw).tiff",
    "B12": "B12_(Raw).tiff",
}

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

DEFAULT_SCL_EXCLUDE_CLASSES = {0, 1, 2, 3, 8, 9, 10, 11}
DEFAULT_CORE_INDICES = ("NDVI", "NDMI", "NBR", "NDRE")


def _scl_color_u16(rgb: Tuple[int, int, int]) -> Tuple[int, int, int]:
    return tuple(int(round(c * 65535.0 / 255.0)) for c in rgb)


SCL_COLORS_U16 = {k: _scl_color_u16(v) for k, v in SCL_COLORS_U8.items()}


def parse_date_from_name(name: str) -> pd.Timestamp:
    """Parse the first YYYY-MM-DD date contained in a scene or member name."""
    m = DATE_RE.search(str(name))
    if not m:
        raise ValueError(f"Could not parse date from: {name}")
    y, mm, dd = map(int, m.groups())
    return pd.Timestamp(year=y, month=mm, day=dd)


def doy_bin_label(doy: int, width: int) -> str:
    """Return a 1-based DOY-bin label, for example 181-210."""
    start = ((int(doy) - 1) // int(width)) * int(width) + 1
    end = min(start + int(width) - 1, 365)
    return f"{start:03d}-{end:03d}"


def doy_bin_center(doy: int, width: int) -> int:
    """Return the integer center of the DOY bin containing the supplied DOY."""
    start = ((int(doy) - 1) // int(width)) * int(width) + 1
    end = min(start + int(width) - 1, 365)
    return int(round((start + end) / 2))


def doy_bin_center_from_label(label: str) -> int:
    """Return the integer center of a DOY-bin label, for example 181-210."""
    a, b = str(label).split("-")
    return int(round((int(a) + int(b)) / 2))


def zip_members_by_tag(
    zip_path: Path,
    required_tags: Sequence[str] = ("B04", "B05", "B08", "B8A", "B11", "B12"),
    include_scl: bool = True,
    include_date_string: bool = False,
) -> Dict[str, str | None]:
    """Find EO Browser raw-band and optional SCL members inside a ZIP scene.

    Parameters
    ----------
    zip_path:
        EO Browser ZIP file.
    required_tags:
        Sentinel-2 band tags that must exist in the ZIP.
    include_scl:
        Whether to also return the first SCL-like member as key ``SCL``.
    include_date_string:
        Whether to store the B04 member as ``DATE_STRING`` for compatibility with
        the original Algorithm #1 implementation.
    """
    zip_path = Path(zip_path)
    out: Dict[str, str | None] = {}
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        lower_names = {n.lower(): n for n in names}
        for tag in required_tags:
            if tag not in S2_RAW_MEMBER_PATTERNS:
                raise KeyError(f"Unsupported Sentinel-2 raw-band tag: {tag}")
            pattern = S2_RAW_MEMBER_PATTERNS[tag].lower()
            matches = [orig for low, orig in lower_names.items() if pattern in low]
            if not matches:
                raise FileNotFoundError(f"Could not find {S2_RAW_MEMBER_PATTERNS[tag]} in {zip_path.name}")
            out[tag] = matches[0]
        if include_scl:
            scl_matches = [
                orig for low, orig in lower_names.items()
                if "scene_classification_map" in low or "_scl" in low or "scl" in low
            ]
            out["SCL"] = scl_matches[0] if scl_matches else None
        if include_date_string:
            out["DATE_STRING"] = out.get("B04")
    return out


def vsizip_path(zip_path: Path, inner_member: str) -> str:
    """Return a GDAL /vsizip/ path for a member inside a ZIP file."""
    return f"/vsizip/{Path(zip_path)}/{inner_member}"


def read_single_band(zip_path: Path, inner_member: str):
    """Read a single-band raster from a ZIP member and return array, profile, mask."""
    path = vsizip_path(zip_path, inner_member)
    with rasterio.open(path) as ds:
        arr = ds.read(1)
        profile = ds.profile.copy()
        valid = ds.read_masks(1) > 0
    return arr, profile, valid


def same_grid(profile_a: dict, profile_b: dict) -> bool:
    """Return True when two raster profiles share dimensions, transform, and CRS."""
    return (
        profile_a.get("height") == profile_b.get("height")
        and profile_a.get("width") == profile_b.get("width")
        and profile_a.get("transform") == profile_b.get("transform")
        and str(profile_a.get("crs")) == str(profile_b.get("crs"))
    )


def align_multiband_to(ref_profile: dict, src_path: str, band_indexes: List[int]) -> np.ndarray:
    """Read selected bands and reproject them to a reference raster profile."""
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


def _decode_rendered_scl_classes(rgb: np.ndarray) -> np.ndarray:
    """Decode an RGB-rendered Sentinel-2 SCL raster to class IDs."""
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


def load_scl_classes(zip_path: Path, scl_member: str | None, ref_profile: dict) -> np.ndarray:
    """Load a single-band or rendered-RGB SCL layer and align it to ref_profile."""
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


def build_clear_mask(
    data_mask: np.ndarray,
    scl_classes: np.ndarray,
    exclude_classes: Iterable[int] = DEFAULT_SCL_EXCLUDE_CLASSES,
    allow_unknown_scl_colors: bool = False,
) -> np.ndarray:
    """Build a clear-pixel mask from a valid-data mask and SCL class raster."""
    invalid = np.isin(scl_classes, list(exclude_classes))
    if not allow_unknown_scl_colors:
        invalid |= (scl_classes == 255)
    return data_mask & (~invalid)


def build_scene_clear_mask(
    zip_path: Path,
    scl_member: str | None,
    ref_profile: dict,
    data_mask: np.ndarray,
    use_scl_if_available: bool = True,
    exclude_classes: Iterable[int] = DEFAULT_SCL_EXCLUDE_CLASSES,
    allow_unknown_scl_colors: bool = False,
) -> Tuple[np.ndarray, bool]:
    """Return a clear-pixel mask and whether SCL was actually used.

    Missing SCL is treated as valid-data-only, which keeps Algorithm #1 and
    Algorithm #2 behavior consistent.
    """
    if use_scl_if_available and scl_member is not None:
        scl_classes = load_scl_classes(zip_path, scl_member, ref_profile)
        return build_clear_mask(data_mask, scl_classes, exclude_classes, allow_unknown_scl_colors), True
    return data_mask.copy(), False


def safe_norm_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute (a-b)/(a+b), returning NaN where the denominator is zero."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    denom = a + b
    out = np.full(a.shape, np.nan, dtype=np.float32)
    valid = denom != 0
    out[valid] = (a[valid] - b[valid]) / denom[valid]
    return out


def compute_indices(
    bands: Dict[str, np.ndarray],
    indices: Sequence[str] | None = None,
) -> Dict[str, np.ndarray]:
    """Compute selected Sentinel-2 normalized-difference forest indices."""
    indices = tuple(indices) if indices is not None else DEFAULT_CORE_INDICES
    out: Dict[str, np.ndarray] = {}
    if "NDVI" in indices:
        out["NDVI"] = safe_norm_diff(bands["B08"], bands["B04"])
    if "NDMI" in indices:
        out["NDMI"] = safe_norm_diff(bands["B08"], bands["B11"])
    if "NBR" in indices:
        out["NBR"] = safe_norm_diff(bands["B08"], bands["B12"])
    if "NDRE" in indices:
        out["NDRE"] = safe_norm_diff(bands["B8A"], bands["B05"])
    unknown = set(indices) - {"NDVI", "NDMI", "NBR", "NDRE"}
    if unknown:
        raise ValueError(f"Unsupported index names for compute_indices: {sorted(unknown)}")
    return out


def compute_ndsi(bands: Dict[str, np.ndarray]) -> np.ndarray:
    """Compute NDSI = (B03-B11)/(B03+B11)."""
    return safe_norm_diff(bands["B03"], bands["B11"])


def write_uint8_mask(path: Path, arr: np.ndarray, ref_profile: dict, nodata: int = 255) -> None:
    """Write a compressed UInt8 single-band GeoTIFF mask."""
    profile = ref_profile.copy()
    profile.update(dtype=rasterio.uint8, count=1, nodata=nodata, compress="deflate")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(np.uint8), 1)


def write_int16(path: Path, arr: np.ndarray, ref_profile: dict, nodata: int = -32768) -> None:
    """Write a compressed Int16 single-band GeoTIFF raster."""
    profile = ref_profile.copy()
    profile.update(dtype=rasterio.int16, count=1, nodata=nodata, compress="deflate")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(np.int16), 1)




def write_float32(path: Path, arr: np.ndarray, ref_profile: dict, nodata: float = -9999.0) -> None:
    """Write a compressed Float32 single-band GeoTIFF raster."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = ref_profile.copy()
    profile.update(dtype=rasterio.float32, count=1, nodata=float(nodata), compress="deflate", predictor=3)
    out = np.asarray(arr, dtype=np.float32)
    out = np.where(np.isfinite(out), out, np.float32(nodata)).astype(np.float32)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out, 1)

def rasterize_groups(
    group_gdf,
    ref_profile: dict,
    group_code_field: str,
    group_label_field: str,
    all_touched: bool = False,
    rename_lut_columns: bool = False,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Rasterize analytical group polygons to a reference raster grid."""
    if group_gdf.crs != ref_profile["crs"]:
        group_gdf = group_gdf.to_crs(ref_profile["crs"])
    groups = (
        group_gdf[[group_code_field, group_label_field]]
        .drop_duplicates()
        .sort_values([group_code_field])
        .reset_index(drop=True)
        .copy()
    )
    groups["group_id"] = np.arange(1, len(groups) + 1, dtype=np.int32)

    if rename_lut_columns:
        lut = groups[["group_id", group_code_field, group_label_field]].rename(
            columns={group_code_field: "group_code", group_label_field: "group_label"}
        )
        merge_right = lut
        left_on = [group_code_field, group_label_field]
        right_on = ["group_code", "group_label"]
    else:
        lut = groups.copy()
        merge_right = groups
        left_on = [group_code_field, group_label_field]
        right_on = [group_code_field, group_label_field]

    gdf_with_id = group_gdf.merge(merge_right, left_on=left_on, right_on=right_on, how="left")
    shapes = (
        (geom, int(gid))
        for geom, gid in zip(gdf_with_id.geometry, gdf_with_id["group_id"])
        if geom is not None and not geom.is_empty and pd.notna(gid)
    )
    group_raster = rasterize(
        shapes=shapes,
        out_shape=(ref_profile["height"], ref_profile["width"]),
        transform=ref_profile["transform"],
        fill=0,
        dtype="int32",
        all_touched=all_touched,
    )
    return group_raster, lut


def norm_text(x) -> str:
    """Normalize text identifiers for robust FMU/workbook joins."""
    s = "" if pd.isna(x) else str(x)
    s = s.strip().upper().replace(" ", "")
    if s in {"", "NONE", "NAN", "NULL"}:
        return ""
    return s


def norm_ua_from_parts(parc, subp, ua) -> str:
    """Normalize FMU unit IDs from parcel, subparcel, and UA components."""
    ua_n = norm_text(ua)
    parc_n = norm_text(parc)
    subp_n = norm_text(subp)
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


def resolve_group_sheet(workbook_path: Path, preferred_sheet: str, required_columns: Iterable[str]) -> str:
    """Return preferred sheet if available, otherwise first sheet with required columns."""
    xls = pd.ExcelFile(workbook_path)
    if preferred_sheet in xls.sheet_names:
        return preferred_sheet
    required = set(required_columns)
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


def zip_output_directory(output_dir: Path, zip_path: Path, overwrite: bool = True) -> None:
    """Create a ZIP bundle with output_dir.name as top-level archive folder."""
    output_dir = Path(output_dir)
    zip_path = Path(zip_path)

    if not output_dir.exists() or not output_dir.is_dir():
        raise FileNotFoundError(f"Cannot create ZIP because output directory does not exist: {output_dir}")

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists() and not overwrite:
        raise FileExistsError(f"Output ZIP already exists: {zip_path}")

    tmp_zip = zip_path.with_name(zip_path.name + ".tmp")
    if tmp_zip.exists():
        tmp_zip.unlink()

    root_name = output_dir.name
    files = sorted(p for p in output_dir.rglob("*") if p.is_file())

    with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for file_path in files:
            try:
                same_as_zip = file_path.resolve() == zip_path.resolve()
                same_as_tmp = file_path.resolve() == tmp_zip.resolve()
            except FileNotFoundError:
                same_as_zip = False
                same_as_tmp = False
            if same_as_zip or same_as_tmp:
                continue

            arcname = Path(root_name) / file_path.relative_to(output_dir)
            zf.write(file_path, arcname.as_posix())

    tmp_zip.replace(zip_path)

# =============================================================================
# Phase 3 cache helpers
# =============================================================================

CACHE_FLOAT_NODATA = -9999.0
CACHE_INDEX_BANDS = ("NDVI", "NDMI", "NBR", "NDRE", "NDSI")


def cache_scene_key(scene_name) -> str:
    """Return a filesystem-safe scene key used for cache file names."""
    return Path(str(scene_name)).stem


def cache_scene_paths(cache_root: Path, scene_name) -> Dict[str, Path]:
    """Return standard Phase 3 cache paths for one scene."""
    root = Path(cache_root)
    key = cache_scene_key(scene_name)
    return {
        "index_stack": root / "index_stacks" / f"{key}__index_stack.tif",
        "clear_mask": root / "clear_masks" / f"{key}__clear_mask.tif",
        "scl_class": root / "scl_classes" / f"{key}__scl_class.tif",
    }


def write_float32_stack(
    path: Path,
    arrays: Dict[str, np.ndarray],
    ref_profile: dict,
    band_names: Sequence[str],
    nodata: float = CACHE_FLOAT_NODATA,
) -> None:
    """Write a named Float32 multi-band GeoTIFF stack, replacing NaN with nodata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = ref_profile.copy()
    profile.update(dtype=rasterio.float32, count=len(band_names), nodata=float(nodata), compress="deflate", predictor=3)
    with rasterio.open(path, "w", **profile) as dst:
        for i, name in enumerate(band_names, start=1):
            arr = np.asarray(arrays[name], dtype=np.float32)
            out = np.where(np.isfinite(arr), arr, np.float32(nodata)).astype(np.float32)
            dst.write(out, i)
            try:
                dst.set_band_description(i, str(name))
            except Exception:
                pass


def read_float32_stack(path: Path, requested_band_names: Sequence[str] | None = None, nodata: float = CACHE_FLOAT_NODATA) -> Tuple[Dict[str, np.ndarray], dict]:
    """Read a named Float32 stack written by write_float32_stack."""
    path = Path(path)
    with rasterio.open(path) as ds:
        profile = ds.profile.copy()
        descriptions = list(ds.descriptions)
        names = [d if d else f"band_{i}" for i, d in enumerate(descriptions, start=1)]
        if requested_band_names is None:
            requested_band_names = names
        name_to_band = {str(name): i for i, name in enumerate(names, start=1)}
        out: Dict[str, np.ndarray] = {}
        for name in requested_band_names:
            if name not in name_to_band:
                # Support old/no-description stacks by assuming the standard order.
                if name in CACHE_INDEX_BANDS and len(names) >= CACHE_INDEX_BANDS.index(name) + 1:
                    band_i = CACHE_INDEX_BANDS.index(name) + 1
                else:
                    raise KeyError(f"Band {name!r} not found in cached stack: {path}")
            else:
                band_i = name_to_band[name]
            arr = ds.read(band_i).astype(np.float32)
            nd = ds.nodata if ds.nodata is not None else nodata
            arr[np.isclose(arr, float(nd))] = np.nan
            out[str(name)] = arr
    return out, profile


def read_uint8_raster(path: Path) -> Tuple[np.ndarray, dict]:
    """Read a single-band UInt8 raster and return array plus profile."""
    with rasterio.open(path) as ds:
        return ds.read(1).astype(np.uint8), ds.profile.copy()


def read_int_raster(path: Path) -> Tuple[np.ndarray, dict]:
    """Read a single-band integer raster and return array plus profile."""
    with rasterio.open(path) as ds:
        return ds.read(1), ds.profile.copy()


def write_json(path: Path, payload: dict) -> None:
    """Write JSON with indentation."""
    import json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

# Phase 3 compatibility aliases used by the cache-aware Algorithm #1/#2 scripts.

def read_json(path: Path) -> dict:
    """Read JSON written by write_json."""
    import json
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_uint8_mask(path: Path) -> Tuple[np.ndarray, dict]:
    """Compatibility alias for reading cached UInt8 masks."""
    return read_uint8_raster(path)


# Override the earlier function with an alias-compatible signature.
def read_float32_stack(path: Path, requested_band_names: Sequence[str] | None = None, expected_names: Sequence[str] | None = None, nodata: float = CACHE_FLOAT_NODATA) -> Tuple[Dict[str, np.ndarray], dict]:
    """Read a named Float32 stack.

    Parameters
    ----------
    requested_band_names, expected_names:
        Synonymous parameters. ``expected_names`` is kept for Phase 3 script
        compatibility.
    """
    path = Path(path)
    if requested_band_names is None and expected_names is not None:
        requested_band_names = expected_names
    with rasterio.open(path) as ds:
        profile = ds.profile.copy()
        descriptions = list(ds.descriptions)
        names = [d if d else f"band_{i}" for i, d in enumerate(descriptions, start=1)]
        if requested_band_names is None:
            requested_band_names = names
        name_to_band = {str(name): i for i, name in enumerate(names, start=1)}
        out: Dict[str, np.ndarray] = {}
        for name in requested_band_names:
            name = str(name)
            if name in name_to_band:
                band_i = name_to_band[name]
            elif name in CACHE_INDEX_BANDS and len(names) >= CACHE_INDEX_BANDS.index(name) + 1:
                band_i = CACHE_INDEX_BANDS.index(name) + 1
            else:
                raise KeyError(f"Band {name!r} not found in cached stack: {path}")
            arr = ds.read(band_i).astype(np.float32)
            nd = ds.nodata if ds.nodata is not None else nodata
            arr[np.isclose(arr, float(nd))] = np.nan
            out[name] = arr
    return out, profile

# =============================================================================
# Phase 5 Sentinel-1 descriptor helpers
# =============================================================================

S1_DESCRIPTOR_BANDS = (
    "S1_VH_DB",
    "S1_VV_DB",
    "S1_VH_LINEAR",
    "S1_VV_LINEAR",
    "S1_VV_MINUS_VH_DB",
    "S1_VV_DIV_VH_LINEAR",
    "S1_VH_LOCAL_MEAN3",
    "S1_VH_LOCAL_STD3",
    "S1_VV_LOCAL_MEAN3",
    "S1_VV_LOCAL_STD3",
    "S1_VALID",
)

S1_BASE_TAGS = ("VH_DB", "VV_DB", "VH_LINEAR", "VV_LINEAR")


def s1_cache_scene_key(scene_name) -> str:
    """Return a filesystem-safe Sentinel-1 scene key used for cache products."""
    return Path(str(scene_name)).stem


def s1_cache_scene_paths(cache_root: Path, scene_name) -> Dict[str, Path]:
    """Return standard Phase 5 Sentinel-1 cache paths for one scene."""
    root = Path(cache_root)
    key = s1_cache_scene_key(scene_name)
    return {
        "s1_stack": root / "s1_descriptor_stacks" / f"{key}__s1_descriptor_stack.tif",
        "s1_valid_mask": root / "s1_valid_masks" / f"{key}__s1_valid_mask.tif",
    }


def _s1_pol_match(member_name_lower: str, pol: str) -> bool:
    """Detect an explicit Sentinel-1 polarization token without matching mode text.

    The uploaded EO Browser/Sentinel Hub ZIPs use names like
    ``..._IW_VV+VH_VV_(Raw).tiff``.  This helper intentionally accepts the
    final ``_VV_`` token while avoiding accidental matches inside unrelated
    words.
    """
    pol = pol.lower()
    patterns = [
        rf"(^|[_\s\-]){pol}([_\s\-]|$)",
        rf"(^|[_\s\-]){pol}_-",
        rf"(^|[_\s\-]){pol} -",
        rf"[_\s\-]{pol}\(raw\)",
        rf"[_\s\-]{pol}_\(raw\)",
        rf"[_\s\-]{pol}_raw",
    ]
    return any(re.search(pat, member_name_lower) for pat in patterns)


def _s1_is_raw_layer(member_name_lower: str) -> bool:
    """Return True for EO Browser analytical raw layers, e.g. VV_(Raw).tiff."""
    return "(raw)" in member_name_lower or "_(raw)" in member_name_lower or "_raw" in member_name_lower


def s1_members_by_tag(zip_path: Path) -> Dict[str, str | None]:
    """Find Sentinel-1 EO Browser members for VV/VH descriptors.

    Preferred input is a simple analytical ZIP with raw dual-polarization
    layers, for example::

        ..._IW_VV+VH_VV_(Raw).tiff
        ..._IW_VV+VH_VH_(Raw).tiff

    The raw layers are interpreted as linear backscatter and are preferred over
    visualization layers.  If raw layers are not present, the function can still
    use explicit ``linear gamma0`` or ``decibel gamma0`` layers.  RGB ratio,
    SAR urban, and enhanced visualization products are ignored.
    """
    zip_path = Path(zip_path)
    best: Dict[str, Tuple[int, str] | None] = {tag: None for tag in S1_BASE_TAGS}
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()

    for name in names:
        low = name.lower()
        if not low.endswith((".tif", ".tiff")):
            continue
        if any(skip in low for skip in ("rgb_ratio", "rgb-ratio", "sar_urban", "sar-urban", "enhanced_visualization", "enhanced-visualization")):
            continue

        is_raw = _s1_is_raw_layer(low)
        is_decibel = "decibel" in low or "db" in low
        is_linear = "linear" in low
        has_gamma = "gamma0" in low or "gamma" in low

        # Priority: raw analytical layers are best; explicit linear gamma0 is
        # next; explicit decibel gamma0 is accepted for dB descriptors.
        for pol, prefix in (("vh", "VH"), ("vv", "VV")):
            if not _s1_pol_match(low, pol):
                continue
            if is_raw:
                tag = f"{prefix}_LINEAR"
                cand = (0, name)
                if best[tag] is None or cand[0] < best[tag][0]:
                    best[tag] = cand
            elif has_gamma and is_linear:
                tag = f"{prefix}_LINEAR"
                cand = (10, name)
                if best[tag] is None or cand[0] < best[tag][0]:
                    best[tag] = cand
            elif has_gamma and is_decibel:
                tag = f"{prefix}_DB"
                cand = (20, name)
                if best[tag] is None or cand[0] < best[tag][0]:
                    best[tag] = cand

    return {tag: (rec[1] if rec is not None else None) for tag, rec in best.items()}


def _normalize_rendered_s1(arr: np.ndarray, dtype_name: str, normalize_rendered_uint: bool = True) -> np.ndarray:
    """Convert a Sentinel-1 raster band to Float32.

    Analytical Float32 products are kept as-is.  Rendered integer products are
    normalized to 0..1 only when explicitly enabled.
    """
    out = arr.astype(np.float32)
    if normalize_rendered_uint and np.issubdtype(np.dtype(dtype_name), np.integer):
        info = np.iinfo(np.dtype(dtype_name))
        denom = float(info.max) if info.max else 1.0
        out = out / denom
    return out.astype(np.float32)


def _looks_like_binary_datamask(arr: np.ndarray) -> bool:
    """Return True if a raster band behaves like an EO Browser dataMask."""
    try:
        vals = arr[np.isfinite(arr)] if np.issubdtype(arr.dtype, np.floating) else arr.ravel()
        if vals.size == 0:
            return False
        sample = vals[: min(vals.size, 200000)]
        uniq = np.unique(sample)
        if uniq.size > 4:
            return False
        uniq_f = {float(x) for x in uniq.tolist()}
        return uniq_f.issubset({0.0, 1.0, 255.0})
    except Exception:
        return False


def read_s1_member_to_ref(
    zip_path: Path,
    inner_member: str,
    ref_profile: dict,
    normalize_rendered_uint: bool = True,
    treat_zero_as_nodata: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Read one Sentinel-1 member and align it to a reference grid.

    Returns a Float32 array on the reference grid and a Boolean validity mask.
    The function supports the simple EO Browser raw layout used in the new ZIPs:
    band 1 is the analytical backscatter value and band 2 is the dataMask.
    Invalid pixels are set to NaN in the returned array.
    """
    src_path = vsizip_path(zip_path, inner_member)
    h = int(ref_profile["height"])
    w = int(ref_profile["width"])
    with rasterio.open(src_path) as src:
        src_arr_raw = src.read(1)
        src_valid = src.read_masks(1) > 0

        # EO Browser raw analytical layers commonly contain dataMask as band 2.
        if src.count >= 2:
            try:
                band2 = src.read(2)
                if _looks_like_binary_datamask(band2):
                    src_valid &= band2 > 0
            except Exception:
                pass

        # Rendered RGBA layers may contain alpha as band 4. They are not preferred,
        # but this keeps backward compatibility with older exports.
        if src.count >= 4:
            try:
                src_valid &= src.read(4) > 0
            except Exception:
                pass
        if treat_zero_as_nodata:
            src_valid &= src_arr_raw != 0
        src_arr = _normalize_rendered_s1(src_arr_raw, src.dtypes[0], normalize_rendered_uint=normalize_rendered_uint)
        src_arr = np.where(src_valid, src_arr, np.nan).astype(np.float32)

        dst = np.full((h, w), np.nan, dtype=np.float32)
        valid_dst = np.zeros((h, w), dtype=np.uint8)
        if same_grid(src.profile, ref_profile):
            dst = src_arr.astype(np.float32)
            valid_dst = src_valid.astype(np.uint8)
        else:
            reproject(
                source=src_arr,
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_profile["transform"],
                dst_crs=ref_profile["crs"],
                src_nodata=np.nan,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
            reproject(
                source=src_valid.astype(np.uint8),
                destination=valid_dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_profile["transform"],
                dst_crs=ref_profile["crs"],
                src_nodata=0,
                dst_nodata=0,
                resampling=Resampling.nearest,
            )
        valid = valid_dst > 0
        dst[~valid] = np.nan
        return dst.astype(np.float32), valid


def local_mean_std_3x3(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute 3x3 local mean and standard deviation ignoring NaN values."""
    arr = arr.astype(np.float32)
    valid = np.isfinite(arr)
    arr0 = np.where(valid, arr, 0.0).astype(np.float64)
    sq0 = arr0 * arr0
    val0 = valid.astype(np.float64)
    padded = np.pad(arr0, 1, mode="constant", constant_values=0.0)
    padded_sq = np.pad(sq0, 1, mode="constant", constant_values=0.0)
    padded_valid = np.pad(val0, 1, mode="constant", constant_values=0.0)
    sum_arr = np.zeros(arr.shape, dtype=np.float64)
    sum_sq = np.zeros(arr.shape, dtype=np.float64)
    count = np.zeros(arr.shape, dtype=np.float64)
    for dr in range(3):
        for dc in range(3):
            sl = (slice(dr, dr + arr.shape[0]), slice(dc, dc + arr.shape[1]))
            sum_arr += padded[sl]
            sum_sq += padded_sq[sl]
            count += padded_valid[sl]
    mean = np.full(arr.shape, np.nan, dtype=np.float32)
    std = np.full(arr.shape, np.nan, dtype=np.float32)
    ok = count > 0
    mean[ok] = (sum_arr[ok] / count[ok]).astype(np.float32)
    var = np.maximum((sum_sq[ok] / count[ok]) - (sum_arr[ok] / count[ok]) ** 2, 0.0)
    std[ok] = np.sqrt(var).astype(np.float32)
    return mean, std


def _linear_to_db(arr: np.ndarray) -> np.ndarray:
    """Convert linear backscatter to dB, preserving NaN and non-positive as NaN."""
    out = np.full(arr.shape, np.nan, dtype=np.float32)
    ok = np.isfinite(arr) & (arr > 0)
    out[ok] = (10.0 * np.log10(arr[ok].astype(np.float64))).astype(np.float32)
    return out


def _db_to_linear(arr: np.ndarray) -> np.ndarray:
    """Convert dB backscatter to linear scale."""
    out = np.full(arr.shape, np.nan, dtype=np.float32)
    ok = np.isfinite(arr)
    out[ok] = np.power(10.0, arr[ok].astype(np.float64) / 10.0).astype(np.float32)
    return out


def build_s1_descriptor_stack(
    zip_path: Path,
    ref_profile: dict,
    normalize_rendered_uint: bool = True,
    treat_zero_as_nodata: bool = False,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, Dict[str, str | None]]:
    """Build a Sentinel-1 descriptor stack aligned to the reference grid.

    For the preferred ZIP layout, only raw VV and VH layers are required. They are
    read as linear backscatter and dB descriptors are computed internally.
    """
    members = s1_members_by_tag(zip_path)
    h = int(ref_profile["height"])
    w = int(ref_profile["width"])
    base: Dict[str, np.ndarray] = {}
    valid_any = np.zeros((h, w), dtype=bool)
    for tag in S1_BASE_TAGS:
        member = members.get(tag)
        if member is None:
            base[tag] = np.full((h, w), np.nan, dtype=np.float32)
            continue
        arr, valid = read_s1_member_to_ref(
            zip_path,
            member,
            ref_profile,
            normalize_rendered_uint=normalize_rendered_uint,
            treat_zero_as_nodata=treat_zero_as_nodata,
        )
        base[tag] = arr
        valid_any |= valid & np.isfinite(arr)

    vh_lin = base["VH_LINEAR"]
    vv_lin = base["VV_LINEAR"]
    vh_db = base["VH_DB"]
    vv_db = base["VV_DB"]

    # Preferred raw ZIPs provide only linear VV/VH. Derive dB internally.
    if not np.isfinite(vh_db).any() and np.isfinite(vh_lin).any():
        vh_db = _linear_to_db(vh_lin)
    if not np.isfinite(vv_db).any() and np.isfinite(vv_lin).any():
        vv_db = _linear_to_db(vv_lin)

    # If a user supplies dB-only inputs, recover approximate linear values for
    # ratios and local support.
    if not np.isfinite(vh_lin).any() and np.isfinite(vh_db).any():
        vh_lin = _db_to_linear(vh_db)
    if not np.isfinite(vv_lin).any() and np.isfinite(vv_db).any():
        vv_lin = _db_to_linear(vv_db)

    valid_any = np.isfinite(vh_lin) | np.isfinite(vv_lin) | np.isfinite(vh_db) | np.isfinite(vv_db)

    vv_minus_vh = np.full((h, w), np.nan, dtype=np.float32)
    ok = np.isfinite(vv_db) & np.isfinite(vh_db)
    vv_minus_vh[ok] = vv_db[ok] - vh_db[ok]

    vv_div_vh = np.full((h, w), np.nan, dtype=np.float32)
    ok = np.isfinite(vv_lin) & np.isfinite(vh_lin) & (vh_lin != 0)
    vv_div_vh[ok] = vv_lin[ok] / vh_lin[ok]

    vh_mean, vh_std = local_mean_std_3x3(vh_db)
    vv_mean, vv_std = local_mean_std_3x3(vv_db)

    out = {
        "S1_VH_DB": vh_db.astype(np.float32),
        "S1_VV_DB": vv_db.astype(np.float32),
        "S1_VH_LINEAR": vh_lin.astype(np.float32),
        "S1_VV_LINEAR": vv_lin.astype(np.float32),
        "S1_VV_MINUS_VH_DB": vv_minus_vh.astype(np.float32),
        "S1_VV_DIV_VH_LINEAR": vv_div_vh.astype(np.float32),
        "S1_VH_LOCAL_MEAN3": vh_mean.astype(np.float32),
        "S1_VH_LOCAL_STD3": vh_std.astype(np.float32),
        "S1_VV_LOCAL_MEAN3": vv_mean.astype(np.float32),
        "S1_VV_LOCAL_STD3": vv_std.astype(np.float32),
        "S1_VALID": valid_any.astype(np.float32),
    }
    return out, valid_any, members

def load_s1_inventory(cache_root: Path) -> pd.DataFrame:
    """Load a Phase 5 Sentinel-1 scene inventory if it exists."""
    path = Path(cache_root) / "s1_scene_inventory.csv"
    if not path.exists():
        return pd.DataFrame()
    inv = pd.read_csv(path)
    if "date" in inv.columns:
        inv["date"] = pd.to_datetime(inv["date"], errors="coerce")
    return inv


def find_nearest_s1_scene(s1_inventory: pd.DataFrame, target_date, max_abs_days: int = 45) -> pd.Series | None:
    """Return the nearest S1 inventory row to target_date within max_abs_days."""
    if s1_inventory is None or s1_inventory.empty or "date" not in s1_inventory.columns:
        return None
    target = pd.Timestamp(target_date)
    inv = s1_inventory.copy()
    inv = inv[pd.notna(inv["date"])]
    if inv.empty:
        return None
    if "cache_status" in inv.columns:
        inv = inv[inv["cache_status"].astype(str).str.lower().isin(["created", "rebuilt", "existing"])]
    if inv.empty:
        return None
    inv["abs_offset_days"] = (inv["date"] - target).abs().dt.days
    inv = inv[inv["abs_offset_days"] <= int(max_abs_days)]
    if inv.empty:
        return None
    inv = inv.sort_values(["abs_offset_days", "date", "scene"])
    return inv.iloc[0]


def summarize_stack_on_mask(stack: Dict[str, np.ndarray], mask: np.ndarray, band_names: Sequence[str]) -> Dict[str, float]:
    """Return basic mean/std/min/max summaries for selected stack bands over a mask."""
    out: Dict[str, float] = {}
    for name in band_names:
        arr = stack.get(name)
        if arr is None:
            vals = np.array([], dtype=np.float32)
        else:
            vals = arr[mask]
            vals = vals[np.isfinite(vals)]
        key = str(name).lower()
        if vals.size:
            out[f"{key}_mean"] = float(np.mean(vals))
            out[f"{key}_std"] = float(np.std(vals))
            out[f"{key}_min"] = float(np.min(vals))
            out[f"{key}_max"] = float(np.max(vals))
            out[f"{key}_valid_px"] = int(vals.size)
        else:
            out[f"{key}_mean"] = np.nan
            out[f"{key}_std"] = np.nan
            out[f"{key}_min"] = np.nan
            out[f"{key}_max"] = np.nan
            out[f"{key}_valid_px"] = 0
    return out

# Additional aliases for Algorithm #4 compatibility.
DEFAULT_S1_DESCRIPTOR_BANDS = S1_DESCRIPTOR_BANDS


def match_nearest_scene_by_date(target_date, inventory: pd.DataFrame, max_days: int = 45):
    """Compatibility wrapper returning nearest row in an inventory by date."""
    return find_nearest_s1_scene(inventory, target_date, max_abs_days=max_days)


# =============================================================================
# Phase 5 final compatibility overrides
# =============================================================================
DEFAULT_S1_DESCRIPTOR_BANDS = S1_DESCRIPTOR_BANDS


def s1_cache_scene_paths(cache_root: Path, scene_name) -> Dict[str, Path]:
    """Return canonical and alias paths for a cached Sentinel-1 descriptor scene."""
    root = Path(cache_root)
    key = cache_scene_key(scene_name)
    stack = root / "s1_descriptor_stacks" / f"{key}__s1_descriptor_stack.tif"
    valid = root / "s1_valid_masks" / f"{key}__s1_valid_mask.tif"
    return {
        "s1_stack": stack,
        "descriptor_stack": stack,
        "s1_descriptor_stack": stack,
        "s1_valid_mask": valid,
        "valid_mask": valid,
        "s1_valid": valid,
    }


def load_s1_inventory(cache_root: Path) -> pd.DataFrame:
    """Load Phase 5 Sentinel-1 scene inventory with standard compatibility columns."""
    path = Path(cache_root) / "s1_scene_inventory.csv"
    if not path.exists():
        return pd.DataFrame()
    inv = pd.read_csv(path)
    if "date" in inv.columns:
        inv["date"] = pd.to_datetime(inv["date"], errors="coerce")
    if "descriptor_stack" not in inv.columns:
        if "s1_stack_path" in inv.columns:
            inv["descriptor_stack"] = inv["s1_stack_path"]
        elif "s1_descriptor_stack_path" in inv.columns:
            inv["descriptor_stack"] = inv["s1_descriptor_stack_path"]
    if "valid_mask" not in inv.columns and "s1_valid_mask_path" in inv.columns:
        inv["valid_mask"] = inv["s1_valid_mask_path"]
    if "descriptor_stack_exists" not in inv.columns and "descriptor_stack" in inv.columns:
        inv["descriptor_stack_exists"] = inv["descriptor_stack"].astype(str).map(lambda p: Path(p).exists())
    return inv


def find_nearest_s1_scene(s1_inventory: pd.DataFrame, target_date, max_abs_days: int = 45) -> pd.Series | None:
    """Return nearest cached Sentinel-1 inventory row within max_abs_days."""
    if s1_inventory is None or s1_inventory.empty or "date" not in s1_inventory.columns:
        return None
    inv = s1_inventory.copy()
    inv["date"] = pd.to_datetime(inv["date"], errors="coerce")
    inv = inv.dropna(subset=["date"])
    if "cache_status" in inv.columns:
        inv = inv[inv["cache_status"].astype(str).str.lower().isin(["created", "rebuilt", "existing"])]
    if "descriptor_stack_exists" in inv.columns:
        inv = inv[inv["descriptor_stack_exists"].astype(bool)]
    if inv.empty:
        return None
    target = pd.Timestamp(target_date)
    inv["abs_offset_days"] = (inv["date"] - target).abs().dt.days
    inv = inv[inv["abs_offset_days"] <= int(max_abs_days)]
    if inv.empty:
        return None
    sort_cols = ["abs_offset_days", "date"]
    if "scene" in inv.columns:
        sort_cols.append("scene")
    return inv.sort_values(sort_cols).iloc[0]


def match_nearest_scene_by_date(target_date, inventory: pd.DataFrame, max_days: int = 45, date_col: str = "date", require_available: bool = True):
    """Compatibility wrapper returning nearest row dict by date."""
    if inventory is None or inventory.empty or date_col not in inventory.columns:
        return None
    inv = inventory.copy()
    inv[date_col] = pd.to_datetime(inv[date_col], errors="coerce")
    inv = inv.dropna(subset=[date_col])
    if require_available and "descriptor_stack_exists" in inv.columns:
        inv = inv[inv["descriptor_stack_exists"].astype(bool)]
    if inv.empty:
        return None
    target = pd.Timestamp(target_date)
    inv["_abs_days"] = (inv[date_col] - target).abs().dt.days
    inv["_signed_days"] = (inv[date_col] - target).dt.days
    inv = inv[inv["_abs_days"] <= int(max_days)]
    if inv.empty:
        return None
    sort_cols = ["_abs_days", date_col]
    if "scene" in inv.columns:
        sort_cols.append("scene")
    return inv.sort_values(sort_cols).iloc[0].to_dict()
