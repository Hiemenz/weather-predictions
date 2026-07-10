"""Decode MRMS .grib2.gz files into stored national reflectivity grids.

Needs the `mrms` dependency group (`poetry install --with mrms`) — this imports
cfgrib, which requires the `eccodes` C library installed separately:
  macOS:  brew install eccodes
  Debian: apt install libeccodes-dev

MRMS CONUS composite grid specs (verified against MRMS documentation):
  - 3500 rows × 7000 columns at 0.01° / ~1 km resolution
  - Bottom-left corner: 20.005°N, 129.995°W
  - Top-right corner:   54.995°N,  60.005°W
  - Longitude convention in GRIB2: 0–360 (converted to -180..180 here)

MRMS uses local NCEP GRIB2 parameter tables, so cfgrib may name the variable
"unknown" rather than a standard WMO name — that's expected and handled below.
"""

from __future__ import annotations

import gzip
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

_FILENAME_RE = re.compile(r"MRMS_.+_(\d{8})-(\d{6})\.grib2(?:\.gz)?$")

# MRMS sentinel values: -999 = no data/missing, -99 = no coverage.
_MISSING_THRESHOLD = -900.0


class MrmsProcessingError(RuntimeError):
    pass


def parse_mrms_timestamp(path: Path) -> datetime:
    """Extract UTC timestamp from an MRMS filename,
    e.g. "MRMS_MergedReflectivityQCComposite_00.50_20260710-000040.grib2.gz"
    -> 2026-07-10T00:00:40Z."""
    match = _FILENAME_RE.search(path.name)
    if not match:
        raise MrmsProcessingError(f"Unrecognized MRMS filename: {path.name}")
    return datetime.strptime(f"{match.group(1)}{match.group(2)}", "%Y%m%d%H%M%S").replace(
        tzinfo=timezone.utc
    )


def decode_mrms_grib2(file_path: Path) -> dict[str, Any]:
    """Decode one MRMS .grib2.gz (or .grib2) into a national reflectivity grid.

    Returns a dict with a (nlat, nlon) float32 array in dBZ (NaN where no data),
    plus the lat/lon bounding box needed to place it on a map.
    """
    import cfgrib

    timestamp = parse_mrms_timestamp(file_path)

    # cfgrib cannot read gzip-compressed GRIB2 directly — decompress to a
    # temporary .grib2 file, read, then clean up.
    if file_path.suffix == ".gz":
        raw_bytes = gzip.decompress(file_path.read_bytes())
        tmp = tempfile.NamedTemporaryFile(suffix=".grib2", delete=False)
        try:
            tmp.write(raw_bytes)
            tmp.flush()
            tmp.close()
            return _decode_grib2_file(Path(tmp.name), timestamp)
        finally:
            Path(tmp.name).unlink(missing_ok=True)
    else:
        return _decode_grib2_file(file_path, timestamp)


def _decode_grib2_file(grib_path: Path, timestamp: datetime) -> dict[str, Any]:
    import cfgrib

    # open_datasets (plural) is more permissive than open_dataset — it
    # surfaces each GRIB2 message as a separate xarray Dataset regardless of
    # whether cfgrib can resolve the local NCEP parameter table.
    datasets = cfgrib.open_datasets(str(grib_path), indexpath="")
    if not datasets:
        raise MrmsProcessingError(f"cfgrib found no datasets in {grib_path}")

    ds = datasets[0]
    var_names = list(ds.data_vars)
    if not var_names:
        raise MrmsProcessingError(f"No data variables in MRMS dataset from {grib_path}")

    da = ds[var_names[0]]
    refl = da.values.astype(np.float32)

    # Replace MRMS sentinel missing values with NaN.
    refl[refl <= _MISSING_THRESHOLD] = np.nan

    # Extract 1-D lat/lon arrays from the dataset coordinates.
    lats = ds["latitude"].values
    lons = ds["longitude"].values

    # cfgrib may return 2D coordinate arrays for non-LatLon projections;
    # collapse to 1D by taking the first column/row.
    if lats.ndim == 2:
        lats = lats[:, 0]
        lons = lons[0, :]

    # MRMS GRIB2 stores longitude as 0–360; convert to -180..180.
    if lons.max() > 180:
        lons = lons - 360.0

    # Ensure the array is stored south-to-north (ascending latitude).
    if lats[0] > lats[-1]:
        refl = refl[::-1, :]
        lats = lats[::-1]

    return {
        "source": "MRMS_CONUS",
        "timestamp": timestamp.isoformat(),
        "lat_min": float(lats.min()),
        "lat_max": float(lats.max()),
        "lon_min": float(lons.min()),
        "lon_max": float(lons.max()),
        "nlat": int(refl.shape[0]),
        "nlon": int(refl.shape[1]),
        "reflectivity_dbz": refl,
    }


def save_mrms_grid(frame: dict[str, Any], dest_dir: Path) -> Path:
    """Save a decoded MRMS frame as a compressed .npz, named by timestamp."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    ts_compact = frame["timestamp"].replace(":", "").replace("-", "")
    dest_path = dest_dir / f"MRMS_CONUS_{ts_compact}.npz"
    np.savez_compressed(
        dest_path,
        reflectivity_dbz=frame["reflectivity_dbz"],
        source=frame["source"],
        timestamp=frame["timestamp"],
        lat_min=frame["lat_min"],
        lat_max=frame["lat_max"],
        lon_min=frame["lon_min"],
        lon_max=frame["lon_max"],
        nlat=frame["nlat"],
        nlon=frame["nlon"],
    )
    return dest_path


def load_mrms_grid(path: Path) -> dict[str, Any]:
    """Load a frame saved by `save_mrms_grid` back into the same dict shape."""
    with np.load(path) as data:
        return {
            "source": str(data["source"]),
            "timestamp": str(data["timestamp"]),
            "lat_min": float(data["lat_min"]),
            "lat_max": float(data["lat_max"]),
            "lon_min": float(data["lon_min"]),
            "lon_max": float(data["lon_max"]),
            "nlat": int(data["nlat"]),
            "nlon": int(data["nlon"]),
            "reflectivity_dbz": data["reflectivity_dbz"],
        }
