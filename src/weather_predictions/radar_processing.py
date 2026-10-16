"""Decode a raw NEXRAD Level II volume scan into a fixed-size reflectivity grid.

Raw scans are 3D polar-coordinate sweeps (multiple elevation angles, each a
360° ring of range gates) — not directly usable as a time-series "image" for
a nowcasting model. This projects the lowest elevation sweep onto a flat,
fixed-resolution Cartesian grid centered on the radar, which is the standard
representation for radar-based ML (e.g. the MRMS/nowcasting literature).

Py-ART is only imported inside `decode_reflectivity_grid`, not at module
scope: `save_grid`/`load_grid`/`parse_scan_timestamp` don't need it at all,
and several consumers (radar_nowcast.py, radar_image.py) only ever load
already-decoded grids — an eager top-level `import pyart` would force the
`radar` dependency group (and Cartopy's ARM wheel problem) onto those too.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

_FILENAME_RE = re.compile(r"^(?P<station>[A-Z]{4})(?P<ts>\d{8}_\d{6})_V\d+")


class RadarProcessingError(RuntimeError):
    pass


def parse_scan_timestamp(path: Path) -> tuple[str, datetime]:
    """Extract station id and UTC timestamp from a NEXRAD filename, e.g.
    "KOHX20260704_124343_V06" -> ("KOHX", 2026-07-04T12:43:43Z)."""
    match = _FILENAME_RE.match(path.name)
    if not match:
        raise RadarProcessingError(f"Unrecognized NEXRAD filename: {path.name}")
    station = match.group("station")
    ts = datetime.strptime(match.group("ts"), "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
    return station, ts


def decode_reflectivity_grid(
    file_path: Path,
    grid_km: float = 200,
    resolution_km: float = 1,
) -> dict[str, Any]:
    """Project the lowest-elevation reflectivity sweep onto a square Cartesian grid.

    Returns a dict with a (n, n) float32 array in dBZ (NaN where no data),
    plus enough metadata to place it on a map and line it up with other frames.
    """
    import pyart

    station, timestamp = parse_scan_timestamp(file_path)
    radar = pyart.io.read_nexrad_archive(str(file_path))

    n = int(round(2 * grid_km / resolution_km))
    limit_m = grid_km * 1000
    grid = pyart.map.grid_from_radars(
        (radar,),
        grid_shape=(1, n, n),
        grid_limits=((0, 1000), (-limit_m, limit_m), (-limit_m, limit_m)),
        fields=["reflectivity"],
    )

    refl = grid.fields["reflectivity"]["data"][0]  # drop the single z level
    refl_filled = np.ma.filled(refl, np.nan).astype(np.float32)

    return {
        "station": station,
        "timestamp": timestamp.isoformat(),
        "grid_km": grid_km,
        "resolution_km": resolution_km,
        "latitude": float(grid.origin_latitude["data"][0]),
        "longitude": float(grid.origin_longitude["data"][0]),
        "reflectivity_dbz": refl_filled,
    }


def save_grid(frame: dict[str, Any], dest_dir: Path) -> Path:
    """Save a decoded frame as a compressed .npz, named by station + timestamp."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    ts_compact = frame["timestamp"].replace(":", "").replace("-", "")
    dest_path = dest_dir / f"{frame['station']}_{ts_compact}.npz"
    np.savez_compressed(
        dest_path,
        reflectivity_dbz=frame["reflectivity_dbz"],
        station=frame["station"],
        timestamp=frame["timestamp"],
        grid_km=frame["grid_km"],
        resolution_km=frame["resolution_km"],
        latitude=frame["latitude"],
        longitude=frame["longitude"],
    )
    return dest_path


def load_grid(path: Path) -> dict[str, Any]:
    """Load a frame saved by `save_grid` back into the same dict shape."""
    with np.load(path) as data:
        return {
            "station": str(data["station"]),
            "timestamp": str(data["timestamp"]),
            "grid_km": float(data["grid_km"]),
            "resolution_km": float(data["resolution_km"]),
            "latitude": float(data["latitude"]),
            "longitude": float(data["longitude"]),
            "reflectivity_dbz": data["reflectivity_dbz"],
        }
