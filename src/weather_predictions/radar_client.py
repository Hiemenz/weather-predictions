"""Client for NOAA's raw NEXRAD Level II radar archive on AWS Open Data.

Public bucket, no AWS credentials needed — but list/get still go through the
`aws` CLI rather than boto3/s3fs, for two reasons: (1) `arm-pyart` pulls in
s3fs, whose supported botocore range conflicts with a plain `boto3` pin, and
(2) `aws s3 cp` was measured at full network speed in this environment while
plain HTTPS clients (`requests`, stdlib `urllib`) were throttled to ~1/30th
speed on similarly sized files from other NOAA endpoints — so it's also just
the faster path here.

One volume scan is a full 360° sweep at multiple elevation angles, arriving
roughly every 5 minutes, ~12-15MB each. Docs: https://registry.opendata.aws/noaa-nexrad/
"""

from __future__ import annotations

import shutil
import subprocess
from datetime import date
from pathlib import Path

from weather_predictions.config import RADAR_S3_BUCKET, RADAR_STATION_ID

_TIMEOUT = 120


class RadarClientError(RuntimeError):
    pass


def _require_aws_cli() -> None:
    if not shutil.which("aws"):
        raise RadarClientError(
            "The `aws` CLI is required for NEXRAD access (install via `brew install awscli` "
            "or `apt install awscli`). No AWS account or credentials needed — the bucket is public."
        )


def list_scans(day: date, station: str = RADAR_STATION_ID) -> list[str]:
    """List S3 keys for all volume scans at a station on a given (UTC) date."""
    _require_aws_cli()
    prefix = f"{day:%Y/%m/%d}/{station}/"
    result = subprocess.run(
        ["aws", "s3", "ls", f"s3://{RADAR_S3_BUCKET}/{prefix}", "--no-sign-request"],
        capture_output=True,
        text=True,
        timeout=_TIMEOUT,
    )
    if result.returncode != 0:
        raise RadarClientError(f"aws s3 ls failed for {prefix}: {result.stderr.strip()[:300]}")

    keys = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if not parts:
            continue
        filename = parts[-1]
        # Skip metadata sidecar files (e.g. "..._V06_MDM") — we only want the volume scans.
        if filename.endswith("_MDM"):
            continue
        keys.append(prefix + filename)
    return sorted(keys)


def download_scan(key: str, dest_dir: Path) -> Path:
    """Download one volume scan by its S3 key, returning the local file path."""
    _require_aws_cli()
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / Path(key).name
    result = subprocess.run(
        ["aws", "s3", "cp", f"s3://{RADAR_S3_BUCKET}/{key}", str(dest_path), "--no-sign-request"],
        capture_output=True,
        text=True,
        timeout=_TIMEOUT,
    )
    if result.returncode != 0:
        raise RadarClientError(f"aws s3 cp failed for {key}: {result.stderr.strip()[:300]}")
    return dest_path


def latest_scan_key(station: str = RADAR_STATION_ID) -> str | None:
    """Most recent available volume scan, checking today then falling back to yesterday
    (near UTC midnight, "today" may not have any scans listed yet)."""
    from datetime import timedelta

    today = date.today()
    for day in (today, today - timedelta(days=1)):
        keys = list_scans(day, station)
        if keys:
            return keys[-1]
    return None
