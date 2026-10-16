"""Client for NOAA's MRMS national radar composite on AWS Open Data.

MRMS (Multi-Radar Multi-Sensor) is a pre-mosaiced CONUS reflectivity product
at 1 km / 2-minute resolution — one ~1.5MB .grib2.gz file covers the whole
country, vs ~160 per-station NEXRAD downloads (~2GB total) for the same moment.
Public bucket, no AWS credentials needed.

Docs: https://registry.opendata.aws/noaa-mrms-pds/
"""

from __future__ import annotations

import re
import shutil
import subprocess
from datetime import date, timedelta
from pathlib import Path

from weather_predictions.config import MRMS_PRODUCT, MRMS_REGION, MRMS_S3_BUCKET

_TIMEOUT = 120
_FILE_RE = re.compile(r"^MRMS_.+_\d{8}-\d{6}\.grib2\.gz$")


class MrmsClientError(RuntimeError):
    pass


def _require_aws_cli() -> None:
    if not shutil.which("aws"):
        raise MrmsClientError(
            "The `aws` CLI is required for MRMS access (install via `brew install awscli` "
            "or `apt install awscli`). No AWS account or credentials needed — the bucket is public."
        )


def list_mrms_scans(day: date, product: str = MRMS_PRODUCT) -> list[str]:
    """List S3 keys for all scans of an MRMS product on a given (UTC) date."""
    _require_aws_cli()
    prefix = f"{MRMS_REGION}/{product}/{day:%Y%m%d}/"
    result = subprocess.run(
        ["aws", "s3", "ls", f"s3://{MRMS_S3_BUCKET}/{prefix}", "--no-sign-request"],
        capture_output=True,
        text=True,
        timeout=_TIMEOUT,
    )
    if result.returncode != 0:
        raise MrmsClientError(f"aws s3 ls failed for {prefix}: {result.stderr.strip()[:300]}")

    keys = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if not parts:
            continue
        filename = parts[-1]
        if _FILE_RE.match(filename):
            keys.append(prefix + filename)
    return sorted(keys)


def download_mrms_scan(key: str, dest_dir: Path) -> Path:
    """Download one MRMS scan by its S3 key, returning the local .grib2.gz path."""
    _require_aws_cli()
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / Path(key).name
    result = subprocess.run(
        ["aws", "s3", "cp", f"s3://{MRMS_S3_BUCKET}/{key}", str(dest_path), "--no-sign-request"],
        capture_output=True,
        text=True,
        timeout=_TIMEOUT,
    )
    if result.returncode != 0:
        raise MrmsClientError(f"aws s3 cp failed for {key}: {result.stderr.strip()[:300]}")
    return dest_path


def latest_mrms_scan_key(product: str = MRMS_PRODUCT) -> str | None:
    """Most recent available scan of an MRMS product, checking today then
    falling back to yesterday."""
    today = date.today()
    for day in (today, today - timedelta(days=1)):
        keys = list_mrms_scans(day, product)
        if keys:
            return keys[-1]
    return None
