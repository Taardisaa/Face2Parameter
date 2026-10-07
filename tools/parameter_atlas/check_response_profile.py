"""Reject a calibration if a fresh game geometry snapshot has a different state."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from explore_live_response import (
    calibration_identity,
    canonical_hash,
    identity_mismatches,
)


def check_profile(catalog_path, snapshot_path, baseline):
    raw_catalog = Path(catalog_path).read_bytes()
    catalog = json.loads(raw_catalog.decode("utf-8-sig"))
    if catalog.get("complete") is not True or baseline not in catalog.get(
        "baselines", {}
    ):
        raise ValueError("A completed explorer and an existing baseline are required")
    raw = Path(snapshot_path).read_bytes()
    if Path(snapshot_path).suffix == ".gz":
        raw = gzip.decompress(raw)
    snapshot = json.loads(raw.decode("utf-8-sig"))
    if snapshot.get("schema_version") != 1 or snapshot.get(
        "frame_count"
    ) != snapshot.get("frame_count_end"):
        raise ValueError("A stable schema1 native geometry snapshot is required")
    expected = catalog["baselines"][baseline]["identity"]
    if canonical_hash(expected) != catalog["baselines"][baseline]["identity_sha256"]:
        raise ValueError("Calibration identity SHA-256 mismatch")
    actual = calibration_identity(snapshot, expected)
    changed = identity_mismatches(expected, actual)
    return {
        "compatible_public_configuration": not changed,
        "changed_fields": changed,
        "catalog_sha256": hashlib.sha256(raw_catalog).hexdigest(),
        "snapshot_sha256": hashlib.sha256(raw).hexdigest(),
        "baseline": baseline,
        "scope": "Public input/source compatibility only; new game response and repeat stability still require measurement",
        "action": "Measure fresh local responses before applying suggestions"
        if not changed
        else "Select a matching profile or recalibrate; do not reuse these gains",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", type=Path)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--baseline", default="card_input")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = check_profile(args.catalog, args.snapshot, args.baseline)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))
    raise SystemExit(0 if result["compatible_public_configuration"] else 2)


if __name__ == "__main__":
    main()
