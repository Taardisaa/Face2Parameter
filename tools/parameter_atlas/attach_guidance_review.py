"""Attach independently reviewed points to a fresh explorer, without rebuilding geometry.

The input package, review, held-out manifest and plan remain unchanged. Prediction
failures may be displayed, but incomplete or untrusted source evidence is rejected.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
from pathlib import Path

from explore_live_response import canonical_hash
from explorer_template import TEMPLATE
from live_response import load_receipt
from view_live_response import UnverifiedResponse


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def number(value, *, positive=False, nonnegative=False):
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or (positive and value <= 0)
        or (nonnegative and value < 0)
    ):
        raise UnverifiedResponse("Finite numeric guidance value required")
    return float(value)


def same(a, b, reason):
    if not math.isclose(number(a), number(b), rel_tol=1e-8, abs_tol=1e-12):
        raise UnverifiedResponse(reason)


def check_receipt(receipt, directory):
    return load_receipt(receipt, directory)


def payload_file(entry, directory):
    if not re.fullmatch(r"data/[A-Za-z0-9_.-]+\.js", entry.get("file", "")):
        raise UnverifiedResponse("Unsafe explorer data path")
    path = directory / entry["file"]
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != entry.get("sha256"):
        raise UnverifiedResponse("Explorer data SHA-256 mismatch")
    text = raw.decode("utf-8-sig").strip()
    if not text.startswith("window.HS2_RESPONSE(") or not text.endswith(");"):
        raise UnverifiedResponse("Unsupported explorer callback")
    data = json.loads(text[len("window.HS2_RESPONSE(") : -2])
    for key in ("id", "kind", "baseline_name", "control", "bone", "channel", "axis"):
        if data.get(key) != entry.get(key):
            raise UnverifiedResponse("Explorer entry and payload metadata differ")
    if data.get("mesh") != "o_head":
        raise UnverifiedResponse("Explicit head-surface payload required")
    if data.get("local_response", {}).get("estimate_verified") is not False:
        raise UnverifiedResponse("Arbitrary-target estimates must remain unverified")
    return data


def unique_signs(rows):
    result = {}
    for row in rows:
        key = (row.get("entry_id"), row.get("sign"))
        if (
            not isinstance(key[0], str)
            or type(key[1]) is not int
            or key[1] not in (-1, 1)
            or key in result
        ):
            raise UnverifiedResponse("Duplicate or invalid guidance direction")
        result[key] = row
    return result


def attach_review(catalog_path, review_path, manifest_path, out):
    catalog_path, review_path, manifest_path = (
        Path(value).resolve() for value in (catalog_path, review_path, manifest_path)
    )
    out = Path(out).resolve()
    if out.exists():
        raise ValueError("Use a fresh guidance explorer directory")
    catalog = json.loads(catalog_path.read_text(encoding="utf-8-sig"))
    review = json.loads(review_path.read_text(encoding="utf-8-sig"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if (
        review.get("schema_version") != 1
        or review.get("analysis_mesh_scope") != ["o_head"]
        or review.get("public_configuration_restored") is not True
        or catalog.get("complete") is not True
        or manifest.get("complete") is not True
        or manifest.get("public_configuration_restored") is not True
        or manifest.get("restoration_errors")
    ):
        raise UnverifiedResponse(
            "Complete, restored, explicit head-scope review required"
        )
    if review.get("source_manifest_sha256") != digest(manifest_path):
        raise UnverifiedResponse("Held-out manifest SHA-256 mismatch")
    if review.get("catalog_receipt", {}).get("sha256") != digest(catalog_path):
        raise UnverifiedResponse("Review binds another explorer catalog")
    if review.get("plan_receipt") != manifest.get("plan_receipt"):
        raise UnverifiedResponse("Review and manifest plan receipt differ")
    plan = check_receipt(review["plan_receipt"], manifest_path.parent)
    if plan.get("schema_version") != 1 or plan.get("baselines") != catalog["baselines"]:
        raise UnverifiedResponse("Unsupported plan or changed baseline declarations")
    if plan.get("catalog_receipt") != review.get("catalog_receipt"):
        raise UnverifiedResponse("Plan binds another explorer catalog")
    if check_receipt(plan["catalog_receipt"], manifest_path.parent) != catalog:
        raise UnverifiedResponse("Declared explorer catalog changed")
    training = check_receipt(plan["training_manifest_receipt"], manifest_path.parent)
    training_sha = plan["training_manifest_receipt"]["sha256"]
    if (
        review.get("training_manifest_sha256") != training_sha
        or catalog.get("provenance", {}).get("source_manifest_sha256") != training_sha
        or training.get("complete") is not True
    ):
        raise UnverifiedResponse("Training manifest binding or completeness failed")
    thresholds = {
        "geometry_normalized": 1e-5,
        "repeat_normalized": 1e-5,
        "relative_prediction_error": 0.05,
        "noise_multiplier": 3,
        "baseline_reuse_normalized": 1e-5,
    }
    if review.get("thresholds") != thresholds or plan.get("thresholds") != thresholds:
        raise UnverifiedResponse("Predeclared numerical gates changed")
    if not review.get("source_receipts"):
        raise UnverifiedResponse("Executed reviewer source receipts required")
    for source in review["source_receipts"]:
        source_path = Path(source["path"])
        if not source_path.is_absolute():
            source_path = review_path.parent / source_path
        if digest(source_path) != source["sha256"]:
            raise UnverifiedResponse("Executed reviewer source SHA-256 mismatch")

    entries = {entry["id"]: entry for entry in catalog["entries"]}
    if len(entries) != len(catalog["entries"]):
        raise UnverifiedResponse("Duplicate explorer entry")
    points = unique_signs(review.get("entries", []) + review.get("skipped", []))
    declarations = unique_signs(plan.get("entries", []) + plan.get("skipped", []))
    expected = {(entry, sign) for entry in entries for sign in (-1, 1)}
    if set(points) != expected or set(declarations) != expected:
        raise UnverifiedResponse(
            "Every packaged entry requires both reviewed directions"
        )
    if set(review.get("baselines", {})) != set(catalog["baselines"]):
        raise UnverifiedResponse("Review baseline coverage differs")
    for name, baseline in catalog["baselines"].items():
        checked = review["baselines"][name]
        identity_sha = canonical_hash(baseline["identity"])
        if (
            identity_sha != baseline["identity_sha256"]
            or checked.get("identity_sha256") != identity_sha
        ):
            raise UnverifiedResponse("Baseline identity SHA-256 mismatch")
        if (
            checked.get("baseline_reuse_pass") is not True
            or checked.get("repeat_drift_pass") is not True
        ):
            raise UnverifiedResponse("Fresh baseline reuse or repeat-drift gate failed")
        number(checked["baseline_reuse_normalized"], nonnegative=True)
        if (
            checked["baseline_reuse_normalized"] > 1e-5
            or max(
                number(checked["training_repeat_noise"], nonnegative=True),
                number(checked["fresh_repeat_noise"], nonnegative=True),
            )
            > 1e-5
            or type(checked["repeat_count"]) is not int
            or checked["repeat_count"] < 1
        ):
            raise UnverifiedResponse(
                "Baseline numeric stability contradicts passed gates"
            )
        same(
            checked["head_diagonal"], baseline["head_diagonal"], "Head diagonal differs"
        )
        same(
            checked["training_repeat_noise"],
            baseline["repeat_drift_normalized"],
            "Training noise differs",
        )

    payloads = {}
    skipped_keys = set(unique_signs(review.get("skipped", [])))
    planned_skips = set(unique_signs(plan.get("skipped", [])))
    if skipped_keys != planned_skips:
        raise UnverifiedResponse("Noise skips differ from predeclared plan")
    for entry_id, entry in entries.items():
        data = payload_file(entry, catalog_path.parent)
        baseline = catalog["baselines"][entry["baseline_name"]]
        checked = review["baselines"][entry["baseline_name"]]
        if data.get("identity") != baseline["identity"]:
            raise UnverifiedResponse("Payload complete input identity differs")
        validation = {}
        for sign in (-1, 1):
            key = (entry_id, sign)
            row, declared = points[key], declarations[key]
            gain = number(
                data["local_response"]["left_gain" if sign < 0 else "right_gain"],
                nonnegative=True,
            )
            same(
                row["training_gain"],
                gain,
                "Reviewed training gain differs from payload",
            )
            if key in skipped_keys:
                if (
                    row.get("noise_guard_verified") is not True
                    or row.get("verified_prediction") is not False
                ):
                    raise UnverifiedResponse(
                        "Noise skip must reject a finite verified recommendation"
                    )
                for field, value in declared.items():
                    if row.get(field) != value:
                        raise UnverifiedResponse("Noise skip declaration changed")
                probe = number(
                    data["local_response"]["minus_step" if sign < 0 else "plus_step"],
                    positive=True,
                )
                target = min(baseline["head_diagonal"] * 0.0001, gain * probe * 0.4)
                if (
                    gain > 0
                    and target
                    > 3 * baseline["head_diagonal"] * checked["training_repeat_noise"]
                ):
                    raise UnverifiedResponse(
                        "Noise skip is above the measured noise floor"
                    )
                validation[str(sign)] = {
                    **row,
                    "input_trusted": False,
                    "status": "noise_guarded",
                    "verified_prediction": False,
                }
                continue
            if (
                row.get("input_trusted") is not True
                or row.get("baseline_reuse_pass") is not True
            ):
                raise UnverifiedResponse("Held-out input or baseline gate did not pass")
            if (
                row.get("kind") != entry["kind"]
                or row.get("baseline_name") != entry["baseline_name"]
            ):
                raise UnverifiedResponse("Held-out point changes the entry identity")
            if row.get("baseline_identity_sha256") != baseline["identity_sha256"]:
                raise UnverifiedResponse(
                    "Held-out point binds another complete configuration"
                )
            for field in ("signed_step", "target_units"):
                same(row[field], declared[field], "Held-out point differs from plan")
            same(
                row["heldout_value"],
                declared["value"],
                "Held-out parameter value differs from plan",
            )
            same(
                row["heldout_value"],
                data["baseline_level"] + row["signed_step"],
                "Held-out signed step differs from baseline",
            )
            signed_step = number(row["signed_step"])
            if signed_step * sign <= 0:
                raise UnverifiedResponse("Held-out signed direction differs")
            target = number(row["target_units"], positive=True)
            probe = number(
                data["local_response"]["minus_step" if sign < 0 else "plus_step"],
                positive=True,
            )
            if gain <= 0:
                raise UnverifiedResponse(
                    "Zero gain cannot produce a finite held-out recommendation"
                )
            same(
                target,
                min(baseline["head_diagonal"] * 0.0001, gain * probe * 0.4),
                "Held-out target changed from predeclared local policy",
            )
            same(
                signed_step,
                sign * target / gain,
                "Held-out step differs from training gain",
            )
            same(
                declared["requested_target_percent"],
                0.01,
                "Requested target policy changed",
            )
            same(declared["max_probe_fraction"], 0.4, "Probe fraction policy changed")
            if (
                target
                <= 3 * baseline["head_diagonal"] * checked["training_repeat_noise"]
            ):
                raise UnverifiedResponse(
                    "Finite held-out recommendation is below training noise"
                )
            actual = number(row["actual_max"], nonnegative=True)
            error = number(row["max_vector_error"], nonnegative=True)
            budget = number(row["error_budget"], nonnegative=True)
            same(
                row["relative_target_error"],
                abs(actual - target) / target,
                "Relative target error differs",
            )
            same(
                budget,
                0.05 * target
                + 3
                * baseline["head_diagonal"]
                * max(checked["training_repeat_noise"], checked["fresh_repeat_noise"]),
                "Prediction error budget changed",
            )
            if type(row.get("verified_prediction")) is not bool or row[
                "verified_prediction"
            ] != (error <= budget and abs(actual - target) <= budget):
                raise UnverifiedResponse(
                    "Prediction status contradicts its numerical gates"
                )
            if (
                row.get("training_case_receipt")
                not in data["verification"]["case_receipts"]
            ):
                raise UnverifiedResponse("Training witness absent from payload")
            validation[str(sign)] = copy.deepcopy(row)
        data["local_response"]["heldout_validation_by_sign"] = validation
        data["local_response"]["estimate_verified"] = False
        data["verification"]["guidance_review_receipt"] = {
            "path": str(review_path),
            "sha256": digest(review_path),
        }
        payloads[entry_id] = data

    for name in catalog["baselines"]:
        native = {
            key
            for key, entry in entries.items()
            if entry["kind"] == "native" and entry["baseline_name"] == name
        }
        verified = sum(
            points[(key, sign)].get("verified_prediction") is True
            for key in native
            for sign in (-1, 1)
        )
        declared_coverage = {
            "native_controls": len(native),
            "verified_native_signs": verified,
            "all59_both_signs_verified": len(native) == 59 and verified == 118,
        }
        if review.get("coverage", {}).get(name) != declared_coverage:
            raise UnverifiedResponse(
                "Coverage counters contradict individually reviewed directions"
            )
    result = copy.deepcopy(catalog)
    result["provenance"]["guidance_overlay"] = {
        "catalog_receipt": {"path": str(catalog_path), "sha256": digest(catalog_path)},
        "review_receipt": {"path": str(review_path), "sha256": digest(review_path)},
        "heldout_manifest_receipt": {
            "path": str(manifest_path),
            "sha256": digest(manifest_path),
        },
        "attacher_source_sha256": digest(__file__),
        "template_source_sha256": digest(
            Path(__file__).with_name("explorer_template.py")
        ),
        "coverage": review.get("coverage", {}),
        "scope": "Sign-specific independently sampled points only; arbitrary target estimates and intervals remain unverified",
    }
    (out / "data").mkdir(parents=True)
    for entry in result["entries"]:
        raw = json.dumps(
            payloads[entry["id"]],
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).replace("<", "\\u003c")
        target_path = out / entry["file"]
        target_path.write_text("window.HS2_RESPONSE(" + raw + ");\n", encoding="utf-8")
        entry["training_data_sha256"] = entry["sha256"]
        entry["sha256"] = digest(target_path)
    catalog_json = json.dumps(
        result, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).replace("<", "\\u003c")
    (out / "catalog.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (out / "index.html").write_text(
        TEMPLATE.replace("CATALOG_JSON", catalog_json), encoding="utf-8"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = attach_review(args.catalog, args.review, args.manifest, args.out)
    print(
        json.dumps(
            {
                "html": str(args.out.resolve() / "index.html"),
                "entries": len(result["entries"]),
                "estimate_verified": False,
            }
        )
    )


if __name__ == "__main__":
    main()
