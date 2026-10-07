"""Certify independently replayed actual Apply calls and ordered chain segments.

Input is a stopped complete MakerAbmxTrace export. No bridge or game calls.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
try:
    from .model import (CACHE_FIELDS, FLAG_FIELDS, ReplayRejected, bool_value, cache_state,
                        finite, replay_apply, require, transform)
except ImportError:  # Direct script execution retains the same model/contract.
    from model import (CACHE_FIELDS, FLAG_FIELDS, ReplayRejected, bool_value, cache_state,
                       finite, replay_apply, require, transform)

THRESHOLDS = {"position_max_abs": 1e-6, "scale_max_abs": 1e-6, "quaternion_sign_equivalent_max_abs": 2e-6,
              "rotation_angle_deg": .001, "cache_float_max_abs": 1e-6}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cache_errors(expected, actual):
    expected, actual = cache_state(expected), cache_state(actual)
    floats = {key: float(np.max(np.abs(np.asarray(expected[key], float) - np.asarray(actual[key], float))))
              for key in CACHE_FIELDS - set(FLAG_FIELDS)}
    flags = {key: {"expected": expected[key], "actual": actual[key]} for key in FLAG_FIELDS if expected[key] != actual[key]}
    return {"float_max_abs": max(floats.values()), "float_fields": floats, "flag_mismatches": flags,
            "passed": not flags and max(floats.values()) <= THRESHOLDS["cache_float_max_abs"]}


def trs_errors(expected, actual):
    expected, actual = transform(expected), transform(actual)
    pos = float(np.max(np.abs(expected["local_position"].astype(float) - actual["local_position"])))
    scl = float(np.max(np.abs(expected["local_scale"].astype(float) - actual["local_scale"])))
    q, r = expected["local_rotation_xyzw"].astype(float), actual["local_rotation_xyzw"].astype(float)
    require(np.linalg.norm(q) > 0 and np.linalg.norm(r) > 0, "Cannot compare zero local quaternion")
    equiv = float(min(np.max(np.abs(q-r)), np.max(np.abs(q+r))))
    qa, ra = q/np.linalg.norm(q), r/np.linalg.norm(r)
    # atan2 distance remains well-conditioned at tiny angular residuals.
    if np.dot(qa, ra) < 0:
        ra = -ra
    angle = float(np.degrees(4 * np.arctan2(np.linalg.norm(qa-ra), np.linalg.norm(qa+ra))))
    return {"position_max_abs": pos, "scale_max_abs": scl,
            "quaternion_raw_max_abs": float(np.max(np.abs(q-r))), "quaternion_sign_equivalent_max_abs": equiv,
            "rotation_angle_deg": angle,
            "passed": pos <= THRESHOLDS["position_max_abs"] and scl <= THRESHOLDS["scale_max_abs"]
                       and equiv <= THRESHOLDS["quaternion_sign_equivalent_max_abs"] and angle <= THRESHOLDS["rotation_angle_deg"]}


def local_only(state):
    return {key: state[key] for key in ("local_position", "local_rotation_xyzw", "local_scale")}


def verify_trace_header(trace, contract):
    require(isinstance(trace, dict), "Expected trace object")
    metadata = trace.get("metadata")
    require(isinstance(metadata, dict) and metadata.get("schema_version") == 1, "Unsupported trace schema")
    require(contract.get("schema_version") == 1, "Unsupported independent source contract")
    for field in ("dropped_events", "pending_calls"):
        require(type(trace.get(field)) is int and trace[field] == 0, "Incomplete trace: " + field)
    require(trace.get("observer_errors") == [], "Observer reported errors")
    require(trace.get("trace_complete") is True and trace.get("active") is False, "Trace is not stopped and complete")
    events = trace.get("events")
    require(isinstance(events, list) and len(events) > 0 and type(trace.get("observed_calls")) is int
            and len(events) == trace["observed_calls"], "Call count/record coverage mismatch")
    for field in ("plugin_mvid", "plugin_version", "apply_method_il_sha256"):
        require(metadata.get(field) == contract.get(field) and isinstance(contract.get(field), str), "Installed source mismatch: " + field)
    require(contract["plugin_mvid"] == "5442c72a-f463-4bf9-831a-247be87146c8" and contract["plugin_version"] == "4.4.6.0", "Unsupported installed branch variant")
    owners = metadata.get("pre_existing_patch_owners")
    require(isinstance(owners, list) and owners == contract.get("allowed_pre_existing_patch_owners") == [], "Unsupported external Apply patches")
    require(metadata.get("observation_policy") == "void Prefix last / void Postfix first; no argument, baseline, transform or return writes", "Unknown observer policy")
    require(type(metadata.get("started_frame")) is int and type(metadata.get("stopped_frame")) is int
            and metadata["stopped_frame"] >= metadata["started_frame"], "Missing stopped trace frame bounds")
    require(isinstance(metadata.get("session_id"), str) and bool(metadata["session_id"]), "Missing observation session identity")
    requested = metadata.get("requested_names")
    require(isinstance(requested, list) and requested and len(requested) == len(set(requested)), "Invalid requested bone filter")
    require(isinstance(contract.get("no_rotation_bones"), list), "Independent exclusion list missing")
    return metadata, events


def verify_event(event, metadata, contract, expected_sequence):
    require(type(event.get("sequence")) is int and event["sequence"] == expected_sequence, "Non-contiguous/out-of-order observed call sequence")
    frame = event.get("frame")
    require(type(frame) is int and frame == event.get("completed_frame")
            and metadata["started_frame"] <= frame <= metadata["stopped_frame"], "Call crosses frame or violates trace bounds")
    require(event.get("bone_name") in metadata["requested_names"], "Unexpected bone outside observer filter")
    require(type(event.get("modifier_instance_identity")) is int, "Missing modifier instance identity")
    require(event.get("coordinate_specific") is contract["is_coordinate_specific"] is False, "Unsupported coordinate semantics")
    bool_value(event.get("no_rotation_excluded"), "no_rotation_excluded")
    require(event["no_rotation_excluded"] == (event["bone_name"] in contract["no_rotation_bones"]), "Observed exclusion differs from installed source contract")
    before, after = event["before"], event["after"]
    require(type(before.get("bone_transform_id")) is int and before["bone_transform_id"] == after.get("bone_transform_id"), "Bone identity changed within Apply")
    for phase, state in (("before", before), ("after", after)):
        transform(local_only(state))
        wrapper = state.get("cache")
        require(isinstance(wrapper, dict) and wrapper.get("missing_fields") == [], "Missing private fields in " + phase)
        require(wrapper.get("assembly_mvid") == contract["plugin_mvid"] and wrapper.get("modifier_type") == "KKABMX.Core.BoneModifier", "Cache source differs")
        require(wrapper.get("frame_count") == frame and wrapper.get("bone_transform_id") == before["bone_transform_id"], "Cache is not same-frame/same-bone")
        cache_state(wrapper["fields"])
    return before, after


def replay_event(event, before_local, before_cache):
    require("resolved_modifier" in event and "additional_modifiers" in event, "Actual call modifier arguments missing")
    return replay_apply(before=before_local, cache=before_cache, coordinate_modifiers=[event["resolved_modifier"]],
                        coordinate=event["coordinate"], additional_modifiers=event["additional_modifiers"],
                        bone_exists=True, rotation_excluded=event["no_rotation_excluded"],
                        is_during_h_scene=event["is_during_h_scene"], coordinate_specific=event["coordinate_specific"])


def validate(trace, contract):
    metadata, events = verify_trace_header(trace, contract)
    rows, chains, boundaries = [], {}, []
    previous_frame = metadata["started_frame"]
    for index, event in enumerate(events, start=1):
        row = {"sequence": event.get("sequence"), "frame": event.get("frame"), "bone_name": event.get("bone_name"), "passed": False}
        try:
            before, after = verify_event(event, metadata, contract, index)
            require(event["frame"] >= previous_frame, "Observed frames go backwards")
            previous_frame = event["frame"]
            prediction = replay_event(event, local_only(before), before["cache"]["fields"])
            local_error = trs_errors(prediction["after"], local_only(after))
            private_error = cache_errors(prediction["cache_after"], after["cache"]["fields"])
            key = event["modifier_instance_identity"], before["bone_transform_id"]
            row.update({"bone_transform_id": before["bone_transform_id"], "modifier_instance_identity": key[0],
                        "observed_before": before, "observed_after": after,
                        "resolved_modifier": event["resolved_modifier"], "additional_modifiers": event["additional_modifiers"],
                        "prediction": prediction, "local_error": local_error, "cache_error": private_error})
            previous = chains.get(key)
            if previous:
                observed_local_boundary = trs_errors(local_only(previous["actual_after"]), local_only(before))
                observed_cache_boundary = cache_errors(previous["actual_after"]["cache"]["fields"], before["cache"]["fields"])
                contiguous = observed_local_boundary["passed"] and observed_cache_boundary["passed"]
                row["previous_same_modifier_call_sequence"] = previous["sequence"]
                row["observed_between_call_local_error"] = observed_local_boundary
                row["observed_between_call_cache_error"] = observed_cache_boundary
                if contiguous:
                    # Propagate predicted states, do not fit application counts.
                    chain_prediction = replay_event(event, previous["predicted_after"], previous["predicted_cache"])
                    chain_local = trs_errors(chain_prediction["after"], local_only(after))
                    chain_cache = cache_errors(chain_prediction["cache_after"], after["cache"]["fields"])
                    row["chain_segment"] = previous["segment"]
                    row["chain_prediction"] = chain_prediction
                    row["chain_local_error"] = chain_local
                    row["chain_cache_error"] = chain_cache
                    row["chain_contiguous"] = True
                    prediction = chain_prediction
                else:
                    row["chain_contiguous"] = False
                    row["chain_segment"] = index
                    boundaries.append({"sequence": index, "previous_sequence": previous["sequence"], "key": list(key),
                                       "classification": "observed inter-call state change; writer/cause not certified by Apply-only trace",
                                       "local_error": observed_local_boundary, "cache_error": observed_cache_boundary})
            else:
                row["chain_contiguous"] = None
                row["chain_segment"] = index
            row["passed"] = local_error["passed"] and private_error["passed"]
            if row.get("chain_contiguous"):
                row["passed"] = row["passed"] and row["chain_local_error"]["passed"] and row["chain_cache_error"]["passed"]
            chains[key] = {"actual_after": after, "predicted_after": prediction["after"], "predicted_cache": prediction["cache_after"],
                           "sequence": index, "segment": row["chain_segment"]}
        except (ReplayRejected, KeyError, TypeError, ValueError) as exc:
            row["rejection"] = str(exc)
        rows.append(row)
    return {"schema_version": 1, "trace_metadata": metadata, "status": "passed_specific_observed_calls" if all(row["passed"] for row in rows) else "failed_specific_observed_calls",
            "passed": all(row["passed"] for row in rows), "observed_call_count": len(rows), "passed_call_count": sum(row["passed"] for row in rows),
            "thresholds": THRESHOLDS, "rows": rows, "inter_call_boundaries": boundaries,
            "application_count_observed": True, "application_count_fitted": False,
            "chain_scope": "Recorded calls in exact sequence; independently anchored at observed external state boundaries, propagated within contiguous segments",
            "static_whole_head_model_fixed": False,
            "limitations": ["This certifies observed Apply transitions, not missing events outside a bounded trace window.",
                            "External native/animation/cache writes between calls are explicit boundaries; this trace does not identify their writer.",
                            "No Unity bone/surface fitting or application-count estimation is performed.",
                            "Cached-rest native59+ABMX->whole-head model remains unchanged and is not automatically made runtime-valid.",
                            "Quaternion.Euler native internal Z-X-Y conversion is independently approximated in float32 with fixed declared tolerances."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), "New report path required; preserve previous evidence")
    contract = read(args.contract)
    report = {"trace_path": str(args.trace.resolve()), "trace_sha256": sha(args.trace),
              "contract_path": str(args.contract.resolve()), "contract_sha256": sha(args.contract),
              "implementation_hashes": [{"path": str(path.resolve()), "sha256": sha(path)} for path in (Path(__file__), Path(__file__).with_name("model.py"))]}
    try:
        require(sha(contract["assembly_path"]) == contract["assembly_sha256"], "Installed assembly changed since independent contract read")
        require(all(sha(source["path"]) == source["sha256"] for source in contract["sources"]), "Decompiled installed sources changed since contract read")
        require(sha(contract["unity_core_path"]) == contract["unity_core_sha256"], "Installed Unity core changed")
        report.update(validate(read(args.trace), contract))
    except (ReplayRejected, KeyError, TypeError, ValueError, OSError) as exc:
        report.update({"schema_version": 1, "status": "rejected_incomplete_or_unsupported_evidence", "passed": False, "rejection": str(exc),
                       "static_whole_head_model_fixed": False})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "calls": report.get("observed_call_count"), "passed_calls": report.get("passed_call_count"), "report": str(args.out.resolve())}))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
