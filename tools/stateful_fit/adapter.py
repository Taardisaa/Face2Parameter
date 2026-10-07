"""Candidate native59 + installed state/call protocol -> differentiable head.

Only a clean native-baseline boundary is supported. Persistent length history
stays measured. New candidates are predictions under a declared protocol, not
automatically certified runtime states.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from src.hs2_abmx_torch import FLAGS, apply_sequence, tensor_cache
from src.hs2_deform_torch import TorchHeadRig
from tools.abmx_replay.validate_trace import validate, trs_errors

CHIN = "cf_J_ChinTip_s"
IDENTITY = np.array([1., 1., 1., 1., 0., 0., 0., 0., 0., 0.])


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class NativeBaselineProtocol:
    """Source-bound clean boundary and an explicit finite Apply schedule."""

    def __init__(self, trace, contract, snapshot, *, trace_path, contract_path,
                 snapshot_path, apply_count=None):
        # Independent NumPy validator consumes the complete trace, not a saved
        # green flag. Unknown writes cannot become anchors for candidate fit.
        validation = validate(trace, contract)
        if not validation["passed"] or validation["inter_call_boundaries"]:
            raise ValueError("Candidate protocol requires a complete passing chain without external boundaries")
        from tools.abmx_replay.validate_geometry import select_cursor_prediction
        # Compare the independently propagated cursor state with snapshot locals
        # and cache solely as an evidence guard, never as candidate input.
        _, cursor_validation = select_cursor_prediction(snapshot, trace, validation)
        if trace["metadata"]["requested_names"] != [CHIN]:
            raise ValueError("This fitting protocol requires one observed ChinTip modifier")
        for path, expected in ((contract["assembly_path"], contract["assembly_sha256"]),
                               (contract["unity_core_path"], contract["unity_core_sha256"])):
            if sha(path) != expected:
                raise ValueError("Installed source contract no longer matches assemblies")
        if any(sha(row["path"]) != row["sha256"] for row in contract["sources"]):
            raise ValueError("Installed source contract no longer matches decompiled source")
        cursor = snapshot.get("abmx_trace_cursor", {})
        if cursor.get("session_id") != trace["metadata"]["session_id"]:
            raise ValueError("Geometry/trace session mismatch")
        sequence = cursor.get("last_completed_sequence")
        if type(sequence) is not int or not 1 <= sequence <= len(trace["events"]):
            raise ValueError("Exact snapshot completed-call cursor required")
        if cursor.get("pending_calls") != 0 or cursor.get("dropped_events") != 0 or cursor.get("observer_error_count") != 0:
            raise ValueError("Snapshot has incomplete observation")
        if cursor.get("frame") != snapshot.get("frame_count"):
            raise ValueError("Snapshot/cursor frame mismatch")
        events = trace["events"][:sequence]
        if any(event["completed_frame"] > cursor["frame"] for event in events):
            raise ValueError("Snapshot cursor references future calls")
        first = events[0]
        fields = first["before"]["cache"]["fields"]
        if not fields["_hasBaseline"] or any(fields[key] for key in FLAGS if key != "_hasBaseline"):
            raise ValueError("Explicit clean native-baseline boundary required; flags cannot be invented/reset")
        first_modifier = first["resolved_modifier"]
        first_values = np.r_[first_modifier["scale"], first_modifier["length"], first_modifier["position"], first_modifier["rotation"]]
        if not np.array_equal(first_values, IDENTITY):
            raise ValueError("Observed clean boundary must begin with an identity modifier")
        for event in events:
            if event["additional_modifiers"] or event["is_during_h_scene"] or event["no_rotation_excluded"]:
                raise ValueError("Unsupported additional/scene/rotation-excluded fitting protocol")
            if event["before"]["bone_transform_id"] != first["before"]["bone_transform_id"] or event["modifier_instance_identity"] != first["modifier_instance_identity"]:
                raise ValueError("Modifier/bone identity changed during candidate sequence")
        values = [np.r_[e["resolved_modifier"]["scale"], e["resolved_modifier"]["length"], e["resolved_modifier"]["position"], e["resolved_modifier"]["rotation"]] for e in events]
        starts = [i for i, value in enumerate(values) if not np.array_equal(value, IDENTITY)]
        if not starts or any(not np.array_equal(value, values[starts[0]]) for value in values[starts[0]:]):
            raise ValueError("Require one constant candidate patch after the clean identity boundary")
        measured_count = len(events) - starts[0]
        if apply_count is None:
            apply_count = measured_count
        if type(apply_count) is not int or not 1 <= apply_count <= 10000:
            raise ValueError("Explicit positive Apply count required")
        self.head_id = snapshot["character"]["head_id"]
        self.reference_native = np.asarray(snapshot["character"]["shape_value_face"], float)
        if self.reference_native.shape != (59,) or not np.isfinite(self.reference_native).all():
            raise ValueError("Protocol requires all 59 actual native controls")
        self.fields = copy.deepcopy(fields)
        self.reference_before = {key: first["before"][key] for key in ("local_position", "local_rotation_xyzw", "local_scale")}
        self.reference_modifier = values[starts[0]]
        self.count = apply_count
        self.metadata = {
            "model": "installed_abmx_4.4.6_explicit_native_baseline_and_persistent_history",
            "head_id": self.head_id, "bone": CHIN,
            "trace_path": str(Path(trace_path).resolve()), "trace_sha256": sha(trace_path),
            "contract_path": str(Path(contract_path).resolve()), "contract_sha256": sha(contract_path),
            "geometry_path": str(Path(snapshot_path).resolve()), "geometry_sha256": sha(snapshot_path),
            "cursor_sequence": sequence, "candidate_start_sequence": starts[0] + 1,
            "observed_candidate_apply_count": measured_count, "declared_apply_count": apply_count,
            "application_count_fitted": False, "declared_count_matches_this_observation": apply_count == measured_count,
            "initial_flags": {key: fields[key] for key in FLAGS},
            "persistent_length_baseline": fields["_lenBaseline"], "persistent_position_baseline": fields["_positionBaseline"],
            "candidate_baseline_policy": "native59 -> clean native local TRS and pos/scale/rotation baseline; measured length history unchanged; Apply without unmodeled external writes",
            "new_candidate_runtime_certified": False,
            "snapshot_cursor_state_verified": cursor_validation["snapshot_local_error"]["passed"] and cursor_validation["snapshot_cache_error"]["passed"],
            "scope": "Four measured head/cache boundaries; clean ChinTip protocol. New arbitrary native/ABMX candidate and different call counts require actual runtime validation.",
        }

    @classmethod
    def from_manifest(cls, manifest_path, contract_path, head_id, *, apply_count=None):
        manifest, contract = read(manifest_path), read(contract_path)
        if not all(manifest.get(key) is True for key in ("state_restored", "expression_restored", "bone_restored")):
            raise ValueError("Protocol capture manifest was not completed/restored")
        rows = [row for row in manifest["cases"] if row["head_id"] == head_id]
        if len(rows) != 1:
            raise ValueError("Unique actual head case required")
        row = rows[0]
        if sha(row["trace"]) != row["trace_sha256"] or sha(row["geometry"]["path"]) != row["geometry"]["sha256"]:
            raise ValueError("Producer trace/geometry SHA mismatch")
        return cls(read(row["trace"]), contract, read(row["geometry"]["path"]),
                   trace_path=row["trace"], contract_path=contract_path,
                   snapshot_path=row["geometry"]["path"], apply_count=apply_count)


class StatefulTorchHeadRig(TorchHeadRig):
    """Same native59/FK/LBS with a propagated, stateful ChinTip transition."""

    def __init__(self, rig, protocol, **kwargs):
        super().__init__(rig, **kwargs)
        if rig.head_id != protocol.head_id or CHIN not in self.name2bone or CHIN not in self.ab_names:
            raise ValueError("Protocol does not belong to this head/ChinTip rig")
        self.protocol = protocol
        self.chin_bone = self.name2bone[CHIN]
        self.chin_slot = self.ab_names.index(CHIN)
        with torch.no_grad():
            native = torch.as_tensor(protocol.reference_native[None], device=self.device, dtype=self.dtype)
            pos, quat, scale = super().local_transforms(native)
            local = {"local_position": pos[0, self.chin_bone].cpu().tolist(),
                     "local_rotation_xyzw": quat[0, self.chin_bone].cpu().tolist(),
                     "local_scale": scale[0, self.chin_bone].cpu().tolist()}
            error = trs_errors(local, protocol.reference_before)
            baseline = {"local_position": protocol.fields["_posBaseline"],
                        "local_rotation_xyzw": protocol.fields["_rotBaseline"],
                        "local_scale": protocol.fields["_sclBaseline"]}
            cache_error = trs_errors(local, baseline)
            if not error["passed"] or not cache_error["passed"]:
                raise ValueError("Native59 predictor fails actual clean before/baseline boundary")
        self.protocol_metadata = {**protocol.metadata, "reference_before_error": error,
                                  "reference_cache_trs_error": cache_error,
                                  "transition_arithmetic": "float32; native tables/FK/LBS use requested rig dtype"}

    def local_transforms(self, shape_face=None, ab=None):
        if shape_face is None or shape_face.ndim != 2 or shape_face.shape[1] != 59:
            raise ValueError("Stateful fitting requires explicit batched native59")
        pos, quat, scale = super().local_transforms(shape_face)
        if ab is None:
            raise ValueError("Stateful fitting requires explicit modifiers including identity")
        if ab.shape != (len(shape_face), len(self.ab_names), 10) or not torch.isfinite(ab).all():
            raise ValueError("Explicit complete batched ABMX array required")
        other = [i for i in range(len(self.ab_names)) if i != self.chin_slot]
        identity = torch.as_tensor(IDENTITY, device=ab.device, dtype=ab.dtype)
        if (ab[:, other] != identity).any():
            raise ValueError("Other bones require their own state protocols; static fallback forbidden")
        index = self.chin_bone
        local = (pos[:, index].float(), quat[:, index].float(), scale[:, index].float())
        cache = tensor_cache(self.protocol.fields, device=self.device, batch=len(shape_face))
        # Only these native baselines refresh. Length/direction are historical.
        cache = {**cache, "_posBaseline": local[0], "_rotBaseline": local[1], "_sclBaseline": local[2]}
        predicted, _ = apply_sequence(local, cache, ab[:, self.chin_slot].float(), self.protocol.count)
        indices = torch.tensor([index], device=self.device)
        return tuple(value.index_copy(1, indices, override.to(value.dtype)[:, None])
                     for value, override in zip((pos, quat, scale), predicted))
