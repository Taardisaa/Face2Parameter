"""Explicit ordered full-response view contracts; never rewrites native evidence.

Geometry/replay/target mathematics remain the responsibility of the composed
backend. Diagnostic loss mode certifies view bindings only, never trace coverage.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import struct
import zlib
from pathlib import Path

REVISION = "paired_capture_groups_v3"
POLICIES = {"complete_active", "complete_or_inactive", "diagnostic_loss"}
HEAD_NAMES = {"o_head", "o_eyebase_L", "o_eyebase_R", "o_eyelashes", "o_eyeshadow", "o_namida", "o_tang", "o_tooth"}


class CaptureGroupsRejected(ValueError):
    """A declared capture or original evidence binding is invalid."""


def _require(condition, message):
    if not condition:
        raise CaptureGroupsRejected(message)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _integer(value, label, *, nonnegative=False):
    _require(type(value) is int and (not nonnegative or value >= 0), "Expected exact integer: " + label)
    return value


def _number(value, label):
    _require(type(value) in (int, float) and math.isfinite(value), "Expected finite non-bool number: " + label)
    return value


def _vector(value, length, label):
    _require(type(value) is list and len(value) == length, "Invalid vector length/type: " + label)
    return [_number(v, label) for v in value]


def _exact(actual, expected, label):
    _require(type(actual) is type(expected), "JSON type differs: " + label)
    if type(expected) is dict:
        _require(actual.keys() == expected.keys(), "JSON keys differ: " + label)
        for key in expected:
            _exact(actual[key], expected[key], label + "." + key)
    elif type(expected) is list:
        _require(len(actual) == len(expected), "JSON length differs: " + label)
        for i, value in enumerate(expected):
            _exact(actual[i], value, label + "." + str(i))
    else:
        _require(actual == expected, "JSON value differs: " + label)


def _read_bound(descriptor):
    _require(type(descriptor) is dict and type(descriptor.get("path")) is str
             and type(descriptor.get("sha256")) is str, "Malformed file descriptor")
    path = Path(descriptor["path"])
    _require(path.is_absolute() and path.is_file(), "Missing absolute evidence path")
    data = path.read_bytes()
    _require(hashlib.sha256(data).hexdigest() == descriptor["sha256"], "File SHA mismatch: " + str(path))
    return path.resolve(), json.loads(data)


def make_schedule(*, groups, width, height, phase):
    """Caller freezes this source-independent schedule BEFORE native capture."""
    _integer(width, "width", nonnegative=True)
    _integer(height, "height", nonnegative=True)
    _require(width > 0 and height > 0 and type(phase) is str and bool(phase), "Invalid dimensions/phase")
    _require(type(groups) is list and groups, "Explicit nonempty ordered groups required")
    declared, flat, seen = [], [], set()
    for group in groups:
        _require(type(group) is dict and set(group) == {"id", "yaws"}, "Group must declare id/yaws only")
        name, yaws = group["id"], group["yaws"]
        _require(type(name) is str and bool(name) and name not in seen, "Duplicate/invalid group ID")
        _require(type(yaws) is list and yaws, "Nonempty declared yaw list required")
        for yaw in yaws:
            _number(yaw, "requested yaw")
        start = len(flat)
        declared.append({"id": name, "indices": list(range(start, start + len(yaws))), "yaws": copy.deepcopy(yaws)})
        flat.extend(yaws)
        seen.add(name)
    result = {"revision": REVISION, "phase": phase, "width": width, "height": height,
              "expected_count": len(flat), "groups": declared, "flat_yaws": flat}
    result["binding_sha256"] = _digest(result)
    return result


def _png(path, width, height):
    data = path.read_bytes()
    _require(data[:8] == b"\x89PNG\r\n\x1a\n", "Not actual PNG")
    offset, chunks, dimensions = 8, [], None
    while offset < len(data):
        _require(offset + 12 <= len(data), "Truncated PNG chunk")
        size = struct.unpack(">I", data[offset:offset + 4])[0]
        kind = data[offset + 4:offset + 8]
        end = offset + 12 + size
        _require(end <= len(data), "Truncated PNG payload")
        payload = data[offset + 8:offset + 8 + size]
        crc = struct.unpack(">I", data[offset + 8 + size:end])[0]
        _require(zlib.crc32(kind + payload) & 0xffffffff == crc, "PNG CRC mismatch")
        if not chunks:
            _require(kind == b"IHDR" and size == 13, "PNG first chunk must be IHDR")
            w, h, depth, color, compression, filtering, interlace = struct.unpack(">IIBBBBB", payload)
            _require((w, h) == (width, height), "PNG IHDR dimensions differ from declared capture")
            _require(depth == 8 and color in (2, 6) and compression == filtering == 0 and interlace in (0, 1), "Unsupported PNG encoding")
            dimensions = [w, h]
        else:
            _require(kind != b"IHDR", "Duplicate PNG IHDR")
        chunks.append(kind)
        offset = end
        if kind == b"IEND":
            _require(size == 0 and offset == len(data), "Invalid PNG end/trailing data")
            break
    _require(chunks[-1:] == [b"IEND"] and b"IDAT" in chunks, "Incomplete PNG")
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest(), "ihdr_size": dimensions,
            "scope": "Original PNG bytes, signature, chunk CRCs and IHDR; no image similarity or alignment"}


def _cursor(snapshot, policy):
    cursor = snapshot.get("abmx_trace_cursor")
    _require(type(cursor) is dict, "Actual trace cursor required")
    for name in ("frame", "observed_calls", "completed_calls", "last_completed_sequence", "pending_calls", "dropped_events", "observer_error_count"):
        _integer(cursor.get(name), "cursor." + name, nonnegative=True)
    _require(type(cursor.get("active")) is bool, "Cursor active must be bool")
    _require(cursor["session_id"] is None or (type(cursor["session_id"]) is str and bool(cursor["session_id"])), "Malformed cursor session")
    _require(cursor["frame"] == snapshot["frame_count"], "Cursor/frame mismatch")
    _require(cursor["completed_calls"] == cursor["last_completed_sequence"]
             and cursor["completed_calls"] <= cursor["observed_calls"]
             and cursor["completed_calls"] + cursor["dropped_events"] + cursor["pending_calls"] <= cursor["observed_calls"],
             "Unsupported/inconsistent serial-prefix cursor")
    complete = (cursor["pending_calls"] == cursor["dropped_events"] == cursor["observer_error_count"] == 0
                and cursor["observed_calls"] == cursor["completed_calls"])
    if policy == "complete_active":
        _require(cursor["active"] is True and bool(cursor["session_id"]) and complete, "Candidate cursor inactive/incomplete/lost")
    elif policy == "complete_or_inactive" and cursor["active"]:
        _require(bool(cursor["session_id"]) and complete, "Active source cursor incomplete/lost")
    return {"policy": policy, "actual": copy.deepcopy(cursor), "complete_serial_prefix": complete,
            "trace_coverage_required": policy == "complete_active" or (policy == "complete_or_inactive" and cursor["active"]),
            "diagnostic_only": policy == "diagnostic_loss"}


class PreparedCaptureGroups:
    """Prepared caller-specified schedule/source, not a JSON acceptance token."""

    def __init__(self, schedule, expected_source, cursor_policy):
        _require(cursor_policy in POLICIES, "Unknown explicit cursor policy")
        _require(type(schedule) is dict and type(schedule.get("groups")) is list, "Malformed schedule")
        rebuilt = make_schedule(groups=[{"id": g["id"], "yaws": g["yaws"]} for g in schedule["groups"]],
                                width=schedule["width"], height=schedule["height"], phase=schedule["phase"])
        _exact(schedule, rebuilt, "predeclared schedule")
        _require(type(expected_source) is dict, "Explicit expected source required")
        _require(set(expected_source) == {"actor_transform_id", "head_id", "native59", "bridge_mvid", "head_meshes", "transform_scope"},
                 "Unknown/missing expected source fields (no ignored declarations)")
        for key in ("actor_transform_id", "head_id"):
            _integer(expected_source.get(key), "source." + key)
        _vector(expected_source.get("native59"), 59, "source.native59")
        _require(type(expected_source.get("bridge_mvid")) is str and bool(expected_source["bridge_mvid"]), "Expected actual bridge MVID required")
        _require(expected_source.get("transform_scope") == "actor_hierarchy_and_ancestors", "Complete actor transform scope required")
        meshes = expected_source.get("head_meshes")
        _require(type(meshes) is list and len(meshes) == 8 and {m.get("mesh_name") for m in meshes} == HEAD_NAMES, "Expected eight source-bound head renderer assets")
        _require(len({m.get("renderer_path") for m in meshes}) == 8, "Duplicate expected head renderer")
        for mesh in meshes:
            _require(set(mesh) == {"mesh_name", "renderer_path", "source_geometry_sha256"}
                     and all(type(v) is str and bool(v) for v in mesh.values()), "Malformed expected source renderer")
        self._schedule, self._source = copy.deepcopy(schedule), copy.deepcopy(expected_source)
        self._policy, self._implementation_sha = cursor_policy, _sha(__file__)
        self._binding = _digest({"schedule": schedule, "source": expected_source, "cursor_policy": cursor_policy})

    def _check(self):
        _require(_sha(__file__) == self._implementation_sha, "Prepared helper implementation changed")
        _require(_digest({"schedule": self._schedule, "source": self._source, "cursor_policy": self._policy}) == self._binding,
                 "Prepared schedule/source context mutated")

    def _snapshot(self, snapshot, descriptor):
        frame = _integer(snapshot.get("frame_count"), "geometry.frame", nonnegative=True)
        _require(_integer(snapshot.get("frame_count_end"), "geometry.frame_end", nonnegative=True) == frame, "Geometry spans frames")
        character = snapshot["character"]
        _require(_integer(character.get("transform_id"), "actor ID") == self._source["actor_transform_id"]
                 and _integer(character.get("head_id"), "head ID") == self._source["head_id"], "Actual actor/head differs from declared source")
        native = _vector(character.get("shape_value_face"), 59, "actual.native59")
        _require(all(abs(v) <= 3.4028234663852886e38 for v in native + self._source["native59"]), "Native59 exceeds float32")
        # IEEE signed zero has equal control semantics; never distinguish its raw
        # bit sign when comparing actual numerical native inputs.
        _require(struct.pack("<59f", *[0.0 if v == 0 else v for v in native])
                 == struct.pack("<59f", *[0.0 if v == 0 else v for v in self._source["native59"]]), "Actual native59 differs")
        _require(snapshot.get("include_actor_transforms") is True and descriptor.get("include_actor_transforms") is True
                 and snapshot.get("transform_scope") == descriptor.get("transform_scope") == self._source["transform_scope"], "Full actor/ancestor geometry coverage missing")
        transforms = snapshot.get("transforms")
        _require(type(transforms) is list and transforms, "Actual actor transforms missing")
        ids = [_integer(t.get("id"), "transform ID") for t in transforms]
        _require(len(set(ids)) == len(ids) and self._source["actor_transform_id"] in ids
                 and _integer(character.get("head_root_transform_id"), "head root ID") in ids, "Duplicate/missing actor/head transforms")
        for transform in transforms:
            parent = transform.get("parent_id")
            if parent is not None:
                _require(_integer(parent, "parent ID") in ids, "Actor ancestor coverage incomplete")
        _require(_integer(descriptor.get("transform_count"), "paired transform count", nonnegative=True) == len(ids), "Paired transform count mismatch")
        meshes = snapshot.get("meshes")
        _require(type(meshes) is list, "Actual meshes missing")
        paths = [m.get("renderer_path") for m in meshes]
        _require(len(set(paths)) == len(paths), "Duplicate actual renderer path")
        for expected in self._source["head_meshes"]:
            rows = [m for m in meshes if m.get("renderer_path") == expected["renderer_path"]]
            _require(len(rows) == 1, "Expected source renderer missing")
            mesh = rows[0]
            _exact({k: mesh[k] for k in expected}, expected, "source mesh")
            _require(mesh.get("enabled") is True and mesh.get("active_in_hierarchy") is True, "Expected visible head renderer disabled")
        _require(descriptor.get("pose_signature") == snapshot.get("pose_signature")
                 and descriptor.get("visibility_sample_signature") == snapshot.get("visibility_sample_signature")
                 and _integer(descriptor.get("frame_count"), "paired frame", nonnegative=True) == frame, "Paired descriptor/export signatures mismatch")
        return _cursor(snapshot, self._policy)

    def verify_response(self, response_path, response_sha256, geometry_descriptor, *, snapshot=None):
        """Validate original full response; default rejects incomplete candidate cursor."""
        self._check()
        path, response = _read_bound({"path": str(response_path), "sha256": response_sha256})
        geo_path, actual_snapshot = _read_bound(geometry_descriptor)
        if snapshot is not None:
            _exact(snapshot, actual_snapshot, "caller snapshot versus actual file")
        snapshot = actual_snapshot
        _exact(response.get("paired_geometry"), geometry_descriptor, "original parent geometry descriptor")
        cursor = self._snapshot(snapshot, geometry_descriptor)
        schedule = self._schedule
        count = _integer(response.get("count"), "response.count", nonnegative=True)
        _require(count == schedule["expected_count"], "Response count differs from predeclared schedule")
        views, yaws, paths = response.get("views"), response.get("yaws"), response.get("paths")
        _require(all(type(values) is list and len(values) == count for values in (views, yaws, paths)), "Response arrays omit/add views")
        _require(response.get("capture_state_restored") is True, "Native capture state not restored")
        for key in ("frame_count", "frame_count_before_render"):
            _require(_integer(response.get(key), key, nonnegative=True) == snapshot["frame_count"], "Parent capture frame mismatch")
        rows, seen = [], set()
        groups = {i: (g["id"], j) for g in schedule["groups"] for j, i in enumerate(g["indices"])}
        for index, view in enumerate(views):
            _require(type(view) is dict, "Malformed native view")
            yaw = _number(view.get("yaw"), "view yaw")
            _require(_number(yaws[index], "response yaw") == yaw == schedule["flat_yaws"][index], "View order/yaw differs from predeclared schedule")
            _require(type(view.get("path")) is str and view["path"] == paths[index], "Native paths/view path differs")
            png_path = Path(view["path"])
            _require(png_path.is_absolute() and png_path.is_file() and png_path.resolve().parent == path.parent, "PNG outside actual response directory")
            _require(str(png_path.resolve()).casefold() not in seen, "Duplicate native view PNG path")
            seen.add(str(png_path.resolve()).casefold())
            for name in ("width", "height"):
                _require(_integer(view.get(name), "view." + name, nonnegative=True) == schedule[name], "View dimensions differ")
            for name in ("frame_count", "frame_count_before_render"):
                _require(_integer(view.get(name), name, nonnegative=True) == snapshot["frame_count"], "View spans another frame")
            _exact(view.get("paired_geometry"), geometry_descriptor, "view paired geometry")
            for name in ("paired_pose_unchanged", "paired_visibility_sampled_unchanged", "paired_capture_state_unchanged", "paired_native_face_drivers_unchanged", "paired_lights_unchanged"):
                _require(view.get(name) is True, "Native paired flag false/untyped: " + name)
            for stem, expected in (("pose_signature", snapshot["pose_signature"]), ("visibility_sample_signature", snapshot["visibility_sample_signature"])):
                _require(type(expected) is str and bool(expected) and view.get(stem + "_before_render") == view.get(stem + "_after_render") == expected, "View/geometry signatures mismatch")
            for stem in ("capture_state", "native_face_drivers", "lights"):
                _exact(view.get(stem + "_before_render"), view.get(stem + "_after_render"), "paired " + stem)
            _exact(view["native_face_drivers_before_render"], snapshot["native_face_drivers"], "view/snapshot drivers")
            camera = view.get("capture_camera")
            _require(type(camera) is dict and camera.get("matrix_layout") == "row_major_16; column_vectors", "Unsupported/missing camera metadata")
            _require(camera.get("bridge_mvid") == self._source["bridge_mvid"], "Actual camera bridge MVID differs")
            for name in ("world_to_camera", "camera_to_world", "projection", "gpu_projection_render_texture"):
                _vector(camera.get(name), 16, "camera." + name)
            # This is an exported-camera consistency check, not fitted alignment
            # or a change to the frozen surface/replay mathematical tolerance.
            inverse_error = max(abs(sum(camera["world_to_camera"][r * 4 + k] * camera["camera_to_world"][k * 4 + c]
                                        for k in range(4)) - float(r == c)) for r in range(4) for c in range(4))
            _require(inverse_error <= 1e-5, "Exported camera matrices are not inverse-consistent")
            _require(_vector(camera.get("pixel_rect"), 4, "camera.pixel_rect") == [0, 0, schedule["width"], schedule["height"]], "Camera viewport differs from PNG")
            _require(0 < _number(camera.get("near_clip"), "near clip") < _number(camera.get("far_clip"), "far clip"), "Invalid camera clip interval")
            _require(_number(camera.get("aspect"), "aspect") == schedule["width"] / schedule["height"], "Camera aspect differs")
            group_id, local_index = groups[index]
            rows.append({"original_index": index, "group_id": group_id, "group_local_index": local_index,
                         "requested_yaw": schedule["flat_yaws"][index], "yaw": yaw,
                         **_png(png_path, schedule["width"], schedule["height"]),
                         "view_metadata_sha256": _digest(view), "camera_metadata_sha256": _digest(camera)})
        return {"revision": REVISION, "parent_response": {"path": str(path), "sha256": response_sha256},
                "geometry": {"path": str(geo_path), "sha256": geometry_descriptor["sha256"]},
                "schedule": copy.deepcopy(schedule), "prepared_binding_sha256": self._binding,
                "helper_source": {"path": str(Path(__file__).resolve()), "sha256": self._implementation_sha},
                "frame": snapshot["frame_count"], "cursor": cursor, "views": rows,
                "camera_inverse_consistency_tolerance": 1e-5,
                "all_original_views_covered": True, "original_response_rewritten": False,
                "trace_coverage_eligible": cursor["complete_serial_prefix"] and self._policy != "diagnostic_loss",
                "numerical_runtime_certified": False, "likeness_certified": False}

    def _window(self, window, snapshot, snapshot_sha):
        descriptor = window.get("capture_response")
        _require(type(descriptor) is dict, "Explicit original capture_response path/SHA required")
        _, response = _read_bound(descriptor)
        _exact(window.get("capture"), response, "window versus unmodified original response")
        geometry = window["geometry"]
        _require(geometry["sha256"] == snapshot_sha, "Caller geometry SHA differs")
        return self.verify_response(descriptor["path"], descriptor["sha256"], geometry, snapshot=snapshot)

    def verify_pairs(self, window, snapshot, snapshot_sha):
        """Signature-compatible explicit dependency slot; covers all original views."""
        return self._window(window, snapshot, snapshot_sha)["views"]

    def verify_capture_pair_signatures(self, window, snapshot):
        receipt = self._window(window, snapshot, window["geometry"]["sha256"])
        return {"revision": REVISION, "actual_signature_values_crosschecked": True,
                "pose_signature": snapshot["pose_signature"], "visibility_sample_signature": snapshot["visibility_sample_signature"],
                "parent_response": receipt["parent_response"], "schedule_binding_sha256": self._schedule["binding_sha256"],
                "view_count": len(receipt["views"]), "all_original_views_covered": True, "cursor": receipt["cursor"]}


def prepare_capture_groups(schedule, expected_source, *, cursor_policy="complete_active"):
    return PreparedCaptureGroups(schedule, expected_source, cursor_policy)
