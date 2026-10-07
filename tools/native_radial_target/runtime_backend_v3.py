"""Explicit multigroup wrapper; frozen numerical code runs in private globals."""
from __future__ import annotations

import copy
import types
from pathlib import Path

from src.hs2_controlled_radial import StrictBackend, bound_read, read, save
from src.hs2_ingame_semantics import digest, file_sha, require
from tools.native_radial_target import asset_provider_strict, validate_runtime
from tools.native_radial_target.capture_groups_v3 import prepare_capture_groups

REVISION = "identity_source_assets_multigroup_v3"


def dependencies():
    root = Path(__file__).resolve().parents[2]
    paths = (Path(__file__), root / "tools/native_radial_target/capture_groups_v3.py",
             root / "src/hs2_controlled_radial_v3.py", root / "src/hs2_ingame_eval_v3.py")
    return {**asset_provider_strict.helper_sources(),
            **{str(p.resolve()): file_sha(p) for p in paths}}


class StrictBackendV3(StrictBackend):
    def __init__(self, contract, driver_contract, capture_contracts):
        super().__init__(contract, driver_contract)
        self.capture_contracts = copy.deepcopy(capture_contracts)
        self.source_bindings = dependencies()

    def _sources_v3(self):
        require(self.source_bindings == dependencies(), "V3 runtime dependency closure changed")
        self.provider._check()

    def audit(self, manifest, out):
        self._sources_v3()
        data = read(manifest)
        require(data.get("capture_backend_revision") == REVISION
                and digest(data.get("capture_contracts")) == digest(self.capture_contracts),
                "Explicit predeclared V3 runtime capture contract missing/changed")
        require(data["asset_provider"]["path"] == self.staged["path"]
                and data["asset_provider"]["sha256"] == self.staged["sha256"],
                "Strict V2 asset descriptor changed")
        artifact = bound_read(data["compiler_artifact"])
        self.provider.verify_snapshot_assets(bound_read(artifact["compiler_inputs"]["history"]["geometry"]))
        for case in data["cases"]:
            self.provider.verify_snapshot_assets(bound_read(case["identity_capture"]["paired_geometry"]))
            for row in case["windows"]:
                self.provider.verify_snapshot_assets(bound_read(row["geometry"]))

        # Existing V1 already isolates the frozen mathematical function. This
        # V3 wrapper declares all additional dependencies and injected validators.
        # No module's shared global namespace or native response is modified.
        fn = self.provider._provider._math_function("audit")
        private = fn.__globals__

        def context(window):
            capture = window["capture"]
            descriptor = capture["paired_geometry"]
            key = str(Path(descriptor["path"]).resolve())
            require(key in self.capture_contracts, "Undeclared capture path in runtime audit")
            entry = self.capture_contracts[key]
            require(entry["response_digest"] == digest(capture), "Full original native response changed")
            prepared = prepare_capture_groups(entry["schedule"], entry["expected_source"],
                                              cursor_policy=entry["cursor_policy"])
            envelope = {**window, "geometry": descriptor, "capture_response": entry["response"]}
            return prepared, envelope

        def pairs(window, snapshot, snapshot_sha):
            prepared, envelope = context(window)
            return prepared.verify_pairs(envelope, snapshot, snapshot_sha)

        def signatures(window, snapshot):
            prepared, envelope = context(window)
            return prepared.verify_capture_pair_signatures(envelope, snapshot)

        private["verify_pairs"] = pairs
        private["verify_capture_pair_signatures"] = signatures
        original = validate_runtime.verify_local_restoration
        cloned = types.FunctionType(original.__code__, private, original.__name__,
                                    original.__defaults__, original.__closure__)
        cloned.__kwdefaults__ = original.__kwdefaults__
        private["verify_local_restoration"] = cloned
        private["independent_target"] = self.provider.independent_target
        result = fn(manifest, self.contract, out, source_gaze_policy="frozen_source_locals")
        result["explicit_multigroup_backend"] = {
            "revision": REVISION, "source_bindings": self.source_bindings,
            "capture_contract_digest": digest(self.capture_contracts),
            "frozen_math_sha256": file_sha(validate_runtime.__file__),
            "private_function_globals": True, "shared_globals_modified": False,
            "original_native_responses_rewritten": False,
            "original_arithmetic_and_tolerances_unchanged": True,
            "not_original_frozen_validator_certification": True,
        }
        # The frozen math wrote its own summary; preserve that report intact.
        original_report = Path(out) / "summary.json"
        result["unchanged_math_report"] = {"path": str(original_report.resolve()),
                                           "sha256": file_sha(original_report)}
        save(Path(out) / "summary_v3.json", result)
        return result
