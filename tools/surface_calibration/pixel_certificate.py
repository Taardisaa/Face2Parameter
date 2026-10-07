"""Bind independent marker measurements to a narrowly scoped PNG coordinate contract.

Hashes establish provenance. Accuracy comes from remeasuring actual marker pixels,
not from metadata declarations or an external ``validated`` boolean.
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

if __package__:
    from ..pixel_calibration.analyze import evaluate, unwrap
    from .core import ContractError
else:
    from core import ContractError
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tools.pixel_calibration.analyze import evaluate, unwrap


# Reviewed against the deployed 0.30.0 module and source on 2026-10-04.
# RenderHead creates/configures one camera and ARGB32 RT, then calls cam.Render
# and SaveRenderTexture -> ReadPixels -> EncodeToPNG for both request kinds.
# MakerPixelCalibration.Create changes cullingMask/clearFlags/background and adds
# unlit quads; it does not replace that render/readback path. This review supports
# transfer of the measured coordinate convention only, never material visibility.
REVIEWED_PIPELINES = {
    "68ab80e6-e0db-4ddd-99ca-46ade7e76020": {
        "source": "HS2Mod/plugins/HS2_McpBridge/MakerRenderService.cs",
        "source_sha256": "6ee8aa36a0898fa900bc3b5e1c3b38d373c4bd1dbc003ee8ce079cde74e0f1e5",
        "marker_source": "HS2Mod/plugins/HS2_McpBridge/MakerPixelCalibration.cs",
        "marker_source_sha256": "d4e2e81bfcb28240415d695b856aed0e92c8c9541c6f2126bbd8865e1601d528",
        "path": "MakerRenderService.RenderHead -> Camera.Render -> SaveRenderTexture -> Texture2D.ReadPixels -> EncodeToPNG",
        "review": "2026-10-05: both capture kinds retain shared camera/RT/readback. Optional texture export runs before views and restores active RT and GL.sRGBWrite; it does not replace camera readback. Transfer certifies coordinates only.",
    },
    "f7becf90-9a8d-4997-8188-d9d39f0f9bcc": {
        "source": "HS2Mod/plugins/HS2_McpBridge/MakerRenderService.cs",
        "source_sha256": "aa00121f5922562a26dd5a72b3a5141339ca4d76038552a12c9caef5493f1a07",
        "marker_source": "HS2Mod/plugins/HS2_McpBridge/MakerPixelCalibration.cs",
        "marker_source_sha256": "d4e2e81bfcb28240415d695b856aed0e92c8c9541c6f2126bbd8865e1601d528",
        "path": "MakerRenderService.RenderHead -> Camera.Render -> SaveRenderTexture -> Texture2D.ReadPixels -> EncodeToPNG",
        "review": "Both capture kinds share camera/RT/readback code; isolated markers change scene contents, culling and clear color only.",
    },
}

REVIEWED_PIPELINES["cef081fe-0d8b-40ef-ae76-c13e8c39a234"] = {
    **REVIEWED_PIPELINES["68ab80e6-e0db-4ddd-99ca-46ade7e76020"],
    "review": "2026-10-05: Renderer and marker source hashes unchanged from 68ab80e6. New module adds read-only ABMX cache snapshots before views; camera/RT/readback remains shared. Fresh eight-case marker capture remeasured; coordinate-only transfer remains restricted to matching validated AA1 scopes.",
}

REVIEWED_PIPELINES["f3d8f519-5007-4d64-99d7-16fa608de287"] = {
    **REVIEWED_PIPELINES["cef081fe-0d8b-40ef-ae76-c13e8c39a234"],
    "binary_review": "HS2Mod/artifacts/ocular_visibility_audit_20261005/new_pipeline_review.json",
    "binary_review_sha256": "70f4c38dab594a789aefdc0185728388e80a42e221ead6f8a56eb0a37f41e0d9",
    "binary_review_section": "old_loaded_f3_vs_reviewed_cef_binary_review",
    "assembly_sha256": "53da5fd593690148dd4df812384480a087ec6be1d2bdec491ced9f616d873dfb",
    "resolved_il_sha256": {
        "MakerRenderService::RenderHead": "b06d2f95033f3d9c0377eff49d05929c7dc34409a2f0bb6f5493895ef513abf3",
        "MakerRenderService::SaveRenderTexture": "0f769c45117683f9f4268383083b7c3b15eba85e07ddbc5f6c722f0c51387201",
        "MakerPixelCalibration::Initialize": "829ffe7651fe0dc9a8c326d776353f36031b83bdc8dacb35c54e6670113c28fb",
    },
    "review": "2026-10-06: the actual loaded f3 module retains the reviewed cef camera/marker/readback method bodies. Independent ILSpy review resolves metadata operands and proves exact method-text equality after removing only RVA location comments; instruction offsets, locals, branches and exception regions remain. The associated old RenderService source hash remains 6ee8aa36, not the newer visibility-batch source hash 6d2cc9a8. Fresh f3 512/1024 AA1 saved marker PNGs are remeasured by the strict consumer. Coordinate-only transfer never certifies shader, fragment depth, same-pose factorial effects or anatomy.",
}

REVIEWED_PIPELINES["0096eab0-3abf-4da6-9b23-639e52470577"] = {
    **REVIEWED_PIPELINES["cef081fe-0d8b-40ef-ae76-c13e8c39a234"],
    "source_sha256": "6d2cc9a87768e2e3321ec5bf9600ee7911685a664f0c55e2f9227c2cbe84f706",
    "binary_review": "HS2Mod/artifacts/ocular_visibility_audit_20261005/current_0096_pipeline_review.json",
    "binary_review_sha256": "9fd016c6592cc8fc0689ab67f2678493b1d8a64145716c09ce7a6bc5e78a6902",
    "assembly_sha256": "cb967aaca80ec126527490ece9a7ec7e105fb72f3eb2d035cf2aec526bbd0f23",
    "resolved_il_sha256": {
        "MakerRenderService::RenderHead": "aecadcbc5fcbc5d156e1611b5ff1720225f11b80a19d3d1f9d1832d3fdb75962",
        "MakerRenderService::SaveRenderTexture": "0f769c45117683f9f4268383083b7c3b15eba85e07ddbc5f6c722f0c51387201",
        "MakerPixelCalibration::Initialize": "829ffe7651fe0dc9a8c326d776353f36031b83bdc8dacb35c54e6670113c28fb",
    },
    "review": "2026-10-06: independent current_0096 source/binary review binds the actual 0096 module and 6d2cc9a8 renderer source. Ordinary/marker modes still share camera construction, RenderTexture, Camera.Render and readback; only marker scene isolation differs. Cached framing and synchronous per-variant geometry export do not replace that path. The reviewed compiled marker initializer and PNG readback method bodies equal the earlier reviewed f3 resolved IL. Fresh 512/1024 AA1 marker PNGs are independently remeasured before transfer. This entry certifies coordinates only; same-frame signatures never override rejected BakeMesh consistency or certify shaders, fragment depth or anatomy.",
}

REVIEWED_PIPELINES["246ea304-9142-4b22-8c28-06a1feec0651"] = {
    **REVIEWED_PIPELINES["0096eab0-3abf-4da6-9b23-639e52470577"],
    "source_sha256": "8de6363e889063b560082f3559a9d4215dab961abb9a01b2be8a0428a9445ad3",
    "binary_review": "HS2Mod/artifacts/ocular_visibility_audit_20261005/force_matrix_pipeline_review.json",
    "binary_review_sha256": "bcaf23c5aaef79a802bd1d811c21397589f6a69717dcc8913ff9e7a4c69fa380",
    "assembly_sha256": "41b7be000fb84231ae2bb8c0c6a54e4889a6381b41995f66dabe304b73c631cc",
    "resolved_il_sha256": {
        "MakerRenderService::RenderHead": "a26e82123d9a9083b285eed2cabd07364ce494526a83ecf60a57ad9e68479175",
        "MakerRenderService::SaveRenderTexture": "0f769c45117683f9f4268383083b7c3b15eba85e07ddbc5f6c722f0c51387201",
        "MakerPixelCalibration::Initialize": "829ffe7651fe0dc9a8c326d776353f36031b83bdc8dacb35c54e6670113c28fb",
    },
    "review": "2026-10-06: independent force_matrix source/binary review binds loaded 246 MVID to 8de6363e render source and 41b7be00 DLL. Compiled RenderHead camera-to-readback prefix equals reviewed 0096 IL, with only whole-method code-size comments omitted; marker initializer and PNG readback helpers retain equal resolved IL. The new reversible opt-in skinning flag changes skinning policy and adds actual metadata, not camera projection/readback. Fresh exact-246 AA1 marker PNGs must still be remeasured. Coordinate transfer does not validate native skinning stability, surviving geometry equality, original shader fragment/depth visibility or anatomy.",
}

# This entry is explicit: no prior RenderHead whole-method hash is inherited.
REVIEWED_PIPELINES["ea8e1a9b-2dbd-45bb-83ef-824405408af5"] = {
    "source": "HS2Mod/plugins/HS2_McpBridge/MakerRenderService.cs",
    "source_sha256": "80c865bee392034dc81c6e1cf6f116230815ecb231ddd4e4064dde9bfc2273da",
    "marker_source": "HS2Mod/plugins/HS2_McpBridge/MakerPixelCalibration.cs",
    "marker_source_sha256": "d4e2e81bfcb28240415d695b856aed0e92c8c9541c6f2126bbd8865e1601d528",
    "binary_review": "HS2Mod/artifacts/ocular_visibility_audit_20261005/layer_serializer_pipeline_review_v1/report.json",
    "binary_review_sha256": "6ed93d976d401652ff994800379085054d1cea873e5c5383c6fd19dcb0db154c",
    "assembly_sha256": "cda3d754faa347b23fa5e6a73e4111704617ca5e2221ed19f86f854ebab5a67f",
    "resolved_il_sha256": {
        "MakerRenderService::RenderHead": "897bedddaae4427423c5f145afda3afac0b3029a77da01e8b874e41adbcf1c6d",
        "RenderHeadCoroutine::MoveNext": "3b98a8be2a600439acf74ebd4628c976826ef304979688008c66015a5702555c",
        "MakerPixelCalibration::Initialize": "829ffe7651fe0dc9a8c326d776353f36031b83bdc8dacb35c54e6670113c28fb",
        "MakerPixelCalibration::Create": "d196002b0ddc97e0b6731b66b159a42f1d5170003346e5dddedb1c552a4e3000",
        "MakerPixelCalibration::Dispose": "296c3f3e5ccfc226b11101c39490695c78df14c3c461e77a83b41675a9998f8d",
        "MakerRenderService::SaveRenderTexture": "0f769c45117683f9f4268383083b7c3b15eba85e07ddbc5f6c722f0c51387201",
        "MakerRenderService::CaptureMatrix": "99c38a57b1498f7e996cd2ceb1264b03a46d53d3dee765ba67ab68aec5241a64",
        "MakerRenderService::CharacterMask": "7098807857d6c4bc0fee9ac3797e8808187b67b655fefb50f6e85a6c0777ec7a",
    },
    "path": "MakerRenderService.RenderHead -> Camera.Render -> SaveRenderTexture -> Texture2D.ReadPixels -> EncodeToPNG",
    "review": "2026-10-05: exact ea8e1a9b/CDA module and preserved 80c865be render source are bound by independent offline PE/IL review. The six complete marker/camera/readback helpers equal reviewed 246 and 4f68; RenderHead and its coroutine have their own new whole-method hashes. Changes since 246 add actual capture-state comparisons and reversible isolated-layer filtering; the serializer fix replaces eight JsonConvert calls with existing JsonUtil.Object. Both ordinary and marker paths retain shared camera/RT/readback. Fresh exact-ea8e1a9b saved 512/1024 AA1 marker measurements are required and remeasured for each transfer. Coordinates only: this entry does not certify surviving geometry, conditional RGB effects, shader fragments, depth, restoration or anatomy.",
}

# Separate installed 0.30.1 review; all whole-method hashes are freshly measured.
REVIEWED_PIPELINES["e7aee083-0801-4327-bf13-69e6119a53ea"] = {
    "source": "HS2Mod/plugins/HS2_McpBridge/MakerRenderService.cs",
    "source_sha256": "7eb283e823f562c088da1b9980ae129d1dde729b72fbb0fcd843ea4a2fb10cc1",
    "marker_source": "HS2Mod/plugins/HS2_McpBridge/MakerPixelCalibration.cs",
    "marker_source_sha256": "d4e2e81bfcb28240415d695b856aed0e92c8c9541c6f2126bbd8865e1601d528",
    "binary_review": "HS2Mod/artifacts/ocular_visibility_audit_20261006/asset_provider_review_v1/current_4480_e7_pipeline_review_v2.json",
    "binary_review_sha256": "0f82f2671f5da2eab40aaa87a8c07e1445f0912886bba23bc5808926c43806c0",
    "source_review": "HS2Mod/artifacts/ocular_visibility_audit_20261006/asset_provider_review_v1/current_pipeline_source_addendum.json",
    "source_review_sha256": "d84bb83ffb7e1a10c35bf8525506de1f6bca929e673dc282b3f95e2b100c1c98",
    "assembly_sha256": "4480e6c6aa48332b431c1cdd875170fab3360df3fc3944ee423236234a1c47d2",
    "resolved_il_sha256": {
        "MakerRenderService::RenderHead": "d8153d6508d7f65102e119e7de2028d4070034c773ff000e177ff8284bbffefc",
        "RenderHeadCoroutine::MoveNext": "3b98a8be2a600439acf74ebd4628c976826ef304979688008c66015a5702555c",
        "MakerPixelCalibration::Initialize": "bf4aa335dabec95e4817706917d74d81fe3e46e3075d35e536a1a92c1509e4d0",
        "MakerPixelCalibration::Create": "d196002b0ddc97e0b6731b66b159a42f1d5170003346e5dddedb1c552a4e3000",
        "MakerPixelCalibration::Dispose": "296c3f3e5ccfc226b11101c39490695c78df14c3c461e77a83b41675a9998f8d",
        "MakerRenderService::SaveRenderTexture": "0f769c45117683f9f4268383083b7c3b15eba85e07ddbc5f6c722f0c51387201",
        "MakerRenderService::CaptureMatrix": "99c38a57b1498f7e996cd2ceb1264b03a46d53d3dee765ba67ab68aec5241a64",
        "MakerRenderService::CharacterMask": "7098807857d6c4bc0fee9ac3797e8808187b67b655fefb50f6e85a6c0777ec7a",
    },
    "review": "2026-10-06: exact installed e7/4480 module independently compared to fresh CDA IL and preserved sources. Five camera/marker/readback methods match; marker Initialize differs only in the version metadata literal and has its own new whole-method hash. RenderHead adds read-only native driver snapshots and paired metadata; its coroutine retains whole-method equality. New e7 marker PNGs must independently match every requested camera scope. Coordinate transfer only; no shader depth, anatomy, full actor/private history restoration or likeness certification.",
}

REVIEWED_PIPELINES["ee058847-2d3f-4812-a206-8d46a83bd6ad"] = {
    **REVIEWED_PIPELINES["e7aee083-0801-4327-bf13-69e6119a53ea"],
    "source_sha256": "7cac0ef8c3cf16672f72fa3294853266f9d5e49ecdd2f5b826025527e9683823",
    "binary_review": "HS2Mod/artifacts/ocular_visibility_audit_20261006/controlled_provider_review_v1/actor_pipeline_review.json",
    "binary_review_sha256": "0a3fc0954307934347adda2cde78f0b6d42935ba90f36d19b8ade84561eb96a8",
    "source_review": "HS2Mod/artifacts/ocular_visibility_audit_20261006/controlled_provider_review_v1/actor_pipeline_review.json",
    "source_review_sha256": "0a3fc0954307934347adda2cde78f0b6d42935ba90f36d19b8ade84561eb96a8",
    "assembly_sha256": "ec3fd9e886d22510b2ee1db8fd5b9301b05df2b0074bdbbd10c7b28e3c7624be",
    "resolved_il_sha256": {
        **REVIEWED_PIPELINES["e7aee083-0801-4327-bf13-69e6119a53ea"]["resolved_il_sha256"],
        "MakerPixelCalibration::Initialize": "d40f06dba4049c7f7dc9e147366d73b8af48ff31a2541c60573ecf09836cf46b",
    },
    "review": "2026-10-06: independently decompiled EC3/ee05 and preceding e7 PEs. Five shared camera/marker/readback helpers, RenderHead and its coroutine retain whole-method equality; Initialize changes only version metadata. Opt-in actor-transform export changes observation helpers only. Fresh marker measurements remain required for every ordinary camera scope. Coordinates only; actor stability during callbacks, full live-pose restoration, shader depth and anatomy remain uncertified.",
}

# Fresh installed 0.30.4 PE review; marker version literal has its own hash.
REVIEWED_PIPELINES["4b6a55a3-ceab-42b9-8b2f-6f0305eac3fa"] = {
    "source": "HS2Mod/plugins/HS2_McpBridge/MakerRenderService.cs",
    "source_sha256": "7cac0ef8c3cf16672f72fa3294853266f9d5e49ecdd2f5b826025527e9683823",
    "marker_source": "HS2Mod/plugins/HS2_McpBridge/MakerPixelCalibration.cs",
    "marker_source_sha256": "d4e2e81bfcb28240415d695b856aed0e92c8c9541c6f2126bbd8865e1601d528",
    "binary_review": "HS2Mod/artifacts/ocular_visibility_audit_20261006/breathing_off_pipeline_review_v1/report.json",
    "binary_review_sha256": "5060ee307e3ddb24a78350e3418550481238ed1da87937b6c67a84071d8ea051",
    "source_review": "HS2Mod/artifacts/ocular_visibility_audit_20261006/breathing_off_pipeline_review_v1/report.json",
    "source_review_sha256": "5060ee307e3ddb24a78350e3418550481238ed1da87937b6c67a84071d8ea051",
    "assembly_sha256": "634a9b2bd1c6a30f70f9b87609675194a75b48da1def7c52d7ac574d3882d0ad",
    "resolved_il_sha256": {
        "MakerRenderService::RenderHead": "d8153d6508d7f65102e119e7de2028d4070034c773ff000e177ff8284bbffefc",
        "RenderHeadCoroutine::MoveNext": "3b98a8be2a600439acf74ebd4628c976826ef304979688008c66015a5702555c",
        "MakerPixelCalibration::Initialize": "02a69e3625e78dbd500dae819c58dc415ec5f01a7149b1bc7e458cb6741a9654",
        "MakerPixelCalibration::Create": "d196002b0ddc97e0b6731b66b159a42f1d5170003346e5dddedb1c552a4e3000",
        "MakerPixelCalibration::Dispose": "296c3f3e5ccfc226b11101c39490695c78df14c3c461e77a83b41675a9998f8d",
        "MakerRenderService::SaveRenderTexture": "0f769c45117683f9f4268383083b7c3b15eba85e07ddbc5f6c722f0c51387201",
        "MakerRenderService::CaptureMatrix": "99c38a57b1498f7e996cd2ceb1264b03a46d53d3dee765ba67ab68aec5241a64",
        "MakerRenderService::CharacterMask": "7098807857d6c4bc0fee9ac3797e8808187b67b655fefb50f6e85a6c0777ec7a",
    },
    "review": "2026-10-06: actual 634a/4b6a PE independently decoded and compared to EC3. Current whole resolved IL hashes are measured; changed raw metadata tokens are distinguished from unchanged resolved bodies. Initialize changes only version literal. Renderer source remains 7cac0ef8; diagnostic effect observers do not replace the shared camera/RT/readback path. Fresh current-module512/1024AA1 PNG markers are independently remeasured for matching scopes. Coordinates only; finite breathing-off repeats do not certify continuous pose, full restoration, shaders, fragment depth, anatomy or likeness.",
}

# Explicit current 0.30.5 final PE; marker measurements still required.
REVIEWED_PIPELINES["caed7020-aa1f-4af1-8793-7c4e19f03c63"] = {'source': 'HS2Mod/plugins/HS2_McpBridge/MakerRenderService.cs', 'source_sha256': '7cac0ef8c3cf16672f72fa3294853266f9d5e49ecdd2f5b826025527e9683823', 'marker_source': 'HS2Mod/plugins/HS2_McpBridge/MakerPixelCalibration.cs', 'marker_source_sha256': 'd4e2e81bfcb28240415d695b856aed0e92c8c9541c6f2126bbd8865e1601d528', 'binary_review': 'HS2Mod/artifacts/ocular_visibility_audit_20261006/physics_capture_pipeline_review_v2/report.json', 'binary_review_sha256': 'bc373371ef2146564b9a33cea4b02aff4cbc96cca7a4bd77f05b9f750f7c5e0a', 'source_review': 'HS2Mod/artifacts/ocular_visibility_audit_20261006/physics_capture_pipeline_review_v2/report.json', 'source_review_sha256': 'bc373371ef2146564b9a33cea4b02aff4cbc96cca7a4bd77f05b9f750f7c5e0a', 'assembly_sha256': '6db501e0b2558c1bca154e7cf5aa654712cf81b4daf050e8af94dc9bea635226', 'resolved_il_sha256': {'MakerPixelCalibration::Initialize': '269a250a8b7d058fbc7099a16c9d42a2ca686736475ad80693e7da98e5a3c1d8', 'MakerPixelCalibration::Create': 'd196002b0ddc97e0b6731b66b159a42f1d5170003346e5dddedb1c552a4e3000', 'MakerPixelCalibration::Dispose': '296c3f3e5ccfc226b11101c39490695c78df14c3c461e77a83b41675a9998f8d', 'MakerRenderService::SaveRenderTexture': '0f769c45117683f9f4268383083b7c3b15eba85e07ddbc5f6c722f0c51387201', 'MakerRenderService::CaptureMatrix': '99c38a57b1498f7e996cd2ceb1264b03a46d53d3dee765ba67ab68aec5241a64', 'MakerRenderService::CharacterMask': '7098807857d6c4bc0fee9ac3797e8808187b67b655fefb50f6e85a6c0777ec7a', 'MakerRenderService::RenderHead': 'd8153d6508d7f65102e119e7de2028d4070034c773ff000e177ff8284bbffefc', 'RenderHeadCoroutine::MoveNext': '3b98a8be2a600439acf74ebd4628c976826ef304979688008c66015a5702555c'}, 'review': '2026-10-06: exact installed 0.30.5 final PE independently decoded; current whole resolved and raw method hashes measured separately. Explicit component physics pause is separately reviewed and does not replace shared camera/RT/readback. This registry entry requires fresh matching-module marker measurements and ordinary scope before coordinates certify; no solver/global-writer/restoration/shader/anatomy/likeness claim.'}

CAMERA_SCOPE_KEYS = (
    "bridge_mvid", "graphics_api", "graphics_uv_starts_at_top", "uses_reversed_z_buffer",
    "render_texture_format", "anti_aliasing", "allow_hdr", "allow_msaa",
    "rendering_path", "actual_rendering_path", "quality_antialiasing",
    "matrix_layout", "viewport_origin", "cpu_ndc_depth_range", "pixel_rect",
    "near_clip", "far_clip", "aspect", "roll", "projection",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def payload_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def scope(capture):
    camera = capture.get("capture_camera")
    if not isinstance(camera, dict):
        raise ContractError("Pixel scope requires capture_camera")
    result = {key: camera.get(key) for key in CAMERA_SCOPE_KEYS}
    result.update({key: capture.get(key) for key in ("width", "height", "orthographic", "color_space")})
    if any(value is None for value in result.values()):
        missing = [key for key, value in result.items() if value is None]
        raise ContractError(f"Missing pixel scope fields: {missing}")
    if not isinstance(result["bridge_mvid"], str) or not result["bridge_mvid"]:
        raise ContractError("Pixel scope requires bridge MVID")
    for key in ("orthographic", "allow_msaa", "allow_hdr", "graphics_uv_starts_at_top", "uses_reversed_z_buffer"):
        if type(result[key]) is not bool:
            raise ContractError(f"Pixel scope {key} must be boolean")
    for key in ("width", "height", "anti_aliasing", "quality_antialiasing"):
        if type(result[key]) is not int:
            raise ContractError(f"Pixel scope {key} must be integer")
    return result


def _same(a, b):
    if isinstance(a, (int, float, list)) and not isinstance(a, bool):
        try:
            aa, bb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
            return aa.shape == bb.shape and np.isfinite(aa).all() and np.isfinite(bb).all() and np.allclose(aa, bb, atol=2e-6, rtol=2e-6)
        except (TypeError, ValueError):
            return False
    return type(a) is type(b) and a == b


def _validated_half_pixel(case):
    direction, center, selected = (case.get(key, {}) for key in ("image_y_direction", "pixel_center_convention", "selected_convention"))
    return (
        case.get("status") == "validated"
        and direction.get("status") == "validated" and direction.get("png_y_top_left") is True
        and center.get("status") == "validated" and center.get("edge_coordinate_of_index_zero_center") == .5
        and center.get("integer_index_projection_translation") == -.5
        and selected.get("png_y_top_left") is True and selected.get("pixel_center_edge_offset") == .5
        and case.get("matches_expected_top_left_half_pixel_convention") is True
    )


@dataclass(frozen=True)
class PixelCertificate:
    capture_payload_sha256: str
    capture_png_path: str
    capture_png_sha256: str
    report_path: str
    report_sha256: str
    case_id: str
    source_metadata_sha256: str
    source_png_sha256: str
    matched_scope: dict
    transfer_review: dict | None

    def validate_capture(self, capture):
        if payload_digest(capture) != self.capture_payload_sha256:
            raise ContractError("Pixel certificate belongs to a different capture payload")
        if digest(self.capture_png_path) != self.capture_png_sha256:
            raise ContractError("Pixel-certified capture PNG changed")
        if digest(self.report_path) != self.report_sha256:
            raise ContractError("Pixel certificate report changed")

    def evidence(self):
        return {**self.__dict__, "pixel_contract_validated": True,
                "coordinate_convention": "top_left_pixel_centers_integer",
                "independent_marker_measurement_rechecked": True,
                "material_visibility_validated": False, "anatomical_correspondence_validated": False,
                "effective_multisampling_validated": False,
                "limits": "Coordinates only within matched runtime/camera/readback scope. Hashes bind files; they do not establish accuracy. No character shader, alpha/depth, skinning or anatomical validation."}


def certify_pixel_contract(report_path, capture):
    """Select a validated case, verify its files, remeasure, and bind to this view.

Aggregate report uncertainty does not invalidate individually validated cases.
For ordinary captures, only the explicitly reviewed shared pipeline is eligible.
"""
    report_path = Path(report_path).resolve()
    report = read_json(report_path)
    cases = report.get("cases", [report])
    if not isinstance(cases, list) or not cases:
        raise ContractError("Pixel report needs nonempty cases")
    target_scope = scope(capture)
    png = Path(capture.get("path", "")).resolve()
    if not png.is_file():
        raise ContractError("Pixel certificate requires the actual capture PNG")
    with Image.open(png) as image:
        if image.format != "PNG" or image.size != (capture["width"], capture["height"]):
            raise ContractError("Target PNG format/dimensions do not match capture")
    failures = []
    for case in cases:
        if not isinstance(case, dict) or not _validated_half_pixel(case):
            continue
        try:
            metadata = Path(case["metadata_path"]).resolve()
            source_png = Path(case["png_path"]).resolve()
            if digest(metadata) != case["metadata_sha256"] or digest(source_png) != case["png_sha256"]:
                raise ContractError("Pixel report source metadata/PNG SHA256 mismatch")
            source = unwrap(read_json(metadata))
            source_path = Path(source["path"])
            if not source_path.is_absolute():
                source_path = metadata.parent / source_path
            if source_path.resolve() != source_png:
                raise ContractError("Pixel report PNG does not match source metadata path")
            fresh = evaluate(source, source_png)
            if not _validated_half_pixel(fresh):
                raise ContractError("Independent saved PNG remeasurement does not validate top-left and .5")
            for key in ("image_y_direction", "pixel_center_convention", "selected_convention", "image_size",
                        "renderer_graph_scope", "capture_camera_contract", "capture_request_controls"):
                # A report cannot replace the independently recomputed measurement.
                if key not in case:
                    raise ContractError(f"Pixel report lacks measured scope: {key}")
                measurements = case[key].items() if isinstance(case[key], dict) else [(None, case[key])]
                for subkey, value in measurements:
                    if isinstance(fresh[key], dict):
                        if subkey not in fresh[key]:
                            continue  # New report annotations are not accuracy inputs.
                        actual = fresh[key][subkey]
                    else:
                        actual = fresh[key]
                    equal = value == actual if isinstance(value, dict) else _same(value, actual)
                    if not equal:
                        raise ContractError(f"Pixel report measurement differs from source: {key}/{subkey}")
            source_scope = scope(source)
            mismatches = [key for key in target_scope if not _same(target_scope[key], source_scope[key])]
            if mismatches:
                raise ContractError(f"Pixel runtime/camera scope mismatch: {mismatches}")
            review = None
            if capture.get("capture_kind") == "maker_character":
                review = REVIEWED_PIPELINES.get(target_scope["bridge_mvid"])
                if review is None:
                    raise ContractError("Ordinary capture pipeline/MVID has no reviewed coordinate transfer")
                if target_scope["anti_aliasing"] != 1 or target_scope["allow_msaa"] is not False:
                    raise ContractError("Ordinary coordinate transfer currently supports measured AA1 with allowMSAA=false only")
            elif capture.get("capture_kind") == "pixel_calibration_only":
                if payload_digest(source) != payload_digest(capture):
                    raise ContractError("Direct marker certificate requires its measured capture payload")
            else:
                raise ContractError("Unknown capture graph cannot receive pixel certificate")
            return PixelCertificate(payload_digest(capture), str(png), digest(png), str(report_path),
                                    digest(report_path), str(case.get("case", metadata.stem)),
                                    case["metadata_sha256"], case["png_sha256"], target_scope, review)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            failures.append(str(exc))
    raise ContractError("No independently validated pixel-report case matches this capture: " + "; ".join(failures))
