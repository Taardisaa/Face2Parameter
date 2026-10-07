"""Synthetic parser/raster fixtures; real Unity acceptance is run separately."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from core import Camera, ContractError
from pixel_certificate import certify_pixel_contract, REVIEWED_PIPELINES
from tools.pixel_calibration.analyze import analyze_file


def fixture(root, *, offset=.5, flip=True):
    width, height = 320, 192
    v = np.eye(4)
    v[0, 3], v[1, 3] = -width / 2, -height / 2
    p = np.diag([2 / width, 2 / height, -1, 1])
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    yy, xx = np.mgrid[:height, :width]
    markers = []
    centers = [(44.183, 42.781), (244.632, 35.131), (90.281, 139.713), (192.371, 164.121), (283.573, 90.481), (140.416, 82.263)]
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (0, 1, 1), (1, 0, 1)]
    for i, ((x, y), size, rgb) in enumerate(zip(centers, [9.7, 13.2, 17.1, 7.8, 11.4, 19.5], colors)):
        lo, hi = np.array([x, y]) - size / 2, np.array([x, y]) + size / 2
        corners = [[a, b, 0] for a, b in [(lo[0], lo[1]), (hi[0], lo[1]), (lo[0], hi[1]), (hi[0], hi[1])]]
        markers.append({"id": str(i), "rgb": list(rgb), "world_center": [x, y, 0], "world_corners": corners, "shader": "Unlit/Color"})
        if flip:
            lo[1], hi[1] = height - hi[1], height - lo[1]
        coverage = (xx + offset >= lo[0]) & (xx + offset < hi[0]) & (yy + offset >= lo[1]) & (yy + offset < hi[1])
        pixels[coverage] = np.array(rgb) * 255
    png = root / "source.png"
    Image.fromarray(pixels).save(png)
    source = {
        "path": str(png), "capture_kind": "pixel_calibration_only", "width": width, "height": height,
        "orthographic": True, "color_space": "Linear",
        "capture_camera": {
            "world_to_camera": v.flatten().tolist(), "camera_to_world": np.linalg.inv(v).flatten().tolist(),
            "projection": p.flatten().tolist(), "pixel_rect": [0, 0, width, height],
            "matrix_layout": "row_major_16; column_vectors", "cpu_ndc_depth_range": [-1, 1], "viewport_origin": "bottom_left",
            "image_coordinate_convention": "PNG pixel centers top_left; viewport y must be inverted",
            "image_y_flip_validated": False, "bridge_mvid": next(iter(REVIEWED_PIPELINES)),
            "graphics_api": "Direct3D11", "graphics_uv_starts_at_top": True, "uses_reversed_z_buffer": True,
            "render_texture_format": "ARGB32", "anti_aliasing": 1, "allow_hdr": True, "allow_msaa": False,
            "rendering_path": "UsePlayerSettings", "actual_rendering_path": "Forward", "quality_antialiasing": 0,
            "near_clip": .1, "far_clip": 1000, "aspect": width / height, "roll": 0,
        },
        "pixel_calibration": {"kind": "isolated_world_space_unlit_markers", "markers": markers,
                              "background_rgb": [0, 0, 0], "background_uniform_expected": True, "anti_aliasing": 1},
    }
    metadata = root / "source.json"
    metadata.write_text(json.dumps(source), encoding="utf-8")
    report = {"status": "uncertain", "cases": [{"case": "measured", **analyze_file(metadata)}, {"status": "uncertain"}]}
    report_path = root / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    target = deepcopy(source)
    target.update(capture_kind="maker_character", pixel_calibration=None, paired_pose_unchanged=True,
                  frame_count_before_render=17, frame_count=17, pose_signature_before_render="same",
                  pose_signature_after_render="same", paired_geometry={"pose_signature": "same", "frame_count": 17})
    target_png = root / "ordinary.png"
    Image.fromarray(np.full_like(pixels, 32)).save(target_png)
    target["path"] = str(target_png)
    return report_path, report, source, target


class PixelCertificateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.report_path, self.report, self.source, self.target = fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_valid_measured_case_in_uncertain_aggregate(self):
        cert = certify_pixel_contract(self.report_path, self.target)
        camera = Camera.from_capture(self.target, pixel_certificate=cert)
        self.assertTrue(camera.pixel_contract_validated)
        self.assertTrue(camera.pose_pairing_validated)
        self.assertEqual(cert.case_id, "measured")
        evidence = cert.evidence()
        self.assertFalse(evidence["anatomical_correspondence_validated"])
        self.assertFalse(evidence["material_visibility_validated"])
        self.assertIn("SaveRenderTexture", evidence["transfer_review"]["path"])

    def test_loose_flags_never_certify_pixel_or_pose(self):
        self.target["capture_camera"]["image_y_flip_validated"] = True
        fake = {"pixel_contract_validated": True, "pose_pairing_validated": True}
        with self.assertRaises(ContractError):
            Camera.from_capture(self.target, certification=fake)
        diagnostic = Camera.from_capture(self.target, certification=fake, diagnostic=True)
        self.assertFalse(diagnostic.pixel_contract_validated)
        cert = certify_pixel_contract(self.report_path, self.target)
        self.target["paired_geometry"]["frame_count"] = 18
        cert = certify_pixel_contract(self.report_path, self.target)
        with self.assertRaises(ContractError):
            Camera.from_capture(self.target, certification=fake, pixel_certificate=cert)

    def test_scope_mismatches_rejected(self):
        cases = [("bridge_mvid", "another-module"), ("graphics_api", "Vulkan"), ("render_texture_format", "ARGBHalf"),
                 ("allow_hdr", False), ("allow_msaa", True), ("actual_rendering_path", "DeferredShading"),
                 ("anti_aliasing", 4), ("quality_antialiasing", 4), ("pixel_rect", [1, 0, 319, 192]),
                 ("projection", np.eye(4).flatten().tolist())]
        for key, value in cases:
            with self.subTest(key=key):
                target = deepcopy(self.target)
                target["capture_camera"][key] = value
                with self.assertRaises(ContractError):
                    certify_pixel_contract(self.report_path, target)
        for key, value in (("orthographic", False), ("color_space", "Gamma"), ("width", 321)):
            with self.subTest(key=key):
                target = deepcopy(self.target)
                target[key] = value
                with self.assertRaises(ContractError):
                    certify_pixel_contract(self.report_path, target)

    def test_hash_binding_precedes_measurement(self):
        self.report["cases"][0]["metadata_sha256"] = "0" * 64
        self.report_path.write_text(json.dumps(self.report), encoding="utf-8")
        with self.assertRaisesRegex(ContractError, "SHA256 mismatch"):
            certify_pixel_contract(self.report_path, self.target)

    def test_forged_success_over_wrong_actual_pixel_offset(self):
        path, report, source, target = fixture(self.root, offset=0)
        wrong = report["cases"][0]
        wrong["selected_convention"]["pixel_center_edge_offset"] = .5
        wrong["pixel_center_convention"].update(edge_coordinate_of_index_zero_center=.5, integer_index_projection_translation=-.5)
        wrong["matches_expected_top_left_half_pixel_convention"] = True
        path.write_text(json.dumps(report), encoding="utf-8")
        with self.assertRaisesRegex(ContractError, "remeasurement"):
            certify_pixel_contract(path, target)

    def test_report_direction_only_is_insufficient(self):
        self.report["cases"][0]["pixel_center_convention"]["status"] = "uncertain"
        self.report_path.write_text(json.dumps(self.report), encoding="utf-8")
        with self.assertRaises(ContractError):
            certify_pixel_contract(self.report_path, self.target)

    def test_report_scope_cannot_disagree_with_hash_bound_metadata(self):
        self.report["cases"][0]["capture_camera_contract"]["actual_rendering_path"] = "DeferredShading"
        self.report_path.write_text(json.dumps(self.report), encoding="utf-8")
        with self.assertRaisesRegex(ContractError, "measurement differs"):
            certify_pixel_contract(self.report_path, self.target)

    def test_empty_pose_signature_is_not_structural_pairing(self):
        self.target.update(pose_signature_before_render="", pose_signature_after_render="")
        self.target["paired_geometry"]["pose_signature"] = ""
        cert = certify_pixel_contract(self.report_path, self.target)
        with self.assertRaises(ContractError):
            Camera.from_capture(self.target, pixel_certificate=cert, certification={"pose_pairing_validated": True})

    def test_certificate_reuse_rejects_changed_payload_or_png(self):
        cert = certify_pixel_contract(self.report_path, self.target)
        altered = deepcopy(self.target)
        altered["capture_camera"]["world_to_camera"][3] += 1
        with self.assertRaisesRegex(ContractError, "different capture"):
            Camera.from_capture(altered, pixel_certificate=cert)
        Image.new("RGB", (320, 192), (127, 127, 127)).save(self.target["path"])
        with self.assertRaisesRegex(ContractError, "PNG changed"):
            Camera.from_capture(self.target, pixel_certificate=cert)

    def test_unknown_pipeline_no_transfer_even_when_scope_matches(self):
        self.source["capture_camera"]["bridge_mvid"] = "unreviewed"
        metadata = self.root / "source.json"
        metadata.write_text(json.dumps(self.source), encoding="utf-8")
        self.report_path.write_text(json.dumps({"cases": [analyze_file(metadata)]}), encoding="utf-8")
        self.target["capture_camera"]["bridge_mvid"] = "unreviewed"
        with self.assertRaisesRegex(ContractError, "no reviewed"):
            certify_pixel_contract(self.report_path, self.target)


if __name__ == "__main__":
    unittest.main()
