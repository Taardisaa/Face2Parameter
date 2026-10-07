"""Small parser/raster tests, not substitutes for real Unity capture evidence."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from analyze import CalibrationError, evaluate


def fixture(path, offset=.5, flip=True, aa=1, srgb=False, rect=None, ambiguous=False):
    width, height = 320, 192
    rect = rect or [0, 0, width, height]
    v = np.eye(4)
    v[0, 3], v[1, 3] = -rect[2] / 2, -rect[3] / 2
    p = np.diag([2 / rect[2], 2 / rect[3], -1, 1])
    markers = []
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    centers = [(44.183, 42.781), (244.632, 35.131), (90.281, 139.713),
               (192.371, 164.121), (283.573, 90.481), (140.416, 82.263)]
    sizes = [9.7, 13.2, 17.1, 7.8, 11.4, 19.5]
    if ambiguous:
        centers = [(int(x) + .5, int(y) + .5) for x, y in centers]
        sizes = [10] * 6
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (0, 1, 1), (1, 0, 1)]
    yy, xx = np.mgrid[:height, :width]
    # Fixed regular D3D-style 4x sample locations, averaged before optional sRGB.
    samples = [(.375, .125), (.875, .375), (.125, .625), (.625, .875)] if aa == 4 else [(offset, offset)]
    for i, ((x, y), size, rgb) in enumerate(zip(centers, sizes, colors)):
        low, high = np.array([x, y]) - size / 2, np.array([x, y]) + size / 2
        vertices = [[a - rect[0], b - rect[1], 0] for a, b in
                    [(low[0], low[1]), (high[0], low[1]), (low[0], high[1]), (high[0], high[1])]]
        markers.append({"id": str(i), "rgb": list(rgb), "world_center": [x - rect[0], y - rect[1], 0],
                        "world_corners": vertices, "shader": "Unlit/Color"})
        lo, hi = low.copy(), high.copy()
        if flip:
            lo[1], hi[1] = height - high[1], height - low[1]
        coverage = np.zeros((height, width), float)
        for sx, sy in samples:
            coverage += ((xx + sx >= lo[0]) & (xx + sx < hi[0]) & (yy + sy >= lo[1]) & (yy + sy < hi[1])) / len(samples)
        if srgb:
            coverage = np.where(coverage <= .0031308, coverage * 12.92, 1.055 * coverage ** (1 / 2.4) - .055)
        pixels += np.rint(coverage[:, :, None] * np.array(rgb) * 255).astype(np.uint8)
    Image.fromarray(pixels).save(path)
    return {"path": str(path), "width": width, "height": height,
            "capture_camera": {"world_to_camera": v.flatten().tolist(), "projection": p.flatten().tolist(),
                               "pixel_rect": rect, "matrix_layout": "row_major_16; column_vectors"},
            "pixel_calibration": {"kind": "isolated_world_space_unlit_markers", "markers": markers,
                                  "background_rgb": [0, 0, 0], "background_uniform_expected": True,
                                  "anti_aliasing": aa}}


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "raster.png"

    def tearDown(self):
        self.temp.cleanup()

    def test_half_pixel_and_y_flip_independently_validated(self):
        report = evaluate(fixture(self.path), self.path)
        self.assertEqual(report["status"], "validated")
        self.assertTrue(report["image_y_direction"]["png_y_top_left"])
        self.assertEqual(report["pixel_center_convention"]["edge_coordinate_of_index_zero_center"], .5)

    def test_zero_offset_detected_in_actual_raster(self):
        report = evaluate(fixture(self.path, offset=0), self.path)
        self.assertEqual(report["pixel_center_convention"]["status"], "validated")
        self.assertEqual(report["pixel_center_convention"]["edge_coordinate_of_index_zero_center"], 0)

    def test_bottom_left_raster_detected(self):
        report = evaluate(fixture(self.path, flip=False), self.path)
        self.assertEqual(report["status"], "validated")
        self.assertFalse(report["image_y_direction"]["png_y_top_left"])

    def test_non_square_and_subviewport(self):
        report = evaluate(fixture(self.path, rect=[13, 7, 294, 178]), self.path)
        self.assertEqual(report["status"], "validated")

    def test_bad_color_and_nonblack_background_rejected(self):
        response = fixture(self.path)
        pixels = np.array(Image.open(self.path))
        pixels[4, 4] = [7, 12, 23]
        Image.fromarray(pixels).save(self.path)
        with self.assertRaises(CalibrationError):
            evaluate(response, self.path)

    def test_duplicate_blob_rejected(self):
        response = fixture(self.path)
        pixels = np.array(Image.open(self.path))
        pixels[4:6, 4:6] = [255, 0, 0]
        Image.fromarray(pixels).save(self.path)
        with self.assertRaisesRegex(CalibrationError, "exactly one"):
            evaluate(response, self.path)

    def test_invalid_projection_rejected(self):
        response = fixture(self.path)
        response["capture_camera"]["projection"] = [0] * 16
        with self.assertRaisesRegex(CalibrationError, "Singular"):
            evaluate(response, self.path)

    def test_aa4_linear_and_srgb_not_arbitrary_fitted(self):
        for srgb in (False, True):
            with self.subTest(srgb=srgb):
                report = evaluate(fixture(self.path, aa=4, srgb=srgb), self.path)
                self.assertEqual(report["status"], "validated")
                self.assertEqual({m["coverage_encoding"] for m in report["markers"]}, {"srgb" if srgb else "linear"})

    def test_aa4_blurred_levels_rejected(self):
        response = fixture(self.path, aa=4)
        pixels = np.array(Image.open(self.path))
        target = np.argwhere(np.any((pixels > 0) & (pixels < 255), axis=2))[0]
        pixels[tuple(target)] = [int(v * .8) for v in pixels[tuple(target)]]
        Image.fromarray(pixels).save(self.path)
        with self.assertRaises(CalibrationError):
            evaluate(response, self.path)

    def test_marker_specific_gamma_cannot_self_certify(self):
        response = fixture(self.path, aa=4)
        linear = np.array(Image.open(self.path))
        fixture(self.path, aa=4, srgb=True)
        srgb = np.array(Image.open(self.path))
        red = (linear[:, :, 0] > 0) & np.all(linear[:, :, 1:] == 0, axis=2)
        linear[red] = srgb[red]
        Image.fromarray(linear).save(self.path)
        with self.assertRaisesRegex(CalibrationError, "single capture-graph"):
            evaluate(response, self.path)

    def test_wrong_dimensions_rejected(self):
        response = fixture(self.path)
        response["height"] += 1
        with self.assertRaisesRegex(CalibrationError, "dimensions"):
            evaluate(response, self.path)

    def test_insufficient_half_pixel_precision_is_uncertain(self):
        report = evaluate(fixture(self.path, ambiguous=True), self.path)
        self.assertEqual(report["image_y_direction"]["status"], "validated")
        self.assertEqual(report["pixel_center_convention"]["status"], "uncertain")
        self.assertEqual(report["status"], "uncertain")

    def test_declared_multisampling_without_actual_edge_coverage_uncertain(self):
        for aa in (4, 8):
            with self.subTest(aa=aa):
                response = fixture(self.path)
                response["pixel_calibration"]["anti_aliasing"] = aa
                report = evaluate(response, self.path)
                self.assertEqual(report["status"], "uncertain")
                self.assertFalse(report["antialiasing_evidence"]["effective_multisampling_verified"])
                self.assertFalse(report["antialiasing_evidence"]["fractional_edge_levels_observed"])


if __name__ == "__main__":
    unittest.main()
