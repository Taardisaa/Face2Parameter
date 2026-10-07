import copy
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from face_alignment import utils

from mapping import CoordinateError, bbox_parameters, decode_heatmaps, heatmap_to_raw, trace_crop
from verify import synthetic_heatmaps, verify_case


class MappingTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((160, 200, 3), np.uint8)
        self.center, self.scale = bbox_parameters([55.3, 55.8, 112.5, 111.9])
        self.cropped, self.trace = trace_crop(self.image, self.center, self.scale)

    def test_instrumented_actual_crop_equals_library(self):
        np.testing.assert_array_equal(self.cropped, utils.crop(self.image, self.center, self.scale))
        np.testing.assert_array_equal(self.trace["actual_ul_int"], utils.transform([1, 1], self.center, self.scale, 256, True))
        np.testing.assert_array_equal(self.trace["actual_br_int"], utils.transform([256, 256], self.center, self.scale, 256, True))

    def test_spies_restored_after_success_and_failure(self):
        transform, resize = utils.transform, cv2.resize
        trace_crop(self.image, self.center, self.scale)
        self.assertIs(utils.transform, transform)
        self.assertIs(cv2.resize, resize)
        # An extent rounded to zero causes native cv2.resize to fail.
        with self.assertRaises(cv2.error):
            trace_crop(self.image, [2000, 2000], .000001)
        self.assertIs(utils.transform, transform)
        self.assertIs(cv2.resize, resize)

    def test_source_hash_and_interpolation_changes_rejected(self):
        for field in ("utils_sha256", "api_sha256", "opencv_version"):
            changed = copy.deepcopy(self.trace)
            changed["installed_source"][field] = "changed"
            with self.assertRaises(CoordinateError):
                heatmap_to_raw([[32.5, 32.5]], changed)
        changed = copy.deepcopy(self.trace)
        changed["actual_resize"]["interpolation"] = cv2.INTER_NEAREST
        with self.assertRaises(CoordinateError):
            heatmap_to_raw([[32.5, 32.5]], changed)

    def test_preserves_fractional_actual_grid_not_native_integer(self):
        result = decode_heatmaps(synthetic_heatmaps(), self.trace)
        raw = np.asarray(result["raw_pixel_center_indices"])
        self.assertTrue(np.any(raw != np.trunc(raw)))
        native = np.asarray(result["native_preds_orig_integer"])
        self.assertTrue(np.all(native == np.trunc(native)))
        self.assertGreater(np.abs(native - raw).max(), .1)

    def test_real_decoder_quarter_offset_and_half_pixel(self):
        hm = np.zeros((1, 1, 64, 64), np.float32)
        hm[0, 0, 20, 30] = 2
        hm[0, 0, 20, 31] = 1
        hm[0, 0, 21, 30] = 1
        result = decode_heatmaps(hm, self.trace)
        self.assertEqual(result["heatmap_decoded_edge_coordinates"], [[30.75, 20.75]])
        self.assertEqual(result["crop_pixel_center_indices"], [[122.5, 82.5]])

    def test_heatmap_shapes_nan_and_wrong_coordinate_domain_rejected(self):
        for shape in ((1, 1, 64, 32), (2, 1, 64, 64), (1, 1, 32, 32)):
            with self.assertRaises(CoordinateError):
                decode_heatmaps(np.zeros(shape), self.trace)
        bad = np.zeros((1, 1, 64, 64), np.float32)
        bad[0, 0, 0, 0] = np.nan
        with self.assertRaises(CoordinateError):
            decode_heatmaps(bad, self.trace)
        with self.assertRaises(CoordinateError):
            heatmap_to_raw([[-.5, 10]], self.trace)

    def test_padding_is_uncertain_not_fitted(self):
        center, scale = bbox_parameters([-18.6, -11.2, 78.7, 80.4])
        _, trace = trace_crop(np.zeros((192, 256, 3), np.uint8), center, scale)
        result = heatmap_to_raw([[.5, .5], [40.5, 40.5]], trace)
        self.assertEqual(result["status"], ["uncertain_padding", "mapped_grid_only"])

    def test_actual_encoded_png_crop_reads_distinguish_half_pixel(self):
        with tempfile.TemporaryDirectory() as directory:
            result = verify_case(Path(directory), "test", (320, 320), [102.35, 109.75, 206.8, 211.45])
        self.assertEqual(result["status"], "validated_crop_grid")
        for axis in result["ramp_measurements"]:
            self.assertLess(axis["measured_error"]["max_abs_pixels"], .25)
            self.assertGreater(axis["wrong_plus_half_pixel_convention_error"]["rmse_pixels"], .35)
            self.assertGreater(axis["wrong_minus_half_pixel_convention_error"]["rmse_pixels"], .35)


if __name__ == "__main__":
    unittest.main()
