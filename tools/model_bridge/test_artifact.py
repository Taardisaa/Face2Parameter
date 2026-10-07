"""Check that the bridge cannot silently change original MICA output aliases."""
from pathlib import Path
import json
import tempfile
import unittest

import numpy as np

from .artifact import ModelArtifact, sha
from .export_game_mesh import export


class ArtifactPreservationTests(unittest.TestCase):
    def fixture(self, root, *, change_shape=False, change_mesh=False, change_local=False):
        faces = np.array([[0, 1, 2]], dtype=np.int64)
        shape = np.zeros((1, 300), dtype=np.float32)
        vertices = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=np.float32)
        raw = {"pred_shape_code": shape, "pred_canonical_shape_vertices": vertices,
               "faceid": np.zeros((1, 512), dtype=np.float32)}
        parameters, geometry, local = shape.copy(), vertices.copy(), vertices.copy()
        if change_shape:
            parameters[0, 0] = 1
        if change_mesh:
            geometry[0, 0, 0] = 1
        if change_local:
            local[0, 0, 0] = 1
        payload = {"output__" + key: value for key, value in raw.items()}
        payload.update(param__shape_params=parameters, geometry__vertices=geometry,
                       head_local__vertices=local, faces=faces)
        np.savez(root / "state.npz", faces_tensor=faces)
        np.savez(root / "image.npz", **payload)
        source = root / "source.py"
        source.write_text("# fixture, not a real model\n")
        manifest = {"format": "mica_flame_raw_export_v1", "raw_parameters_modified": False,
            "git_revision": "fixture", "sources": [{"path": str(source), "sha256": sha(source)}],
            "checkpoint": {"sha256": "fixture-checkpoint"},
            "flame_state": {"file": "state.npz", "sha256": sha(root / "state.npz")},
            "images": [{"input_sha256": "fixture-image", "file": "image.npz", "sha256": sha(root / "image.npz"),
                "parameter_fields": {"shape_params": {"shape": [1, 300], "dtype": "float32"}},
                "raw_output_fields": {key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                                      for key, value in raw.items()}}]}
        path = root / "manifest.json"
        path.write_text(json.dumps(manifest))
        return path

    def test_original_outputs_and_provenance_survive_interchange(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = ModelArtifact(self.fixture(root))
            artifact.verify_sources()
            export(artifact, root / "source_head.json", head_local=True)
            result = json.loads((root / "source_head.json").read_text())
            np.testing.assert_array_equal(result["vertices"], artifact.arrays["output__pred_canonical_shape_vertices"][0])
            np.testing.assert_array_equal(result["source_parameters"]["shape_params"], artifact.arrays["output__pred_shape_code"])
            self.assertEqual(result["source"]["decoder_sources"], artifact.manifest["sources"])
            self.assertEqual(result["source"]["checkpoint_sha256"], "fixture-checkpoint")

    def test_updated_artifact_digest_does_not_allow_alias_changes(self):
        for field in ["change_shape", "change_mesh", "change_local"]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):
                    ModelArtifact(self.fixture(Path(directory), **{field: True}))

    def test_changed_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = ModelArtifact(self.fixture(root))
            (root / "source.py").write_text("# different implementation\n")
            with self.assertRaises(ValueError):
                artifact.verify_sources()


if __name__ == "__main__":
    unittest.main()
