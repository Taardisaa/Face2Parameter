"""Check that the bridge cannot silently change original MICA output aliases."""
from pathlib import Path
import copy
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

    def two_images(self, root):
        path = self.fixture(root)
        manifest = json.loads(path.read_text())
        with np.load(root / 'image.npz') as data:
            second = {key: data[key].copy() for key in data.files}
        for key in ('param__shape_params', 'output__pred_shape_code'):
            second[key][0, 0] = 2
        for key in ('geometry__vertices', 'head_local__vertices', 'output__pred_canonical_shape_vertices'):
            second[key][0, :, 2] = .25
        np.savez(root / 'image2.npz', **second)
        row = copy.deepcopy(manifest['images'][0])
        row.update(file='image2.npz', sha256=sha(root / 'image2.npz'), input_sha256='fixture-second-image')
        manifest['images'].append(row)
        path.write_text(json.dumps(manifest))
        return path

    def test_selected_second_image_is_exported_and_resolved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); path = self.two_images(root)
            original = ModelArtifact(path, 1)
            export(original, root / 'second.json', head_local=True)
            source = json.loads((root / 'second.json').read_text())['source']
            self.assertEqual(source['image_index'], 1)
            resolved = ModelArtifact.from_game_source(source)
            self.assertEqual(resolved.image_index, 1)
            np.testing.assert_array_equal(resolved.mesh(head_local=True)[0], original.mesh(head_local=True)[0])
            self.assertFalse(np.array_equal(resolved.parameters['shape_params'], ModelArtifact(path, 0).parameters['shape_params']))
            source.pop('image_index')  # Legacy record: unique image+NPZ hashes still select row 1.
            self.assertEqual(ModelArtifact.from_game_source(source).image_index, 1)

    def test_wrong_index_and_provenance_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); path = self.two_images(root)
            export(ModelArtifact(path, 1), root / 'second.json', head_local=True)
            source = json.loads((root / 'second.json').read_text())['source']
            for key, value in [('image_index', 0), ('image_index', True), ('image_index', 2),
                               ('manifest_sha256', 'wrong'), ('model_revision', 'wrong'),
                               ('artifact_sha256', 'wrong')]:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    ModelArtifact.from_game_source({**source, key: value})

    def test_ambiguous_legacy_record_requires_explicit_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); path = self.fixture(root)
            manifest = json.loads(path.read_text())
            manifest['images'].append(copy.deepcopy(manifest['images'][0]))
            path.write_text(json.dumps(manifest))
            export(ModelArtifact(path, 1), root / 'second.json', head_local=True)
            source = json.loads((root / 'second.json').read_text())['source']
            self.assertEqual(ModelArtifact.from_game_source(source).image_index, 1)
            source.pop('image_index')
            with self.assertRaises(ValueError):
                ModelArtifact.from_game_source(source)


if __name__ == "__main__":
    unittest.main()
