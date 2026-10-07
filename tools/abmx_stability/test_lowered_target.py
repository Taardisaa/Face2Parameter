"""Offline source-only target guards; no game calls or candidate observations."""
import copy
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from tools.abmx_multibone.geometry import DEFAULT_BONES
from tools.abmx_replay.validate_trace import read
from tools.abmx_stability.lowered_target import numpy_native_locals, prepare_target

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT/'outputs/abmx_stable_lowering_20261005/original_combined_source_only/compiled_execution.json'
CONTRACT = ROOT/'outputs/abmx_replay_20261005/installed_contract_v2.json'


class NativeDecompositionTests(unittest.TestCase):
    def make_rig_world(self):
        rig = SimpleNamespace(name2pid={name: i+1 for i, name in enumerate(DEFAULT_BONES)},
                              bones={i+1: {'parent': 0} for i in range(4)})
        parent = np.eye(4)
        parent[:3, :3] = 2*Rotation.from_euler('z', 47, degrees=True).as_matrix()
        parent[:3, 3] = [100, -50, 23]
        world, local = {0: parent}, np.eye(4)
        local[:3, :3] = Rotation.from_euler('xyz', [13, -21, 7], degrees=True).as_matrix() @ np.diag([.8, 1.1, 1.3])
        local[:3, 3] = [.2, .3, -.4]
        for pid in range(1, 5):
            world[pid] = parent @ local
        return rig, world, local

    def test_parent_world_pose_removed_before_local_decomposition(self):
        rig, world, local = self.make_rig_world()
        with patch('tools.abmx_stability.lowered_target._fk_world', return_value=world):
            output = numpy_native_locals(rig, [.5]*59)
        for row in output.values():
            np.testing.assert_allclose(row['local_position'], local[:3, 3], atol=2e-8, rtol=0)
            np.testing.assert_allclose(row['local_scale'], [.8, 1.1, 1.3], atol=6e-8, rtol=0)
            restored = Rotation.from_quat(row['local_rotation_xyzw']).as_matrix() @ np.diag(row['local_scale'])
            np.testing.assert_allclose(restored, local[:3, :3], atol=1e-7, rtol=0)

    def test_shear_and_reflection_rejected(self):
        for mode in ('shear', 'reflection'):
            rig, world, local = self.make_rig_world()
            if mode == 'shear':
                local[0, 1] += .1
            else:
                local[:3, 0] *= -1
            world[1] = world[0] @ local
            with patch('tools.abmx_stability.lowered_target._fk_world', return_value=world), self.assertRaisesRegex(ValueError, 'shear/reflection'):
                numpy_native_locals(rig, [.5]*59)


@unittest.skipUnless(ARTIFACT.exists() and CONTRACT.exists(), 'Local frozen source-only fixture unavailable')
class FrozenSourceGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifact, cls.contract = read(ARTIFACT), read(CONTRACT)

    def test_source_only_first_target_fullhead_native_guard(self):
        target = prepare_target(self.artifact, self.contract)
        self.assertEqual(target['vertices'].shape, (4439, 3))
        self.assertFalse(target['evidence']['target_input_candidate_after_or_counts'])
        self.assertLessEqual(target['evidence']['source_fullhead_guard']['rigid_errors']['max_normalized'], 1e-5)

    def test_stale_source_hash_rejected(self):
        artifact = copy.deepcopy(self.artifact)
        path = next(iter(artifact['provenance']['source_files']))
        artifact['provenance']['source_files'][path] = '0'*64
        with self.assertRaisesRegex(ValueError, 'Frozen compiler/source history changed'):
            prepare_target(artifact, self.contract)

    def test_changed_candidate_native_rejected(self):
        artifact = copy.deepcopy(self.artifact)
        artifact['compiler_inputs']['declared']['native59'][0] = .7
        with self.assertRaisesRegex(ValueError, 'same-native source'):
            prepare_target(artifact, self.contract)

    def test_compiler_numerical_target_is_checked_not_used(self):
        artifact = copy.deepcopy(self.artifact)
        artifact['bone_diagnostics'][DEFAULT_BONES[0]]['logical_first_prediction']['after']['local_position'][0] += .1
        with self.assertRaisesRegex(ValueError, 'independent source-only prediction'):
            prepare_target(artifact, self.contract)


if __name__ == '__main__':
    unittest.main()
