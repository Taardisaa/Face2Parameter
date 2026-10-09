"""Analytic checks for the authoring math; no game parameter sampling."""
import unittest

import numpy as np

from tools.native_head.mother_arap import SpokesARAP
from tools.native_head.mother_candidate_review import section
from tools.native_head.mother_uv_anchors import embed
from tools.native_head.mother_default_pose import controller_weights, dense_frame


class MotherAuthoringTests(unittest.TestCase):
    def test_rigid_motion_is_stationary(self):
        vertices = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        faces = np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]])
        angle = .7
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                             [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        transformed = vertices@rotation.T+[2, 3, -1]
        arap = SpokesARAP(vertices, faces)
        np.testing.assert_allclose(arap.rotations(transformed),
            np.broadcast_to(rotation, (4, 3, 3)), atol=1e-12)
        np.testing.assert_allclose(arap.rhs(transformed),
                                   arap.stiffness@transformed, atol=1e-12)
        np.testing.assert_allclose(arap.stiffness@np.ones(4), 0, atol=1e-12)

    def test_uv_embeds_affine_point_and_rejects_overlapping_chart(self):
        arrays = dict(uv=np.array([[0., 0], [1, 0], [0, 1]]),
                      faces=np.array([[0, 1, 2]]))
        ids, bary, _ = embed([.2, .3], arrays)
        np.testing.assert_allclose(bary, [.5, .2, .3])
        np.testing.assert_allclose(bary@arrays['uv'][ids], [.2, .3])
        arrays['faces'] = np.array([[0, 1, 2], [0, 1, 2]])
        with self.assertRaisesRegex(ValueError, 'got 2'):
            embed([.2, .3], arrays)

    def test_section_keeps_edge_exactly_on_plane(self):
        vertices = np.array([[0., 0, 0], [0, 1, 0], [1, 0, 1]])
        segments = section(vertices, np.array([[0, 1, 2]]))
        self.assertEqual(len(segments), 1)
        np.testing.assert_array_equal(segments[0], [[0, 0], [0, 1]])

    def test_native_float32_open_weight_and_shared_close_open_channel(self):
        control = dict(OpenMin=0, OpenMax=0.8999999761581421, FixedRate=-.1,
            FBSTarget=[dict(ObjTarget={'m_PathID': 1}, PtnSet=[dict(Close=1, Open=0)])])
        target, weights = controller_weights(control, 1)[0]
        self.assertEqual(target, 1)
        self.assertEqual(weights, {1: 10., 0: 90.})
        control['FBSTarget'][0]['PtnSet'][0] = dict(Close=0, Open=0)
        self.assertEqual(controller_weights(control, 1)[0][1], {0: 100.})

    def test_sparse_frame_keeps_normal_only_delta(self):
        shapes = dict(channels=[dict(frameIndex=0, frameCount=1)], fullWeights=[100.],
            shapes=[dict(firstVertex=0, vertexCount=1)],
            vertices=[dict(index=1, vertex=dict(x=0, y=0, z=0),
                normal=dict(x=.1, y=.2, z=.3), tangent=dict(x=.3, y=.2, z=.1))])
        frame = dense_frame(shapes, 0, 3)
        np.testing.assert_array_equal(frame['vertex'], np.zeros((3, 3)))
        np.testing.assert_allclose(frame['normal'][1], [.1, .2, .3])


if __name__ == '__main__':
    unittest.main()
