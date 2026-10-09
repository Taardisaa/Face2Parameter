"""Analytic checks for the authoring math; no game parameter sampling."""
import unittest
from types import SimpleNamespace

import numpy as np

from tools.native_head.mother_arap import SpokesARAP
from tools.native_head.mother_candidate_review import section
from tools.native_head.mother_uv_anchors import embed
from tools.native_head.mother_default_pose import controller_weights, dense_frame
from tools.native_head.mother_component_adaptation import adapt_frames
from tools.native_head.mother_reference_bindings import complete_hierarchy
from tools.native_head.mother_surface_warp import SurfaceWarp, direction_maps
from tools.native_head.mother_oriented_surface import OrientedSurface
from src.hs2_mesh_deform import HeadRig


class MotherAuthoringTests(unittest.TestCase):
    def test_near_wrong_side_cannot_hide_further_compatible_surface(self):
        triangles=np.array([[[0.,0,0],[0,1,0],[1,0,0]],
                            [[0,0,1],[1,0,1],[0,1,1]]])
        surface=OrientedSurface(triangles)
        point=np.array([[.2,.2,.1]])
        self.assertEqual(surface.closest(point)[1][0],0)
        closest, ids, _=surface.closest_oriented(point,[[0.,0,1]])
        self.assertEqual(ids[0],1)
        np.testing.assert_allclose(closest,[[.2,.2,1]])
        with self.assertRaisesRegex(ValueError,'No target triangle'):
            surface.closest_oriented(point,[[1.,0,0]])

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

    def test_surface_similarity_transports_offsets_and_shader_frame(self):
        source = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0]])
        angle = .45
        rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                             [-np.sin(angle), 0, np.cos(angle)]])
        scale, translation = 1.8, np.array([1., -2, .7])
        target = scale*source@rotation.T+translation
        warp = SurfaceWarp(source, target, [[0, 1, 2]])
        points = np.array([[.2, .3, .4], [.3, .2, -.2]])
        mapped, jac, _ = warp.map(points)
        np.testing.assert_allclose(mapped, scale*points@rotation.T+translation, atol=1e-12)
        np.testing.assert_allclose(jac, np.broadcast_to(scale*rotation, jac.shape), atol=1e-12)
        normals = np.tile([0., 0, 1], (2, 1))
        tangents = np.tile([1., 0, 0], (2, 1))
        nmap, tmap = direction_maps(jac, normals, tangents)
        np.testing.assert_allclose(np.einsum('nij,nj->ni', nmap, normals), normals@rotation.T)
        np.testing.assert_allclose(np.einsum('nij,nj->ni', tmap, tangents), tangents@rotation.T)

    def test_adaptation_unbakes_native_default_and_retains_sparse_structure(self):
        arrays = dict(verts=np.array([[0., 0, 0], [1, 0, 0]]),
            normals=np.tile([0., 0, 1], (2, 1)),
            tangents=np.tile([1., 0, 0, -1], (2, 1)))
        shapes = dict(channels=[dict(name='nativeClose', frameIndex=0, frameCount=1)], fullWeights=[100.],
            shapes=[dict(firstVertex=0, vertexCount=1)],
            vertices=[dict(index=1, vertex=dict(x=.2, y=0, z=0),
                normal=dict(x=.1, y=0, z=0), tangent=dict(x=0, y=.1, z=0))])
        target = np.array([[2., 3, 4], [4.4, 3, 4]])
        jac = np.broadcast_to(2*np.eye(3), (2, 3, 3))
        bind, adapted, reference, error = adapt_frames(arrays, shapes, {'0': 100}, target, jac)
        frame = dense_frame(adapted, 0, 2)
        np.testing.assert_allclose(bind['verts']+frame['vertex'], target)
        np.testing.assert_allclose(bind['normals']+frame['normal'], reference['normal'])
        np.testing.assert_allclose(bind['tangents'][:, :3]+frame['tangent'], reference['tangent'])
        self.assertLess(max(error.values()), 1e-12)
        self.assertEqual(adapted['channels'], shapes['channels'])
        self.assertEqual(adapted['shapes'], shapes['shapes'])
        self.assertEqual(adapted['vertices'][0]['index'], 1)
        np.testing.assert_array_equal(bind['tangents'][:, 3], [-1., -1])

    def test_complete_hierarchy_extends_skin_cache_and_rejects_mismatch(self):
        root = dict(name='root', parent='0', pos=[0, 0, 0], rot=[0, 0, 0, 1], scale=[1, 1, 1])
        rig = SimpleNamespace(bones={'1': root.copy()})
        rig._toposort = lambda: HeadRig._toposort(rig)
        renderer = dict(name='renderer', parent=1, pos=[0, 0, 0], rot=[0, 0, 0, 1], scale=[1, 1, 1])
        complete_hierarchy(rig, {'2': renderer, '1': root})
        self.assertEqual(rig._topo, ['1', '2'])
        self.assertEqual(rig.bones['2']['parent'], '1')
        renderer['pos'] = [1, 0, 0]
        with self.assertRaisesRegex(ValueError, 'TRS differs'):
            complete_hierarchy(rig, {'2': renderer, '1': root})


if __name__ == '__main__':
    unittest.main()
