"""Unity 2018 uncompressed mesh writing with explicit vertex/skin channels."""
from __future__ import annotations

import copy
import numpy as np


def basis(vertices, faces, uv):
    vertices = np.asarray(vertices, float); faces = np.asarray(faces, int)
    normals = np.zeros_like(vertices); tangent = np.zeros_like(vertices); bitangent = np.zeros_like(vertices)
    xyz = vertices[faces]; tex = np.asarray(uv)[faces]
    e1, e2 = xyz[:, 1]-xyz[:, 0], xyz[:, 2]-xyz[:, 0]
    cross = np.cross(e1, e2)
    d1, d2 = tex[:, 1]-tex[:, 0], tex[:, 2]-tex[:, 0]
    det = d1[:, 0]*d2[:, 1]-d1[:, 1]*d2[:, 0]
    inv = np.divide(1., det, out=np.zeros_like(det), where=np.abs(det)>1e-12)
    t = (e1*d2[:, 1, None]-e2*d1[:, 1, None])*inv[:, None]
    b = (e2*d1[:, 0, None]-e1*d2[:, 0, None])*inv[:, None]
    for i in range(3):
        np.add.at(normals, faces[:, i], cross)
        np.add.at(tangent, faces[:, i], t); np.add.at(bitangent, faces[:, i], b)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-20)
    tangent -= normals*np.sum(normals*tangent, axis=1, keepdims=True)
    invalid = np.linalg.norm(tangent, axis=1)<1e-12
    axes = np.eye(3)[np.argmin(np.abs(normals[invalid]), axis=1)]
    tangent[invalid] = np.cross(axes, normals[invalid])
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-20)
    sign = np.where(np.sum(np.cross(normals, tangent)*bitangent, axis=1)<0, -1., 1.)
    return normals.astype('<f4'), np.column_stack([tangent, sign]).astype('<f4')


def aabb(vertices):
    low, high = np.asarray(vertices).min(0), np.asarray(vertices).max(0)
    return {'m_Center': dict(zip('xyz', map(float, (low+high)/2))),
            'm_Extent': dict(zip('xyz', map(float, (high-low)/2)))}


def write_mesh(reader, vertices, faces, uv, indices, weights, bindposes, *, uv1=None, colors=None, normals=None):
    tree = copy.deepcopy(reader.read_typetree())
    vertices = np.asarray(vertices, '<f4'); faces = np.asarray(faces, '<u2')
    uv = np.asarray(uv, '<f4'); indices = np.asarray(indices, '<u4'); weights = np.asarray(weights, '<f4')
    count = len(vertices)
    if count>65535 or len(vertices)==0 or faces.max()>=count or not np.isfinite(vertices).all():
        raise ValueError('Finite, nonempty UInt16 mesh required')
    if indices.shape!=(count, 4) or weights.shape!=(count, 4) or indices.max()>=len(bindposes):
        raise ValueError('Four valid native bone influences per vertex required')
    if not np.allclose(weights.sum(1), 1, atol=2e-7) or (weights<0).any():
        raise ValueError('Normalized nonnegative skin weights required')
    computed_normals, tangents = basis(vertices, faces, uv)
    normals = computed_normals if normals is None else np.asarray(normals, '<f4')
    uv1 = uv if uv1 is None else np.asarray(uv1, '<f4')
    colors = np.ones((count, 4), '<f4') if colors is None else np.asarray(colors, '<f4')
    channels = [dict(stream=0, offset=0, format=0, dimension=0) for _ in range(14)]
    # All channels float32 except bone indices (Unity 2018 UInt32 format=11).
    arrays = [(0, vertices, 0), (1, normals, 0), (2, tangents, 0), (3, colors, 0),
              (4, uv, 0), (5, uv1, 0), (6, uv, 0), (12, weights, 0), (13, indices, 11)]
    dtype = []; offset = 0
    for channel, values, fmt in arrays:
        width = values.shape[1]
        channels[channel] = dict(stream=0, offset=offset, format=fmt, dimension=width)
        dtype.append((str(channel), '<u4' if fmt==11 else '<f4', (width,)))
        offset += width*4
    stream = np.empty(count, dtype=dtype)
    for channel, values, _ in arrays:
        stream[str(channel)] = values
    tree['m_VertexData'] = {'m_VertexCount': count, 'm_Channels': channels, 'm_DataSize': stream.tobytes()}
    tree['m_IndexFormat'] = 0; tree['m_IndexBuffer'] = list(faces.tobytes())
    bounds = aabb(vertices)
    tree['m_SubMeshes'] = [dict(firstByte=0, indexCount=faces.size, topology=0, baseVertex=0,
                               firstVertex=0, vertexCount=count, localAABB=bounds)]
    tree['m_LocalAABB'] = bounds; tree['m_MeshCompression'] = 0; tree['m_IsReadable'] = True
    tree['m_Shapes'] = dict(vertices=[], shapes=[], channels=[], fullWeights=[])
    tree['m_BindPose'] = [{f'e{r}{c}': float(matrix[r, c]) for r in range(4) for c in range(4)} for matrix in bindposes]
    tree['m_StreamData'] = dict(offset=0, size=0, path='')
    tree['m_BakedConvexCollisionMesh'] = []; tree['m_BakedTriangleCollisionMesh'] = []
    for packed in tree['m_CompressedMesh'].values():
        if isinstance(packed, dict) and packed.get('m_NumItems', 0):
            raise ValueError('Compressed template not supported')
    if len(tree['m_BoneNameHashes']) != len(bindposes):
        # Runtime rebinding uses explicit renderer transform names, not imported
        # model-optimization hashes. Added boundary root is deliberately unhashed.
        tree['m_BoneNameHashes'] = tree['m_BoneNameHashes'][:len(bindposes)] + [0]*max(0,len(bindposes)-len(tree['m_BoneNameHashes']))
    reader.save_typetree(tree)
    return dict(vertices=count, triangles=len(faces), bones=len(bindposes), expression_channels=0)
