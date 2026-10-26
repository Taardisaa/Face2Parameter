"""Input-assembler bindings and draw arguments of a generated mesh draw (D3D11, Unity 2018.4.11f1).

Transcribed from the installed UnityPlayer.dll (dumps under
``E:/HS2_offline_renderer_data/external_closure_20261025/disasm/ia/``):

* ``GfxDeviceD3D11Base::BindBuffersForDraw`` 0x5438e0: ``IASetVertexBuffers(0, streamCount,
  stream[i].buffer, stream[i].stride, offset 0)`` -- slots at and above ``streamCount`` keep the
  device's earlier bindings; ``IASetIndexBuffer(ib, R16/R32, 0)``.
* ``ExtractMeshBuffers`` 0x7a3f40: the rendering data's vertex streams in stream order (a
  skinned renderer's rendering data holds the skinning output as stream 0, Init 0x7a5560 from
  ``SkinnedMeshRenderer::AddAsRenderNode``); with an extra stream (the motion-vector previous
  positions) the list is truncated to ``CalculateRequiredStreamCount`` 0x793d50 (1 + the highest
  stream holding a required channel) and the extra stream appended, adding channel 8 (TexCoord4)
  to the required and available masks; then ``AddDefaultStreamsToMeshBuffers`` 0x793760 appends
  ``GfxDevice::GetDefaultVertexBuffer`` 0x8cd0c0 when (required & ~available) has Normal,
  Tangent or Color, or has texcoords while the mesh has none, or the lowest missing texcoord is
  below the mesh's lowest texcoord (no default when TexCoord0 is present).
* Draw arguments (``SubMesh``): IndexCount = indexCount, StartIndexLocation = firstByte /
  index size, BaseVertexLocation = baseVertex.

The program's required channel mask is taken as the vertex program's serialized
``ParserBindChannels.m_SourceMap``; the D3D11 VertexInputMasks the engine passes is not traced
(probe v8 request), so every derived slot is accepted only when it equals the captured binding.
"""
from __future__ import annotations

SHADER_CHANNEL_COUNT = 14            # 2018.4 Mesh m_VertexData channels (BlendWeight 12, BlendIndices 13)
CH_TEXCOORD4 = 8

_MESH_CACHE = {}


def lowest_bit(v):
    """``LowestBit`` 0x602df0 on an 8-bit value (index of the lowest set bit)."""
    v &= 0xFF
    for i in range(8):
        if v >> i & 1:
            return i
    return 8


def needs_default_stream(required, available):
    """``AddDefaultStreamsToMeshBuffers`` 0x793760 condition."""
    missing = ~available & required
    if not missing:
        return False
    if missing & 0xE:
        return True
    if not missing & 0xFF0:
        return False
    if (available >> 4) & 1:
        return False
    if not available & 0xFF0:
        return True
    return lowest_bit(missing >> 4) < lowest_bit(available >> 4)


def required_stream_count(channels, required):
    """``MeshVertexFormat::CalculateRequiredStreamCount`` 0x793d50 (channels: [(stream, present)])."""
    mask = required & available_mask(channels)
    n = 0
    for i, (stream, present) in enumerate(channels):
        if mask >> i & 1:
            n = max(n, stream + 1)
    return n


def available_mask(channels):
    return sum(1 << i for i, (stream, present) in enumerate(channels) if present)


def mesh_format(spec, path_id):
    """(channels [(stream, present)], stream strides, submeshes, index size) of an installed Mesh."""
    key = (spec, int(path_id))
    if key in _MESH_CACHE:
        return _MESH_CACHE[key]
    from .material_sources import load_env
    from .resources import mesh_streams_from_tree
    tree = None
    for o in load_env(spec).objects:
        if o.path_id == int(path_id) and o.type.name == 'Mesh':
            tree = o.read_typetree()
            break
    if tree is None:
        raise KeyError('mesh %s/%s not found' % (spec, path_id))
    ch = [(int(c['stream']), bool(c['dimension'])) for c in tree['m_VertexData']['m_Channels']]
    streams, n = mesh_streams_from_tree(tree)
    out = {'channels': ch, 'streams': [s['stride'] for s in streams], 'vertex_count': n,
           'submeshes': tree['m_SubMeshes'], 'index_size': 2 if int(tree['m_IndexFormat']) == 0 else 4,
           'name': tree['m_Name']}
    _MESH_CACHE[key] = out
    return out


_STREAM_CACHE = {}


def stream_bytes(spec, path_id, k):
    """Vertex stream ``k`` of an installed Mesh as uploaded (hs2pipe.resources rule: 16-byte aligned
    stream offsets in m_DataSize)."""
    key = (spec, int(path_id), int(k))
    if key not in _STREAM_CACHE:
        from .material_sources import load_env
        from .resources import mesh_streams_from_tree
        tree = next(o.read_typetree() for o in load_env(spec).objects
                    if o.path_id == int(path_id) and o.type.name == 'Mesh')
        data = bytes(tree['m_VertexData']['m_DataSize'])
        streams, n = mesh_streams_from_tree(tree)
        s = [x for x in streams if x['stream'] == int(k)]
        _STREAM_CACHE[key] = data[s[0]['offset']:s[0]['offset'] + s[0]['size']] if s else None
    return _STREAM_CACHE[key]


def draw_args(fmt, subset):
    """DrawIndexed arguments of submesh ``subset``."""
    sm = fmt['submeshes'][int(subset)]
    return {'IndexCount': int(sm['indexCount']), 'StartIndexLocation': int(sm['firstByte']) // fmt['index_size'],
            'BaseVertexLocation': int(sm['baseVertex'])}


def stream_plan(fmt, required, *, skinned, previous_stream=False):
    """Vertex streams the draw sets, as [('skin'|'skin_previous'|('mesh', k)|'default')]."""
    k = len(fmt['streams'])
    streams = [('mesh', i) for i in range(k)]
    if skinned:
        streams[0] = 'skin'
    req, avail = int(required), available_mask(fmt['channels'])
    if previous_stream:
        n = required_stream_count(fmt['channels'], req)
        streams = streams[:n] + ['skin_previous']
        req |= 1 << CH_TEXCOORD4
        avail |= 1 << CH_TEXCOORD4
    if needs_default_stream(req, avail):
        streams.append('default')
    return streams
