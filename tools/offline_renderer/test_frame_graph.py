"""CPU checks of the frame-graph interpreter on synthetic RenderDoc XML fixtures.

The fixtures are hand-built D3D11 chunk streams (not HS2 evidence): they cover
DXBC declaration parsing, driver event binding, capture-start buffer records,
Map/Unmap provenance, render-target producer edges, pass grouping/segments,
head-surface binding by exact index lists and explicit rejection branches.
"""
import json
import struct
import tempfile
import unittest
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from .frame_graph import (FrameGraph, ShaderLibrary, Blobs, UnresolvedFrame, load_chunks, load_driver_listing,
                          parse_declarations, reflect, group_passes, segment_passes, bind_surfaces, resource_edges,
                          view_subresources, decode_element, Repeated, temporal_dependencies)


# ------------------------------------------------------------ DXBC fixtures
def shader_tokens(program_type, declarations):
    body = []
    for d in declarations:
        body += d
    body.append(62 | (1 << 24))  # ret
    return [(program_type << 16) | 0x50, 2 + len(body)] + body


def dcl_cb(slot, float4):
    return [89 | (4 << 24), 2 | (8 << 12) | (2 << 20), slot, float4]


def dcl_texture2d(slot):
    return [88 | (3 << 11) | (4 << 24), (7 << 12) | (1 << 20), slot, 0x5555]


def dcl_sampler(slot):
    return [90 | (3 << 24), (6 << 12) | (1 << 20), slot]


def signature(names):
    count = len(names)
    head = struct.pack('<II', count, 8)
    strings = b''
    rows = b''
    base = 8 + 24 * count
    for i, n in enumerate(names):
        rows += struct.pack('<5IBBH', base + len(strings), 0, 0, 3, i, 0x7, 0x7, 0)
        strings += n.encode() + b'\0'
    return head + rows + strings


def dxbc(tokens, input_names=None):
    chunks = [(b'SHDR', struct.pack('<%dI' % len(tokens), *tokens))]
    if input_names:
        chunks.insert(0, (b'ISGN', signature(input_names)))
    header = 32 + 4 * len(chunks)
    offsets, payload = [], b''
    for fourcc, data in chunks:
        offsets.append(header + len(payload))
        payload += fourcc + struct.pack('<I', len(data)) + data
    total = header + len(payload)
    return b'DXBC' + b'\0' * 16 + struct.pack('<III', 1, total, len(chunks)) + struct.pack('<%dI' % len(offsets), *offsets) + payload


# ------------------------------------------------------------- XML builder
class X:
    @staticmethod
    def u(name, v):
        e = ET.Element('uint', name=name) if name else ET.Element('uint')
        e.text = str(v)
        return e

    @staticmethod
    def rid(name, v):
        e = ET.Element('ResourceId', name=name) if name else ET.Element('ResourceId')
        e.text = str(v)
        return e

    @staticmethod
    def f(name, v):
        e = ET.Element('float', name=name) if name else ET.Element('float')
        e.text = repr(float(v))
        return e

    @staticmethod
    def b(name, v):
        e = ET.Element('bool', name=name)
        e.text = 'true' if v else 'false'
        return e

    @staticmethod
    def s(name, v):
        e = ET.Element('string', name=name)
        e.text = v
        return e

    @staticmethod
    def null(name):
        return ET.Element('null', name=name)

    @staticmethod
    def buf(name, index, length):
        e = ET.Element('buffer', name=name, byteLength=str(length))
        e.text = str(index)
        return e

    @staticmethod
    def arr(name, items):
        e = ET.Element('array', name=name)
        e.extend(items)
        return e

    @staticmethod
    def st(name, children):
        e = ET.Element('struct', name=name) if name else ET.Element('struct')
        e.extend(children)
        return e


def ids(name, values):
    return X.arr(name, [X.rid(None, v) for v in values])


def uints(name, values):
    return X.arr(name, [X.u(None, v) for v in values])


def stage_state(name):
    return X.st(name, [X.rid('Object', 0), ids('ConstantBuffers', [0] * 14), uints('CBOffsets', [0] * 14),
                       uints('CBCounts', [4096] * 14), ids('SRVs', [0] * 128), ids('Samplers', [0] * 16),
                       ids('Instances', [0] * 256), X.u('NumInstances', 0)])


def initial_state():
    return X.st('state', [
        X.st('IA', [X.rid('Layout', 0), X.u('Topo', 4), ids('VBs', [0] * 32), uints('Strides', [0] * 32),
                    uints('Offsets', [0] * 32), X.rid('IndexBuffer', 0), X.u('IndexFormat', 57), X.u('IndexOffset', 0)]),
        *[stage_state(s) for s in ('VS', 'HS', 'DS', 'GS', 'PS', 'CS')],
        ids('CSUAVs', [0] * 64),
        X.st('SO', [ids('Buffers', [0] * 4), uints('Offsets', [0] * 4)]),
        X.st('RS', [X.u('NumViews', 0), X.u('NumScissors', 0), X.arr('Viewports', []), X.arr('Scissors', []), X.rid('State', 0)]),
        X.st('OM', [X.rid('DepthStencilState', 0), X.u('StencRef', 0), X.rid('BlendState', 0), X.arr('BlendFactor', []),
                    X.u('SampleMask', 0xffffffff), X.rid('DepthView', 0), ids('RenderTargets', [0] * 8),
                    X.u('UAVStartSlot', 0), ids('UAVs', [0] * 8)]),
        X.rid('Predicate', 0), X.b('PredicateValue', False)])


class Fixture:
    """Builds zip.xml + zip + driver listing; event ids follow RenderDoc (one per API chunk)."""

    def __init__(self):
        self.root = ET.Element('rdc')
        header = ET.SubElement(self.root, 'header')
        ET.SubElement(header, 'driver').text = 'D3D11'
        self.chunks = ET.SubElement(self.root, 'chunks', version='20')
        self.blobs = []
        self.index = 0
        self.captured = False
        self.event = 0
        self.api = []
        self.actions = []

    def blob(self, data):
        self.blobs.append(data)
        return len(self.blobs) - 1

    def chunk(self, name, children, action=None):
        e = ET.SubElement(self.chunks, 'chunk', chunkIndex=str(self.index), name=name)
        if name.startswith('ID3D11DeviceContext::') or name == 'IDXGISwapChain::Present':
            e.append(X.rid('Context', 7))
        e.extend(children)
        if self.captured and (name.startswith('ID3D11DeviceContext::') or name == 'IDXGISwapChain::Present'):
            self.event += 1
            self.api.append({'event_id': self.event, 'chunk_index': self.index})
            if action:
                self.actions.append({'event': 'action', 'event_id': self.event, 'name': name + '()', 'api_events': self.api})
                self.api = []
        self.index += 1
        return self.event

    def write(self, directory):
        directory = Path(directory)
        xml = directory / 'f.zip.xml'
        ET.ElementTree(self.root).write(xml)
        with zipfile.ZipFile(directory / 'f.zip', 'w') as z:
            for i, data in enumerate(self.blobs):
                z.writestr('%06d' % i, data)
        listing = directory / 'inspect.bin'
        rows = [{'event': 'opened', 'driver': 'D3D11', 'version': '1.46'}] + self.actions
        listing.write_bytes(b'\n'.join(json.dumps(r).encode() for r in rows) + b'\n')
        return xml, directory / 'f.zip', listing


def create_buffer(fx, ident, data, bind=1, usage=0, name=None):
    children = [X.st('pDesc', [X.u('ByteWidth', len(data)), X.u('Usage', usage), X.u('BindFlags', bind),
                               X.u('CPUAccessFlags', 65536 if usage == 2 else 0), X.u('MiscFlags', 0), X.u('StructureByteStride', 0)])]
    children.append(X.buf('InitialData', fx.blob(data), len(data)))
    children.append(X.rid('pBuffer', ident))
    fx.chunk('ID3D11Device::CreateBuffer', children)
    if name:
        fx.chunk('ID3D11Resource::SetDebugName', [X.rid('pResource', ident), X.s('Name', name)])


def create_rt(fx, tex, rtv, srv, w=4, h=4, name=None):
    fx.chunk('ID3D11Device::CreateTexture2D', [
        X.st('Descriptor', [X.u('Width', w), X.u('Height', h), X.u('MipLevels', 1), X.u('ArraySize', 1), X.u('Format', 28),
                            X.st('SampleDesc', [X.u('Count', 1), X.u('Quality', 0)]), X.u('Usage', 0), X.u('BindFlags', 40),
                            X.u('CPUAccessFlags', 0), X.u('MiscFlags', 0)]),
        X.arr('pInitialData', []), X.rid('pTexture', tex)])
    fx.chunk('ID3D11Device::CreateRenderTargetView', [X.rid('pResource', tex), X.st('pDesc', [X.u('Format', 28), X.u('ViewDimension', 4), X.st('Texture2D', [X.u('MipSlice', 0)])]), X.rid('pView', rtv)])
    fx.chunk('ID3D11Device::CreateShaderResourceView', [X.rid('pResource', tex), X.st('pDesc', [X.u('Format', 28), X.u('ViewDimension', 4), X.st('Texture2D', [X.u('MostDetailedMip', 0), X.u('MipLevels', 1)])]), X.rid('pView', srv)])
    if name:
        fx.chunk('ID3D11Resource::SetDebugName', [X.rid('pResource', tex), X.s('Name', name)])


VB, IB, CB, RT_A, RTV_A, SRV_A, RT_B, RTV_B, SRV_B, VS, PS, LAYOUT, VB_PREV = 100, 101, 102, 110, 111, 112, 120, 121, 122, 130, 131, 132, 103


def standard_fixture(*, unmap_before_capture=False, second_ib=None):
    fx = Fixture()
    fx.chunk('Internal::DeviceInitialisation', [X.rid('ImmediateContext', 7)])
    create_buffer(fx, VB, struct.pack('<12f', *range(12)), usage=2, name='Buffer-1-48')
    create_buffer(fx, IB, struct.pack('<6H', 0, 1, 2, 2, 1, 3), bind=2, name='o_head(Clone)')
    create_buffer(fx, CB, b'\0' * 64, bind=4, usage=2, name='ConstantBuffer-1-64')
    if second_ib:
        create_buffer(fx, 104, struct.pack('<%dH' % len(second_ib), *second_ib), bind=2, name='o_tang(Clone)')
    create_rt(fx, RT_A, RTV_A, SRV_A, name='TempBuffer A')
    create_rt(fx, RT_B, RTV_B, SRV_B, name='TempBuffer B')
    vs = dxbc(shader_tokens(1, [dcl_cb(0, 4)]), ['POSITION'])
    ps = dxbc(shader_tokens(0, [dcl_sampler(0), dcl_texture2d(0)]))
    fx.chunk('ID3D11Device::CreateVertexShader', [X.buf('pShaderBytecode', fx.blob(vs), len(vs)), X.u('BytecodeLength', len(vs)), X.rid('pClassLinkage', 0), X.rid('pShader', VS)])
    fx.chunk('ID3D11Resource::SetDebugName', [X.rid('pResource', VS), X.s('Name', 'AIT/Test')])
    fx.chunk('ID3D11Device::CreatePixelShader', [X.buf('pShaderBytecode', fx.blob(ps), len(ps)), X.u('BytecodeLength', len(ps)), X.rid('pClassLinkage', 0), X.rid('pShader', PS)])
    fx.chunk('ID3D11Resource::SetDebugName', [X.rid('pResource', PS), X.s('Name', 'AIT/Test')])
    fx.chunk('ID3D11Device::CreateInputLayout', [X.arr('pInputElementDescs', [X.st(None, [
        X.s('SemanticName', 'POSITION'), X.u('SemanticIndex', 0), X.u('Format', 6), X.u('InputSlot', 0),
        X.u('AlignedByteOffset', 0), X.u('InputSlotClass', 0), X.u('InstanceDataStepRate', 0)])]),
        X.u('NumElements', 1), X.buf('pShaderBytecodeWithInputSignature', fx.blob(vs), len(vs)), X.u('BytecodeLength', len(vs)), X.rid('pInputLayout', LAYOUT)])
    fx.chunk('Internal::List of Initial Contents Resources', [X.arr('NeededInitials', [X.st(None, [X.rid('id', VB), X.b('written', True)])])])
    fx.chunk('Internal::Beginning of Capture', [initial_state(), X.arr('HiddenStreamOutCounters', []), X.u('numAnnotations', 0)])
    fx.captured = True
    ev = {}
    if unmap_before_capture:
        fx.chunk('ID3D11DeviceContext::Unmap', [X.rid('pResource', CB), X.u('Subresource', 0), X.u('MapType', 4), X.u('MapFlags', 0),
                                                X.u('Byte offset to start of written data', 0), X.u('Byte offset to end of written data', 64),
                                                X.buf('MapWrittenData', fx.blob(b'P' * 64), 64)])
    fx.chunk('ID3D11DeviceContext::Map', [X.rid('pResource', CB), X.u('Subresource', 0), X.u('MapType', 4), X.u('MapFlags', 0)])
    ev['unmap1'] = fx.chunk('ID3D11DeviceContext::Unmap', [X.rid('pResource', CB), X.u('Subresource', 0), X.u('MapType', 4), X.u('MapFlags', 0),
                                                           X.u('Byte offset to start of written data', 0), X.u('Byte offset to end of written data', 64),
                                                           X.buf('MapWrittenData', fx.blob(b'A' * 64), 64)])
    fx.chunk('ID3D11DeviceContext::OMSetRenderTargets', [X.u('NumViews', 1), ids('ppRenderTargetViews', [RTV_A]), X.rid('pDepthStencilView', 0)])
    fx.chunk('ID3D11DeviceContext::RSSetViewports', [X.u('NumViewports', 1), X.arr('pViewports', [X.st(None, [
        X.f('TopLeftX', 0), X.f('TopLeftY', 0), X.f('Width', 4), X.f('Height', 4), X.f('MinDepth', 0), X.f('MaxDepth', 1)])])])
    ev['clear'] = fx.chunk('ID3D11DeviceContext::ClearRenderTargetView', [X.rid('pRenderTargetView', RTV_A), X.arr('ColorRGBA', [X.f(None, 0)] * 4)], action=True)
    fx.chunk('ID3D11DeviceContext::IASetInputLayout', [X.rid('pInputLayout', LAYOUT)])
    fx.chunk('ID3D11DeviceContext::IASetVertexBuffers', [X.u('StartSlot', 0), X.u('NumBuffers', 1), ids('ppVertexBuffers', [VB]), uints('pStrides', [12]), uints('pOffsets', [0])])
    fx.chunk('ID3D11DeviceContext::IASetIndexBuffer', [X.rid('pIndexBuffer', IB), X.u('Format', 57), X.u('Offset', 0)])
    fx.chunk('ID3D11DeviceContext::VSSetShader', [X.rid('pShader', VS), X.arr('ppClassInstances', []), X.u('NumClassInstances', 0)])
    fx.chunk('ID3D11DeviceContext::PSSetShader', [X.rid('pShader', PS), X.arr('ppClassInstances', []), X.u('NumClassInstances', 0)])
    fx.chunk('ID3D11DeviceContext::VSSetConstantBuffers', [X.u('StartSlot', 0), X.u('NumBuffers', 1), ids('ppConstantBuffers', [CB])])
    fx.chunk('ID3D11DeviceContext::PSSetShaderResources', [X.u('StartSlot', 0), X.u('NumViews', 1), ids('ppShaderResourceViews', [SRV_B])])
    ev['draw1'] = fx.chunk('ID3D11DeviceContext::DrawIndexed', [X.u('IndexCount', 6), X.u('StartIndexLocation', 0), X.u('BaseVertexLocation', 0)], action=True)
    ev['draw2'] = fx.chunk('ID3D11DeviceContext::DrawIndexed', [X.u('IndexCount', 3), X.u('StartIndexLocation', 3), X.u('BaseVertexLocation', 0)], action=True)
    fx.chunk('ID3D11DeviceContext::Map', [X.rid('pResource', CB), X.u('Subresource', 0), X.u('MapType', 4), X.u('MapFlags', 0)])
    ev['unmap2'] = fx.chunk('ID3D11DeviceContext::Unmap', [X.rid('pResource', CB), X.u('Subresource', 0), X.u('MapType', 4), X.u('MapFlags', 0),
                                                           X.u('Byte offset to start of written data', 0), X.u('Byte offset to end of written data', 64),
                                                           X.buf('MapWrittenData', fx.blob(b'B' * 64), 64)])
    fx.chunk('ID3D11DeviceContext::OMSetRenderTargets', [X.u('NumViews', 1), ids('ppRenderTargetViews', [RTV_B]), X.rid('pDepthStencilView', 0)])
    fx.chunk('ID3D11DeviceContext::PSSetShaderResources', [X.u('StartSlot', 0), X.u('NumViews', 1), ids('ppShaderResourceViews', [SRV_A])])
    if second_ib:
        fx.chunk('ID3D11DeviceContext::IASetIndexBuffer', [X.rid('pIndexBuffer', 104), X.u('Format', 57), X.u('Offset', 0)])
        ev['draw3'] = fx.chunk('ID3D11DeviceContext::DrawIndexed', [X.u('IndexCount', len(second_ib)), X.u('StartIndexLocation', 0), X.u('BaseVertexLocation', 0)], action=True)
    else:
        ev['draw3'] = fx.chunk('ID3D11DeviceContext::DrawIndexed', [X.u('IndexCount', 6), X.u('StartIndexLocation', 0), X.u('BaseVertexLocation', 0)], action=True)
    ev['present'] = fx.chunk('IDXGISwapChain::Present', [X.u('SyncInterval', 0), X.u('Flags', 0)], action=True)
    fx.captured = False
    fx.chunk('Internal::End of Capture', [])
    return fx, ev


def build(fx, tmp):
    xml, zip_path, listing = fx.write(tmp)
    graph = FrameGraph(load_chunks(xml), load_driver_listing(listing), Blobs(zip_path), ShaderLibrary()).build()
    return graph


class DeclarationTests(unittest.TestCase):
    def test_declaration_tokens(self):
        t = shader_tokens(0, [dcl_cb(3, 27), dcl_sampler(1), dcl_texture2d(5)])
        d = parse_declarations(struct.pack('<%dI' % len(t), *t))
        self.assertEqual(d['cb'], {3: {'declaration': 'dcl_constantbuffer', 'float4_count': 27}})
        self.assertEqual(d['srv'][5]['dimension'], 'texture2d')
        self.assertIn(1, d['sampler'])

    def test_signature_and_container(self):
        r = reflect(dxbc(shader_tokens(1, [dcl_cb(0, 4)]), ['POSITION', 'TEXCOORD']))
        self.assertEqual([x['semantic'] for x in r['input_signature']], ['POSITION', 'TEXCOORD'])
        self.assertEqual(r['container_chunks'], ['ISGN', 'SHDR'])

    def test_nonimmediate_index_is_rejected(self):
        bad = [89 | (4 << 24), 2 | (8 << 12) | (2 << 20) | (1 << 22), 0, 4]
        with self.assertRaises(UnresolvedFrame):
            parse_declarations(struct.pack('<%dI' % len(shader_tokens(0, [bad])), *shader_tokens(0, [bad])))

    def test_repeated_fields_are_retained(self):
        e = ET.fromstring('<struct name="s"><uint name="RowPitch">4</uint><uint name="RowPitch">8</uint></struct>')
        value = decode_element(e)
        self.assertIsInstance(value['RowPitch'], Repeated)
        self.assertEqual(list(value['RowPitch']), [4, 8])


class FrameTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_events_match_driver_listing(self):
        fx, ev = standard_fixture()
        g = build(fx, self.tmp.name)
        v = g.validate_events()
        self.assertEqual(v['mismatched_driver_actions'], [])
        self.assertEqual(v['interpreted_actions'], 5)
        self.assertEqual([a['event_id'] for a in g.actions if a['kind'] == 'DrawIndexed'], [ev['draw1'], ev['draw2'], ev['draw3']])

    def test_capture_start_buffer_and_unmap_versions(self):
        fx, ev = standard_fixture()
        g = build(fx, self.tmp.name)
        draws = [a for a in g.actions if a['kind'] == 'DrawIndexed']
        vb = draws[0]['vertex_buffers'][0]
        self.assertEqual(vb['producers'][0]['producer'], 'capture_start')
        self.assertEqual(vb['producers'][0]['kind'], 'capture_start_bytes_in_CreateBuffer_record')
        self.assertEqual(draws[0]['shaders']['VS']['constant_buffers'][0]['producers'][0]['producer'], ev['unmap1'])
        self.assertEqual(draws[2]['shaders']['VS']['constant_buffers'][0]['producers'][0]['producer'], ev['unmap2'])
        # Declared cb0[4] limits the read to 64 bytes.
        self.assertEqual(draws[0]['shaders']['VS']['constant_buffers'][0]['bytes'], [0, 64])

    def test_render_target_producers(self):
        fx, ev = standard_fixture()
        g = build(fx, self.tmp.name)
        draws = [a for a in g.actions if a['kind'] == 'DrawIndexed']
        srv = draws[2]['shaders']['PS']['srvs'][0]
        self.assertEqual(srv['resource'], RT_A)
        self.assertEqual(srv['producers'][0]['base'], ev['clear'])
        self.assertEqual(srv['producers'][0]['partial_writers_since_base'], [ev['draw1'], ev['draw2']])
        unread = draws[0]['shaders']['PS']['srvs'][0]
        self.assertEqual(unread['producers'][0]['base'], 'capture_start')
        edges = resource_edges(g)
        latest = [e for e in edges if e['consumer'] == ev['draw3'] and e['resource'] == RT_A and e['latest_writer']]
        self.assertEqual([e['producer'] for e in latest], [ev['draw2']])

    def test_passes_and_segments(self):
        fx, ev = standard_fixture()
        g = build(fx, self.tmp.name)
        passes = group_passes(g)
        first = [p for p in passes if ev['draw1'] in p['events']][0]
        self.assertEqual(first['events'], [ev['draw1'], ev['draw2']])
        self.assertEqual(first['clear_events'], [ev['clear']])
        segments = segment_passes(passes, g, boundary_resource=RT_A)
        self.assertEqual(len(segments), 2)
        self.assertEqual(passes[segments[-1][-1]]['label'], 'present')
        t = temporal_dependencies(g, passes, segments, resource_edges(g))
        self.assertTrue(any(r['resource'] == RT_A and r['consumer_pass'] for r in t['cross_segment_reads']))

    def test_unmap_of_precapture_map_is_applied(self):
        fx, ev = standard_fixture(unmap_before_capture=True)
        g = build(fx, self.tmp.name)
        self.assertTrue(any('before Beginning of Capture' in n['note'] for n in g.notes))
        unmaps = [a for a in g.actions if a['kind'] == 'Unmap']
        self.assertTrue(unmaps[0]['map_issued_before_capture'])

    def test_surface_binding_uses_exact_index_lists(self):
        tang = [0, 2, 1]
        fx, ev = standard_fixture(second_ib=tang)
        g = build(fx, self.tmp.name)
        mesh = lambda name, idx, path: {'mesh_name': name, 'renderer_path': path, 'vertex_count': 4, 'triangle_count': len(idx) // 3,
                                        'active_in_hierarchy': True, 'enabled': True,
                                        'source': {'submeshes': [{'indices': idx}], 'vertices': [[0, 0, 0]] * 4}}
        geometry = {'meshes': [mesh('o_head', [0, 1, 2, 2, 1, 3], '/h'), mesh('o_tang', [0, 1, 2], '/t1'),
                               mesh('o_tang', tang, '/t2')]}
        passes = group_passes(g)
        b = bind_surfaces(g, geometry, passes, segment_passes(passes, g))
        self.assertEqual([d['event_id'] for d in b['surfaces'][0]['draws']], [ev['draw1']])
        # Debug name o_tang(Clone) alone does not bind /t1; the exact list binds /t2.
        self.assertEqual(b['surfaces'][1]['draws'], [])
        self.assertEqual([d['event_id'] for d in b['surfaces'][2]['draws']], [ev['draw3']])
        # draw2 uses a 3-index sub-range of the head IB that matches no submesh.
        self.assertNotIn(ev['draw2'], [d['event_id'] for s in b['surfaces'] for d in s['draws']])

    def test_unknown_context_command_is_rejected(self):
        fx, ev = standard_fixture()
        fx.captured = True
        e = ET.SubElement(fx.chunks, 'chunk', chunkIndex=str(fx.index), name='ID3D11DeviceContext::SOSetTargets')
        e.append(X.rid('Context', 7))
        with self.assertRaises(UnresolvedFrame):
            build(fx, self.tmp.name)


class ViewTests(unittest.TestCase):
    def test_all_remaining_array_slices(self):
        resources = {1: {'kind': 'CreateTexture2D', 'descriptor': {'Width': 4, 'Height': 4, 'MipLevels': 2, 'ArraySize': 3}},
                     2: {'kind': 'CreateUnorderedAccessView', 'base': 1,
                         'descriptor': {'Format': 28, 'ViewDimension': 5,
                                        'Texture2DArray': {'MipSlice': 1, 'FirstArraySlice': 1, 'ArraySize': 0xffffffff}}}}
        self.assertEqual(view_subresources(resources, 2), (1, {'subresources': [3, 5]}))


if __name__ == '__main__':
    unittest.main()
