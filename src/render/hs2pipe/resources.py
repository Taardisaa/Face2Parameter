"""Exact GPU resource bytes for the HS2 offline renderer (Unity 2018.4.11f1, D3D11).

Given scene state (the frame-pair probe export) and a resource request -- a
renderer's mesh, a material texture slot, or a texture identity -- this module
returns frame-program resource entries: a verbatim D3D11 descriptor, the exact
``D3D11_SUBRESOURCE_DATA`` bytes of every subresource, the shader-resource-view
descriptor Unity creates, and a provenance record saying where the bytes come
from:

``asset``             read from an installed Unity serialized file (bundle /
                      ``.assets`` / zipmod member) by ``path_id`` and converted
                      exactly as Unity's D3D11 texture / mesh upload does;
``computed``          produced by a recovered CPU computation (UnityPlayer.dll
                      built-ins, PPv2 script objects), with the code location;
``external_runtime``  produced at runtime by code whose inputs are not in the
                      scene state (or whose GPU computation is not yet emitted):
                      no bytes are invented; the entry names the producer and
                      the raw-readback channel that must supply them.

Nothing is decoded for display, gamma-converted, re-mipped or filled.  Every
conversion is one Unity performs on upload and is cited where it is defined.

Recovered upload rules (see ``docs/hs2_offline_resources.md`` for evidence):

* **Textures** (``Texture2D``/``Cubemap`` assets).  ``m_TextureFormat`` maps to
  the DXGI storage format through ``UNITY_TEXTURE_FORMATS``; the ``_SRGB``
  variant is used when the asset's ``m_ColorSpace`` is sRGB (1) and the
  project is linear (``QualitySettings.activeColorSpace``).  ``MipLevels`` is
  ``m_MipCount``; Cubemaps are 6-slice Texture2D arrays with
  ``D3D11_RESOURCE_MISC_TEXTURECUBE``.  Image data is image-major then
  mip-major, each mip tightly packed (4x4 blocks for BC formats), which is the
  D3D11 subresource order ``mip + slice * mips``.  RGB24 is expanded to RGBA32
  with alpha 255 and ARGB32 is reordered to RGBA (D3D11 has neither layout).
  Data stored in ``m_StreamData`` is read from the ``.resS`` member/file.
* **Meshes** (``Mesh`` assets, ``m_MeshCompression`` 0).  ``m_VertexData``
  holds the streams back to back, each starting at a 16-byte aligned offset;
  one immutable-content DEFAULT vertex buffer is created per stream with the
  stream's bytes, and the index buffer is ``m_IndexBuffer`` verbatim.  For a
  ``SkinnedMeshRenderer`` the stream holding POSITION/NORMAL/TANGENT is
  replaced at draw time by the skinning output buffer (``hs2pipe`` skinning
  owners); its static bytes are still returned but marked
  ``bound_by_frame=False``.
* **Built-ins** from ``builtintex::GenerateBuiltinTextures`` (UnityPlayer.dll
  RVA 0x754620) and ``GfxDevice::CreateDefaultVertexBuffer`` (RVA 0x8cb000).

Usage::

    from src.render.hs2pipe.resources import SourceIndex, SceneResources
    index = SourceIndex.load(INDEX_JSON)                     # built once by build_index()
    res = SceneResources(scene_state_dict, index)
    entry = res.texture(231706)                              # scene texture id
    entries = res.renderer_mesh(-301430)                     # VB streams + IB
    rid, views = entry.add_to(builder)                       # frame_program.ProgramBuilder
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import io
import json
import os
import struct
import zipfile
from dataclasses import dataclass, field
from math import ceil
from pathlib import Path

GENERATOR = 'hs2pipe.resources/1'
GAME_ROOT = Path(os.environ.get('HS2_DIR', r'E:\HoneySelect2_ArcticFox'))
UNITYPLAYER_SHA256 = '5034cdc2bff71dc8d9e038cb71059f8965d0d879f8098af0f3c726ea06e13923'
UNITYPLAYER_PDB = {'guid': 'BFA91A8C44734BB4AABC5A975B304CFE', 'age': 1}


class ResourceUnresolved(RuntimeError):
    """A request whose exact bytes cannot be produced; never approximated."""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ----------------------------------------------------------------------- DXGI
DXGI = {'R32G32B32A32_FLOAT': 2, 'R16G16B16A16_FLOAT': 10, 'R16G16B16A16_UNORM': 11, 'R32G32_FLOAT': 16,
        'R8G8B8A8_TYPELESS': 27, 'R8G8B8A8_UNORM': 28, 'R8G8B8A8_UNORM_SRGB': 29, 'R16G16_FLOAT': 34,
        'R32_FLOAT': 41, 'R8G8_UNORM': 49, 'R16_FLOAT': 54, 'R16_UNORM': 56, 'R8_UNORM': 61, 'A8_UNORM': 65,
        'BC1_UNORM': 71, 'BC1_UNORM_SRGB': 72, 'BC2_UNORM': 74, 'BC2_UNORM_SRGB': 75, 'BC3_UNORM': 77,
        'BC3_UNORM_SRGB': 78, 'BC4_UNORM': 80, 'BC5_UNORM': 83, 'B8G8R8A8_UNORM': 87, 'B8G8R8A8_UNORM_SRGB': 91,
        'BC6H_UF16': 95, 'BC7_UNORM': 98, 'BC7_UNORM_SRGB': 99}

# Unity TextureFormat -> (name, linear DXGI, sRGB DXGI or None, source bytes/texel or block,
#                         conversion).  Only formats whose D3D11 upload is recovered are listed;
# others raise ResourceUnresolved instead of guessing.
UNITY_TEXTURE_FORMATS = {
    1: ('Alpha8', 'A8_UNORM', None, 'texel', 1, None),
    3: ('RGB24', 'R8G8B8A8_UNORM', 'R8G8B8A8_UNORM_SRGB', 'texel', 3, 'rgb24_to_rgba32'),
    4: ('RGBA32', 'R8G8B8A8_UNORM', 'R8G8B8A8_UNORM_SRGB', 'texel', 4, None),
    5: ('ARGB32', 'R8G8B8A8_UNORM', 'R8G8B8A8_UNORM_SRGB', 'texel', 4, 'argb32_to_rgba32'),
    9: ('R16', 'R16_UNORM', None, 'texel', 2, None),
    10: ('DXT1', 'BC1_UNORM', 'BC1_UNORM_SRGB', 'block', 8, None),
    12: ('DXT5', 'BC3_UNORM', 'BC3_UNORM_SRGB', 'block', 16, None),
    14: ('BGRA32', 'B8G8R8A8_UNORM', 'B8G8R8A8_UNORM_SRGB', 'texel', 4, None),
    15: ('RHalf', 'R16_FLOAT', None, 'texel', 2, None),
    16: ('RGHalf', 'R16G16_FLOAT', None, 'texel', 4, None),
    17: ('RGBAHalf', 'R16G16B16A16_FLOAT', None, 'texel', 8, None),
    18: ('RFloat', 'R32_FLOAT', None, 'texel', 4, None),
    19: ('RGFloat', 'R32G32_FLOAT', None, 'texel', 8, None),
    20: ('RGBAFloat', 'R32G32B32A32_FLOAT', None, 'texel', 16, None),
    24: ('BC6H', 'BC6H_UF16', None, 'block', 16, None),
    25: ('BC7', 'BC7_UNORM', 'BC7_UNORM_SRGB', 'block', 16, None),
    26: ('BC4', 'BC4_UNORM', None, 'block', 8, None),
    27: ('BC5', 'BC5_UNORM', None, 'block', 16, None),
    62: ('RG16', 'R8G8_UNORM', None, 'texel', 2, None),
    63: ('R8', 'R8_UNORM', None, 'texel', 1, None),
}

# Unity VertexFormat (2018.4 m_Channels[].format) -> bytes per component.
VERTEX_FORMAT_BYTES = {0: 4, 1: 2, 2: 1, 3: 1, 4: 2, 5: 2, 6: 1, 7: 1, 8: 2, 9: 2, 10: 4, 11: 4}
VERTEX_CHANNELS = ('Vertex', 'Normal', 'Tangent', 'Color', 'TexCoord0', 'TexCoord1', 'TexCoord2', 'TexCoord3',
                   'TexCoord4', 'TexCoord5', 'TexCoord6', 'TexCoord7', 'BlendWeight', 'BlendIndices')
SKIN_HOT_CHANNELS = (0, 1, 2)   # position/normal/tangent are replaced by the skinning output


def tight_layouts(kind, desc):
    """D3D11 subresource order, tightly packed (RenderDoc/D3D11 creation layout)."""
    from src.render.frame_program import subresource_layouts
    return subresource_layouts(kind, desc)


# ----------------------------------------------------------------- entries
@dataclass
class ResourceEntry:
    """One frame-program resource with exact bytes, its default views and provenance."""
    kind: str                       # buffer | tex2d | tex3d | texcube
    desc: dict
    initial: list | None            # [{'subresource', 'data', 'row_pitch', 'slice_pitch'}] or None
    name: str
    provenance: dict
    srv: dict | None = None         # Unity's SRV descriptor (None for buffers)
    role: str = ''                  # e.g. 'vertex_stream_1', 'index_buffer', 'texture'
    extra: dict = field(default_factory=dict)

    def subresource_hashes(self):
        return None if self.initial is None else [sha256(s['data']) for s in self.initial]

    def content_hash(self):
        """SHA-256 over the concatenated tight subresource bytes (for compact receipts)."""
        if self.initial is None:
            return None
        h = hashlib.sha256()
        for s in self.initial:
            h.update(s['data'])
        return h.hexdigest()

    def program_initial(self, builder):
        if self.initial is None:
            return None
        return [{'subresource': s['subresource'], 'blob': builder.blob(s['data']), 'row_pitch': s['row_pitch'],
                 'slice_pitch': s['slice_pitch']} for s in self.initial]

    def add_to(self, builder):
        """Add to a ``frame_program.ProgramBuilder``; returns (resource id, {'srv': view id} or {})."""
        rid = builder.resource(self.kind, self.desc, self.program_initial(builder), name=self.name)
        builder.program['resources'][-1]['provenance'] = self.provenance
        views = {}
        if self.srv is not None:
            views['srv'] = builder.view('srv', rid, self.srv, name=self.name)
        return rid, views

    def summary(self):
        return {'kind': self.kind, 'name': self.name, 'role': self.role, 'desc': self.desc, 'srv': self.srv,
                'content_sha256': self.content_hash(), 'subresources': None if self.initial is None else len(self.initial),
                'provenance': self.provenance, **({'extra': self.extra} if self.extra else {})}


# ----------------------------------------------------------- serialized files
class SourceFiles:
    """Opens installed Unity serialized files with UnityPy, read-only.

    A source spec is an absolute path (bundle / ``.assets``) or
    ``ZIPPATH::MEMBER`` for an uncompressed-in-place zipmod member."""

    def __init__(self, max_open=3):
        from collections import OrderedDict
        self._env = OrderedDict()          # LRU: whole decompressed bundles are large
        self._objs = {}
        self._sha = {}
        self.max_open = max_open

    @staticmethod
    def split(spec):
        return spec.split('::', 1) if '::' in spec else (spec, None)

    def raw(self, spec):
        path, member = self.split(spec)
        if member is None:
            return Path(path).read_bytes()
        with zipfile.ZipFile(path) as z:
            return z.read(member)

    def file_sha256(self, spec):
        if spec not in self._sha:
            path, member = self.split(spec)
            if member is None:
                h = hashlib.sha256()
                with open(path, 'rb') as fh:
                    for chunk in iter(lambda: fh.read(1 << 22), b''):
                        h.update(chunk)
                self._sha[spec] = h.hexdigest()
            else:
                self._sha[spec] = sha256(self.raw(spec))
        return self._sha[spec]

    def env(self, spec):
        if spec in self._env:
            self._env.move_to_end(spec)
            return self._env[spec]
        import gc
        import UnityPy
        while len(self._env) >= self.max_open:
            old, _ = self._env.popitem(last=False)
            self._objs.pop(old, None)
            gc.collect()
        path, member = self.split(spec)
        self._env[spec] = UnityPy.load(str(path)) if member is None else UnityPy.load(self.raw(spec))
        return self._env[spec]

    def objects(self, spec):
        env = self.env(spec)
        if spec not in self._objs:
            self._objs[spec] = {o.path_id: o for o in env.objects}
        return self._objs[spec]

    def object(self, spec, path_id):
        objs = self.objects(spec)
        if path_id not in objs:
            raise ResourceUnresolved('path_id %d not in %s' % (path_id, spec))
        return objs[path_id]

    def stream_data(self, spec, obj, stream):
        """Bytes of an ``m_StreamData`` record (``.resS`` inside the bundle, or the
        sibling file of a ``.assets``)."""
        if not stream or not stream.get('size'):
            return None
        name = Path(stream['path'].replace('archive:/', '').replace('\\', '/')).name
        path, member = self.split(spec)
        data = None
        # Bundle members (CAB-*.resS) live in the same BundleFile.
        parent = getattr(obj.assets_file, 'parent', None)
        files = getattr(parent, 'files', None) or {}
        for key, f in files.items():
            if Path(key).name == name:
                reader = getattr(f, 'bytes', None)
                if reader is None and hasattr(f, 'reader'):
                    f.reader.Position = 0
                    reader = f.reader.read_bytes(f.reader.Length)
                if reader is None and hasattr(f, 'get_data'):
                    reader = f.get_data()
                data = bytes(reader)
                break
        if data is None and member is None:
            sibling = Path(path).parent / name
            if sibling.exists():
                with open(sibling, 'rb') as fh:
                    fh.seek(stream['offset'])
                    out = fh.read(stream['size'])
                if len(out) != stream['size']:
                    raise ResourceUnresolved('short .resS read for %s' % spec)
                return out
        if data is None:
            raise ResourceUnresolved('stream data %s not found for %s' % (stream['path'], spec))
        out = data[stream['offset']:stream['offset'] + stream['size']]
        if len(out) != stream['size']:
            raise ResourceUnresolved('short stream data for %s' % spec)
        return out


# --------------------------------------------------------------- name index
INDEX_TYPES = ('Texture2D', 'Cubemap', 'Texture3D', 'Mesh')
INDEX_SCHEMA = 'hs2pipe.resources.source_index/2'


# abdata folders holding textures/meshes the Maker scene can load (characters, Maker UI,
# list thumbnails, maps/backgrounds, sprites, shared etc. bundles).  Studio, audio,
# animation-controller, ADV-script, list and H-scene folders are not Maker render inputs.
MAKER_ABDATA_DIRS = ('chara', 'custom', 'map', 'spr', 'etcetra', 'search', 'probe', 'prefabs', 'title', 'home',
                     'lobby', 'gamedata', 'fursroom', 'hscene', 'vr')


def default_index_sources(game_root=GAME_ROOT, zipmods=()):
    """Every installed serialized file the HS2 renderer can load textures/meshes from
    in the Maker scene: the ``MAKER_ABDATA_DIRS`` bundles, the player data files and
    the given zipmods' ``.unity3d`` members."""
    g = Path(game_root)
    out = []
    for sub in MAKER_ABDATA_DIRS:
        out += sorted(str(p) for p in (g / 'abdata' / sub).rglob('*.unity3d'))
    data = g / 'HoneySelect2_Data'
    out += sorted(str(p) for p in data.glob('*.assets'))
    out += sorted(str(p) for p in data.glob('level*') if p.suffix == '')
    out += [str(data / 'Resources' / 'unity_builtin_extra')]
    for zm in zipmods:
        with zipfile.ZipFile(zm) as z:
            out += ['%s::%s' % (zm, n) for n in z.namelist() if n.endswith('.unity3d')]
    return out


def _serialized_names(env):
    """Names of the SerializedFiles inside a loaded source (``CAB-...`` for bundles),
    the targets of external PPtr references."""
    out = []
    for key, f in env.files.items():
        if hasattr(f, 'objects'):                 # a SerializedFile (.assets / level file)
            out.append(Path(str(key)).name)
            continue
        for k, v in (getattr(f, 'files', None) or {}).items():    # BundleFile members
            if hasattr(v, 'objects'):
                out.append(Path(str(k)).name)
    return out


def _index_one(spec):
    import UnityPy  # noqa: F401  (worker process)
    files = SourceFiles()
    rows = []
    try:
        env = files.env(spec)
        for o in env.objects:
            t = o.type.name
            if t in INDEX_TYPES:
                rows.append([t, o.peek_name(), o.path_id, o.byte_size])
        cabs = _serialized_names(env)
    except Exception as e:  # noqa: BLE001 - recorded, never silently dropped
        return spec, None, repr(e)
    path, member = SourceFiles.split(spec)
    st = os.stat(path)
    return spec, {'size': st.st_size, 'mtime_ns': st.st_mtime_ns, 'objects': rows, 'serialized_files': cabs}, None


class SourceIndex:
    """(type, name) -> [(source spec, path_id, byte_size)] over installed files."""

    def __init__(self, doc):
        if doc.get('schema') != INDEX_SCHEMA:
            raise ResourceUnresolved('source index schema mismatch')
        self.doc = doc
        self.by_name = {}
        self.by_serialized = {}
        for spec, rec in doc['sources'].items():
            for t, name, pid, size in rec['objects']:
                self.by_name.setdefault((t, name), []).append((spec, pid, size))
            for cab in rec.get('serialized_files', []):
                self.by_serialized.setdefault(cab.lower(), []).append(spec)

    @classmethod
    def build(cls, sources, out_json, workers=16):
        doc = {'schema': INDEX_SCHEMA, 'types': list(INDEX_TYPES), 'sources': {}, 'errors': {}}
        with concurrent.futures.ProcessPoolExecutor(workers) as ex:
            for spec, rec, err in ex.map(_index_one, sources, chunksize=4):
                if err:
                    doc['errors'][spec] = err
                else:
                    doc['sources'][spec] = rec
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(doc), encoding='utf-8')
        return cls(doc)

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding='utf-8')))

    def stale(self):
        """Sources whose size/mtime changed since indexing (the index must be rebuilt)."""
        out = []
        for spec, rec in self.doc['sources'].items():
            path, _ = SourceFiles.split(spec)
            try:
                st = os.stat(path)
            except FileNotFoundError:
                out.append(spec)
                continue
            if st.st_size != rec['size'] or st.st_mtime_ns != rec['mtime_ns']:
                out.append(spec)
        return out

    def find(self, type_name, name):
        return list(self.by_name.get((type_name, name), []))


# ------------------------------------------------------------ texture upload
def _mip_bytes(fmt_row, w, h):
    kind, size = fmt_row[3], fmt_row[4]
    if kind == 'block':
        return ceil(w / 4) * ceil(h / 4) * size
    return w * h * size


def _convert_texels(conversion, data):
    if conversion is None:
        return data
    if conversion == 'rgb24_to_rgba32':
        if len(data) % 3:
            raise ResourceUnresolved('RGB24 data length not a multiple of 3')
        out = bytearray(len(data) // 3 * 4)
        out[0::4] = data[0::3]
        out[1::4] = data[1::3]
        out[2::4] = data[2::3]
        out[3::4] = b'\xff' * (len(data) // 3)
        return bytes(out)
    if conversion == 'argb32_to_rgba32':
        out = bytearray(len(data))
        out[0::4] = data[1::4]
        out[1::4] = data[2::4]
        out[2::4] = data[3::4]
        out[3::4] = data[0::4]
        return bytes(out)
    raise ResourceUnresolved('unknown texel conversion ' + conversion)


def texture_entry_from_tree(tree, image, *, linear_project=True, provenance=None, unity_type='Texture2D'):
    """D3D11 creation of a Texture2D/Cubemap asset: descriptor, subresources, SRV.

    ``image`` is the complete serialized image data (inline ``image data`` or the
    ``m_StreamData`` bytes)."""
    fmt = tree['m_TextureFormat']
    if fmt not in UNITY_TEXTURE_FORMATS:
        raise ResourceUnresolved('Unity TextureFormat %d upload is not recovered' % fmt)
    row = UNITY_TEXTURE_FORMATS[fmt]
    name, lin, srgb_name, _, _, conversion = row
    srgb = tree.get('m_ColorSpace', 0) == 1 and linear_project and srgb_name is not None
    dxgi = DXGI[srgb_name if srgb else lin]
    w, h, mips = tree['m_Width'], tree['m_Height'], tree['m_MipCount']
    images = tree.get('m_ImageCount', 1)
    dim = tree.get('m_TextureDimension', 2)
    cube = unity_type == 'Cubemap' or dim == 4
    if cube and images != 6:
        raise ResourceUnresolved('Cubemap with %d images' % images)
    if not cube and images != 1:
        raise ResourceUnresolved('Texture2D with %d images (array upload not recovered)' % images)
    per_image = sum(_mip_bytes(row, max(1, w >> m), max(1, h >> m)) for m in range(mips))
    if per_image != tree['m_CompleteImageSize']:
        raise ResourceUnresolved('m_CompleteImageSize %d differs from mip chain %d for %s'
                                 % (tree['m_CompleteImageSize'], per_image, tree['m_Name']))
    if len(image) != per_image * images:
        raise ResourceUnresolved('image data %d bytes, expected %d for %s' % (len(image), per_image * images, tree['m_Name']))
    desc = {'Width': w, 'Height': h, 'MipLevels': mips, 'ArraySize': images, 'Format': dxgi,
            'SampleDesc': {'Count': 1, 'Quality': 0}, 'Usage': 0, 'BindFlags': 8, 'CPUAccessFlags': 0,
            'MiscFlags': 4 if cube else 0}
    kind = 'texcube' if cube else 'tex2d'
    layouts = tight_layouts(kind, desc)
    initial = []
    pos = 0
    for img in range(images):
        for m in range(mips):
            n = _mip_bytes(row, max(1, w >> m), max(1, h >> m))
            data = _convert_texels(conversion, image[pos:pos + n])
            pos += n
            lay = layouts[m + img * mips]
            if len(data) != lay['bytes']:
                raise ResourceUnresolved('converted mip size mismatch for %s' % tree['m_Name'])
            initial.append({'subresource': lay['subresource'], 'data': data, 'row_pitch': lay['row_bytes'],
                            'slice_pitch': lay['slice_bytes']})
    if cube:
        srv = {'Format': dxgi, 'ViewDimension': 9, 'TextureCube': {'MostDetailedMip': 0, 'MipLevels': mips}}
    else:
        srv = {'Format': dxgi, 'ViewDimension': 4, 'Texture2D': {'MostDetailedMip': 0, 'MipLevels': mips}}
    ts = tree.get('m_TextureSettings', {})
    extra = {'unity_format': name, 'unity_color_space': tree.get('m_ColorSpace'), 'conversion': conversion,
             'texture_settings': {k: ts.get(k) for k in ('m_FilterMode', 'm_Aniso', 'm_MipBias', 'm_WrapU', 'm_WrapV', 'm_WrapW')}}
    return ResourceEntry(kind, desc, initial, tree['m_Name'], provenance or {}, srv=srv, role='texture', extra=extra)


def asset_texture_entry(files: SourceFiles, spec, path_id, *, linear_project=True):
    obj = files.object(spec, path_id)
    if obj.type.name not in ('Texture2D', 'Cubemap'):
        raise ResourceUnresolved('%s/%d is %s, not a texture' % (spec, path_id, obj.type.name))
    tree = obj.read_typetree()
    image = bytes(tree.get('image data') or b'')
    stream = tree.get('m_StreamData')
    streamed = bool(stream and stream.get('size'))
    if not image and streamed:
        image = files.stream_data(spec, obj, stream)
    prov = {'class': 'asset', 'source': spec, 'source_sha256': files.file_sha256(spec), 'path_id': path_id,
            'unity_type': obj.type.name, 'name': tree['m_Name'],
            'image_location': 'm_StreamData' if streamed else 'image data',
            'rule': 'Unity D3D11 texture upload (resources.texture_entry_from_tree)', 'generator': GENERATOR}
    return texture_entry_from_tree(tree, image, linear_project=linear_project, provenance=prov, unity_type=obj.type.name)


# --------------------------------------------------------------- mesh upload
def mesh_streams_from_tree(tree):
    """[(stream, stride, byte offset in m_DataSize, size, channels)] of a 2018.4 Mesh."""
    vd = tree['m_VertexData']
    n = vd['m_VertexCount']
    streams = {}
    for ci, c in enumerate(vd['m_Channels']):
        if not c['dimension']:
            continue
        if c['format'] not in VERTEX_FORMAT_BYTES:
            raise ResourceUnresolved('vertex format %d' % c['format'])
        end = c['offset'] + VERTEX_FORMAT_BYTES[c['format']] * (c['dimension'] & 0xF)
        s = streams.setdefault(c['stream'], {'stride': 0, 'channels': []})
        s['stride'] = max(s['stride'], end)
        s['channels'].append({'channel': VERTEX_CHANNELS[ci], 'index': ci, **c})
    out = []
    off = 0
    for si in sorted(streams):
        if si != len(out):
            raise ResourceUnresolved('non-contiguous vertex stream numbering')
        size = streams[si]['stride'] * n
        out.append({'stream': si, 'stride': streams[si]['stride'], 'offset': off, 'size': size,
                    'channels': streams[si]['channels']})
        off = (off + size + 15) // 16 * 16
    return out, n


def mesh_entries_from_tree(tree, *, skinned, provenance=None):
    """Vertex buffer per stream + index buffer, exactly as Unity creates them."""
    if tree.get('m_MeshCompression', 0) != 0:
        raise ResourceUnresolved('compressed mesh upload (m_MeshCompression=%d) not recovered' % tree['m_MeshCompression'])
    data = bytes(tree['m_VertexData']['m_DataSize'])
    streams, n = mesh_streams_from_tree(tree)
    if streams:
        last = streams[-1]
        if last['offset'] + last['size'] != len(data):
            raise ResourceUnresolved('vertex stream layout does not cover m_DataSize exactly for %s' % tree['m_Name'])
    prov = dict(provenance or {})
    entries = []
    for s in streams:
        hot = any(c['index'] in SKIN_HOT_CHANNELS for c in s['channels'])
        vb = data[s['offset']:s['offset'] + s['size']]
        desc = {'ByteWidth': len(vb), 'Usage': 0, 'BindFlags': 1, 'CPUAccessFlags': 0, 'MiscFlags': 0,
                'StructureByteStride': 0}
        extra = {'stride': s['stride'], 'vertex_count': n, 'channels': s['channels'],
                 'bound_by_frame': not (skinned and hot)}
        if skinned and hot:
            extra['replaced_by'] = 'skinning output (dynamic VB written by the skinning generator)'
        entries.append(ResourceEntry('buffer', desc, [{'subresource': 0, 'data': vb, 'row_pitch': len(vb),
                                                       'slice_pitch': len(vb)}], tree['m_Name'],
                                     dict(prov, rule='m_VertexData stream %d (16-byte aligned stream offsets)' % s['stream']),
                                     role='vertex_stream_%d' % s['stream'], extra=extra))
    ib = bytes(tree['m_IndexBuffer'])
    desc = {'ByteWidth': len(ib), 'Usage': 0, 'BindFlags': 2, 'CPUAccessFlags': 0, 'MiscFlags': 0, 'StructureByteStride': 0}
    entries.append(ResourceEntry('buffer', desc, [{'subresource': 0, 'data': ib, 'row_pitch': len(ib), 'slice_pitch': len(ib)}],
                                 tree['m_Name'], dict(prov, rule='m_IndexBuffer verbatim'), role='index_buffer',
                                 extra={'index_format': 'u16' if tree['m_IndexFormat'] == 0 else 'u32',
                                        'submeshes': tree['m_SubMeshes']}))
    return entries


def asset_mesh_entries(files: SourceFiles, spec, path_id, *, skinned):
    obj = files.object(spec, path_id)
    if obj.type.name != 'Mesh':
        raise ResourceUnresolved('%s/%d is %s, not a Mesh' % (spec, path_id, obj.type.name))
    tree = obj.read_typetree()
    sd = tree.get('m_StreamData')
    if sd and sd.get('size'):
        raise ResourceUnresolved('streamed mesh vertex data not recovered for %s' % tree['m_Name'])
    prov = {'class': 'asset', 'source': spec, 'source_sha256': files.file_sha256(spec), 'path_id': path_id,
            'unity_type': 'Mesh', 'name': tree['m_Name'], 'generator': GENERATOR}
    return mesh_entries_from_tree(tree, skinned=skinned, provenance=prov)


# ------------------------------------------------------------ engine built-ins
UNITYPLAYER_CITE = {'dll': 'UnityPlayer.dll', 'dll_sha256': UNITYPLAYER_SHA256, 'pdb': UNITYPLAYER_PDB}
# builtintex::GenerateBuiltinTextures (RVA 0x754620): BuildTexture<uint8>(name, w, h, TextureFormat,
# fill callback, TextureColorSpace, mipmaps).  Each fill callback stores one dword per texel:
#   EmptyNormalMap 0x754220 'mov dword [rdx], 0xffff7f7f'; White 0x75e620 0xffffffff;
#   Black 0x74f8b0 0; Empty2D 0x754210 0x80808080.  Cubemaps via 0x7504c0 (ClearImage RGBA32
#   per face, size 1); Texture3D UnityDefault3D: InitTexture(1,1,1,RGBA32) + ClearImage 0x80808080.
BUILTIN_TEXTURES = {
    'UnityNormalMap': {'kind': 'tex2d', 'size': 4, 'dword': 0xffff7f7f, 'srgb': False, 'fill': 'EmptyNormalMap@0x754220'},
    'UnityWhite': {'kind': 'tex2d', 'size': 4, 'dword': 0xffffffff, 'srgb': True, 'fill': 'White@0x75e620'},
    'UnityBlack': {'kind': 'tex2d', 'size': 4, 'dword': 0x00000000, 'srgb': True, 'fill': 'Black@0x74f8b0'},
    'UnityDefault2D': {'kind': 'tex2d', 'size': 16, 'dword': 0x80808080, 'srgb': True, 'fill': 'Empty2D@0x754210'},
    'UnityBlackCube': {'kind': 'texcube', 'size': 1, 'dword': 0x00000000, 'srgb': True, 'fill': 'cube builder 0x7504c0 color 0'},
    'UnityDefaultCube': {'kind': 'texcube', 'size': 1, 'dword': 0x80808080, 'srgb': True, 'fill': 'cube builder 0x7504c0 color 0x80808080'},
    'UnityDefault3D': {'kind': 'tex3d', 'size': 1, 'dword': 0x80808080, 'srgb': False, 'fill': 'Texture3D ClearImage 0x80808080'},
}


def builtin_texture_entry(name):
    b = BUILTIN_TEXTURES.get(name)
    if b is None:
        raise ResourceUnresolved('unknown Unity built-in texture ' + name)
    fmt = DXGI['R8G8B8A8_UNORM_SRGB' if b['srgb'] else 'R8G8B8A8_UNORM']
    n = b['size']
    texel = struct.pack('<I', b['dword'])
    prov = {'class': 'computed', 'producer': 'builtintex::GenerateBuiltinTextures (RVA 0x754620)',
            'fill': b['fill'], 'code': UNITYPLAYER_CITE, 'generator': GENERATOR}
    if b['kind'] == 'tex3d':
        desc = {'Width': 1, 'Height': 1, 'Depth': 1, 'MipLevels': 1, 'Format': fmt, 'Usage': 0, 'BindFlags': 8,
                'CPUAccessFlags': 0, 'MiscFlags': 0}
        srv = {'Format': fmt, 'ViewDimension': 8, 'Texture3D': {'MostDetailedMip': 0, 'MipLevels': 1}}
        init = [{'subresource': 0, 'data': texel, 'row_pitch': 4, 'slice_pitch': 4}]
        return ResourceEntry('tex3d', desc, init, name, prov, srv=srv, role='texture')
    cube = b['kind'] == 'texcube'
    desc = {'Width': n, 'Height': n, 'MipLevels': 1, 'ArraySize': 6 if cube else 1, 'Format': fmt,
            'SampleDesc': {'Count': 1, 'Quality': 0}, 'Usage': 0, 'BindFlags': 8, 'CPUAccessFlags': 0,
            'MiscFlags': 4 if cube else 0}
    init = [{'subresource': s, 'data': texel * (n * n), 'row_pitch': 4 * n, 'slice_pitch': 4 * n * n}
            for s in range(6 if cube else 1)]
    srv = ({'Format': fmt, 'ViewDimension': 9, 'TextureCube': {'MostDetailedMip': 0, 'MipLevels': 1}} if cube else
           {'Format': fmt, 'ViewDimension': 4, 'Texture2D': {'MostDetailedMip': 0, 'MipLevels': 1}})
    return ResourceEntry(b['kind'], desc, init, name, prov, srv=srv, role='texture')


# SkyboxGenerator::GetCubemapSkyboxVB (RVA 0x641740) base octahedron: 24 float3 at RVA 0x1242710.
SKYBOX_BASE_RVA = 0x1242710
SKYBOX_BASE = ((0, 1, 0), (0, 0, -1), (1, 0, 0), (0, 1, 0), (1, 0, 0), (0, 0, 1), (0, 1, 0), (0, 0, 1), (-1, 0, 0),
               (0, 1, 0), (-1, 0, 0), (0, 0, -1), (0, -1, 0), (1, 0, 0), (0, 0, -1), (0, -1, 0), (0, 0, 1), (1, 0, 0),
               (0, -1, 0), (-1, 0, 0), (0, 0, 1), (0, -1, 0), (0, 0, -1), (-1, 0, 0))


def _sky_normalize(x, y, z):
    """``v / sqrtf(x*x + y*y + z*z)`` with SSE single-precision ops in the disassembled order."""
    import numpy as np
    f = np.float32
    d = f(f(f(x * x) + f(y * y)) + f(z * z))
    s = np.sqrt(d, dtype=np.float32)       # CRT sqrtf = sqrtss (RVA 0x116c720), correctly rounded
    return (f(x / s), f(y / s), f(z / s))


def _sky_mid(a, b):
    """``normalize(b * 0.5f + a * 0.5f)`` (each component scaled first, then added)."""
    import numpy as np
    f = np.float32
    h = f(0.5)
    return _sky_normalize(f(f(b[0] * h) + f(a[0] * h)), f(f(b[1] * h) + f(a[1] * h)), f(f(b[2] * h) + f(a[2] * h)))


def skybox_cubemap_vertices():
    """Positions of ``SkyboxGenerator::GetCubemapSkyboxVB`` (UnityPlayer.dll RVA 0x641740):
    the 8-triangle octahedron (normalized), 3 passes of 4-way subdivision (RVA 0x63d6e0:
    (v0,m01,m20) (m01,v1,m12) (m12,m20,m01) (v2,m20,m12)), then 2 adaptive passes with
    threshold ``powf(0.5, pass + 1)``: a triangle whose largest |y| is <= threshold is split
    across its edge of largest y extent (RVA 0x63dbb0), the others are kept."""
    import numpy as np
    f = np.float32
    tris = [_sky_normalize(f(x), f(y), f(z)) for x, y, z in SKYBOX_BASE]
    for _ in range(3):
        out = []
        for i in range(0, len(tris), 3):
            v0, v1, v2 = tris[i:i + 3]
            m01, m12, m20 = _sky_mid(v0, v1), _sky_mid(v1, v2), _sky_mid(v0, v2)
            # midpoints: m01 = norm(v1*.5 + v0*.5); m12 = norm(v2*.5 + v1*.5); m20 = norm(v2*.5 + v0*.5)
            out += [v0, m01, m20, m01, v1, m12, m12, m20, m01, v2, m20, m12]
        tris = out
    for p in range(2):
        threshold = f(0.5) ** f(p + 1)      # powf(0.5, 1.0) = 0.5, powf(0.5, 2.0) = 0.25 (exact)
        out = []
        for i in range(0, len(tris), 3):
            a, b, c = tris[i:i + 3]
            ya, yb, yc = abs(a[1]), abs(b[1]), abs(c[1])
            m = ya if not (yb > ya) else yb
            m = m if not (yc > m) else yc
            if m > threshold:
                out += [a, b, c]
                continue
            d_ab = abs(f(b[1] - a[1]))
            d_bc = abs(f(b[1] - c[1]))
            d_ca = abs(f(c[1] - a[1]))
            if d_bc > d_ab and d_ca > d_ab:
                P, Q, Rv = c, a, b
            elif d_ab > d_bc and d_ca > d_bc:
                P, Q, Rv = a, b, c
            else:
                P, Q, Rv = b, c, a
            mQ = _sky_mid(P, Q)        # normalize(Q*.5 + P*.5)
            mR = _sky_mid(P, Rv)       # normalize(R*.5 + P*.5)

            def dist2(u, v):
                dx, dy, dz = f(u[0] - v[0]), f(u[1] - v[1]), f(u[2] - v[2])
                return f(f(f(dy * dy) + f(dx * dx)) + f(dz * dz))
            if dist2(mR, Q) > dist2(mQ, Rv):
                out += [P, mQ, mR, mQ, Q, Rv, mR, mQ, Rv]
            else:
                out += [P, mQ, mR, mR, mQ, Q, mR, Q, Rv]
        tris = out
    return tris


def skybox_cubemap_vertex_buffer_entry():
    """The skybox sphere vertex buffer Unity draws ``Skybox/*`` cubemap skyboxes with:
    CubemapSkyboxVertex {float3 pos; uint32 color = 0xffffffff; float3 tex = pos}."""
    verts = skybox_cubemap_vertices()
    data = b''.join(struct.pack('<3fI3f', float(v[0]), float(v[1]), float(v[2]), 0xffffffff,
                                float(v[0]), float(v[1]), float(v[2])) for v in verts)
    desc = {'ByteWidth': len(data), 'Usage': 0, 'BindFlags': 1, 'CPUAccessFlags': 0, 'MiscFlags': 0,
            'StructureByteStride': 0}
    prov = {'class': 'computed', 'producer': 'SkyboxGenerator::GetCubemapSkyboxVB (RVA 0x641740; helpers 0x63d6e0, '
                                             '0x63dbb0; base table RVA 0x1242710)', 'code': UNITYPLAYER_CITE,
            'vertices': len(verts), 'generator': GENERATOR}
    return ResourceEntry('buffer', desc, [{'subresource': 0, 'data': data, 'row_pitch': len(data), 'slice_pitch': len(data)}],
                         'Unity cubemap skybox mesh', prov, role='vertex_stream_0', extra={'stride': 28})


def default_vertex_buffer_entry(buffer_type=0, count=1):
    """``GfxDevice::CreateDefaultVertexBuffer`` (RVA 0x8cb000), bound with stride 0 for
    shader inputs a mesh lacks.  Type 0 elements are 16 bytes: dwords 0x00ff0000,
    0xff0000ff, 0xffffffff, 0; type 1 elements are 28 bytes (0,0,-1 | 1,0,0,-1).  The
    element count is the next power of two of max(1, requested) (``GetDefaultVertexBuffer``
    RVA 0x8cd0c0)."""
    n = 1
    while n < max(1, count):
        n <<= 1
    if buffer_type == 0:
        elem = struct.pack('<4I', 0x00ff0000, 0xff0000ff, 0xffffffff, 0)
    elif buffer_type == 1:
        elem = struct.pack('<7f', 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, -1.0)
    else:
        raise ResourceUnresolved('default vertex buffer type %r' % buffer_type)
    data = elem * n
    desc = {'ByteWidth': len(data), 'Usage': 0, 'BindFlags': 1, 'CPUAccessFlags': 0, 'MiscFlags': 0,
            'StructureByteStride': 0}
    prov = {'class': 'computed', 'producer': 'GfxDevice::CreateDefaultVertexBuffer (RVA 0x8cb000)',
            'code': UNITYPLAYER_CITE, 'buffer_type': buffer_type, 'elements': n, 'generator': GENERATOR}
    return ResourceEntry('buffer', desc, [{'subresource': 0, 'data': data, 'row_pitch': len(data), 'slice_pitch': len(data)}],
                         'Unity default vertex buffer', prov, role='default_vertex_stream')


# ----------------------------------------------------------- prefab bindings
def renderer_mesh_bindings(files: SourceFiles, spec):
    """Every renderer GameObject of a serialized file (prefab content):
    [{'go', 'path': [ancestor names..., go], 'mesh': (file_id, path_id) | None,
      'materials': [(file_id, path_id)...], 'component', 'component_path_id'}].
    Mesh from SkinnedMeshRenderer.m_Mesh or MeshFilter.m_Mesh; materials from
    SkinnedMeshRenderer/MeshRenderer.m_Materials.  Used to bind a scene renderer
    (GameObject path below the instantiated prefab root) to the assets its prefab references."""
    objs = files.objects(spec)
    trees = {}

    def tree(pid):
        if pid not in trees:
            trees[pid] = objs[pid].read_typetree()
        return trees[pid]
    go_name, go_transform, parent = {}, {}, {}
    for pid, o in objs.items():
        if o.type.name == 'GameObject':
            go_name[pid] = o.peek_name()
        elif o.type.name in ('Transform', 'RectTransform'):
            t = tree(pid)
            go_transform[t['m_GameObject']['m_PathID']] = pid
            parent[pid] = t['m_Father']['m_PathID']
    tr_go = {v: k for k, v in go_transform.items()}

    def path_of(go):
        names = []
        tr = go_transform.get(go)
        seen = set()
        while tr and tr not in seen:
            seen.add(tr)
            names.append(go_name.get(tr_go.get(tr), '?'))
            tr = parent.get(tr)
        return list(reversed(names))
    per_go = {}
    for pid, o in objs.items():
        if o.type.name not in ('SkinnedMeshRenderer', 'MeshFilter', 'MeshRenderer'):
            continue
        t = tree(pid)
        go = t['m_GameObject']['m_PathID']
        row = per_go.setdefault(go, {'go': go_name.get(go), 'path': path_of(go), 'mesh': None, 'materials': [],
                                     'component': None, 'component_path_id': None})
        if 'm_Mesh' in t:
            row['mesh'] = (t['m_Mesh']['m_FileID'], t['m_Mesh']['m_PathID'])
        if 'm_Materials' in t:
            row['materials'] = [(m['m_FileID'], m['m_PathID']) for m in t['m_Materials']]
            row['component'] = o.type.name
            row['component_path_id'] = pid
    return list(per_go.values())


def resolve_pptr(files: SourceFiles, index, spec, owner_obj, file_id, path_id):
    """(spec, path_id) of a PPtr held by ``owner_obj`` in ``spec``.  External
    references go through the owner file's externals table to the serialized file
    name, which the index maps to its installed source."""
    if path_id == 0:
        return None
    if file_id == 0:
        return spec, path_id
    ext = owner_obj.assets_file.externals[file_id - 1]
    name = Path(ext.path.replace('archive:/', '').replace('\\', '/')).name.lower()
    specs = index.by_serialized.get(name, []) if index else []
    if len(specs) != 1:
        raise ResourceUnresolved('external reference %s resolves to %d installed sources' % (name, len(specs)))
    return specs[0], path_id


def prefab_material_textures(files: SourceFiles, index, spec, material_pptr, owner_obj):
    """{property: (spec, path_id) | None} of a prefab material's m_TexEnvs."""
    loc = resolve_pptr(files, index, spec, owner_obj, *material_pptr)
    if loc is None:
        return {}
    mspec, mpid = loc
    mobj = files.object(mspec, mpid)
    mt = mobj.read_typetree()
    out = {}
    for name, env in mt['m_SavedProperties']['m_TexEnvs']:
        tex = env['m_Texture']
        out[name] = resolve_pptr(files, index, mspec, mobj, tex['m_FileID'], tex['m_PathID'])
    return out


def _scene_path_names(path):
    """'/A[0]/B[3]/C[0]' -> ['A', 'B', 'C']."""
    out = []
    for part in path.strip('/').split('/'):
        out.append(part.rsplit('[', 1)[0] if part.endswith(']') else part)
    return out


# --------------------------------------------------------------- runtime ones
def runtime_texture_entry(t, producer=None):
    """A scene texture whose bytes are produced at runtime: no bytes, explicit producer.

    The descriptor is filled only where the D3D11 creation rule is recovered
    (``runtime_texture_desc``); otherwise ``desc`` is None and the entry cannot be
    added to a program until its owner supplies it."""
    desc, srv = runtime_texture_desc(t)
    prov = {'class': 'external_runtime', 'scene_texture_id': t['id'], 'unity_type': t['type'], 'name': t['name'],
            'producer': producer or classify_runtime_texture(t), 'generator': GENERATOR}
    kind = 'tex3d' if desc and 'Depth' in desc else 'tex2d'
    return ResourceEntry(kind, desc, None, t['name'], prov, srv=srv, role='runtime_texture')


def runtime_texture_desc(t):
    """D3D11 descriptor of a script-created RenderTexture with mips (the ChaControl /
    CustomTextureCreate composites): ``RenderTexture(w, h, 0, ARGB32)`` with
    ``useMipMap``/``autoGenerateMips`` -> TYPELESS RGBA8, SRV|RTV, GENERATE_MIPS, full
    mip chain; the SRV/RTV use the sRGB or linear view per ``sRGB``.  Other runtime
    textures return (None, None) here: their owners (PPv2, Unity camera, CTAA,
    DHH) create them."""
    p = t['properties']
    if not t['type'].endswith('RenderTexture'):
        return None, None
    fmt = p.get('format')
    if fmt != 'RenderTextureFormat.ARGB32=0' or not p.get('useMipMap') or p.get('antiAliasing') != 1 \
            or p.get('dimension') != 'TextureDimension.Tex2D=2' or p.get('depth'):
        return None, None
    w, h = p['width'], p['height']
    mips = max(w, h).bit_length()
    view = DXGI['R8G8B8A8_UNORM_SRGB'] if p.get('sRGB') else DXGI['R8G8B8A8_UNORM']
    desc = {'Width': w, 'Height': h, 'MipLevels': mips, 'ArraySize': 1, 'Format': DXGI['R8G8B8A8_TYPELESS'],
            'SampleDesc': {'Count': 1, 'Quality': 0}, 'Usage': 0, 'BindFlags': 40, 'CPUAccessFlags': 0,
            'MiscFlags': 1 if p.get('autoGenerateMips') else 0}
    srv = {'Format': view, 'ViewDimension': 4, 'Texture2D': {'MostDetailedMip': 0, 'MipLevels': mips}}
    return desc, srv


CHACONTROL_DLL = 'HoneySelect2_Data/Managed/Assembly-CSharp.dll'
KSOX_DLL = 'BepInEx/Plugins/HS2_Plugins/ManlyMarco/HS2_OverlayMods.dll'
FACE_RENDERERS = ('o_head',)
BODY_RENDERERS = ('o_body_cf', 'o_body_cm', 'o_body_a')
CLOTHES_ROOTS = ('ct_clothesTop', 'ct_clothesBot', 'ct_inner_t', 'ct_inner_b', 'ct_gloves', 'ct_panst', 'ct_socks',
                 'ct_shoes')

# Recovered producers (decompiled Assembly-CSharp AIChara.ChaControl / CustomTextureCreate,
# HS2_OverlayMods KoiSkinOverlayX.Hooks.OverlayBlit).  'a' = reproducible from installed
# assets + card data by a recovered GPU/CPU computation; 'b' = needs a raw runtime export.
RUNTIME_PRODUCERS = {
    ('face', '_MainTex'): {
        'kind': 'chacontrol_face_composite', 'class': 'a',
        'producer': 'ChaControl.CreateFaceTexture -> customTexCtrlFace.SetNewCreateTexture(0, SkinTex) -> '
                    'CustomTextureCreate.RebuildTextureAndSetMaterial (KSOX OverlayBlit) -> Graphics.Blit(texMain, '
                    'createTex, create_skin_face, 0) into RenderTexture(2048, 2048, 0, ARGB32){useMipMap}, '
                    'GL.Clear(Color.clear) and sRGBWrite first; mips auto-generated',
        'inputs': 'mm_base create_skin_face material; ft_skin_f MainTex (skinId); st_eyeshadow/st_paint/st_cheek/'
                  'st_lip/st_mole AddTex; fileBody.skinColor and makeup colors/layouts; KSOX face overlays',
        'code': [CHACONTROL_DLL, KSOX_DLL]},
    ('face', '_DetailMainTex'): {
        'kind': 'chacontrol_face_detail_composite', 'class': 'a',
        'producer': 'ChaControl.CreateFaceTexture -> SetNewCreateTexture(1, SkinCreateDetailTex) -> Graphics.Blit('
                    'etc black2048, createTex, "create_skin detail_face", 0)',
        'inputs': 'mm_base "create_skin detail_face"; chara/etc.unity3d black2048; makeup GlossTex + gloss values',
        'code': [CHACONTROL_DLL, KSOX_DLL]},
    ('body', '_MainTex'): {
        'kind': 'chacontrol_body_composite', 'class': 'a',
        'producer': 'ChaControl.CreateBodyTexture -> SetNewCreateTexture(0, SkinTex) -> KSOX OverlayBlit: underlay '
                    'textures (UncensorSelector uncensorUnderlay) composited into a temporary first, overlays '
                    '(uncensorOverlay) after the create-material Blit (KoiSkinOverlayController.ApplyOverlays)',
        'inputs': 'mm_base body create material (UncensorSelector BodyMaterialCreate), body MainTex, skin colors, '
                  'paint/sunburn/nip layers, KSOX additional textures from the selected uncensor bundle',
        'code': [CHACONTROL_DLL, KSOX_DLL]},
    ('body', '_DetailMainTex'): {
        'kind': 'chacontrol_body_detail_composite', 'class': 'a',
        'producer': 'ChaControl.CreateBodyTexture -> SetNewCreateTexture(1, SkinCreateDetailTex)',
        'inputs': 'mm_base body detail create material, gloss layers', 'code': [CHACONTROL_DLL, KSOX_DLL]},
    ('clothes', '_MainTex'): {
        'kind': 'chacontrol_clothes_composite', 'class': 'a',
        'producer': 'ChaControl.ChangeCustomClothes -> CustomTextureCreate.RebuildTextureAndSetMaterial (clothes '
                    'create material, colour-mask recolour; KCOX ChangeCustomClothes post-hook overlays)',
        'inputs': 'clothes bundle create material + MainTex/ColorMask, card clothes colors/patterns',
        'code': [CHACONTROL_DLL, KSOX_DLL]},
    ('clothes', '_DetailMainTex'): {
        'kind': 'chacontrol_clothes_detail_composite', 'class': 'a',
        'producer': 'ChaControl.ChangeCustomClothes -> second CustomTextureCreate (detail/gloss)',
        'inputs': 'clothes bundle create material, card clothes parameters', 'code': [CHACONTROL_DLL, KSOX_DLL]},
}


def classify_runtime_texture(t, users=()):
    """Producer record for a runtime texture from the scene renderers/materials that use it."""
    for r, slot, prop in users:
        group = ('face' if r['name'] in FACE_RENDERERS else 'body' if r['name'] in BODY_RENDERERS else
                 'clothes' if any('/%s[' % c in r['path'] for c in CLOTHES_ROOTS) else None)
        rec = RUNTIME_PRODUCERS.get((group, prop))
        if rec is not None and t['type'].endswith('RenderTexture'):
            return dict(rec, renderer=r['name'], slot=slot, property=prop)
    if t['type'].endswith('RenderTexture'):
        return {'kind': 'render_target', 'class': 'in_frame_or_previous_frame',
                'producer': 'camera / PPv2 / CTAA / DHH render targets (owned by the frame generators)',
                'unity_type': t['type'], 'name': t['name']}
    return {'kind': 'script_texture', 'class': 'b', 'unity_type': t['type'], 'name': t['name'],
            'producer': 'Texture2D created at runtime by script (no installed asset of this identity)'}


# PPv2 script objects (Unity.Postprocessing.Runtime 2.1.7 RuntimeUtilities), reproduced by
# hs2pipe.ppv2 as well; bytes recovered from the C# source recorded in docs/hs2_offline_ppv2.md.
def ppv2_fullscreen_triangle_entries():
    """``RuntimeUtilities.fullscreenTriangle``: Mesh{vertices (-1,-1,0),(-1,3,0),(3,-1,0);
    SetIndices([0,1,2], Triangles, 0, false); UploadMeshData(false)} -> one position stream."""
    vb = struct.pack('<9f', -1, -1, 0, -1, 3, 0, 3, -1, 0)
    ib = struct.pack('<3H', 0, 1, 2)
    prov = {'class': 'computed', 'producer': 'PPv2 RuntimeUtilities.fullscreenTriangle', 'generator': GENERATOR}
    out = []
    for data, bind, role in ((vb, 1, 'vertex_stream_0'), (ib, 2, 'index_buffer')):
        desc = {'ByteWidth': len(data), 'Usage': 0, 'BindFlags': bind, 'CPUAccessFlags': 0, 'MiscFlags': 0,
                'StructureByteStride': 0}
        out.append(ResourceEntry('buffer', desc, [{'subresource': 0, 'data': data, 'row_pitch': len(data),
                                                   'slice_pitch': len(data)}], 'Fullscreen Triangle', dict(prov), role=role))
    return out


def ppv2_white_texture_entry(linear_project=True):
    """``RuntimeUtilities.whiteTexture``: new Texture2D(1, 1, ARGB32, false) (sRGB storage in
    a linear project), SetPixel(0, 0, Color.white), Apply()."""
    fmt = DXGI['R8G8B8A8_UNORM_SRGB' if linear_project else 'R8G8B8A8_UNORM']
    desc = {'Width': 1, 'Height': 1, 'MipLevels': 1, 'ArraySize': 1, 'Format': fmt, 'SampleDesc': {'Count': 1, 'Quality': 0},
            'Usage': 0, 'BindFlags': 8, 'CPUAccessFlags': 0, 'MiscFlags': 0}
    srv = {'Format': fmt, 'ViewDimension': 4, 'Texture2D': {'MostDetailedMip': 0, 'MipLevels': 1}}
    prov = {'class': 'computed', 'producer': 'PPv2 RuntimeUtilities.whiteTexture', 'generator': GENERATOR}
    return ResourceEntry('tex2d', desc, [{'subresource': 0, 'data': b'\xff\xff\xff\xff', 'row_pitch': 4, 'slice_pitch': 4}],
                         'White Texture', prov, srv=srv, role='texture')


# ------------------------------------------------------ plugin selections
class NetRandom:
    """``System.Random`` (Mono / .NET Framework reference source): Knuth's subtractive
    generator.  ``Random(0).Next() == 1559595546``."""
    MBIG = 2147483647
    MSEED = 161803398

    def __init__(self, seed):
        sub = 2147483647 if seed == -2147483648 else abs(seed)
        mj = self.MSEED - sub
        a = [0] * 56
        a[55] = mj
        mk = 1
        for i in range(1, 55):
            ii = (21 * i) % 55
            a[ii] = mk
            mk = mj - mk
            if mk < 0:
                mk += self.MBIG
            mj = a[ii]
        for _ in range(4):
            for i in range(1, 56):
                a[i] = ((a[i] - a[1 + (i + 30) % 55]) + 2 ** 31) % 2 ** 32 - 2 ** 31
                if a[i] < 0:
                    a[i] += self.MBIG
        self.a, self.inext, self.inextp = a, 0, 21

    def internal_sample(self):
        inext = self.inext + 1 if self.inext + 1 < 56 else 1
        inextp = self.inextp + 1 if self.inextp + 1 < 56 else 1
        r = self.a[inext] - self.a[inextp]
        if r == self.MBIG:
            r -= 1
        if r < 0:
            r += self.MBIG
        self.a[inext], self.inext, self.inextp = r, inext, inextp
        return r

    def next(self, max_value):
        return int(self.internal_sample() * (1.0 / self.MBIG) * max_value)


UNCENSOR_SELECTOR_DLL = 'BepInEx/Plugins/HS2_Plugins/DeathWeasel1337/HS2_UncensorSelector.dll'
UNCENSOR_BODY_NAMES = ('o_body_a', 'o_body_cf', 'o_body_cm')


def uncensor_selector_body(card, sideloader, game_root=GAME_ROOT, ext_body_guid=None):
    """Body uncensor UncensorSelector 3.11.5 uses for a character (decompiled
    ``UncensorSelectorController.BodyData``): the card's ``BodyGUID`` extended data if
    present, else ``DefaultData.GetDefaultOrRandomBody``: the configured default body,
    else ``GetRandomBody`` = eligible bodies (``Sex``, ``AllowRandom``, not in the
    ``Random Excluded`` config) in ``BodyDictionary`` insertion order (built-in default
    first, then ``Sideloader.Manifests.Values``), index
    ``new System.Random(birthDay + personality + (int)(voicePitch * 100f)).Next(n)``.

    ``card``: {'sex', 'birth_day', 'personality', 'voice_pitch'} (ChaFileParameter)."""
    import numpy as np
    from xml.etree import ElementTree as ET
    g = Path(game_root)
    cfg = (g / 'BepInEx/config/com.deathweasel.bepinex.uncensorselector.cfg').read_text(encoding='utf-8', errors='replace')

    def cfg_value(key):
        for line in cfg.splitlines():
            if line.split('=', 1)[0].strip() == key:
                return line.split('=', 1)[1].strip()
        raise ResourceUnresolved('UncensorSelector config key missing: ' + key)
    female = card['sex'] == 1
    default_name = cfg_value('Default Female Body' if female else 'Default Male Body')
    excluded = {x.strip().lower() for x in cfg_value('Female Body' if female else 'Male Body').split(',') if x.strip()}
    bodies = [{'guid': 'Default.Body.Female', 'sex': 1, 'allow_random': False, 'oo_base': 'chara/oo_base.unity3d',
               'asset': 'p_cf_body_00', 'display': 'Default Body F'}]
    for r in sideloader.loaded:
        if not r.get('uncensor_selector'):
            continue
        root = ET.fromstring(r['manifest_xml'])
        us = root.find('KK_UncensorSelector')
        for b in us.findall('body'):
            guid = (b.findtext('guid') or '').strip() or None
            disp = (b.findtext('displayName') or '').strip() or guid
            sex = 1 if (b.findtext('sex') or '').lower() == 'female' else 0
            ar = (b.findtext('allowRandom') or '').lower() not in ('false', '0')
            oo = b.find('oo_base')
            oo_file = (oo.findtext('file') or '').strip() if oo is not None else ''
            asset = (oo.findtext('asset') or '').strip() if oo is not None else ''
            oo_file = oo_file or 'chara/oo_base.unity3d'
            if guid is None or disp is None or oo_file == 'chara/oo_base.unity3d':
                continue
            row = {'guid': guid, 'sex': sex, 'allow_random': ar, 'oo_base': oo_file,
                   'asset': asset or ('p_cm_body_00' if sex == 0 else 'p_cf_body_00'), 'display': disp,
                   'manifest': r['path'], 'mod_guid': r['guid']}
            for i, old_row in enumerate(bodies):
                if old_row['guid'] == guid:
                    bodies[i] = row     # Dictionary[key] = value keeps the first insertion position
                    break
            else:
                bodies.append(row)
    evidence = {'producer': 'UncensorSelector 3.11.5 BodyData / GetDefaultOrRandomBody',
                'dll': UNCENSOR_SELECTOR_DLL, 'default_config': default_name}
    chosen = None
    if ext_body_guid:
        chosen = next((b for b in bodies if b['guid'] == ext_body_guid and b['sex'] == card['sex']), None)
        evidence['rule'] = 'card extended data BodyGUID'
    if chosen is None:
        by_display = {('[%s] %s' % ('Male' if b['sex'] == 0 else 'Female', b['display'])): b['guid'] for b in bodies}
        key = by_display.get(default_name, default_name)
        chosen = next((b for b in bodies if b['guid'] == key), None)
        evidence['rule'] = 'configured default body'
    if chosen is None:
        eligible = [b for b in bodies if b['sex'] == card['sex'] and b['allow_random'] and b['guid'].lower() not in excluded]
        if not eligible:
            chosen = bodies[0]
            evidence['rule'] = 'no eligible random body: built-in default'
        else:
            from .sideloader_assets import culture_order
            mods = [b.get('mod_guid') for b in eligible if b.get('mod_guid')]
            if culture_order(mods) != mods:
                raise ResourceUnresolved('eligible uncensor bodies are not in a determined Sideloader load order')
            pitch = np.float32(card['voice_pitch']) * np.float32(100.0)
            seed = int(card['birth_day']) + int(card['personality']) + int(np.trunc(pitch))
            idx = NetRandom(seed).next(len(eligible))
            chosen = eligible[idx]
            evidence.update(rule='GetRandomBody', seed=seed, eligible=[b['guid'] for b in eligible], index=idx)
    if chosen['oo_base'] == 'chara/oo_base.unity3d':
        source = None
    else:
        regs = sideloader.bundle_sources(chosen['oo_base'])
        if not regs:
            raise ResourceUnresolved('uncensor bundle %s not registered by any loaded zipmod' % chosen['oo_base'])
        source = '%s::%s' % (regs[0][0], regs[0][1])
        evidence['bundle_registrations'] = [r[2] for r in regs]
    return dict(chosen, source=source, evidence=evidence)


# ------------------------------------------------- XUnity texture translation
XUNITY_DLL = 'BepInEx/Plugins/XUnity.AutoTranslator/XUnity.AutoTranslator.Plugin.Core.dll'


class XUnityTextureTranslations:
    """XUnity.AutoTranslator texture replacement for UI textures (decompiled
    ``TextureTranslationCache`` / ``TextureTranslationInfo`` / ``LoadImageImageLoader``):
    every ``*.png`` below ``BepInEx/<TextureDirectory>`` (``{Lang}`` substituted) whose file
    name ends in ``[KEY-DATAHASH]`` is registered under KEY; it is applied when its own
    SHA-1 prefix differs from DATAHASH (or ``LoadUnmodifiedTextures``).  With
    ``TextureHashGenerationStrategy=FromImageName`` KEY is the first 10 upper-case hex digits
    of SHA-1(UTF-8 texture name).  The PNG is loaded with ``ImageConversion.LoadImage`` into
    the texture: decoded RGBA8, stored bottom-up (Unity texture row order)."""

    def __init__(self, game_root=GAME_ROOT):
        g = Path(game_root)
        cfg = (g / 'BepInEx/config/AutoTranslatorConfig.ini').read_text(encoding='utf-8-sig', errors='replace')
        vals = {}
        for line in cfg.splitlines():
            if '=' in line and not line.lstrip().startswith((';', '#', '[')):
                k, v = line.split('=', 1)
                vals[k.strip()] = v.strip()
        self.config = {k: vals.get(k) for k in ('Language', 'TextureDirectory', 'EnableTextureTranslation',
                                                 'TextureHashGenerationStrategy', 'LoadUnmodifiedTextures')}
        if self.config['TextureHashGenerationStrategy'] != 'FromImageName':
            raise ResourceUnresolved('XUnity texture key strategy %r not recovered' % self.config['TextureHashGenerationStrategy'])
        self.enabled = self.config['EnableTextureTranslation'] == 'True'
        rel = self.config['TextureDirectory'].replace('{Lang}', self.config['Language']).replace('\\', '/')
        self.root = g / 'BepInEx' / rel
        self.files = {}
        self.duplicates = {}
        if self.enabled and self.root.exists():
            for f in sorted(self.root.rglob('*')):
                if f.suffix.lower() == '.zip':
                    raise ResourceUnresolved('zipped XUnity texture packs are not reproduced: %s' % f)
                if f.suffix.lower() != '.png':
                    continue
                stem = f.stem
                a, b = stem.rfind('['), stem.rfind(']')
                if a < 0 or b < a:
                    continue
                parts = stem[a + 1:b].split('-')
                if len(parts) not in (1, 2):
                    continue
                key = parts[0]
                if key in self.files:
                    self.duplicates.setdefault(key, [self.files[key][0]]).append(str(f))
                self.files[key] = (str(f), parts[-1])

    @staticmethod
    def key(name):
        return hashlib.sha1(name.encode('utf-8')).hexdigest()[:10].upper()

    def apply(self, entry, linear_project=True):
        if not self.enabled:
            return entry
        key = self.key(entry.name)
        if key not in self.files:
            return entry
        if key in self.duplicates:
            raise ResourceUnresolved('XUnity texture key %s has several files (GetFiles order decides): %r'
                                     % (key, self.duplicates[key]))
        path, data_hash = self.files[key]
        data = Path(path).read_bytes()
        own = hashlib.sha1(data).hexdigest()[:10].upper()
        if own.lower() == data_hash.lower() and self.config['LoadUnmodifiedTextures'] != 'True':
            return entry
        from PIL import Image
        im = Image.open(io.BytesIO(data))
        if im.size != (entry.desc['Width'], entry.desc['Height']) or entry.desc['MipLevels'] != 1 or \
                entry.desc['Format'] not in (DXGI['R8G8B8A8_UNORM'], DXGI['R8G8B8A8_UNORM_SRGB']):
            raise ResourceUnresolved('XUnity LoadImage into %s (%r) changes size/format/mips: not recovered'
                                     % (entry.name, entry.desc))
        import numpy as np
        rgba = np.asarray(im.convert('RGBA'), dtype=np.uint8)[::-1]
        out = rgba.tobytes()
        prov = {'class': 'computed', 'producer': 'XUnity.AutoTranslator texture translation (LoadImage)',
                'dll': XUNITY_DLL, 'png': path, 'png_sha256': sha256(data), 'key': key,
                'replaced_asset': entry.provenance, 'generator': GENERATOR}
        return ResourceEntry(entry.kind, dict(entry.desc), [{'subresource': 0, 'data': out, 'row_pitch': entry.desc['Width'] * 4,
                                                            'slice_pitch': len(out)}], entry.name, prov, srv=entry.srv,
                             role=entry.role, extra=dict(entry.extra, xunity=True))


# --------------------------------------------------------------- scene level


class SceneResources:
    """Resolve scene-state requests (frame-pair probe export) to exact resource entries.

    ``preferred_sources``: source specs the game loaded for this scene (from the
    card's list rows / zipmods); used only when the installed files hold several
    *different-content* assets with the same identity.  Equal-content duplicates are
    reported in ``provenance['equivalent_sources']``."""

    def __init__(self, scene, index: SourceIndex, files: SourceFiles | None = None, preferred_sources=(),
                 sideloader=None, plugin_meshes=None, game_root=GAME_ROOT, xunity=None):
        self.scene = scene
        self.index = index
        self.files = files or SourceFiles()
        self.preferred = [str(s) for s in preferred_sources]
        self.sideloader = sideloader            # sideloader_assets.SideloaderState or None (vanilla only)
        self.plugin_meshes = dict(plugin_meshes or {})   # renderer name -> {'source', 'asset', 'producer'}
        self.game_root = Path(game_root)
        self._contains = {}
        self.xunity = xunity                    # XUnityTextureTranslations or None
        cs = scene.get('statics', {}).get('UnityEngine.QualitySettings', {}).get('activeColorSpace')
        if cs not in ('ColorSpace.Linear=1', 'ColorSpace.Gamma=0'):
            raise ResourceUnresolved('scene QualitySettings.activeColorSpace missing/unknown: %r' % cs)
        self.linear = cs == 'ColorSpace.Linear=1'
        self.textures = {t['id']: t for t in scene.get('textures', [])}
        self.renderers = {r['id']: r for r in scene.get('renderers', [])}
        self.materials = {m['id']: m for m in scene.get('materials', [])}
        self._bindings = {}
        self._renderer_binding = {}
        self._cache = {}
        self.texture_users = {}
        for r in scene.get('renderers', []):
            for slot, mref in enumerate(r.get('shared_materials') or []):
                m = self.materials.get(mref['ref_id']) if mref else None
                for prop, sl in ((m or {}).get('texture_properties') or {}).items():
                    if sl.get('texture'):
                        self.texture_users.setdefault(sl['texture']['ref_id'], []).append((r['id'], slot, prop))

    # ------------------------------------------------- Sideloader resolution
    def bundle_path_of(self, spec):
        """Asset-bundle path a source is loaded under (``chara/x.unity3d``), or None for
        player data files (``HoneySelect2_Data``), which Sideloader does not redirect."""
        path, member = SourceFiles.split(spec)
        if member is not None:
            return member[member.index('/') + 1:] if '/' in member else member
        rel = os.path.relpath(path, self.game_root / 'abdata')
        if rel.startswith('..'):
            return None
        return rel.replace('\\', '/')

    def contains(self, spec, asset_name):
        """``AssetBundle.Contains(name)``: the bundle's ``m_Container`` lists an asset whose
        file name without extension equals ``name`` (case-insensitive short-name lookup)."""
        key = spec
        if key not in self._contains:
            names = set()
            for o in self.files.objects(spec).values():
                if o.type.name == 'AssetBundle':
                    for path, _ in o.read_typetree()['m_Container']:
                        base = path.rsplit('/', 1)[-1]
                        names.add(base.rsplit('.', 1)[0].lower() if '.' in base else base.lower())
            self._contains[key] = names
        return asset_name.lower() in self._contains[key]

    def effective_source(self, spec, asset_name):
        """True when a load of ``asset_name`` from this source's bundle path returns this
        source (``Sideloader.RedirectHook`` -> ``BundleManager.TryGetObjectFromName``)."""
        bp = self.bundle_path_of(spec)
        if bp is None or self.sideloader is None:
            return SourceFiles.split(spec)[1] is None
        for zpath, member, guid in self.sideloader.bundle_sources(bp):
            reg = '%s::%s' % (zpath, member)
            if self.contains(reg, asset_name):
                return reg == spec
        return SourceFiles.split(spec)[1] is None

    # ---------------------------------------------------------------- choice
    def _choose(self, what, groups):
        """groups: {content_hash: [(entry, spec, path_id)]} -> entry (+ equivalent sources)."""
        if not groups:
            raise ResourceUnresolved('no installed asset matches %s' % what)
        total = len(groups)
        if len(groups) > 1:
            pref = {h: g for h, g in groups.items() if any(spec in self.preferred for _, spec, _ in g)}
            if len(pref) != 1:
                raise ResourceUnresolved('%s: %d different-content installed candidates %s; preferred sources select %d'
                                         % (what, len(groups), {h[:12]: [(s, p) for _, s, p in g] for h, g in groups.items()},
                                            len(pref)))
            groups = pref
        (h, g), = groups.items()
        g = sorted(g, key=lambda r: (r[1] not in self.preferred, r[1], r[2]))
        payload = g[0][0]
        for e in (payload if isinstance(payload, list) else [payload]):
            e.provenance['equivalent_sources'] = [{'source': s, 'path_id': p} for _, s, p in g[1:]]
            e.provenance['different_content_candidates'] = total
            e.provenance['selected_by_preferred_source'] = total > 1
        return payload

    # -------------------------------------------------------------- textures
    def texture(self, texture_id):
        if ('tex', texture_id) in self._cache:
            return self._cache[('tex', texture_id)]
        t = self.textures.get(texture_id)
        if t is None:
            raise ResourceUnresolved('scene texture %r not exported' % texture_id)
        typ = t['type'].split('.')[-1]
        p = t['properties']
        if typ == 'RenderTexture' or t['name'] in ('',) or typ not in ('Texture2D', 'Cubemap'):
            users = [(self.renderers[rid], slot, prop) for rid, slot, prop in self.texture_users.get(texture_id, [])]
            entry = runtime_texture_entry(t, classify_runtime_texture(t, users))
        elif t['name'] in BUILTIN_TEXTURES and texture_id < 0:
            entry = builtin_texture_entry(t['name'])
        elif t['name'] == 'White Texture' and texture_id < 0 and (p['width'], p['height'], p['format']) == \
                (1, 1, 'TextureFormat.ARGB32=5'):
            entry = ppv2_white_texture_entry(self.linear)
        elif texture_id < 0 and not self.index.find(typ, t['name']):
            users = [(self.renderers[rid], slot, prop) for rid, slot, prop in self.texture_users.get(texture_id, [])]
            entry = runtime_texture_entry(t, classify_runtime_texture(t, users))
        else:
            want = (p['width'], p['height'], int(p['format'].rsplit('=', 1)[1]), p['mipmapCount'])
            what = 'texture %s %dx%d fmt %d mips %d' % ((t['name'],) + want)
            entry = self._texture_from_prefab_material(texture_id, t, typ, want)
            if entry is None:
                groups = {}
                shadowed = []
                for spec, pid, _ in self.index.find(typ, t['name']):
                    if not self.effective_source(spec, t['name']):
                        shadowed.append(spec)
                        continue
                    e = self._asset_texture_if(spec, pid, want)
                    if e is not None:
                        groups.setdefault(e.content_hash(), []).append((e, spec, pid))
                entry = self._choose(what, groups)
                entry.provenance['binding'] = 'installed name index (no prefab material reference)'
                entry.provenance['shadowed_by_sideloader'] = shadowed
            entry.provenance['scene_texture_id'] = texture_id
        self._cache[('tex', texture_id)] = entry
        return entry

    def _asset_texture_if(self, spec, pid, want):
        obj = self.files.object(spec, pid)
        if obj.type.name not in ('Texture2D', 'Cubemap'):
            return None
        tree = obj.read_typetree()
        if (tree['m_Width'], tree['m_Height'], tree['m_TextureFormat'], tree['m_MipCount']) != want:
            return None
        return asset_texture_entry(self.files, spec, pid, linear_project=self.linear)

    def _texture_from_prefab_material(self, texture_id, t, typ, want):
        """Exact source of a texture the prefab material itself references: the scene
        renderer is bound to its prefab GameObject (``renderer_mesh``), whose material at
        that slot names the texture by PPtr.  None when no user renderer's prefab
        material holds a texture of this identity (it was assigned at runtime)."""
        found = {}
        for rid, slot, prop in self.texture_users.get(texture_id, []):
            r = self.renderers[rid]
            if rid not in self._renderer_binding:
                if not r['type'].startswith('UnityEngine.SkinnedMeshRenderer'):
                    continue
                try:
                    self.renderer_mesh(rid)
                except ResourceUnresolved:
                    continue
            spec, bound = self._renderer_binding[rid]
            for b in bound:
                if slot >= len(b['materials']):
                    continue
                owner = self.files.object(spec, b['component_path_id'])
                texs = prefab_material_textures(self.files, self.index, spec, b['materials'][slot], owner)
                loc = texs.get(prop)
                if loc is None:
                    continue
                tobj = self.files.object(*loc)
                if tobj.peek_name() != t['name']:
                    continue
                e = self._asset_texture_if(loc[0], loc[1], want)
                if e is not None:
                    e.provenance['binding'] = {'kind': 'prefab_material', 'renderer': rid, 'slot': slot, 'property': prop,
                                               'component_path_id': b['component_path_id']}
                    found.setdefault(e.content_hash(), []).append((e, loc[0], loc[1]))
        if not found:
            return None
        if len(found) > 1:
            raise ResourceUnresolved('texture %s: prefab materials reference %d different contents' % (t['name'], len(found)))
        return self._choose('texture %s via prefab material' % t['name'], found)

    def texture_identity(self, name, width, height, dxgi_format, mip_levels, unity_type='Texture2D', bundle_path=None,
                         ui=False):
        """A texture the scene export does not list (UI sprites, list thumbnails): resolved by
        identity (name, size, mip count and the D3D11 format its upload produces) among the
        installed sources Sideloader would read.  ``bundle_path`` (e.g. a list row's
        ``ThumbAB``) restricts the load to that asset-bundle path, as ``CommonLib.LoadAsset``
        does."""
        key = ('ident', name, width, height, dxgi_format, mip_levels, unity_type, bundle_path, ui)
        if key in self._cache:
            return self._cache[key]
        if name == 'White Texture' and (width, height, mip_levels) == (1, 1, 1):
            entry = ppv2_white_texture_entry(self.linear)
        elif name in BUILTIN_TEXTURES:
            entry = builtin_texture_entry(name)
        else:
            groups, shadowed, errors = {}, [], []
            for spec, pid, _ in self.index.find(unity_type, name):
                if bundle_path is not None and self.bundle_path_of(spec) != bundle_path:
                    continue
                if not self.effective_source(spec, name):
                    shadowed.append(spec)
                    continue
                obj = self.files.object(spec, pid)
                tree = obj.read_typetree()
                if (tree['m_Width'], tree['m_Height'], tree['m_MipCount']) != (width, height, mip_levels):
                    continue
                row = UNITY_TEXTURE_FORMATS.get(tree['m_TextureFormat'])
                if row is None:
                    errors.append({'source': spec, 'path_id': pid, 'error': 'TextureFormat %d' % tree['m_TextureFormat']})
                    continue
                srgb = tree.get('m_ColorSpace', 0) == 1 and self.linear and row[2] is not None
                if DXGI[row[2] if srgb else row[1]] != dxgi_format:
                    continue
                try:
                    e = asset_texture_entry(self.files, spec, pid, linear_project=self.linear)
                except ResourceUnresolved as err:
                    errors.append({'source': spec, 'path_id': pid, 'error': str(err)})
                    continue
                groups.setdefault(e.content_hash(), []).append((e, spec, pid))
            what = 'texture identity %s %dx%d DXGI %d mips %d' % (name, width, height, dxgi_format, mip_levels)
            if errors and not groups:
                raise ResourceUnresolved('%s: no convertible candidate; %r' % (what, errors[:4]))
            entry = self._choose(what, groups)
            entry.provenance['binding'] = 'texture identity request (not in the scene export)'
            entry.provenance['shadowed_by_sideloader'] = shadowed
            entry.provenance['unconvertible_candidates'] = errors
        if ui and self.xunity is not None:
            entry = self.xunity.apply(entry, self.linear)
        self._cache[key] = entry
        return entry

    def material_texture(self, material_id, prop):
        m = self.materials.get(material_id)
        if m is None:
            raise ResourceUnresolved('scene material %r not exported' % material_id)
        slot = m['texture_properties'].get(prop)
        if slot is None:
            raise ResourceUnresolved('material %s has no texture property %s' % (m['name'], prop))
        if slot['texture'] is None:
            return None
        return self.texture(slot['texture']['ref_id'])

    def renderer_textures(self, renderer_id):
        r = self.renderers[renderer_id]
        out = []
        for slot, mref in enumerate(r['shared_materials']):
            m = self.materials.get(mref['ref_id']) if mref else None
            row = {}
            if m:
                for prop, s in m['texture_properties'].items():
                    row[prop] = None if s['texture'] is None else self.texture(s['texture']['ref_id'])
            out.append(row)
        return out

    # ---------------------------------------------------------------- meshes
    def _bindings_of(self, spec):
        if spec not in self._bindings:
            self._bindings[spec] = renderer_mesh_bindings(self.files, spec)
        return self._bindings[spec]

    def _scene_file_bindings(self, spec, pid, go_name, scene_path):
        """MeshFilters of objects placed in a player scene file (``HoneySelect2_Data/level*``)
        whose full hierarchy path equals the scene renderer's and whose ``m_Mesh`` resolves
        (through the level's externals) to ``(spec, pid)``."""
        out = []
        for lvl in sorted(self.index.doc['sources']):
            if SourceFiles.split(lvl)[1] is not None or not Path(lvl).name.startswith('level') or Path(lvl).suffix:
                continue
            for b in self._bindings_of(lvl):
                if b['go'] != go_name or b['mesh'] is None or b['path'] != scene_path:
                    continue
                owner = None
                for o in self.files.objects(lvl).values():
                    if o.type.name == 'MeshFilter' and o.read_typetree()['m_Mesh'] == {'m_FileID': b['mesh'][0], 'm_PathID': b['mesh'][1]}:
                        owner = o
                        break
                if owner is None:
                    continue
                if resolve_pptr(self.files, self.index, lvl, owner, *b['mesh']) == (spec, pid):
                    out.append(dict(b, level=lvl))
        return out

    def renderer_mesh(self, renderer_id, mesh_name=None):
        """Static vertex streams + index buffer of a renderer's mesh.

        SkinnedMeshRenderer: the mesh name comes from ``sharedMesh``.  MeshRenderer:
        the probe does not export ``MeshFilter.sharedMesh``; pass ``mesh_name``
        (explicit gap, see docs)."""
        key = ('mesh', renderer_id, mesh_name)
        if key in self._cache:
            return self._cache[key]
        r = self.renderers.get(renderer_id)
        if r is None:
            raise ResourceUnresolved('scene renderer %r not exported' % renderer_id)
        skinned = r['type'].startswith('UnityEngine.SkinnedMeshRenderer')
        if skinned:
            sm = r['properties'].get('sharedMesh')
            if not isinstance(sm, dict):
                raise ResourceUnresolved('renderer %s has no sharedMesh' % r['name'])
            name = sm['ref_name']
        elif mesh_name is None:
            raise ResourceUnresolved('MeshRenderer %s: MeshFilter.sharedMesh is not in the scene export; '
                                     'pass mesh_name' % r['name'])
        else:
            name = mesh_name
        scene_path = _scene_path_names(r['path'])
        groups, bindings = {}, {}
        plugin = self.plugin_meshes.get(r['name'])
        skipped = []
        for spec, pid, _ in self.index.find('Mesh', name):
            if plugin is not None and spec != plugin['source']:
                continue
            bound = [b for b in self._bindings_of(spec) if b['mesh'] == (0, pid)
                     and b['go'] == r['name'] and len(b['path']) >= 2
                     and (scene_path[-(len(b['path']) - 1):] == b['path'][1:] if plugin is None
                          else b['path'][0] == plugin['asset'])]
            if not bound and plugin is None and self.bundle_path_of(spec) is None:
                bound = self._scene_file_bindings(spec, pid, r['name'], scene_path)
                for b in bound:
                    b['scene_file'] = True
            if not bound:
                continue
            if plugin is None and not bound[0].get('scene_file'):
                bound = [b for b in bound if self.effective_source(spec, b['path'][0])]
                if not bound:
                    skipped.append(spec)
                    continue
            entries = asset_mesh_entries(self.files, spec, pid, skinned=skinned)
            for e in entries:
                e.provenance['prefab_binding'] = [{'component': b['component'], 'component_path_id': b['component_path_id'],
                                                   'path': '/'.join(b['path']), **({'scene_file': b['level']} if b.get('level') else {})}
                                                  for b in bound]
            h = sha256(b''.join(e.content_hash().encode() for e in entries))
            groups.setdefault(h, []).append((entries, spec, pid))
            bindings[(spec, pid)] = bound
        entries = self._choose('mesh %s of renderer %s' % (name, r['path']), groups)
        for e in entries:
            e.provenance['shadowed_by_sideloader'] = skipped
            if plugin is not None:
                e.provenance['plugin_mesh'] = plugin
        src = (entries[0].provenance['source'], entries[0].provenance['path_id'])
        self._renderer_binding[renderer_id] = (src[0], bindings[src])
        for e in entries:
            e.provenance['scene_renderer_id'] = renderer_id
            e.provenance['scene_mesh_id'] = (r['properties'].get('sharedMesh') or {}).get('ref_id') if skinned else None
        self._cache[key] = entries
        return entries
