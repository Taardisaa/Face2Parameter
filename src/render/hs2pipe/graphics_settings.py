"""Installed GraphicsSettings built-in shader references (globalgamemanagers).

``GraphicsSettings`` stores, per built-in shader setting, ``{m_Mode, m_Shader}``: mode 1
(Builtin) uses the referenced shader object, a PPtr into ``m_Externals`` (for HS2:
``resources/unity_builtin_extra``).  ``Camera::RenderDepthNormalsTexture`` uses
``GetGraphicsSettings().GetBuiltinShader(kDepthNormals)``, the motion-vector loop setting 5
(MotionVectors) and the collect pass the ScreenSpaceShadows setting.  ``builtin_shader_objects``
resolves those PPtrs to (installed file, path id) so a shader-library copy can be checked to be
that object.
"""
from __future__ import annotations

from pathlib import Path

SETTINGS = ('m_Deferred', 'm_DeferredReflections', 'm_ScreenSpaceShadows', 'm_LegacyDeferred', 'm_DepthNormals',
            'm_MotionVectors', 'm_LightHalo', 'm_LensFlare')
_CACHE = {}


def builtin_shader_objects(game_root='E:/HoneySelect2_ArcticFox'):
    """{setting: {'mode', 'file', 'path_id'}} from ``<Data>/globalgamemanagers``."""
    key = str(game_root)
    if key in _CACHE:
        return _CACHE[key]
    import UnityPy
    data = Path(game_root) / 'HoneySelect2_Data'
    env = UnityPy.load(str(data / 'globalgamemanagers'))
    out = {}
    for o in env.objects:
        if o.type.name != 'GraphicsSettings':
            continue
        t = o.read_typetree()
        ext = [e.path for e in o.assets_file.externals]
        for k in SETTINGS:
            v = t.get(k)
            if not v:
                continue
            fid = int(v['m_Shader']['m_FileID'])
            rel = ext[fid - 1] if fid > 0 else 'globalgamemanagers'
            # 'resources/unity_builtin_extra' -> <Data>/Resources/unity_builtin_extra
            parts = rel.split('/')
            path = data.joinpath(*[p if p != 'resources' else 'Resources' for p in parts])
            out[k] = {'mode': int(v['m_Mode']), 'file': str(path), 'path_id': int(v['m_Shader']['m_PathID'])}
    _CACHE[key] = out
    return out


def library_copy_is_setting(copy, setting, game_root='E:/HoneySelect2_ArcticFox'):
    """True when a shader-library copy is the object GraphicsSettings.<setting> references (Builtin mode)."""
    ref = builtin_shader_objects(game_root).get(setting)
    if not ref or ref['mode'] != 1:
        return False
    return (int(copy.record.get('path_id')) == ref['path_id']
            and str(copy.source_label).replace('/', '\\').lower() == ref['file'].replace('/', '\\').lower())
