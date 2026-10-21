"""PPv2 generator inputs from the frame-pair probe scene state (``scene_*.json``).

The probe (``tools/offline_renderer/frame_pair_probe``) reflects, per frame, the live
fields of every PostProcessLayer / PostProcessVolume (instantiated profiles), the
layer's bundle settings and renderer state, ``Dithering`` (index + ``System.Random``
seed array), ``ColorGradingRenderer.m_Pixels``, cameras, globals, QualitySettings and
RenderTexture objects.  The snapshot of frame N is taken after that frame's
``OnPreCull`` (Spline ``frameCount`` == N, Dithering already stepped), so the
generator for frame N takes

* inputs that PPv2 only reads (volumes, profiles, camera, globals) from snapshot N,
* PPv2's own persistent state (Dithering, ``m_Pixels``) from snapshot N-1,

and snapshot N's layer bundle settings / Dithering / ``m_Pixels`` are acceptance
references for the blend and the persistent-state update.

AnimationCurve keys are not reflected (``<depth limit>``); they are taken from the
installed profile assets (``extract_ppv2_scene_state``) for scene volumes; the DHH
splines are ``new AnimationCurve()`` in ``DHH_Main.ApplyGraphicSetting`` (no keys).
"""
from __future__ import annotations

from .ppv2_settings import DEFAULTS, EffectSettings, SplineValue, TextureValue, Volume, _f, effect_from_scene
from .ppv2_unity import F, SystemRandom, TextureRef


def enum_value(v):
    if isinstance(v, str) and '=' in v:
        return int(v.rsplit('=', 1)[1])
    return int(v)


def flag_names(v):
    return v.split('=', 1)[0].split('.', 1)[1] if isinstance(v, str) else v


class ProbeScene:
    def __init__(self, scene):
        self.scene = scene
        self.textures = {t['id']: t for t in scene['textures']}
        self.components = scene['effect_components']

    def component(self, type_suffix, name):
        rows = [c for c in self.components if c['type'].split(',')[0].endswith(type_suffix) and c['name'] == name]
        if len(rows) != 1:
            raise KeyError('%s %s: %d matches' % (type_suffix, name, len(rows)))
        return rows[0]

    def camera(self, name='Main Camera'):
        rows = [c for c in self.scene['cameras'] if c['name'] == name]
        if len(rows) != 1:
            raise KeyError(name)
        return rows[0]['properties']

    def texture_value(self, ref):
        if ref is None:
            return None
        t = self.textures.get(ref['ref_id'])
        w = t['properties'].get('width') if t else None
        h = t['properties'].get('height') if t else None
        return TextureValue('probe:%d:%s' % (ref['ref_id'], ref.get('ref_name')), w, h)

    def texture_attrs(self, name):
        rows = [t for t in self.scene['textures'] if t['name'] == name]
        if len(rows) != 1:
            raise KeyError('texture %r: %d matches' % (name, len(rows)))
        p = rows[0]['properties']
        return {'filter_mode': flag_names(p['filterMode']), 'wrap_mode': flag_names(p['wrapModeU']),
                'aniso_level': p['anisoLevel']}


def _param(kind, entry, probe, static_spline=None, owner=''):
    state = entry['ParameterOverride.overrideState']
    v = entry['ParameterOverride`1.value']
    ds = enum_value(entry['TextureParameter.defaultState']) if 'TextureParameter.defaultState' in entry else None
    if kind == 'float':
        val = F(v)
    elif kind == 'int':
        val = int(v)
    elif kind == 'bool':
        val = bool(v)
    elif kind == 'enum':
        val = enum_value(v)
    elif kind in ('color', 'vector2', 'vector4'):
        val = tuple(F(x) for x in v)
    elif kind == 'texture':
        val = probe.texture_value(v)
    elif kind == 'spline':
        if v is None:
            val = None
        else:
            keys = static_spline.keys if static_spline is not None else []
            val = SplineValue(keys, F(v['Spline.m_ZeroValue']), v['Spline.m_Loop'], F(v['Spline.m_Range']), None,
                              name=owner)
            val.cache(-2147483648)
    else:
        raise KeyError(kind)
    return state, val, ds


def effect_from_probe(cls, fields, probe, static_effect=None, active_override=None):
    s = EffectSettings(cls, active=fields['PostProcessEffectSettings.active'] if active_override is None else active_override)
    en = fields['PostProcessEffectSettings.enabled']
    s.params['enabled'].override_state = en['ParameterOverride.overrideState']
    s.params['enabled'].value = bool(en['ParameterOverride`1.value'])
    for name, kind, default in DEFAULTS[cls]:
        entry = fields['%s.%s' % (cls, name)]
        static = static_effect.params[name].value if (static_effect is not None and kind == 'spline') else None
        state, val, ds = _param(kind, entry, probe, static, '%s.%s' % (cls, name))
        p = s.params[name]
        p.override_state = state
        p.value = val
        if ds is not None:
            p.default_state = ds
    return s


def volumes_from_probe(scene, static_scene, order=('PostProcessVolume2D', 'PostProcessVolume3D', 'DHH_Post')):
    """Volumes (instantiated profiles) in the given registration order.

    ``Behaviour.enabled`` of the volume components is not reflected by the probe;
    ``active_in_hierarchy`` is used (gap: ``PostProcessVolume.enabled``)."""
    probe = ProbeScene(scene)
    static = {}
    for k, row in static_scene['effects'].items():
        static[(row['file'], row['path_id'])] = effect_from_scene(row, static_scene.get('textures'))
    profile_static = {}
    for key, prof in static_scene['profiles'].items():
        profile_static[prof['m_Name']] = [static[(prof['file'], p['path_id'])] for p in prof['settings']]
    vols = []
    for name in order:
        c = probe.component('PostProcessVolume', name)
        f = c['fields']
        shared = f.get('PostProcessVolume.sharedProfile')
        prof_name = shared['ref']['ref_name'] if shared else None
        statics = {s.cls: s for s in profile_static.get(prof_name, [])}
        settings = []
        for st in f['PostProcessVolume.m_InternalProfile']['fields']['PostProcessProfile.settings']:
            cls = st['ref']['ref_type'].split('.')[-1]
            settings.append(effect_from_probe(cls, st['fields'], probe, statics.get(cls)))
        vols.append(Volume(name=name, layer=c['layer'], priority=F(f['PostProcessVolume.priority']),
                           weight=F(f['PostProcessVolume.weight']), is_global=f['PostProcessVolume.isGlobal'],
                           enabled=c['active_in_hierarchy'], settings=settings,
                           blend_distance=F(f['PostProcessVolume.blendDistance']),
                           provenance={'probe_id': c['id'], 'path': c['path'], 'shared_profile': prof_name}))
    return vols


def layer_fields(scene, camera='Main Camera'):
    return ProbeScene(scene).component('PostProcessLayer', camera)['fields']


def layer_config(scene, camera='Main Camera'):
    from .ppv2 import LayerConfig
    f = layer_fields(scene, camera)
    bundles = f['PostProcessLayer.m_BeforeTransparentBundles'] + f['PostProcessLayer.m_BeforeStackBundles'] + \
        f['PostProcessLayer.m_AfterStackBundles']
    return LayerConfig(volume_layer=f['PostProcessLayer.volumeLayer']['LayerMask.m_Mask'],
                       stop_nan_propagation=f['PostProcessLayer.stopNaNPropagation'],
                       antialiasing_mode=enum_value(f['PostProcessLayer.antialiasingMode']),
                       smaa_quality=enum_value(f['PostProcessLayer.subpixelMorphologicalAntialiasing']['SubpixelMorphologicalAntialiasing.quality']),
                       fxaa_fast_mode=f['PostProcessLayer.fastApproximateAntialiasing']['FastApproximateAntialiasing.fastMode'],
                       fxaa_keep_alpha=f['PostProcessLayer.fastApproximateAntialiasing']['FastApproximateAntialiasing.keepAlpha'],
                       break_before_color_grading=f['PostProcessLayer.breakBeforeColorGrading'],
                       fog_enabled=f['PostProcessLayer.fog']['Fog.enabled'], custom_bundles=tuple(bundles),
                       debug_overlay=enum_value(f['PostProcessLayer.debugLayer']['PostProcessDebugLayer.<debugOverlay>k__BackingField']))


def dithering_state(scene, camera='Main Camera'):
    """(m_NoiseTextureIndex, System.Random) of ``PostProcessLayer.dithering``."""
    d = layer_fields(scene, camera)['PostProcessLayer.dithering']
    r = d['Dithering.m_Random']
    rnd = SystemRandom(0)
    rnd.seed_array = [int(x) for x in r['Random.SeedArray']]
    rnd.inext = r['Random.inext']
    rnd.inextp = r['Random.inextp']
    if len(rnd.seed_array) != 56:
        raise ValueError('System.Random SeedArray must have 56 entries')
    return d['Dithering.m_NoiseTextureIndex'], rnd


def curve_pixels(scene, camera='Main Camera'):
    cg = layer_fields(scene, camera)['PostProcessLayer.m_Bundles']['UnityEngine.Rendering.PostProcessing.ColorGrading']
    px = cg['PostProcessBundle.m_Renderer']['ColorGradingRenderer.m_Pixels']
    if len(px) != 256:
        raise ValueError('m_Pixels must have 256 colors')
    return [tuple(F(c) for c in p) for p in px]


def blended_settings(scene, camera='Main Camera'):
    """The layer's bundle settings after this frame's volume blending (reference)."""
    probe = ProbeScene(scene)
    out = {}
    for typ, b in layer_fields(scene, camera)['PostProcessLayer.m_Bundles'].items():
        cls = typ.split('.')[-1]
        if cls not in DEFAULTS:
            continue
        st = b['PostProcessBundle.<settings>k__BackingField']['fields']
        out[cls] = st
    return out, probe


def camera_state(scene, camera='Main Camera'):
    from .ppv2 import CameraState
    p = ProbeScene(scene).camera(camera)
    proj = p['projectionMatrix']          # row-major 4x4 as reflected
    return CameraState(pixel_width=p['pixelWidth'], pixel_height=p['pixelHeight'], near_clip=F(p['nearClipPlane']),
                       far_clip=F(p['farClipPlane']), projection_m00=F(proj[0]), allow_hdr=p['allowHDR'],
                       rendering_path=flag_names(p['actualRenderingPath']).split('.')[-1],
                       stereo=p['stereoEnabled'])


def render_settings(scene):
    from .ppv2 import RenderSettingsState
    rs = scene['statics']['UnityEngine.RenderSettings']
    return RenderSettingsState(fog=rs['fog'], fog_density=F(rs['fogDensity']), fog_start=F(rs['fogStartDistance']),
                               fog_end=F(rs['fogEndDistance']))


def anisotropic_filtering(scene):
    return flag_names(scene['statics']['UnityEngine.QualitySettings']['anisotropicFiltering']).split('.')[-1]


def global_vector(scene, name):
    v = scene['global_uniforms'][name]['value']
    return tuple(F(x) for x in v) if isinstance(v, list) else F(v)
