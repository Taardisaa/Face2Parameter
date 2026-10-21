"""PostProcessing v2 (2.1.7) settings, volume blending and the HS2 configuration.

Source (installed ``Unity.Postprocessing.Runtime.dll`` SHA-256 f18eac42..., identified
as package tag 2.1.7, commit 283c8187): ``PostProcessEffectSettings`` and the effect
classes (defaults below), ``ParameterOverride<T>.Interp`` and its overrides,
``SplineParameter``, ``TextureParameter``, ``PostProcessManager.UpdateSettings``
(``ReplaceData`` + ``OverrideSettings``), ``PostProcessVolume``.

HS2 configuration sources:
* CharaCustom scene (``level3``): two global volumes on layer 8,
  ``PostProcessVolume2D`` (profile "Custom Camera Profile") and
  ``PostProcessVolume3D`` (profile "Main Camera Profile Custom"), both priority 0,
  weight 1 (``tools/offline_renderer/extract_ppv2_scene_state.py``);
  ``CameraEffector.ConfigEffectorVolume`` sets the Bloom/AO/SSR/Vignette
  ``active`` flags from ``Manager.Config.GraphicData`` (UserData/config/system.xml).
* DHH (``DHH_HS2.dll`` SHA-256 44137712...): ``DHH_Main.Init`` adds the global
  ``DHH_Post`` volume (layer 8, priority 1000); ``ApplyGraphicSetting`` writes the
  parameters from ``DHH_Data/GraphicSetting.dhs`` (parsed with ``GetDataWithTag``)
  and sets the volume weight to ``GraphicFloat[45]``.

All arithmetic is float32 (``ppv2_unity.F``).
"""
from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from .ppv2_unity import F, dv, evaluate_curve, mathf_clamp01, mathf_sqrt, r4

K_PRECISION = 128
K_STEP = F(1) / F(128)


# ------------------------------------------------------------------ value types
@dataclass
class TextureValue:
    """Identity of a texture referenced by a TextureParameter (asset or runtime)."""
    key: str
    width: int | None = None
    height: int | None = None


class SplineValue:
    """``Spline`` (k_Precision 128).  ``cached`` mirrors ``cachedData``; ``frame_count``
    mirrors the private ``frameCount`` used by ``Cache``."""

    def __init__(self, keys, zero_value, loop, range_, cached=None, name=''):
        self.keys = [dict(k) for k in keys]
        self.zero = F(zero_value)
        self.loop = bool(loop)
        self.range = F(range_)
        self.cached = [F(x) for x in cached] if cached is not None else [F(0)] * K_PRECISION
        self.frame_count = -1
        self.name = name

    @classmethod
    def new(cls, keys, zero_value, loop, bounds=(0.0, 1.0), name=''):
        bx, by = dv(bounds[0]), dv(bounds[1])
        # Vector2.magnitude = Mathf.Sqrt(x * x + y * y)
        return cls(keys, zero_value, loop, mathf_sqrt(r4(bx * bx + by * by)), None, name)

    def _looping_keys(self):
        last = dict(self.keys[-1])
        last['time'] = r4(dv(last['time']) - dv(self.range))
        first = dict(self.keys[0])
        first['time'] = r4(dv(first['time']) + dv(self.range))
        return sorted([last] + self.keys + [first], key=lambda k: float(k['time']))

    def evaluate(self, t):
        n = len(self.keys)
        if n == 0:
            return self.zero
        if not self.loop or n == 1:
            return evaluate_curve(self.keys, t)
        return evaluate_curve(self._looping_keys(), t)

    def cache(self, frame):
        if frame == self.frame_count:
            return
        for i in range(K_PRECISION):
            self.cached[i] = self.evaluate(r4(dv(i) * dv(K_STEP)))
        self.frame_count = frame


@dataclass
class Param:
    kind: str                 # float int bool enum color vector2 vector4 spline texture
    override_state: bool
    value: object
    default_state: int | None = None   # TextureParameter.defaultState

    def set_value(self, other: 'Param', frame):
        """``ParameterOverride<T>.SetValue`` (+ SplineParameter re-cache)."""
        if self.kind in ('color', 'vector2', 'vector4'):
            self.value = tuple(other.value)   # struct copy
        else:
            self.value = other.value          # reference for Spline/Texture objects
        if self.kind == 'spline' and self.value is not None:
            self.value.cache(frame)

    def interp(self, frm: 'Param', to: 'Param', t, frame):
        """``ParameterOverride.Interp(from, to, t)`` with the per-type overrides."""
        t = F(t)
        k = self.kind
        if k == 'float':
            a, b = dv(frm.value), dv(to.value)
            self.value = r4(a + (b - a) * dv(t))
        elif k == 'int':
            a, b = int(frm.value), int(to.value)
            self.value = int(dv(a) + dv(b - a) * dv(t))   # (int)((float)from + (float)(to - from) * t)
        elif k in ('color', 'vector2', 'vector4'):
            self.value = tuple(r4(dv(a) + (dv(b) - dv(a)) * dv(t)) for a, b in zip(frm.value, to.value))
        elif k == 'spline':
            a, b = frm.value, to.value
            if a is None or b is None:
                self.value = b if t > F(0) else a
                return
            a.cache(frame)
            b.cache(frame)
            for i in range(K_PRECISION):
                x, y = dv(a.cached[i]), dv(b.cached[i])
                self.value.cached[i] = r4(x + (y - x) * dv(t))
        elif k == 'texture':
            self.value = texture_interp(frm.value, to.value, t, self.default_state)
        else:  # bool / enum: base ParameterOverride<T>.Interp
            self.value = to.value if t > F(0) else frm.value


TEXTURE_DEFAULT = {'None': 0, 'Black': 1, 'White': 2, 'Transparent': 3, 'Lut2D': 4}   # TextureParameterDefault


def texture_interp(frm, to, t, default_state):
    """``TextureParameter.Interp``.  The GPU ``TextureLerper`` path (0 < t < 1 with
    two different textures) is not reproduced and raises (gap ``texture_lerper``)."""
    if frm is None and to is None:
        return None
    if frm is not None and to is not None:
        return lerp_texture(frm, to, t)
    if default_state == TEXTURE_DEFAULT['Lut2D']:
        h = frm.height if frm is not None else to.height
        strip = TextureValue('RuntimeUtilities.GetLutStrip(%d)' % h, h * h, h)
        frm = frm or strip
        to = to or strip
        if frm.width != to.width or frm.height != to.height:
            return None
        return lerp_texture(frm, to, t)
    if default_state not in (1, 2, 3):
        return to if t > F(0) else frm          # base ParameterOverride<T>.Interp
    color = {1: 'black', 2: 'white', 3: 'clear'}[default_state]
    to2 = TextureValue('RuntimeUtilities.%sTexture' % color, 1, 1)
    if frm is None:
        return lerp_texture(to, to2, r4(1.0 - dv(t)))
    return lerp_texture(frm, to2, t)


def lerp_texture(frm, to, t):
    if frm == to or frm.key == to.key:
        return frm
    if t <= F(0):
        return frm
    if t >= F(1):
        return to
    raise NotImplementedError('TextureLerper blend of %s -> %s at t=%r (gap texture_lerper)' % (frm.key, to.key, t))


# -------------------------------------------------------------- effect defaults
def _identity_curve():
    return [{'time': F(0), 'value': F(0), 'inSlope': F(1), 'outSlope': F(1), 'weightedMode': 0,
             'inWeight': F(0), 'outWeight': F(0)},
            {'time': F(1), 'value': F(1), 'inSlope': F(1), 'outSlope': F(1), 'weightedMode': 0,
             'inWeight': F(0), 'outWeight': F(0)}]


# Keyframe(time, value, inTangent, outTangent): weightedMode None, weights 1/3 in the
# managed ctor; the defaults are non-weighted, weights irrelevant for evaluation.
def _defaults():
    W = (1.0, 1.0, 1.0, 1.0)
    return {
        'AmbientOcclusion': [('mode', 'enum', 1), ('intensity', 'float', 0.0), ('color', 'color', (0.0, 0.0, 0.0, 1.0)),
                             ('ambientOnly', 'bool', True), ('noiseFilterTolerance', 'float', 0.0),
                             ('blurTolerance', 'float', -4.6), ('upsampleTolerance', 'float', -12.0),
                             ('thicknessModifier', 'float', 1.0), ('directLightingStrength', 'float', 0.0),
                             ('radius', 'float', 0.25), ('quality', 'enum', 2)],
        'DepthOfField': [('focusDistance', 'float', 10.0), ('aperture', 'float', 5.6), ('focalLength', 'float', 50.0),
                         ('kernelSize', 'enum', 1)],
        'ColorGrading': [('gradingMode', 'enum', 1), ('externalLut', 'texture', (None, 1)), ('tonemapper', 'enum', 0),
                         ('toneCurveToeStrength', 'float', 0.0), ('toneCurveToeLength', 'float', 0.5),
                         ('toneCurveShoulderStrength', 'float', 0.0), ('toneCurveShoulderLength', 'float', 0.5),
                         ('toneCurveShoulderAngle', 'float', 0.0), ('toneCurveGamma', 'float', 1.0),
                         ('ldrLut', 'texture', (None, 4)), ('ldrLutContribution', 'float', 1.0),
                         ('temperature', 'float', 0.0), ('tint', 'float', 0.0), ('colorFilter', 'color', W),
                         ('hueShift', 'float', 0.0), ('saturation', 'float', 0.0), ('brightness', 'float', 0.0),
                         ('postExposure', 'float', 0.0), ('contrast', 'float', 0.0),
                         ('mixerRedOutRedIn', 'float', 100.0), ('mixerRedOutGreenIn', 'float', 0.0),
                         ('mixerRedOutBlueIn', 'float', 0.0), ('mixerGreenOutRedIn', 'float', 0.0),
                         ('mixerGreenOutGreenIn', 'float', 100.0), ('mixerGreenOutBlueIn', 'float', 0.0),
                         ('mixerBlueOutRedIn', 'float', 0.0), ('mixerBlueOutGreenIn', 'float', 0.0),
                         ('mixerBlueOutBlueIn', 'float', 100.0),
                         ('lift', 'vector4', (1.0, 1.0, 1.0, 0.0)), ('gamma', 'vector4', (1.0, 1.0, 1.0, 0.0)),
                         ('gain', 'vector4', (1.0, 1.0, 1.0, 0.0)),
                         ('masterCurve', 'spline', ('identity', 0.0, False)), ('redCurve', 'spline', ('identity', 0.0, False)),
                         ('greenCurve', 'spline', ('identity', 0.0, False)), ('blueCurve', 'spline', ('identity', 0.0, False)),
                         ('hueVsHueCurve', 'spline', ('empty', 0.5, True)), ('hueVsSatCurve', 'spline', ('empty', 0.5, True)),
                         ('satVsSatCurve', 'spline', ('empty', 0.5, False)), ('lumVsSatCurve', 'spline', ('empty', 0.5, False))],
        'Bloom': [('intensity', 'float', 0.0), ('threshold', 'float', 1.0), ('softKnee', 'float', 0.5),
                  ('clamp', 'float', 65472.0), ('diffusion', 'float', 7.0), ('anamorphicRatio', 'float', 0.0),
                  ('color', 'color', W), ('fastMode', 'bool', False), ('dirtTexture', 'texture', (None, 1)),
                  ('dirtIntensity', 'float', 0.0)],
        'Vignette': [('mode', 'enum', 0), ('color', 'color', (0.0, 0.0, 0.0, 1.0)), ('center', 'vector2', (0.5, 0.5)),
                     ('intensity', 'float', 0.0), ('smoothness', 'float', 0.2), ('roundness', 'float', 1.0),
                     ('rounded', 'bool', False), ('mask', 'texture', (None, 1)), ('opacity', 'float', 1.0)],
        'ScreenSpaceReflections': [('preset', 'enum', 2), ('maximumIterationCount', 'int', 16), ('resolution', 'enum', 0),
                                   ('thickness', 'float', 8.0), ('maximumMarchDistance', 'float', 100.0),
                                   ('distanceFade', 'float', 0.5), ('vignette', 'float', 0.5)],
        'AutoExposure': [('filtering', 'vector2', (50.0, 95.0)), ('minLuminance', 'float', 0.0),
                         ('maxLuminance', 'float', 0.0), ('keyValue', 'float', 1.0), ('eyeAdaptation', 'enum', 0),
                         ('speedUp', 'float', 2.0), ('speedDown', 'float', 1.0)],
        'Grain': [('colored', 'bool', True), ('intensity', 'float', 0.0), ('size', 'float', 1.0), ('lumContrib', 'float', 0.8)],
        'ChromaticAberration': [('spectralLut', 'texture', (None, 1)), ('intensity', 'float', 0.0), ('fastMode', 'bool', False)],
        'MotionBlur': [('shutterAngle', 'float', 270.0), ('sampleCount', 'int', 10)],
        'LensDistortion': [('intensity', 'float', 0.0), ('intensityX', 'float', 1.0), ('intensityY', 'float', 1.0),
                           ('centerX', 'float', 0.0), ('centerY', 'float', 0.0), ('scale', 'float', 1.0)],
    }


DEFAULTS = _defaults()
# Enum value notes (installed enums): AmbientOcclusionMode SAO=0 MSVO=1; AmbientOcclusionQuality
# Lowest..Ultra=0..4; KernelSize Small..VeryLarge=0..3; GradingMode LDR=0 HDR=1 External=2;
# Tonemapper None=0 Neutral=1 ACES=2 Custom=3; VignetteMode Classic=0 Masked=1;
# ScreenSpaceReflectionPreset Lower=0 Low=1 Medium=2 High=3 Higher=4 Ultra=5 Overkill=6 Custom=7;
# SSR resolution Downsampled=0; EyeAdaptation Progressive=0.
EFFECT_TYPES = tuple(DEFAULTS)


def _make_value(kind, default, name):
    if kind == 'float':
        return F(default)
    if kind in ('color', 'vector2', 'vector4'):
        return tuple(F(x) for x in default)
    if kind == 'spline':
        what, zero, loop = default
        return SplineValue.new(_identity_curve() if what == 'identity' else [], zero, loop, name=name)
    if kind == 'texture':
        return None
    return default


class EffectSettings:
    """One ``PostProcessEffectSettings`` instance (``active`` + parameters)."""

    def __init__(self, cls, active=True):
        self.cls = cls
        self.active = active
        self.params = {'enabled': Param('bool', True, False)}
        for name, kind, default in DEFAULTS[cls]:
            ds = default[1] if kind == 'texture' else None
            self.params[name] = Param(kind, False, _make_value(kind, default, name), ds)

    def __getitem__(self, name):
        return self.params[name].value

    def override(self, name, value):
        """``ParameterOverride<T>.Override(x)``."""
        p = self.params[name]
        p.override_state = True
        p.value = value

    def set(self, name, value, override_state=True):
        p = self.params[name]
        p.override_state = override_state
        p.value = value


def base_settings():
    """``PostProcessManager.ReloadBaseTypes``: defaults with every override on."""
    out = {}
    for cls in EFFECT_TYPES:
        s = EffectSettings(cls)
        for p in s.params.values():
            p.override_state = True
        for p in s.params.values():
            if p.kind == 'spline':
                p.value.cache(-2147483648)   # SplineParameter.OnEnable -> Cache(int.MinValue)
        out[cls] = s
    return out


# ----------------------------------------------------------------- volumes
@dataclass
class Volume:
    name: str
    layer: int
    priority: float
    weight: float
    is_global: bool
    enabled: bool                       # component enabled and GameObject active in hierarchy
    settings: list = field(default_factory=list)   # EffectSettings of profileRef, in profile order
    blend_distance: float = 0.0
    provenance: dict = field(default_factory=dict)


def sort_by_priority(volumes):
    """``PostProcessManager.SortByPriority``: stable insertion sort on priority."""
    v = list(volumes)
    for i in range(1, len(v)):
        cur = v[i]
        j = i - 1
        while j >= 0 and F(v[j].priority) > F(cur.priority):
            v[j + 1] = v[j]
            j -= 1
        v[j + 1] = cur
    return v


class VolumeBlender:
    """``PostProcessManager.UpdateSettings`` for one PostProcessLayer.

    ``base`` and ``layer`` persist across renders exactly like the singleton base
    settings and the layer's bundle settings (Spline objects are shared by
    reference after ``SetValue``)."""

    def __init__(self):
        self.base = base_settings()
        self.layer = {cls: EffectSettings(cls) for cls in EFFECT_TYPES}
        for s in self.layer.values():
            for p in s.params.values():
                if p.kind == 'spline':
                    p.value.cache(-2147483648)

    def update(self, volume_layer_mask, volumes, frame, volume_trigger_position=None):
        for cls, base in self.base.items():
            dst = self.layer[cls]
            for name, p in base.params.items():
                dst.params[name].set_value(p, frame)
        applied = []
        for vol in sort_by_priority([v for v in volumes if volume_layer_mask & (1 << v.layer)]):
            if not vol.enabled or F(vol.weight) <= F(0):
                continue
            if not vol.is_global:
                if volume_trigger_position is None:
                    continue
                raise NotImplementedError('local volumes with colliders (gap local_volume_blend)')
            w = mathf_clamp01(vol.weight)
            self.override_settings(vol.settings, w, frame)
            applied.append((vol.name, float(w)))
        return applied

    def override_settings(self, settings, t, frame):
        for s in settings:
            if not s.active:
                continue
            dst = self.layer[s.cls]
            for name, p in s.params.items():
                if p.override_state:
                    q = dst.params[name]
                    q.interp(q, p, t, frame)


# ------------------------------------------------------- scene state (static)
def _f(x):
    return F(x['f']) if isinstance(x, dict) and 'f' in x else x


def effect_from_scene(row, textures=None):
    """EffectSettings from ``extract_ppv2_scene_state`` rows (exact serialized values)."""
    s = EffectSettings(row['class'], active=row['active'])
    for name, p in row['parameters'].items():
        q = s.params[name]
        q.override_state = p['overrideState']
        v = p['value']
        if q.kind == 'float':
            q.value = _f(v)
        elif q.kind in ('color', 'vector2', 'vector4'):
            q.value = tuple(_f(x) for x in v)
        elif q.kind == 'spline':
            keys = [{k: (_f(x[k]) if k != 'weightedMode' else x[k]) for k in x} for x in v['curve']['m_Curve']]
            for wrap in ('m_PreInfinity', 'm_PostInfinity'):
                if v['curve'][wrap] != 2:
                    raise NotImplementedError('AnimationCurve wrap mode %r' % v['curve'][wrap])
            q.value = SplineValue(keys, _f(v['m_ZeroValue']), v['m_Loop'], _f(v['m_Range']),
                                  [_f(c) for c in v['cachedData']], name=name)
            q.value.serialized_cache = [_f(c) for c in v['cachedData']]
            q.value.cache(-2147483648)   # SplineParameter.OnEnable at load
        elif q.kind == 'texture':
            q.default_state = p['defaultState']
            ptr = v
            if ptr['path_id'] == 0:
                q.value = None
            else:
                key = '%s:%d:%d' % (row['file'], ptr['file_id'], ptr['path_id'])
                info = (textures or {}).get(key, {})
                q.value = TextureValue(key, info.get('width'), info.get('height'))
        elif q.kind == 'bool':
            q.value = bool(v)
        else:
            q.value = int(v)
    return s


@dataclass
class GraphicData:
    """``Config.GraphicSystem`` flags used by ``ConfigEffectorVolume`` (system.xml)."""
    Bloom: bool = True
    SSAO: bool = True
    SSR: bool = True
    Vignette: bool = True


def scene_volumes(scene, graphic: GraphicData, draw3d=True, textures=None):
    """CharaCustom volumes in registration order with ConfigEffectorVolume applied.

    ``draw3d``: ``CharaCustom.CustomControl.draw3D`` (obj3DTop = Map3D active) --
    PostProcessVolume3D sits under ``CharaCustom/CustomControl/Map3D``.  The
    registration order of the two priority-0 volumes is runtime OnEnable order;
    the default [2D, 3D] is the order the retry capture's DoF/LUT constants select
    (docs gap ``volume_registration_order``)."""
    effects = scene['effects']
    by_vol = {}
    for c in scene['other_scripts']:
        if c['class'] == 'ConfigEffectorVolume':
            by_vol[c['postProcessVolume']['path_id']] = c
    out = []
    for v in scene['volumes']:
        prof = scene['profiles']['%s:%d' % ('sharedassets3.assets', v['sharedProfile']['path_id'])]
        settings = []
        for ptr in prof['settings']:
            row = effects['%s:%d' % ('sharedassets3.assets', ptr['path_id'])]
            settings.append(effect_from_scene(row, textures))
        cev = by_vol.get(v['path_id'])
        if cev is not None and cev['m_Enabled']:
            # ConfigEffectorVolume.Refresh (not an H scene): active = GraphicData flag
            for s in settings:
                if s.cls == 'Bloom' and cev['isBloom']:
                    s.active = graphic.Bloom
                elif s.cls == 'AmbientOcclusion' and cev['isAo']:
                    s.active = graphic.SSAO
                elif s.cls == 'ScreenSpaceReflections' and cev['isSsr']:
                    s.active = graphic.SSR
                elif s.cls == 'Vignette' and cev['isVignette']:
                    s.active = graphic.Vignette
        go = v['game_object']
        name = go['m_Name']
        enabled = bool(v['m_Enabled'] and go['m_IsActive'])
        if name == 'PostProcessVolume3D':
            enabled = enabled and draw3d
        out.append(Volume(name=name, layer=go['m_Layer'], priority=_f(v['priority']), weight=_f(v['weight']),
                          is_global=v['isGlobal'], enabled=enabled, settings=settings,
                          blend_distance=_f(v['blendDistance']),
                          provenance={'scene_path_id': v['path_id'], 'profile': prof['m_Name']}))
    order = {'PostProcessVolume2D': 0, 'PostProcessVolume3D': 1}
    out.sort(key=lambda vol: order.get(vol.name, 99))
    return out


# --------------------------------------------------------------------- DHH
DHH_FLOATS, DHH_BOOLS, DHH_VECTORS = 100, 30, 4


def _dhh_set_default(fb, bb):
    """``DHH_Main.SetDefault`` (backup defaults)."""
    for i, v in ((46, 80), (49, 500), (50, 1), (51, 1), (53, 1), (54, 0.3), (55, 0.1), (56, 1), (57, 1), (58, 5),
                 (59, 1.2), (60, 0.1), (61, 0.1), (62, 0.1), (63, 0.1), (64, 2), (65, 2), (66, 2)):
        fb[i] = F(v)
    for i, v in ((8, False), (13, True), (14, False), (15, True), (16, True), (17, True), (18, False), (19, True),
                 (20, False), (21, True), (22, True), (23, False), (24, False), (25, False), (26, True), (27, True),
                 (28, True), (29, True)):
        bb[i] = v


def _tag_slice(tag, data):
    """``GetDataWithTag``: first IndexOf("?tag="+tag), value up to the next "&&"."""
    i = data.find('?tag=' + tag)
    if i == -1:
        return None
    i += len(tag) + 7
    j = data.find('&&', i)
    if j == -1:
        return None
    return data[i:j]


def _parse_float(s):
    """``float.TryParse`` (failure -> 0)."""
    try:
        s2 = s.strip()
        if not s2 or s2.lower() in ('nan', 'inf', '-inf', 'infinity'):
            raise ValueError
        return F(float(s2))
    except ValueError:
        return F(0)


def parse_dhh_graphic_setting(text):
    """``DHH_Main.GetGraphicSetting`` -> ``SetGraphicSettingString`` on GraphicSetting.dhs."""
    fb = [F(0)] * DHH_FLOATS
    bb = [False] * DHH_BOOLS
    vb = [(F(0), F(0), F(0))] * DHH_VECTORS
    _dhh_set_default(fb, bb)
    floats, bools, vecs = list(fb), list(bb), list(vb)
    for i in range(DHH_FLOATS):
        s = _tag_slice('GSfloat_%d' % i, text)
        floats[i] = fb[i] if s is None else _parse_float(s)
    for i in range(DHH_BOOLS):
        s = _tag_slice('GSbool_%d' % i, text)
        if s is None:
            bools[i] = bb[i]
        else:
            low = s.strip().lower()
            bools[i] = True if low == 'true' else False   # bool.TryParse; failure -> false
    for i in range(DHH_VECTORS):
        s = _tag_slice('GSvector3_%d' % i, text)
        if s is None:
            vecs[i] = vb[i]
            continue
        parts = s[1:-1].split(',')
        try:
            if len(parts) > 2:
                vals = [F(float(p.strip())) for p in parts[:3]]
                vecs[i] = tuple(vals)
            else:
                vecs[i] = vb[i]
        except ValueError:
            vecs[i] = vb[i]
    sky = _tag_slice('SkyboxID', text)
    return {'floats': floats, 'bools': bools, 'vectors': vecs,
            'skybox_id': math.floor(float(_parse_float(sky))) if sky is not None else 0,
            'cus_hdri': _tag_slice('CusHDRI', text) or ''}


@dataclass
class DHHRuntime:
    """Live DHH state not stored in GraphicSetting.dhs."""
    graphic_enable: bool = True
    focus_distance: float | None = None   # value written to depthOfFieldLayer.focusDistance
    ldr_lut_texture: TextureValue | None = None   # MyLutTexture (LutTextures/Lut_<GSfloat_43>.png)


def dhh_focus_distance(cfg):
    """``FocusDis`` when autofocus (GraphicBool[9]) is off: GraphicFloat[41] * 14.6."""
    if cfg['bools'][9]:
        raise ValueError('DHH autofocus uses Time.deltaTime/MoveTowards: supply DHHRuntime.focus_distance')
    return r4(dv(cfg['floats'][41]) * dv(14.6))


def dhh_volume(cfg, rt: DHHRuntime):
    """``DHH_Main.Init`` volume + the PostProcessing part of ``ApplyGraphicSetting``.

    Returns (Volume, layer overrides).  Effect instances are created by Init's first
    ApplyGraphicSetting (ScriptableObject.CreateInstance -> defaults, only ``enabled``
    overridden) and filled by every later call."""
    fl, bo, ve = cfg['floats'], cfg['bools'], cfg['vectors']
    ge = rt.graphic_enable
    ao = EffectSettings('AmbientOcclusion')
    ao.override('quality', 4)                               # Ultra
    ao.set('mode', 1 if bo[11] else 0)
    ao.set('enabled', bo[2])
    ao.set('intensity', fl[8])
    ao.set('radius', fl[9])
    ao.set('color', (r4(dv(ve[1][0]) * dv(0.005)), r4(dv(ve[1][1]) * dv(0.005)), r4(dv(ve[1][2]) * dv(0.005)), F(1)))
    ao.override('thicknessModifier', F(1))
    ao.override('directLightingStrength', F(1))
    bloom = EffectSettings('Bloom')
    bloom.set('enabled', bo[1])
    bloom.set('intensity', fl[3])
    bloom.set('threshold', fl[4])
    bloom.set('softKnee', fl[5])
    bloom.set('diffusion', fl[6])
    bloom.set('anamorphicRatio', fl[7])
    bloom.set('color', (r4(dv(ve[0][0]) * dv(0.005)), r4(dv(ve[0][1]) * dv(0.005)), r4(dv(ve[0][2]) * dv(0.005)), F(1)))
    dof = EffectSettings('DepthOfField')
    dof.set('enabled', bo[0])
    dof.set('aperture', fl[0])
    dof.set('focalLength', fl[1])
    k = fl[2]
    dof.set('kernelSize', 0 if k == F(0) else (1 if k == F(1) else (2 if k == F(2) else 3)))
    dof.set('focusDistance', F(rt.focus_distance) if rt.focus_distance is not None else dhh_focus_distance(cfg))
    cg = EffectSettings('ColorGrading')
    cg.set('enabled', bo[3])
    cg.set('gradingMode', 1 if fl[42] == F(0) else 0)
    cg.set('ldrLut', rt.ldr_lut_texture, override_state=fl[43] > F(0))
    cg.set('ldrLutContribution', fl[44])
    cg.set('tonemapper', 1 if fl[10] == F(0) else 2)
    for name, idx in (('temperature', 11), ('tint', 12), ('postExposure', 13), ('hueShift', 14),
                      ('saturation', 15), ('contrast', 16)):
        cg.set(name, fl[idx])
    cg.set('lift', (F(1), F(1), F(1), fl[17]))
    cg.set('gamma', (F(1), F(1), F(1), fl[18]))
    cg.set('gain', (F(1), F(1), F(1), fl[19]))
    for name in ('hueVsHueCurve', 'satVsSatCurve', 'lumVsSatCurve'):
        cg.override(name, SplineValue.new([], 0.5, True, (0.0, 1.0), name='DHH ' + name))
    ca = EffectSettings('ChromaticAberration')
    ca.set('enabled', bo[4])
    ca.set('intensity', fl[20])
    vig = EffectSettings('Vignette')
    vig.set('enabled', bo[5])
    vig.set('intensity', fl[21])
    vig.set('smoothness', fl[22])
    mb = EffectSettings('MotionBlur')
    mb.set('enabled', bo[6])
    mb.set('shutterAngle', fl[23])
    ssr = EffectSettings('ScreenSpaceReflections')
    ssr.set('enabled', bo[7])
    ssr.set('vignette', fl[24])
    ssr.set('maximumMarchDistance', fl[25])
    ssr.set('preset', 4)                                    # Higher
    # Profile order = Init's TryGetSettings failures followed by AddSettings in
    # ApplyGraphicSetting order: AO, Bloom, DoF, ColorGrading, CA, Vignette, MotionBlur, SSR.
    vol = Volume(name='DHH_Post', layer=8, priority=F(1000) if ge else F(0), weight=fl[45] if ge else F(0),
                 is_global=True, enabled=ge, settings=[ao, bloom, dof, cg, ca, vig, mb, ssr], blend_distance=F(100),
                 provenance={'source': 'DHH_Main.Init/ApplyGraphicSetting'})
    mode = fl[26]
    layer = {'antialiasingMode': 1 if mode == F(0) else (2 if mode == F(1) else 3),
             'fastApproximateAntialiasing.fastMode': False,
             'subpixelMorphologicalAntialiasing.quality': 2}
    if mode == F(2):
        layer.update({'temporalAntialiasing.jitterSpread': fl[27], 'temporalAntialiasing.stationaryBlending': fl[28],
                      'temporalAntialiasing.motionBlending': fl[29], 'temporalAntialiasing.sharpness': fl[30]})
    return vol, layer


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))
