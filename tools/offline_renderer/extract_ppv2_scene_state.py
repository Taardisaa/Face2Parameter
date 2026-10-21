"""Static PPv2 configuration of the installed HS2 CharaCustom scene and resources.

Read-only.  The player build carries no type trees for MonoBehaviours, so the
script objects are decoded with field layouts taken from the decompiled
installed ``Unity.Postprocessing.Runtime.dll`` (PPv2 2.1.7) in declaration
order, using Unity's serialization rules (each bool/uint8 field and each
string/array is followed by 4-byte alignment).  Every object must be consumed
to its exact serialized length, otherwise the decode is rejected.

Outputs one JSON document with:
* the CharaCustom PostProcessLayer components and their camera GameObjects,
* every PostProcessVolume (GameObject layer/active/tag, profile, weight,...),
* every referenced PostProcessProfile and its effect settings (all parameters
  with overrideState/value; splines with keys and serialized cachedData),
* the PostProcessResources asset (blue-noise, SMAA LUT, shader and compute
  shader references resolved to names).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import UnityPy

DATA = Path(r'E:\HoneySelect2_ArcticFox\HoneySelect2_Data')
SCENE = 'level3'  # BuildSettings scene 3 = Assets/Illusion/Scene/CharaCustom.unity


class Reader:
    def __init__(self, data):
        self.d = data
        self.o = 0

    def u8(self):
        v = self.d[self.o]
        self.o += 1
        return v

    def align(self):
        self.o = (self.o + 3) & ~3

    def i32(self):
        v = struct.unpack_from('<i', self.d, self.o)[0]
        self.o += 4
        return v

    def u32(self):
        v = struct.unpack_from('<I', self.d, self.o)[0]
        self.o += 4
        return v

    def f32(self):
        v = struct.unpack_from('<f', self.d, self.o)[0]
        raw = self.d[self.o:self.o + 4].hex()
        self.o += 4
        return {'f': v, 'hex': raw}

    def boolean(self):
        v = self.u8()
        self.align()
        if v not in (0, 1):
            raise ValueError('non-boolean byte %d at %d' % (v, self.o))
        return bool(v)

    def pptr(self):
        fid = self.i32()
        pid = struct.unpack_from('<q', self.d, self.o)[0]
        self.o += 8
        return {'file_id': fid, 'path_id': pid}

    def string(self):
        n = self.i32()
        s = self.d[self.o:self.o + n].decode('utf-8')
        self.o += n
        self.align()
        return s

    def array(self, fn):
        n = self.i32()
        out = [fn() for _ in range(n)]
        self.align()
        return out


def header(r):
    return {'m_GameObject': r.pptr(), 'm_Enabled': r.boolean(), 'm_Script': r.pptr(), 'm_Name': r.string()}


# ---- parameter value readers (PPv2 2.1.7 ParameterOverride<T> = overrideState, value[, extra])
def color(r):
    return [r.f32() for _ in range(4)]


def keyframe(r):
    return {'time': r.f32(), 'value': r.f32(), 'inSlope': r.f32(), 'outSlope': r.f32(),
            'weightedMode': r.i32(), 'inWeight': r.f32(), 'outWeight': r.f32()}


def spline(r):
    curve = {'m_Curve': r.array(lambda: keyframe(r)), 'm_PreInfinity': r.i32(), 'm_PostInfinity': r.i32(),
             'm_RotationOrder': r.i32()}
    return {'curve': curve, 'm_Loop': r.boolean(), 'm_ZeroValue': r.f32(), 'm_Range': r.f32(),
            'cachedData': r.array(r.f32)}


VALUE = {
    'Float': lambda r: r.f32(), 'Int': lambda r: r.i32(), 'Bool': lambda r: r.boolean(), 'Enum': lambda r: r.i32(),
    'Color': color, 'Vector2': lambda r: [r.f32(), r.f32()], 'Vector4': lambda r: [r.f32() for _ in range(4)],
    'Spline': spline,
}


def parameter(r, kind):
    state = r.boolean()
    if kind == 'Texture':
        return {'overrideState': state, 'value': r.pptr(), 'defaultState': r.i32()}
    return {'overrideState': state, 'value': VALUE[kind](r)}


# Declaration order from the decompiled installed classes (base PostProcessEffectSettings: active, enabled).
EFFECT_FIELDS = {
    'AmbientOcclusion': [('mode', 'Enum'), ('intensity', 'Float'), ('color', 'Color'), ('ambientOnly', 'Bool'),
                         ('noiseFilterTolerance', 'Float'), ('blurTolerance', 'Float'), ('upsampleTolerance', 'Float'),
                         ('thicknessModifier', 'Float'), ('directLightingStrength', 'Float'), ('radius', 'Float'),
                         ('quality', 'Enum')],
    'DepthOfField': [('focusDistance', 'Float'), ('aperture', 'Float'), ('focalLength', 'Float'), ('kernelSize', 'Enum')],
    'ColorGrading': [('gradingMode', 'Enum'), ('externalLut', 'Texture'), ('tonemapper', 'Enum'),
                     ('toneCurveToeStrength', 'Float'), ('toneCurveToeLength', 'Float'),
                     ('toneCurveShoulderStrength', 'Float'), ('toneCurveShoulderLength', 'Float'),
                     ('toneCurveShoulderAngle', 'Float'), ('toneCurveGamma', 'Float'), ('ldrLut', 'Texture'),
                     ('ldrLutContribution', 'Float'), ('temperature', 'Float'), ('tint', 'Float'),
                     ('colorFilter', 'Color'), ('hueShift', 'Float'), ('saturation', 'Float'), ('brightness', 'Float'),
                     ('postExposure', 'Float'), ('contrast', 'Float'),
                     ('mixerRedOutRedIn', 'Float'), ('mixerRedOutGreenIn', 'Float'), ('mixerRedOutBlueIn', 'Float'),
                     ('mixerGreenOutRedIn', 'Float'), ('mixerGreenOutGreenIn', 'Float'), ('mixerGreenOutBlueIn', 'Float'),
                     ('mixerBlueOutRedIn', 'Float'), ('mixerBlueOutGreenIn', 'Float'), ('mixerBlueOutBlueIn', 'Float'),
                     ('lift', 'Vector4'), ('gamma', 'Vector4'), ('gain', 'Vector4'),
                     ('masterCurve', 'Spline'), ('redCurve', 'Spline'), ('greenCurve', 'Spline'), ('blueCurve', 'Spline'),
                     ('hueVsHueCurve', 'Spline'), ('hueVsSatCurve', 'Spline'), ('satVsSatCurve', 'Spline'),
                     ('lumVsSatCurve', 'Spline')],
    'Bloom': [('intensity', 'Float'), ('threshold', 'Float'), ('softKnee', 'Float'), ('clamp', 'Float'),
              ('diffusion', 'Float'), ('anamorphicRatio', 'Float'), ('color', 'Color'), ('fastMode', 'Bool'),
              ('dirtTexture', 'Texture'), ('dirtIntensity', 'Float')],
    'Vignette': [('mode', 'Enum'), ('color', 'Color'), ('center', 'Vector2'), ('intensity', 'Float'),
                 ('smoothness', 'Float'), ('roundness', 'Float'), ('rounded', 'Bool'), ('mask', 'Texture'),
                 ('opacity', 'Float')],
    'ScreenSpaceReflections': [('preset', 'Enum'), ('maximumIterationCount', 'Int'), ('resolution', 'Enum'),
                               ('thickness', 'Float'), ('maximumMarchDistance', 'Float'), ('distanceFade', 'Float'),
                               ('vignette', 'Float')],
    'AutoExposure': [('filtering', 'Vector2'), ('minLuminance', 'Float'), ('maxLuminance', 'Float'),
                     ('keyValue', 'Float'), ('eyeAdaptation', 'Enum'), ('speedUp', 'Float'), ('speedDown', 'Float')],
    'Grain': [('colored', 'Bool'), ('intensity', 'Float'), ('size', 'Float'), ('lumContrib', 'Float')],
    'ChromaticAberration': [('spectralLut', 'Texture'), ('intensity', 'Float'), ('fastMode', 'Bool')],
    'MotionBlur': [('shutterAngle', 'Float'), ('sampleCount', 'Int')],
    'LensDistortion': [('intensity', 'Float'), ('intensityX', 'Float'), ('intensityY', 'Float'),
                       ('centerX', 'Float'), ('centerY', 'Float'), ('scale', 'Float')],
}


def effect(r, cls):
    row = header(r)
    row['active'] = r.boolean()
    row['parameters'] = {'enabled': parameter(r, 'Bool')}
    for name, kind in EFFECT_FIELDS[cls]:
        row['parameters'][name] = parameter(r, kind)
    return row


def layer(r):
    row = header(r)
    row.update(volumeTrigger=r.pptr(), volumeLayer=r.u32(), stopNaNPropagation=r.boolean(),
               finalBlitToCameraTarget=r.boolean(), antialiasingMode=r.i32(),
               temporalAntialiasing={'jitterSpread': r.f32(), 'sharpness': r.f32(), 'stationaryBlending': r.f32(),
                                     'motionBlending': r.f32()},
               subpixelMorphologicalAntialiasing={'quality': r.i32()},
               fastApproximateAntialiasing={'fastMode': r.boolean(), 'keepAlpha': r.boolean()},
               fog={'enabled': r.boolean(), 'excludeSkybox': r.boolean()},
               debugLayer={'lightMeter': {'width': r.i32(), 'height': r.i32(), 'showCurves': r.boolean()},
                           'histogram': {'width': r.i32(), 'height': r.i32(), 'channel': r.i32()},
                           'waveform': {'exposure': r.f32(), 'height': r.i32()},
                           'vectorscope': {'size': r.i32(), 'exposure': r.f32()},
                           'overlaySettings': {'linearDepth': r.boolean(), 'motionColorIntensity': r.f32(),
                                               'motionGridSize': r.i32(), 'colorBlindnessType': r.i32(),
                                               'colorBlindnessStrength': r.f32()}},
               m_Resources=r.pptr(), m_ShowToolkit=r.boolean(), m_ShowCustomSorter=r.boolean(),
               breakBeforeColorGrading=r.boolean(),
               m_BeforeTransparentBundles=r.array(r.string), m_BeforeStackBundles=r.array(r.string),
               m_AfterStackBundles=r.array(r.string))
    return row


def volume(r):
    row = header(r)
    row.update(sharedProfile=r.pptr(), isGlobal=r.boolean(), blendDistance=r.f32(), weight=r.f32(), priority=r.f32())
    return row


def profile(r):
    row = header(r)
    row['settings'] = r.array(r.pptr)
    return row


def resources(r):
    row = header(r)
    row['blueNoise64'] = r.array(r.pptr)
    row['blueNoise256'] = r.array(r.pptr)
    row['smaaLuts'] = {'area': r.pptr(), 'search': r.pptr()}
    row['shaders'] = {n: r.pptr() for n in (
        'bloom', 'copy', 'copyStd', 'copyStdFromTexArray', 'copyStdFromDoubleWide', 'discardAlpha', 'depthOfField',
        'finalPass', 'grainBaker', 'motionBlur', 'temporalAntialiasing', 'subpixelMorphologicalAntialiasing',
        'texture2dLerp', 'uber', 'lut2DBaker', 'lightMeter', 'gammaHistogram', 'waveform', 'vectorscope',
        'debugOverlays', 'deferredFog', 'scalableAO', 'multiScaleAO', 'screenSpaceReflections')}
    row['computeShaders'] = {n: r.pptr() for n in (
        'autoExposure', 'exposureHistogram', 'lut3DBaker', 'texture3dLerp', 'gammaHistogram', 'waveform',
        'vectorscope', 'multiScaleAODownsample1', 'multiScaleAODownsample2', 'multiScaleAORender',
        'multiScaleAOUpsample', 'gaussianDownsample')}
    return row


def decode(obj, fn, *args):
    data = obj.get_raw_data()
    r = Reader(data)
    row = fn(r, *args)
    if r.o != len(data):
        raise ValueError('decode of path %d consumed %d of %d bytes' % (obj.path_id, r.o, len(data)))
    row['path_id'] = obj.path_id
    row['raw_sha256'] = hashlib.sha256(data).hexdigest()
    return row


class Files:
    def __init__(self, data_dir):
        self.dir = Path(data_dir)
        self.envs = {}

    def file(self, name):
        if name not in self.envs:
            env = UnityPy.load(str(self.dir / name))
            sf = list(env.files.values())[0]
            self.envs[name] = (env, sf, {o.path_id: o for o in env.objects})
        return self.envs[name]

    def resolve(self, owner, ptr):
        """(file name, object) addressed by a PPtr stored in ``owner``."""
        if ptr['path_id'] == 0:
            return None, None
        env, sf, objs = self.file(owner)
        name = owner if ptr['file_id'] == 0 else Path(sf.externals[ptr['file_id'] - 1].path).name
        if name in ('unity default resources', 'unity_builtin_extra'):
            return name, None
        return name, self.file(name)[2].get(ptr['path_id'])

    def script_class(self, owner, obj):
        hdr = Reader(obj.get_raw_data())
        header(hdr)
        hdr.o = 0
        h = header(hdr)
        name, sobj = self.resolve(owner, h['m_Script'])
        t = sobj.read_typetree()
        return t['m_ClassName'], t.get('m_Namespace')


def game_object(files, owner, ptr):
    name, obj = files.resolve(owner, ptr)
    t = obj.read_typetree()
    comps = []
    for c in t['m_Component']:
        cn, co = files.resolve(name, {'file_id': c['component']['m_FileID'], 'path_id': c['component']['m_PathID']})
        comps.append({'path_id': c['component']['m_PathID'], 'type': co.type.name if co else None})
    return {'file': name, 'path_id': obj.path_id, 'm_Name': t['m_Name'], 'm_Layer': t['m_Layer'],
            'm_TagString': t.get('m_TagString'), 'm_Tag': t.get('m_Tag'), 'm_IsActive': t['m_IsActive'],
            'components': comps}


def run(data_dir, out):
    out = Path(out)
    if out.exists():
        raise FileExistsError('Fresh output file required')
    files = Files(data_dir)
    env, sf, objs = files.file(SCENE)
    result = {'schema': 'hs2_ppv2_scene_state/1', 'scene': {'file': SCENE, 'build_scene': 'Assets/Illusion/Scene/CharaCustom.unity',
              'sha256': hashlib.sha256((Path(data_dir) / SCENE).read_bytes()).hexdigest()},
              'layers': [], 'volumes': [], 'profiles': {}, 'effects': {}, 'resources': None, 'other_scripts': []}
    for obj in env.objects:
        if obj.type.name != 'MonoBehaviour':
            continue
        try:
            cls, ns = files.script_class(SCENE, obj)
        except Exception:
            continue
        if cls == 'PostProcessLayer':
            row = decode(obj, layer)
            row['game_object'] = game_object(files, SCENE, row['m_GameObject'])
            result['layers'].append(row)
        elif cls == 'PostProcessVolume':
            row = decode(obj, volume)
            row['game_object'] = game_object(files, SCENE, row['m_GameObject'])
            result['volumes'].append(row)
        elif cls in ('ConfigEffectorVolume',):
            data = obj.get_raw_data()
            r = Reader(data)
            row = header(r)
            row.update(postProcessVolume=r.pptr(), isBloom=r.boolean(), isAo=r.boolean(), isSsr=r.boolean(),
                       isVignette=r.boolean(), rp=r.pptr(), isHSceneSettingUse=r.boolean(), particle=r.pptr())
            if r.o != len(data):
                raise ValueError('ConfigEffectorVolume decode length')
            row['path_id'] = obj.path_id
            row['class'] = cls
            row['game_object'] = game_object(files, SCENE, row['m_GameObject'])
            result['other_scripts'].append(row)
    for v in result['volumes']:
        name, pobj = files.resolve(SCENE, v['sharedProfile'])
        key = '%s:%d' % (name, pobj.path_id)
        if key in result['profiles']:
            continue
        prow = decode(pobj, profile)
        prow['file'] = name
        result['profiles'][key] = prow
        for s in prow['settings']:
            ename, eobj = files.resolve(name, s)
            ecls, _ = files.script_class(ename, eobj)
            erow = decode(eobj, effect, ecls)
            erow['class'] = ecls
            erow['file'] = ename
            result['effects']['%s:%d' % (ename, eobj.path_id)] = erow
            for pname, prm in erow['parameters'].items():
                v = prm.get('value')
                if isinstance(v, dict) and 'path_id' in v and 'file_id' in v and v['path_id']:
                    tname, tobj = files.resolve(ename, v)
                    t = tobj.read_typetree() if tobj is not None else {}
                    result.setdefault('textures', {})['%s:%d:%d' % (ename, v['file_id'], v['path_id'])] = {
                        'file': tname, 'path_id': v['path_id'], 'type': tobj.type.name if tobj else None,
                        'name': t.get('m_Name'), 'width': t.get('m_Width'), 'height': t.get('m_Height'),
                        'format': t.get('m_TextureFormat')}
    lay = result['layers'][0]
    rname, robj = files.resolve(SCENE, lay['m_Resources'])
    res = decode(robj, resources)
    res['file'] = rname

    def named(ptr):
        n, o = files.resolve(rname, ptr)
        if o is None:
            return {'ptr': ptr, 'file': n, 'name': None}
        t = o.read_typetree()
        nm = t.get('m_Name') or (t.get('m_ParsedForm') or {}).get('m_Name')
        return {'ptr': ptr, 'file': n, 'path_id': o.path_id, 'type': o.type.name, 'name': nm,
                'width': t.get('m_Width'), 'height': t.get('m_Height'), 'format': t.get('m_TextureFormat')}
    res['blueNoise64'] = [named(p) for p in res['blueNoise64']]
    res['blueNoise256'] = [named(p) for p in res['blueNoise256']]
    res['smaaLuts'] = {k: named(p) for k, p in res['smaaLuts'].items()}
    res['shaders'] = {k: named(p) for k, p in res['shaders'].items()}
    res['computeShaders'] = {k: named(p) for k, p in res['computeShaders'].items()}
    result['resources'] = res
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1), encoding='utf-8')
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', type=Path, default=DATA)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)
    r = run(args.data, args.out)
    print(json.dumps({'layers': len(r['layers']), 'volumes': len(r['volumes']), 'profiles': len(r['profiles']),
                      'effects': len(r['effects'])}))


if __name__ == '__main__':
    main()
