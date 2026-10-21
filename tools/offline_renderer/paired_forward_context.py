"""Offline CapturedHeadForward inputs for a FramePairProbe frame (no game).

Context fields come only from the same session's recorded data:
  * ABMX ordered modifiers + private caches: bridge geometry export
    ``abmx_runtime.bones[].runtime_baseline`` (controller.Modifiers order);
  * native59 / body values: the loaded-state snapshot;
  * transforms: the probe frame's float32 local TRS (instance ids from the
    same process), ancestors from the bridge export;
  * effect state: an explicitly supplied recorded context whose effects are
    checked to be identity or disjoint from every head skin chain.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

GAME = Path('E:/HoneySelect2_ArcticFox')
SOURCES = {'game_sha256': 'HoneySelect2_Data/Managed/Assembly-CSharp.dll',
           'slider_unlocker_sha256': 'BepInEx/Plugins/HS2_BepisPlugins/HS2_SliderUnlocker.dll',
           'abmx_sha256': 'BepInEx/Plugins/HS2ABMX.dll',
           'HS2_AdditionalAccessoryControls_sha256': 'BepInEx/Plugins/HS2_Plugins/OrangeSpork/HS2_AdditionalAccessoryControls.dll',
           'PushUpAI_sha256': 'BepInEx/Plugins/HS2_Plugins/Mikke/PushUpHS2_2.1.4.dll',
           'HS2_AdvIKPlugin_sha256': 'BepInEx/Plugins/HS2_Plugins/OrangeSpork/HS2_AdvIKPlugin.dll'}
P = 'KKABMX.Core.BoneModifier.'
D = 'KKABMX.Core.BoneModifierData.'


def build(geometry_path, loaded_state_path, effect_context_path, probe_window, frame_count):
    geo = json.loads(Path(geometry_path).read_text(encoding='utf-8'))
    state = json.loads(Path(loaded_state_path).read_text(encoding='utf-8'))
    effect = json.loads(Path(effect_context_path).read_text(encoding='utf-8'))['identity']
    from src.render.paired_skin import ProbeWindow
    win = probe_window if isinstance(probe_window, ProbeWindow) else ProbeWindow(probe_window)
    frame = win.by_frame[frame_count]
    ids = [t['id'] for t in win.static['transforms']]
    T = frame['T']
    geo = copy.deepcopy(geo)
    by_id = {t['id']: t for t in geo['transforms']}
    replaced = 0
    for i, tid in enumerate(ids):
        if tid in by_id:
            row = by_id[tid]
            row['local_position'] = T[i, 0:3].astype(float).tolist()
            row['local_rotation_xyzw'] = T[i, 3:7].astype(float).tolist()
            row['local_scale'] = T[i, 7:10].astype(float).tolist()
            row['local_to_world'] = T[i, 17:33].astype(float).tolist()
            row['world_to_local'] = np.linalg.inv(T[i, 17:33].reshape(4, 4).astype(float)).ravel().tolist()
            replaced += 1
    mods = []
    for b in geo['abmx_runtime']['bones']:
        rb = b['runtime_baseline']
        if rb['missing_fields']:
            raise ValueError('Incomplete ABMX private cache: ' + b['name'])
        row = {'bone_instance_id': rb['bone_transform_id']}
        for k, v in rb['fields'].items():
            row[P + k] = v
        row[P + '<BoneName>k__BackingField'] = b['name']
        row[P + '<CoordinateModifiers>k__BackingField'] = [
            {D + 'ScaleModifier': c['scale'], D + 'LengthModifier': c['length'],
             D + 'PositionModifier': c['position'], D + 'RotationModifier': c['rotation']} for c in b['coordinate_modifiers']]
        mods.append(row)
    snap = state['snapshot']
    native = [it['value'] for it in sorted(snap['face_shapes']['items'], key=lambda x: x['index'])]
    body = [it['value'] for it in sorted(snap['body_shapes']['items'], key=lambda x: x['index'])]
    identity = {
        'head_id': geo['character']['head_id'], 'frame_count': frame_count, 'native59': native, 'body_values': body,
        'abmx_runtime': geo['abmx_runtime'], 'abmx_ordered_modifiers': mods,
        'abmx_coordinate': geo['abmx_runtime']['coordinate'], 'abmx_during_h_scene': False,
        'abmx_baseline_refresh_pending': False,
        'reviewed_sources': {k: hashlib.sha256((GAME / v).read_bytes()).hexdigest() for k, v in SOURCES.items()},
        'effect_state': effect['effect_state'], 'effect_globals': effect['effect_globals'],
        'mouth_shape_mask': False, 'skin_quality': 'FourBones'}
    return geo, {'identity': identity}, {'transforms_replaced_from_probe': replaced, 'modifiers': len(mods)}
