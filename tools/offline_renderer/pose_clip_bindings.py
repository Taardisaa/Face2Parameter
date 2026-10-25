"""Which transforms the Maker pose Animator rewrites (also inside the bridge PoseFreeze).

CPU only. Inputs: a FramePairProbe v7 window (static.json and the scene export's ``animation``
block, which identifies the Animator, its controller, the current state and the Maker pose list
entry with its bundle file) and the installed asset bundle that entry names.

Mechanism (source):
  * CustomBase.ChangeAnimation (decompiled_acs/CharaCustom/CustomBase.cs 1464-1526) loads the
    RuntimeAnimatorController ``MainData`` from ``AssetBundleManager.BaseDownloadingURL + MainAB``
    (ChaControl.LoadAnimation) and plays state ``Clip``.
  * The bridge PoseFreeze sets speed 0, Play(fullPathHash, 0, 0) and Animator.Update(0).
  * A Mecanim generic Animator binds the controller clips' ``m_ClipBindingConstant.genericBindings``
    to the hierarchy below its GameObject; ``path`` is the CRC-32 of the transform path relative to
    that GameObject (root = ""), ``typeID`` 4 = Transform with attribute 1 position, 2 rotation,
    3 scale, 4 euler rotation. A path with no transform is not bound (reported, never dropped).
  * The current state's clips are its blend-tree nodes' ``m_ClipID`` into the controller's
    ``m_AnimationClips``; ``m_WriteDefaultValues`` says whether bound properties not animated by
    the current clips are rewritten with their default values.

    python tools/offline_renderer/pose_clip_bindings.py --window E:/.../window_<nonce> --out FRESH.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zlib
from pathlib import Path

TRANSFORM_ATTRIBUTES = {1: 'localPosition', 2: 'localRotation', 3: 'localScale', 4: 'localEulerAngles'}


class BindingError(RuntimeError):
    pass


def u32(v):
    return int(v) & 0xFFFFFFFF


def relative_paths(transforms, root_index):
    """{crc32(path): (index, path)} for every transform at or below root_index."""
    by_i = {t['i']: t for t in transforms}
    out = {}
    for t in transforms:
        parts, i = [], t['i']
        while i != root_index:
            if i < 0:
                parts = None
                break
            parts.append(by_i[i]['name'])
            i = by_i[i]['parent']
        if parts is None:
            continue
        path = '/'.join(reversed(parts))
        h = zlib.crc32(path.encode('utf-8'))
        if h in out and out[h][1] != path:
            raise BindingError('CRC-32 collision between %r and %r' % (out[h][1], path))
        out[h] = (t['i'], path)
    return out


def load_bundle(path):
    import UnityPy
    env = UnityPy.load(str(path))
    objs = {o.path_id: o for o in env.objects}
    controllers = [o for o in env.objects if o.type.name == 'AnimatorController']
    return objs, controllers


def resolve_clip(clip_tree, paths):
    rows, unbound = [], []
    for b in clip_tree['m_ClipBindingConstant']['genericBindings']:
        h = u32(b['path'])
        hit = paths.get(h)
        entry = {'path_crc32': h, 'type_id': b['typeID'], 'attribute': b['attribute'],
                 'custom_type': b['customType'], 'is_pptr_curve': b['isPPtrCurve']}
        if b['typeID'] == 4:
            entry['property'] = TRANSFORM_ATTRIBUTES.get(b['attribute'], 'attribute_%d' % b['attribute'])
        if hit is None:
            unbound.append(entry)
        else:
            entry.update(transform_index=hit[0], path=hit[1])
            rows.append(entry)
    return rows, unbound


def analyse(window: Path, scene_file: Path | None = None):
    static = json.loads((window / 'static.json').read_text(encoding='utf-8'))
    if scene_file is None:
        receipt = json.loads((window / 'capture_receipt.json').read_text(encoding='utf-8'))
        frames = receipt['frame_counts_by_index']
        cap = receipt.get('expected_capture_frame')
        if cap not in frames:
            raise BindingError('captured frame %s has no export' % cap)
        scene_file = window / 'frames' / ('scene_%03d.json' % frames.index(cap))
    scene = json.loads(Path(scene_file).read_text(encoding='utf-8'))
    anim = scene.get('animation')
    if not isinstance(anim, dict):
        raise BindingError('scene export has no animation block (probe < 7.0.0 or %s)' % scene.get('animation_error'))
    pose = anim.get('maker_pose_source')
    if not isinstance(pose, dict) or not isinstance(pose.get('entry'), dict):
        raise BindingError('no Maker pose list entry: %s' % pose)
    entry = pose['entry']
    bundle = Path(entry['bundle_file'])
    if not bundle.is_file():
        raise BindingError('pose bundle not found: %s' % bundle)
    index_of_id = {t['id']: t['i'] for t in static['transforms']}
    animators = [a for a in anim['animators'] if isinstance(a, dict) and a.get('controller')]
    objs, controllers = load_bundle(bundle)
    out = {'schema': 'hs2_pose_clip_bindings_v1', 'source': __doc__, 'window': str(window), 'scene_export': str(scene_file),
           'bundle': str(bundle), 'bundle_sha256': hashlib.sha256(bundle.read_bytes()).hexdigest(),
           'pose_entry': entry, 'pose_no': pose.get('pose_no'), 'animators': []}
    for a in animators:
        cname = a['controller']['ref_name']
        if cname != entry['MainData']:
            raise BindingError('runtime controller %s is not the pose entry MainData %s' % (cname, entry['MainData']))
        cands = [o for o in controllers if o.read_typetree()['m_Name'] == cname]
        if len(cands) != 1:
            raise BindingError('controller %s found %d times in %s' % (cname, len(cands), bundle))
        ct = cands[0].read_typetree()
        tos = {u32(k): v for k, v in ct['m_TOS']}
        clip_ptrs = ct['m_AnimationClips']
        clips = []
        for p in clip_ptrs:
            if p['m_FileID'] != 0 or p['m_PathID'] not in objs:
                raise BindingError('controller clip outside this bundle: %s' % p)
            clips.append(objs[p['m_PathID']].read_typetree())
        root = index_of_id.get(a['transform']['ref_id'])
        if root is None:
            raise BindingError('Animator transform %s is not in static.json' % a['transform'])
        paths = relative_paths(static['transforms'], root)
        cur = a.get('layer0_current') or {}
        full = u32(cur.get('full_path_hash', 0))
        states = [s['data'] for sm in ct['m_Controller']['m_StateMachineArray'] for s in sm['data']['m_StateConstantArray']]
        match = [s for s in states if u32(s['m_FullPathID']) == full]
        if len(match) != 1:
            raise BindingError('state fullPathHash %d matches %d controller states' % (full, len(match)))
        st = match[0]
        clip_ids = sorted({n['data']['m_ClipID'] for bt in st['m_BlendTreeConstantArray'] for n in bt['data']['m_NodeArray']
                           if n['data'].get('m_ClipID', 0xFFFFFFFF) != 0xFFFFFFFF})
        state_clips = [clips[i]['m_Name'] for i in clip_ids]
        runtime_clips = [c['clip']['ref_name'] for c in a.get('layer0_current_clips') or []]
        if sorted(runtime_clips) != sorted(state_clips):
            raise BindingError('runtime current clips %s differ from the state blend-tree clips %s' % (runtime_clips, state_clips))
        per_clip = []
        union = set()
        for i, c in enumerate(clips):
            rows, unbound = resolve_clip(c, paths)
            union |= {r['transform_index'] for r in rows if r['type_id'] == 4}
            per_clip.append({'clip': c['m_Name'], 'controller_clip_index': i, 'in_current_state': i in clip_ids,
                             'bindings': len(rows) + len(unbound), 'bound': rows, 'unbound': unbound})
        current = set().union(*[{r['transform_index'] for r in pc['bound'] if r['type_id'] == 4} for pc in per_clip if pc['in_current_state']])
        out['animators'].append({
            'animator': a['animator'], 'animator_transform_index': root, 'controller': cname,
            'state': {'name': tos.get(u32(st['m_NameID'])), 'full_path_hash': full, 'write_default_values': bool(st['m_WriteDefaultValues']),
                      'speed': st['m_Speed'], 'loop': bool(st['m_Loop']), 'clip_ids': clip_ids, 'clips': state_clips},
            'clips': per_clip,
            'transforms_written_by_current_state': sorted(current),
            'transforms_bound_by_any_controller_clip': sorted(union),
            'note': ('write_default_values true: bound properties not in the current clips are rewritten with defaults'
                     if st['m_WriteDefaultValues'] else 'write_default_values false: only the current clips write'),
        })
    if not out['animators']:
        raise BindingError('no Animator with a controller in the export')
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--window', type=Path, required=True)
    ap.add_argument('--scene', type=Path, help='scene export to use (default: the captured frame)')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(args.out)
    res = analyse(args.window, args.scene)
    args.out.write_text(json.dumps(res, indent=1), encoding='utf-8')
    for a in res['animators']:
        print(a['controller'], a['state']['name'], 'written', len(a['transforms_written_by_current_state']),
              'bound(any clip)', len(a['transforms_bound_by_any_controller_clip']),
              'unbound paths', sum(len(c['unbound']) for c in a['clips'] if c['in_current_state']))


if __name__ == '__main__':
    sys.exit(main())
