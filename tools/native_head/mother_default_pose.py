"""Recover the neutral closed-mouth / open-eye reference from native controllers.

This is an explicit source-derived static state: pattern 0, settled transition,
blink open rate 1, gaze correction zero and voice rate 0. It is not an animation
simulation, a skinned capture or a claim that all-zero weights mean neutral.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from tools.model_bridge.artifact import sha
from tools.native_head.mother_template_inputs import save_json, source_file


def controller_weights(control, open_rate, pattern=0):
    # FBSBase.CalculateBlendShape's float32 Lerp -> float32 * 100 -> int
    # matters: 0.8999999761581421 * 100 in double would truncate to 89.
    lo, hi, rate = map(np.float32, (control['OpenMin'], control['OpenMax'], open_rate))
    fraction = np.float32(lo+np.float32(hi-lo)*np.clip(rate, np.float32(0), np.float32(1)))
    if control['FixedRate'] >= 0:
        fraction = np.float32(control['FixedRate'])
    value = int(np.clip(np.float32(fraction*100), 0, 100))
    result = []
    for target in control['FBSTarget']:
        weights = {int(p[k]): 0. for p in target['PtnSet'] for k in ('Close', 'Open')}
        p = target['PtnSet'][pattern]
        weights[p['Close']] += 100-value
        weights[p['Open']] += value
        result.append((target['ObjTarget']['m_PathID'], weights))
    return result


def dense_frame(shapes, channel_index, count):
    channel = shapes['channels'][channel_index]
    i = channel['frameIndex']
    if channel['frameCount'] != 1 or shapes['fullWeights'][i] != 100:
        raise ValueError('Current donor frame contract changed; multi-frame evaluation is not implemented')
    frame = shapes['shapes'][i]
    rows = shapes['vertices'][frame['firstVertex']:frame['firstVertex']+frame['vertexCount']]
    result = {key: np.zeros((count, 3)) for key in ('vertex', 'normal', 'tangent')}
    ids = [r['index'] for r in rows]
    if len(set(ids)) != len(ids) or any(i < 0 or i >= count for i in ids):
        raise ValueError('Invalid sparse frame vertex indices')
    for key in result:
        result[key][ids] = [[r[key][axis] for axis in 'xyz'] for r in rows]
    return result


def export(inputs, out, controller_source):
    if out.exists():
        raise FileExistsError('Preserve previous reference states; use a new directory')
    receipt = json.loads((inputs/'receipt.json').read_text())
    audit_path = receipt['reference_audit']['path']
    if sha(audit_path) != receipt['reference_audit']['sha256']:
        raise ValueError('Input source audit changed')
    audit = json.loads(Path(audit_path).read_text(encoding='utf-8-sig'))
    for item in audit['assemblies']+audit['decompiled_sources']:
        if sha(item['path']) != item['sha256']:
            raise ValueError('Native controller implementation changed')
    prefab_row = receipt['native_prefab_export']
    if sha(prefab_row['path']) != prefab_row['sha256']:
        raise ValueError('Native prefab changed')
    prefab = json.loads(Path(prefab_row['path']).read_text())
    controller = next(b['original_typetree'] for b in prefab['behaviours'] if b['class_name']=='FaceBlendShape')
    weights = {}
    # FaceBlendShape.OnLateUpdate order; later controllers overwrite shared
    # channel entries, they do not add to the previous controller's values.
    for name, rate in [('EyebrowCtrl', 1), ('EyesCtrl', 1), ('MouthCtrl', 0)]:
        for target, values in controller_weights(controller[name], rate):
            weights.setdefault(target, {}).update(values)
    out.mkdir(parents=True)
    exported = []
    for renderer in prefab['renderers']:
        row = renderer['array_export']
        shape_row = renderer['blendshape_export']
        if sha(row['path']) != row['sha256'] or sha(shape_row['path']) != shape_row['sha256']:
            raise ValueError('Native component source changed')
        arrays = dict(np.load(row['path'], allow_pickle=False))
        shape = json.loads(Path(shape_row['path']).read_text())
        target = renderer['original_renderer_typetree']['m_GameObject']['m_PathID']
        w = weights.get(target, {})
        delta = {k: np.zeros_like(arrays['verts'], dtype=float) for k in ('vertex', 'normal', 'tangent')}
        for index, value in w.items():
            if index == -1 or value == 0:
                continue
            frame = dense_frame(shape, index, len(arrays['verts']))
            for key in delta:
                delta[key] += frame[key]*value/100
        path = out/(renderer['mesh']+'.npz')
        np.savez_compressed(path, **arrays, closed_reference_vertices=arrays['verts']+delta['vertex'],
            closed_reference_normals=arrays['normals']+delta['normal'],
            closed_reference_tangent_xyz=arrays['tangents'][:, :3]+delta['tangent'])
        exported.append(dict(mesh=renderer['mesh'], source=row, frames=shape_row,
                             weights=w, reference=source_file(path)))
    save_json(out/'receipt.json', dict(format='native_mother_default_reference_v1',
        code=source_file(Path(__file__)), inputs=source_file(inputs/'receipt.json'),
        implementation_sources=audit['decompiled_sources']+[
            source_file(controller_source/(name+'.cs')) for name in ('FBSCtrlEyes', 'FBSCtrlEyebrow')],
        state='Pattern0; settled; blink1; gaze0; voice0; source controller sequence',
        skinning_evaluated=False, bind_mesh_modified=False, game_mutated=False,
        renderer_references=exported,
        limitations=['Static controller reference only; no animation/eye-look trajectory simulation',
                     'Original skin transforms and float attributes remain separate from blend shape evaluation']))
    print(str(out.resolve()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--controller-source', type=Path,
        default=Path('../HS2Mod/artifacts/native_mother_template_20261008/controller_source'))
    args = parser.parse_args()
    export(args.inputs.resolve(), args.out.resolve(), args.controller_source.resolve())
