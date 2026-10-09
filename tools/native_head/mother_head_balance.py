"""Rebalance head placement and occipital volume on an intact native asset.

This is explicit mesh authoring, not anatomical pose inference or HS2 math.
Start from the joined-neck candidate, move the complete head reference forward,
lower/bring forward the rear bulge, then rebuild the native clamped collar.
All native components and expression frames must subsequently be regenerated
by mother_component_adaptation and the complete binding/bundle pipeline.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np

from tools.model_bridge.artifact import sha
from tools.native_head.mother_neck_transition import collar, review
from tools.native_head.mother_occipital_adjustment import smooth
from tools.native_head.mother_surface_warp import SurfaceWarp
from tools.native_head.mother_template_inputs import save_json, source_file


def author(candidate, out, forward_fraction=.10, rear_down_fraction=.24,
           rear_forward_fraction=.22):
    if out.exists():
        raise FileExistsError('Use a fresh output; preserve previous versions')
    receipt = json.loads((candidate/'receipt.json').read_text())
    neck = receipt['neck_transition']
    for row in (receipt['candidate_arrays'], neck['domain'], neck['actual_head_body']):
        if sha(row['path']) != row['sha256']:
            raise ValueError('Input provenance changed: '+row['path'])
    if 'occipital_adjustment' in receipt or 'head_balance' in receipt:
        raise ValueError('Start from the joined neck, not accumulated rear edits')
    arrays = dict(np.load(candidate/'o_head_candidate.npz', allow_pickle=False))
    regions = json.loads((Path(receipt['inputs']['path']).parent/'native_regions.json').read_text())
    domain = json.loads(Path(neck['domain']['path']).read_text())
    v = arrays['verts']
    names = arrays['bone_names'].tolist()
    ear_bones = [i for i, name in enumerate(names) if 'Ear' in name]
    ear_support = (arrays['bone_w']*np.isin(arrays['bone_idx'], ear_bones)).sum(1)
    ears = v[ear_support > .5]
    span = float(np.ptp(ears[:, 1]))
    low = float(v[domain['lower_native_render_ids'], 1].max())
    full_low = float(ears[:, 1].mean())
    full_high = float(ears[:, 1].max()+.3*span)
    top = float(v[:, 1].max())
    front, rear = float(ears[:, 2].min()), float(v[:, 2].min())
    if not low < full_low < full_high < top or not rear < front:
        raise ValueError('Invalid native support bounds')
    weight = smooth((front-v[:, 2])/(front-rear))
    weight *= smooth((v[:, 1]-low)/(full_low-low))
    weight *= 1-smooth((v[:, 1]-full_high)/(top-full_high))
    protected_ids = np.asarray(domain['protected_feature_render_ids'], dtype=int)
    weight[protected_ids] = 0
    translation = np.array([0., 0., forward_fraction*span])
    rear_delta = span*np.array([0., -rear_down_fraction, rear_forward_fraction])
    target = v+translation+weight[:, None]*rear_delta
    # Reuse the native endpoint/tangent contract; never move the body interface.
    canon = np.array([int(regions['graph_aliases'].get(str(i), i)) for i in range(len(v))])
    unique, inverse = np.unique(canon, return_inverse=True)
    protected = np.zeros(len(v), dtype=bool)
    protected[protected_ids] = True
    result, rebuilt = collar(arrays['original_vertices'][unique], target[unique],
        inverse[arrays['faces']], np.unique(inverse[regions['physical_neck_boundary']]),
        protected[unique], np.zeros(3), neck['transition_rings'])
    final = result[inverse]
    if not np.array_equal(final[protected_ids], (v+translation)[protected_ids]):
        raise ValueError('Facial/ear shape changed beyond whole-head translation')
    lower = np.flatnonzero(rebuilt['lower'][inverse])
    if not np.array_equal(final[lower], arrays['original_vertices'][lower]):
        raise ValueError('Native collar positions or outgoing direction changed')
    for group in regions['identical_bind_position_and_skin_groups']:
        if not np.all(final[group] == final[group[0]]):
            raise ValueError('Physical UV copies separated')
    for key in arrays:
        if not np.isfinite(arrays[key]).all() if arrays[key].dtype.kind in 'fc' else False:
            raise ValueError('Nonfinite source array')
    field = SurfaceWarp(arrays['original_vertices'], final, arrays['faces'])
    field.vertex_gradients({int(k): val for k, val in regions['graph_aliases'].items()})
    out.mkdir(parents=True)
    np.savez_compressed(out/'o_head_candidate.npz', **{**arrays, 'verts': final,
        'reference_vertices': arrays['reference_vertices']+translation})
    np.savez_compressed(out/'head_balance_field.npz', before_vertices=v,
        prescribed_vertices=target, posterior_weight=weight, translation=translation,
        posterior_delta=rear_delta, protected_render_ids=protected_ids)
    save_json(out/'neck_domain.json', dict(
        lower_native_render_ids=lower.tolist(),
        free_render_ids=np.flatnonzero(rebuilt['free'][inverse]).tolist(),
        rigid_exterior_render_ids=np.flatnonzero(rebuilt['upper'][inverse]).tolist(),
        protected_feature_render_ids=protected_ids.tolist(),
        all_render_copies_included=True, body_interface=domain['body_interface']))
    updated = copy.deepcopy(receipt)
    updated['placement']['translation'] = (np.asarray(receipt['placement']['translation'])+translation).tolist()
    updated['neck_transition']['domain'] = source_file(out/'neck_domain.json')
    updated['neck_transition']['outside_collar_rigid_translation_only'] = False
    updated.update(candidate_arrays=source_file(out/'o_head_candidate.npz'),
        code=source_file(Path(__file__)), head_balance=dict(
            input_candidate=source_file(candidate/'receipt.json'),
            field=source_file(out/'head_balance_field.npz'),
            ear_height=span, complete_head_translation=translation.tolist(),
            rear_displacement=rear_delta.tolist(),
            bounds=dict(lower=low, full_low=full_low, full_high=full_high,
                        top=top, front=front, rear=rear),
            authoring_fractions=dict(forward=forward_fraction, rear_down=rear_down_fraction,
                                    rear_forward=rear_forward_fraction),
            pitch_unchanged=True, no_global_scale=True,
            face_ears_and_flame_eye_reference_translate_together=True,
            neck_lower_two_rows_literal=True, native_topology_unchanged=True,
            method='Whole-reference translation; C2 posterior bulge lowering/forward shift; native clamped collar rebuild',
            anatomical_pose_inference=False),
        installed=False, game_mutated=False, deliverable=False)
    save_json(out/'receipt.json', updated)
    actual = json.loads(Path(neck['actual_head_body']['path']).read_text(encoding='utf-8-sig'))
    body = next(m for m in actual['meshes'] if m['mesh_name'] == 'o_body_cf')
    review(v, final, arrays['faces'], body, out)
    print(json.dumps(dict(output=str(out.resolve()), native_interface_unchanged=True,
                          facial_shape_preserved=True, full_head_position_considered=True)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--forward-fraction', type=float, default=.10)
    p.add_argument('--rear-down-fraction', type=float, default=.24)
    p.add_argument('--rear-forward-fraction', type=float, default=.22)
    a = p.parse_args()
    if not np.isfinite([a.forward_fraction, a.rear_down_fraction, a.rear_forward_fraction]).all():
        raise ValueError('Finite authoring displacements required')
    author(a.candidate.resolve(), a.out.resolve(), a.forward_fraction,
           a.rear_down_fraction, a.rear_forward_fraction)
