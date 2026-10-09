"""Build bindings against the recovered native neutral shape path.

Preserve all original controller data, animation tables and skin weights. Adapt
the two undriven gaze-parent TRS and invert actual nominal bone matrices; do not
alter a driven bone position that ShapeHeadInfoFemale.Update would overwrite.
This static authoring contract has 59 face values 0.5, no ABMX and isolated head
ancestor coordinates. It does not certify arbitrary animation/modifier states.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np

from scripts.hs2_extract_head import parse_anm_shape, parse_customhead, CUSTOMSHAPE
from src.hs2_mesh_deform import HeadRig, _fk_world
from tools.model_bridge.artifact import sha
from tools.native_head.mother_component_adaptation import verified_json
from tools.native_head.mother_template_inputs import save_json, source_file


def source_audit(path):
    audit = json.loads(path.read_text())
    for row in audit['assemblies']+audit['cache_tables']+audit['decompiled_sources']:
        p = Path(row['path'])
        if not p.is_absolute():
            p = path.parent.parent/p
        if sha(p) != row['sha256']:
            raise ValueError('Native deformation oracle changed: '+str(p))
    if not audit['all_tables_match']:
        raise ValueError('Native equation audit does not match installed source')


def complete_hierarchy(rig, transforms):
    """Validate the cached skin subset, then use the intact donor hierarchy.

    HeadRig's extraction intentionally includes skin bones and ancestors only.
    Renderer transforms and gaze/accessory descendants come from the original
    prefab, rather than being fabricated or silently omitted from FK.
    """
    full = {str(pid): copy.deepcopy(row) for pid, row in transforms.items()}
    for row in full.values():
        row['parent'] = str(row['parent'])
    for pid, cached in rig.bones.items():
        if pid not in full:
            raise ValueError('Cached bone absent from complete donor: '+pid)
        transform = full[pid]
        if cached['name'] != transform['name'] or str(cached['parent']) != transform['parent']:
            raise ValueError('Cached hierarchy differs from donor')
        for field in ('pos', 'rot', 'scale'):
            if cached[field] != transform[field]:
                raise ValueError('Cached rest TRS differs from donor')
    # Detect cycles before invoking the original parent-first ordering routine.
    for pid in full:
        path, current = set(), pid
        while current in full:
            if current in path:
                raise ValueError('Cycle in complete donor hierarchy')
            path.add(current)
            current = full[current]['parent']
    rig.bones = full
    rig.name2pid = {}
    for pid, row in full.items():
        if row['name'] in rig.name2pid:
            raise ValueError('Ambiguous transform name in donor: '+row['name'])
        rig.name2pid[row['name']] = pid
    rig._topo = rig._toposort()


def bindings(inputs, components, out, audit_path):
    if out.exists():
        raise FileExistsError('Preserve previous binding stage; use fresh output')
    source_audit(audit_path)
    source = json.loads((inputs/'receipt.json').read_text())
    prefab = verified_json(source['native_prefab_export'])
    native_audit = verified_json(source['reference_audit'])
    receipt = json.loads((components/'receipt.json').read_text())
    if receipt['inputs']['sha256'] != sha(inputs/'receipt.json'):
        raise ValueError('Component adaptation belongs to different inputs')
    rig = HeadRig(2, sampling_profile='slider_unlocker_18_2')
    original = np.load(inputs/'o_head.npz')
    if not np.array_equal(rig.verts, original['verts']) or rig.skin_bone_names != original['bone_names'].tolist():
        raise ValueError('Cached native rig is not complete donor')
    parsed = parse_anm_shape(source['native_bundle']['path'],
                             native_audit['selected_installed_list_row']['ShapeAnime'])
    if json.loads(json.dumps(parsed)) != rig.anm or parse_customhead() != rig.customhead:
        raise ValueError('Cached animation/category tables differ from actual installed asset')
    complete_hierarchy(rig, prefab['transforms'])
    authored_prefab = copy.deepcopy(prefab)
    face = np.full(59, .5)
    original_world = _fk_world(rig, face)
    head_renderer = next(r for r in prefab['renderers'] if r['mesh']=='o_head')
    mesh_world = original_world[str(head_renderer['transform'])]
    driven_positions = {rig.enums['dst'][e['dst']] for e in rig.eqns if e['target']=='pos'}
    edited = []
    for side, eye in receipt['eye_authoring'].items():
        name = 'cf_J_look_'+side
        if name in driven_positions:
            raise ValueError('Native shape updater would overwrite gaze-parent position')
        pid = rig.name2pid[name]
        parent = rig.bones[pid]['parent']
        target_world = mesh_world@np.r_[eye['target_pivot'], 1.]
        local = np.linalg.inv(original_world[parent])@target_world
        rig.bones[pid]['pos'] = local[:3].tolist()
        rig.bones[pid]['scale'] = (np.asarray(rig.bones[pid]['scale'])*eye['scale']).tolist()
        authored_prefab['transforms'][pid]['pos'] = rig.bones[pid]['pos']
        authored_prefab['transforms'][pid]['scale'] = rig.bones[pid]['scale']
        edited.append(dict(name=name, transform=pid, local_position=local[:3].tolist(),
                           local_scale=rig.bones[pid]['scale'], native_shape_position_driven=False))
    world = _fk_world(rig, face)
    out.mkdir(parents=True)
    reports = []
    for row in receipt['renderer_outputs']:
        if sha(row['arrays']['path']) != row['arrays']['sha256']:
            raise ValueError('Adapted component changed')
        arrays = dict(np.load(row['arrays']['path'], allow_pickle=False))
        renderer = next(r for r in prefab['renderers'] if r['mesh']==row['mesh'])
        model = world[str(renderer['transform'])]
        bones = np.stack([world[rig.name2pid[n]] for n in arrays['bone_names']])
        arrays['bindpose'] = np.linalg.inv(bones)@model
        skin = np.linalg.inv(model)[None]@bones@arrays['bindpose']
        identity_error = float(np.max(np.abs(skin-np.eye(4))))
        if identity_error > 1e-12:
            raise ValueError('Nominal native skin matrices do not preserve authored reference')
        path = out/(row['mesh']+'.npz')
        np.savez_compressed(path, **arrays)
        reports.append(dict(mesh=row['mesh'], source=row['arrays'], arrays=source_file(path),
                             frames=row['frames'], nominal_skin_identity_error=identity_error))
    save_json(out/'native_prefab_authored.json', authored_prefab)
    save_json(out/'receipt.json', dict(format='native_mother_reference_bindings_v1',
        code=source_file(Path(__file__)), source_audit=source_file(audit_path),
        inputs=source_file(inputs/'receipt.json'), components=source_file(components/'receipt.json'),
        native_animation_bundle=source['native_bundle'], native_animation=native_audit['selected_installed_list_row']['ShapeAnime'],
        native_category_bundle=source_file(Path(CUSTOMSHAPE)),
        computation_sources=[source_file(Path(p)) for p in ('src/hs2_mesh_deform.py','src/hs2_sampling.py')],
        authored_prefab=source_file(out/'native_prefab_authored.json'), edited_transforms=edited,
        renderer_outputs=reports, shape_reference=face.tolist(), abmx_reference=None,
        original_controllers_and_tables_retained=True, game_mutated=False, installed=False, deliverable=False,
        limitations=['Native nominal shape path only; no ABMX or external ancestor replay',
                     'Eye gaze parent/pupil offsets adapted; full facial bone pivot authoring remains unverified',
                     'Bundle packing, actual BP interface and complete candidate acceptance still required']))
    print(json.dumps(dict(output=str(out.resolve()), native_nominal_bindings=True,
                          gaze_parents_adapted=True, installed=False, deliverable=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--components', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--audit', type=Path, default=Path('../HS2Mod/docs/hs2_native_source_audit.json'))
    args = parser.parse_args()
    bindings(args.inputs.resolve(), args.components.resolve(), args.out.resolve(), args.audit.resolve())
