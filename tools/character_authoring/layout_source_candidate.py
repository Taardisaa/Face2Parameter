"""Author feature placement using captured native head and both eye meshes.

Uses installed cached length-one position branches, actual descendants/weights,
and perspective surface attachments. Image marks are reviewed observations;
they do not recover hidden anatomy or certify shader visibility.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from chin_source_candidate import SourceForward, read, sections
from portrait_compare import measure


FIELDS = [('cf_J_FaceUp_ty', 'position', 0), ('cf_J_FaceUp_ty', 'position', 1),
          ('cf_J_NoseBase_trs', 'position', 0), ('cf_J_NoseBase_trs', 'position', 1)]
FIELD_PROFILES = {'upper4': FIELDS,
                  'local8': [(n, 'position', axis) for n in
                             ['cf_J_Eye_s_L', 'cf_J_Eye_s_R', 'cf_J_NoseBase_trs', 'cf_J_MouthBase_tr']
                             for axis in (0, 1)]}


def attachment(model, pixel):
    points = model.pixels(model.base)
    triangles = model.triangles
    tri = points[triangles]
    a, b, c = tri[:, 0], tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]
    q = np.asarray(pixel)-a
    den = b[:, 0]*c[:, 1]-b[:, 1]*c[:, 0]
    valid = abs(den) > 1e-12
    u, v = np.full(len(tri), np.nan), np.full(len(tri), np.nan)
    u[valid] = (q[valid, 0]*c[valid, 1]-q[valid, 1]*c[valid, 0])/den[valid]
    v[valid] = (b[valid, 0]*q[valid, 1]-b[valid, 1]*q[valid, 0])/den[valid]
    inside = np.flatnonzero((u >= 0) & (v >= 0) & (u+v <= 1))
    if not len(inside):
        raise ValueError('Observed feature pixel has no recorded mesh intersection')
    h = model.head_vertices(model.base)
    pose = model.c['head_pose']
    linear = Rotation.from_quat(pose['rotation_xyzw']).as_matrix() @ np.diag(pose['scale'])
    world = h @ linear.T + pose['position']
    depth = model.c['ortho_size']/np.tan(np.deg2rad(model.fov/2))-(world-model.c['target'])[:, 2]
    if np.any(depth <= 0):
        raise ValueError('Source mesh behind capture camera')
    screen_bary = np.c_[1-u[inside]-v[inside], u[inside], v[inside]]
    perspective = screen_bary/depth[triangles[inside]]
    chosen = int(np.argmin(1/perspective.sum(1)))
    ids = triangles[inside[chosen]]
    bary = perspective[chosen]/perspective[chosen].sum()
    return {'triangle': int(inside[chosen]), 'vertex_ids': ids.tolist(),
            'perspective_barycentric': bary.tolist(), 'observed_pixel': list(pixel)}


def project_attachment(model, x, binding):
    head = np.asarray(binding['perspective_barycentric']) @ model.head_vertices(x)[binding['vertex_ids']]
    pose = model.c['head_pose']
    world = (Rotation.from_quat(pose['rotation_xyzw']).as_matrix() @ np.diag(pose['scale'])) @ head + pose['position']
    rel = world-model.c['target']
    tan = np.tan(np.deg2rad(model.fov/2))
    depth = model.c['ortho_size']/tan-rel[2]
    focal = model.height/2/tan
    return np.array([model.width/2-rel[0]*focal/depth, model.height/2-rel[1]*focal/depth])


def gaze_contract(drivers):
    eye = drivers['eye_controller']
    if eye['resolved_type']['name'] != 'NO_LOOK' or not all(
            eye[k] for k in ('enabled','active_and_enabled','script_enabled','global_enabled')):
        raise ValueError('Recorded native enabled NO_LOOK controller required')
    angles = np.asarray(eye['fixed_angles_xyzw'])
    if angles.shape != (2,4) or not np.all(np.isfinite(angles)):
        raise ValueError('Actual two-eye cached fixAngle required')
    return {'actor_transform_id': drivers['actor_transform_id'],
            'bridge_mvid': drivers['bridge_mvid'], 'game_mvid': drivers['game_mvid'],
            'pattern': eye['pattern'], 'resolved_type': eye['resolved_type'],
            'fixed_angles_xyzw': eye['fixed_angles_xyzw']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--geometry', type=Path, required=True)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--marks', type=Path, required=True)
    p.add_argument('--jaw-candidate', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--controls', choices=list(FIELD_PROFILES), default='upper4')
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError('Fresh candidate required')
    d, gs = read(a.geometry)
    driver_contract = gaze_contract(d['native_face_drivers'])
    capture, cs = read(a.capture)
    marks, ms = read(a.marks)
    jaw, js = read(a.jaw_candidate)
    registration, rs = read(Path(marks['registration_measurements']))
    for name in ('reference', 'candidate'):
        r = marks[name]
        if hashlib.sha256(Path(r['path']).read_bytes()).hexdigest() != r['image_sha256']:
            raise ValueError('Reviewed image hash mismatch')
    if hashlib.sha256(Path(capture['capture']['path']).read_bytes()).hexdigest() != marks['candidate']['image_sha256']:
        raise ValueError('Capture does not bind reviewed current portrait')
    rr = np.asarray(registration['reference']['landmarks_pixels'])
    gg = np.asarray(registration['candidate']['landmarks_pixels'])
    _, _, _, _, sr, Rr = measure(rr)
    _, _, _, _, sg, Rg = measure(gg)
    mr, mg = rr[[62, 66]].mean(0), gg[[62, 66]].mean(0)
    def observations(row):
        return np.array([np.mean(row['eye_left_contour_xy'], axis=0),
                         np.mean(row['eye_right_contour_xy'], axis=0),
                         row['nose_base_xy'], row['mouth_seam_xy']])
    observed = observations(marks['candidate'])
    target = ((observations(marks['reference'])-mr) @ Rr.T/sr) @ Rg*sg+mg
    fields = FIELD_PROFILES[a.controls]
    models = {n: SourceForward(d, capture['capture'], parameter_fields=fields, mesh_name=n)
              for n in ('o_head', 'o_eyebase_L', 'o_eyebase_R')}
    # Anatomical left corresponds to image right; require actual projected eye
    # centers rather than relying on the artist's screen-left naming convention.
    eye_order = sorted(['o_eyebase_L', 'o_eyebase_R'],
                       key=lambda n: models[n].pixels(models[n].base)[:, 0].mean())
    feature_meshes = eye_order + ['o_head', 'o_head']
    bindings = [attachment(models[n], point) for n, point in zip(feature_meshes, observed)]
    base = models['o_head'].base
    def predict(x):
        return np.array([project_attachment(models[n], x, b)
                         for n, b in zip(feature_meshes, bindings)])
    def residual(x):
        return (predict(x)-target).ravel()
    # Author limits only; no assertion of native parameter/likeness limits.
    result = least_squares(residual, base, bounds=(base-.10, base+.10),
                           ftol=1e-11, xtol=1e-11, gtol=1e-11)
    head = models['o_head']
    px0, px1 = head.pixels(base), head.pixels(result.x)
    norm0, norm1 = (px0-mg) @ Rg.T/sg, (px1-mg) @ Rg.T/sg
    levels = np.asarray(jaw['target_levels_per_eye_distance'])
    tx = np.asarray(jaw['target_terminal_x_per_eye_distance'])
    boundaries = sections(norm1, head.triangles, levels)
    terminal = sections(norm1[:, [1, 0]], head.triangles, tx)[:, 1]
    patch = {n: {'name': n, 'position': head.mods[n]['position'].copy(), 'length': 1.0}
             for n, _, _ in fields}
    for value, (n, _, axis) in zip(result.x, fields):
        patch[n]['position'][axis] = float(value)
    errors = {}
    for name, model in models.items():
        m = model.m
        baked = np.c_[m['baked']['vertices'], np.ones(m['vertex_count'])]
        observed_head = (baked @ np.array(m['baked']['world_candidates']['scale_free_trs']['matrix']).reshape(4, 4).T @ model.head_inv.T)[:, :3]
        e = np.linalg.norm(model.head_vertices(base)-observed_head, axis=1)
        errors[name] = {'max': float(e.max()), 'mean': float(e.mean()), 'vertex_count': len(e)}
    report = {'status': 'Source author candidate; actual image positions and complete jaw lock must be reviewed before acceptance',
              'inputs_sha256': {'geometry': gs, 'capture': cs, 'marks': ms, 'registration': rs, 'jaw': js},
              'source_gaze_contract': driver_contract,
              'fields': fields, 'author_controls': a.controls, 'base': base.tolist(), 'candidate': result.x.tolist(),
              'optimizer_success': bool(result.success), 'optimizer_message': result.message,
              'feature_meshes': feature_meshes, 'feature_bindings': bindings,
              'observed_feature_pixels': observed.tolist(), 'target_feature_pixels': target.tolist(),
              'candidate_feature_pixels': predict(result.x).tolist(),
              'base_signed_error_pixels': residual(base).reshape(-1, 2).tolist(),
              'candidate_signed_error_pixels': residual(result.x).reshape(-1, 2).tolist(),
              'baseline_actual_mesh_errors_game_units': errors,
              'jaw_upper_target_errors_reference_pixels': ((boundaries-jaw['target_boundaries_per_eye_distance'])*sr).tolist(),
              'jaw_terminal_target_errors_reference_pixels': ((terminal-jaw['target_terminal_y_per_eye_distance'])*sr).tolist(),
              'jaw_upper_displacement_reference_pixels': ((boundaries-sections(norm0, head.triangles, levels))*sr).tolist(),
              'jaw_terminal_displacement_reference_pixels': ((terminal-sections(norm0[:, [1, 0]], head.triangles, tx)[:, 1])*sr).tolist(),
              'candidate_head_vertices': head.head_vertices(result.x).tolist(),
              'patch': {'bones': list(patch.values())},
              'scope': __doc__,
              'registration_scope': marks['scope'],
              'attachment_scope': 'Closest positive-depth recorded mesh intersection at reviewed pixels; source geometry attachment, not an independently certified anatomical label or shader visibility. Full-eye shape, texture, gaze and hidden art 3D are separate requirements.'}
    with a.out.open('x', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: report[k] for k in ['candidate', 'candidate_signed_error_pixels', 'patch']}, indent=2))


if __name__ == '__main__':
    main()
