"""Author a chin candidate against full projected triangle boundaries.

Fits appearance using the captured source forward model; it never learns game
deformation gains from samples. Unsupported active blendshape frames fail.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from portrait_compare import measure
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.hs2_mesh_deform import euler_zxy_quat, _qmul, _quat_to_mat


def read(path):
    raw = path.read_bytes()
    return json.loads(raw.decode('utf-8-sig')), hashlib.sha256(raw).hexdigest()


class SourceForward:
    def __init__(self, d, capture, width=768, height=1024, controls='chin8',
                 parameter_fields=None, mesh_name='o_head'):
        meshes = [m for m in d['meshes'] if m['mesh_name'] == mesh_name]
        if len(meshes) != 1:
            raise ValueError('One recorded native mesh required: '+mesh_name)
        self.m = m = meshes[0]
        self.t = t = {r['id']: r for r in d['transforms']}
        heads = [r for r in t.values() if r['name'] == 'cf_J_Head']
        if len(heads) != 1:
            raise ValueError('Unique actual head required')
        self.head_inv = np.asarray(heads[0]['world_to_local']).reshape(4, 4)
        self.world = {i: np.asarray(r['local_to_world']).reshape(4, 4) for i, r in t.items()}
        self.local = {i: np.linalg.inv(self.world[r['parent_id']]) @ self.world[i]
                      for i, r in t.items() if r['parent_id'] is not None}
        self.mods = {r['name']: r for r in d['abmx_runtime']['bones']}
        if controls not in ('chin8', 'jaw14', 'jaw18', 'jaw22'):
            raise ValueError('Unknown author controls')
        names = ['cf_J_ChinTip_s', 'cf_J_ChinLow', 'cf_J_Chin_rs']
        if controls in ('jaw14', 'jaw18', 'jaw22'):
            names += ['cf_J_CheekLow_L', 'cf_J_CheekLow_R']
        if parameter_fields is not None:
            if not parameter_fields or any(field not in ('scale','position','rotation') or
                    axis not in (0,1,2) for _,field,axis in parameter_fields):
                raise ValueError('Explicit nonempty cached modifier fields required')
            names = list(dict.fromkeys(n for n,_,_ in parameter_fields))
        for name in names:
            cache = self.mods[name]['runtime_baseline']
            if cache['missing_fields'] or not cache['fields']['_hasBaseline']:
                raise ValueError('Actual baseline missing: '+name)
        if parameter_fields is None and any(self.mods[n]['length'] != 1 for n in ('cf_J_ChinTip_s', 'cf_J_ChinLow')):
            raise ValueError('Position patches require installed length-one branch')
        self.ids = {n: self.mods[n]['runtime_baseline']['bone_transform_id'] for n in names}
        v = np.asarray(m['source']['vertices'], float).copy()
        for s in m['blendshapes']:
            if not s['current_weight']:
                continue
            frames = s.get('frames', [])
            if len(frames) != 1 or frames[0]['weight'] != 100 or not 0 <= s['current_weight'] <= 100:
                raise ValueError('Unsupported active blendshape sampling')
            v += np.asarray(frames[0]['delta_vertices']) * s['current_weight']/100
        self.vertices = np.c_[v, np.ones(len(v))]
        self.indices = np.asarray(m['source']['bone_indices'])
        self.weights = np.asarray(m['source']['bone_weights'])
        self.bind = np.asarray(m['source']['bindposes']).reshape(-1, 4, 4)
        self.triangles = np.asarray(m['source']['triangles']).reshape(-1, 3)
        self.c = capture
        if capture['pipeline'] != 'actual_main_camera' or capture['yaw'] != 0 or capture['projection'] != 'perspective':
            raise ValueError('Front actual-main-camera perspective capture required')
        self.fov = d['main_camera']['fov_degrees']
        self.width, self.height = width, height
        self.parameter_fields = [
            ('cf_J_Chin_rs','scale',0), ('cf_J_Chin_rs','scale',1),
            ('cf_J_ChinTip_s','scale',0), ('cf_J_ChinTip_s','scale',1),
            ('cf_J_ChinTip_s','position',1), ('cf_J_ChinLow','scale',0),
            ('cf_J_ChinLow','scale',1), ('cf_J_ChinLow','position',1)]
        if controls in ('jaw14', 'jaw18', 'jaw22'):
            # Actual cached-scale branch only for the cheeks: their non-unit
            # length, rotation and position are retained, never approximated.
            self.parameter_fields += [
                ('cf_J_ChinTip_s','position',0), ('cf_J_ChinLow','position',0),
                ('cf_J_CheekLow_L','scale',0), ('cf_J_CheekLow_L','scale',1),
                ('cf_J_CheekLow_R','scale',0), ('cf_J_CheekLow_R','scale',1)]
        if controls in ('jaw18', 'jaw22'):
            self.parameter_fields += [
                ('cf_J_CheekLow_L','rotation',1), ('cf_J_CheekLow_L','rotation',2),
                ('cf_J_CheekLow_R','rotation',1), ('cf_J_CheekLow_R','rotation',2)]
        if controls == 'jaw22':
            self.parameter_fields += [
                ('cf_J_CheekLow_L','position',0), ('cf_J_CheekLow_L','position',1),
                ('cf_J_CheekLow_R','position',0), ('cf_J_CheekLow_R','position',1)]
        if parameter_fields is not None:
            self.parameter_fields = list(parameter_fields)
        for name,field,_ in self.parameter_fields:
            if field == 'position' and (self.mods[name]['length'] != 1 or
                    self.mods[name]['runtime_baseline']['fields']['_lenModForceUpdate']):
                raise ValueError('Position authoring requires captured steady length-one branch: '+name)
            if field == 'rotation' and (d['abmx_runtime'].get('rotation_excluded_bones') is None or
                    name in d['abmx_runtime']['rotation_excluded_bones']):
                raise ValueError('Rotation authoring requires recorded supported bone: '+name)
        self.rotation_base = {}
        for name,field,_ in self.parameter_fields:
            if field != 'rotation' or name in self.rotation_base:
                continue
            q = self.mods[name]['runtime_baseline']['fields']['_rotBaseline']
            self.rotation_base[name] = _quat_to_mat(_qmul(q, euler_zxy_quat(*self.mods[name]['rotation'])))
        self.base = np.array([self.mods[n][field][axis] for n,field,axis in self.parameter_fields])

    def head_vertices(self, x):
        local = {i: self.local[i].copy() for i in self.ids.values()}
        # Installed cached-baseline scale multiplication; unchanged rotations and
        # lengths retained. Affine columns correspond to the local scale axes.
        for j,(name,field,axis) in enumerate(self.parameter_fields):
            if field == 'scale':
                local[self.ids[name]][:3,axis] *= x[j]/self.base[j]
        position_names = {n for n,field,_ in self.parameter_fields if field == 'position'}
        for name in position_names:
            position = self.mods[name]['position'].copy()
            for value,(n,field,axis) in zip(x,self.parameter_fields):
                if n == name and field == 'position':
                    position[axis] = value
            cache = self.mods[name]['runtime_baseline']['fields']
            # Installed HasPosition is an exact component comparison. An old
            # length change can leave a different current position when there
            # is no offset and _changedPosition is false. Preserve that branch;
            # nonzero offsets use the actual cache, not current-position+delta.
            if any(v != 0 for v in position) or cache['_changedPosition']:
                local[self.ids[name]][:3,3] = np.asarray(cache['_posBaseline']) + position
        for name, base_rotation in self.rotation_base.items():
            euler = self.mods[name]['rotation'].copy()
            for value,(n,field,axis) in zip(x,self.parameter_fields):
                if n == name and field == 'rotation':
                    euler[axis] = value
            q = self.mods[name]['runtime_baseline']['fields']['_rotBaseline']
            proposed_rotation = _quat_to_mat(_qmul(q, euler_zxy_quat(*euler)))
            local[self.ids[name]][:3,:3] = proposed_rotation @ np.linalg.inv(base_rotation) @ local[self.ids[name]][:3,:3]
        changed, result = {}, {}
        def affected(i):
            if i not in changed:
                parent = self.t[i]['parent_id']
                changed[i] = i in local or (parent is not None and affected(parent))
            return changed[i]
        def world(i):
            if i not in result:
                if not affected(i):
                    result[i] = self.world[i]
                else:
                    result[i] = world(self.t[i]['parent_id']) @ local.get(i, self.local[i])
            return result[i]
        matrices = np.asarray([world(i) @ bp for i, bp in zip(self.m['bone_transform_ids'], self.bind)])
        skinned = np.einsum('vki,vk->vi', np.einsum(
            'vkij,vj->vki', matrices[self.indices], self.vertices), self.weights)
        return (skinned @ self.head_inv.T)[:, :3]

    def pixels(self, x):
        h = self.head_vertices(x)
        pose = self.c['head_pose']
        linear = Rotation.from_quat(pose['rotation_xyzw']).as_matrix() @ np.diag(pose['scale'])
        world = h @ linear.T + pose['position']
        tan = np.tan(np.deg2rad(self.fov/2))
        rel = world - self.c['target']
        depth = self.c['ortho_size']/tan-rel[:, 2]
        if np.any(depth <= 0):
            raise ValueError('Vertex behind capture camera')
        focal = self.height/2/tan
        return np.c_[self.width/2-rel[:, 0]*focal/depth,
                     self.height/2-rel[:, 1]*focal/depth]


def sections(points, triangles, levels):
    """Exact horizontal intersections of all projected triangle edges.

    Min/max of actual triangle union, not a convex hull or handpicked bone/region.
    This geometric silhouette does not certify all shader/depth visibility.
    """
    tri = points[triangles]
    a, b = tri, np.roll(tri, -1, axis=1)
    ans = []
    for y in levels:
        crosses = ((a[:, :, 1] <= y) & (b[:, :, 1] > y)) | ((b[:, :, 1] <= y) & (a[:, :, 1] > y))
        aa, bb = a[crosses], b[crosses]
        if not len(aa):
            raise ValueError('Section outside the projected mesh')
        hit = aa[:, 0] + (bb[:, 0]-aa[:, 0])*(y-aa[:, 1])/(bb[:, 1]-aa[:, 1])
        ans.append([hit.min(), hit.max()])
    return np.asarray(ans)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--geometry', type=Path, required=True)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--measurements', type=Path, required=True)
    p.add_argument('--trace', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--scale-span', type=float, default=.30)
    p.add_argument('--position-span', type=float, default=.15)
    p.add_argument('--scale-min', type=float, default=.65)
    p.add_argument('--scale-max', type=float, default=1.50)
    p.add_argument('--max-iterations', type=int, default=200)
    p.add_argument('--controls', choices=['chin8', 'jaw14', 'jaw18', 'jaw22'], default='chin8')
    p.add_argument('--rotation-span', type=float, default=10)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError('Fresh candidate output required')
    if not (0 < a.scale_span <= 5 and 0 < a.position_span <= 5 and
            .001 <= a.scale_min < a.scale_max <= 20 and 1 <= a.max_iterations <= 2000 and
            0 < a.rotation_span <= 180):
        raise ValueError('Finite positive author bounds and 1..2000 iterations required')
    d, gs = read(a.geometry)
    c, cs = read(a.capture)
    obs, ms = read(a.measurements)
    trace, ts = read(a.trace)
    for name in ('reference', 'candidate'):
        if obs[name]['sha256'] != trace[name]['image_sha256'] or hashlib.sha256(Path(obs[name]['path']).read_bytes()).hexdigest() != obs[name]['sha256']:
            raise ValueError('Hash-bound measurement/trace mismatch')
    if c['capture']['path'] != obs['candidate']['path']:
        # Windows slash spelling may differ; identity still checked through file hash.
        if hashlib.sha256(Path(c['capture']['path']).read_bytes()).hexdigest() != obs['candidate']['sha256']:
            raise ValueError('Capture and fixed alignment image differ')
    rr = np.asarray(obs['reference']['landmarks_pixels'])
    gg = np.asarray(obs['candidate']['landmarks_pixels'])
    _, _, _, _, sr, Rr = measure(rr)
    _, _, _, _, sg, Rg = measure(gg)
    mr, mg = rr[[62, 66]].mean(0), gg[[62, 66]].mean(0)
    left = (np.asarray(trace['reference']['left_xy'])-mr) @ Rr.T/sr
    right = (np.asarray(trace['reference']['right_xy'])-mr) @ Rr.T/sr
    chin = (np.asarray(trace['reference']['chin_xy'])-mr) @ Rr.T/sr
    # Full below-mouth contour; fixed source-derived view and fixed eye/mouth anchors.
    # Horizontal sections cover the upper/mid jaw. The terminal section is
    # represented as Y at X, including its actual bottom, so an optimizer cannot
    # hide a broad flat end by ending just above the last horizontal sample.
    fractions = np.linspace(.10, .85, 30)
    levels = fractions*chin[1]
    target = np.c_[np.interp(levels, left[:, 1], left[:, 0]),
                   np.interp(levels, right[:, 1], right[:, 0])]
    terminal_x = np.linspace(chin[0]-.30, chin[0]+.30, 41)
    terminal_y = np.where(terminal_x <= chin[0],
                          np.interp(terminal_x, left[:,0], left[:,1]),
                          np.interp(terminal_x, right[::-1,0], right[::-1,1]))
    model = SourceForward(d, c['capture'], controls=a.controls)
    def normalized(x):
        return (model.pixels(x)-mg) @ Rg.T/sg
    def objective(x):
        pts = normalized(x)
        top = float(pts[:, 1].max())
        if not np.all(levels < top):
            raise ValueError('Candidate has no boundary on a required upper/mid section')
        residual = sections(pts, model.triangles, levels)-target
        terminal = sections(pts[:,[1,0]], model.triangles, terminal_x)[:,1]-terminal_y
        return np.r_[residual.ravel(), terminal, top-chin[1]]
    # Explicit author bounds, not native engine limits or compatibility thresholds.
    lower = [max(a.scale_min,v-a.scale_span) if field=='scale' else
             v-(a.rotation_span if field=='rotation' else a.position_span)
             for v,(_,field,_) in zip(model.base,model.parameter_fields)]
    upper = [min(a.scale_max,v+a.scale_span) if field=='scale' else
             v+(a.rotation_span if field=='rotation' else a.position_span)
             for v,(_,field,_) in zip(model.base,model.parameter_fields)]
    def appearance_cost(x):
        errors = objective(x)[:-1]
        return float(errors @ errors)
    # Preserve target terminal height as an explicit geometric constraint. It
    # cannot be traded against many mid-jaw samples to shorten a flat bottom.
    result = minimize(appearance_cost, model.base, method='SLSQP',
                      bounds=list(zip(lower,upper)),
                      constraints=[{'type':'eq','fun':lambda x: objective(x)[-1]}],
                      options={'maxiter':a.max_iterations,'ftol':1e-9,'eps':1e-5})
    x = result.x
    baked = np.c_[model.m['baked']['vertices'], np.ones(model.m['vertex_count'])]
    observed_head = (baked @ np.asarray(model.m['baked']['world_candidates']['scale_free_trs']['matrix']).reshape(4,4).T @ model.head_inv.T)[:, :3]
    baseline_error = np.linalg.norm(model.head_vertices(model.base)-observed_head, axis=1)
    def normals(v):
        tri = v[model.triangles]
        return np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    original_normals = normals(model.head_vertices(model.base))
    candidate_normals = normals(model.head_vertices(x))
    reversed_ids = np.flatnonzero(np.einsum('ij,ij->i',original_normals,candidate_normals)<0)
    patches = {n:{'name':n} for n in model.ids}
    for v,(name,field,axis) in zip(x,model.parameter_fields):
        patches[name].setdefault(field,model.mods[name][field].copy())[axis] = float(v)
        if field == 'position':
            # Include the actual required author branch in the replayable patch;
            # do not rely on a separate preparatory game mutation being remembered.
            patches[name]['length'] = 1.0
    patch = {'bones':list(patches.values())}
    out = {'status':'source-model author candidate only; requires actual game/complete appearance review, not accepted or locked',
           'inputs': {'geometry_sha256':gs, 'capture_sha256':cs, 'measurements_sha256':ms, 'trace_sha256':ts},
           'source_geometry_sha256': model.m['source_geometry_sha256'],
           'scope': __doc__, 'projection_scope':'Reported frozen head TRS with recorded FOV23 and 768x1024; excludes CTAA jitter/shader depth, arbitrary shear and general pose compatibility.',
           'parameters': ['.'.join([n,field,str(axis)]) for n,field,axis in model.parameter_fields],
           'base':model.base.tolist(), 'candidate':x.tolist(), 'author_bounds':[lower,upper],
           'author_bound_config': {'scale_span':a.scale_span,'position_span':a.position_span,'scale_min':a.scale_min,'scale_max':a.scale_max,'rotation_span_degrees':a.rotation_span,'max_iterations':a.max_iterations},
           'author_controls':a.controls,
           'reversed_triangle_normal_indices':reversed_ids.tolist(),
           'normal_check_scope':'Normals from actual source topology and proposed positions; a changed sign is a diagnostic, not a universal collision/visibility/animation certificate.',
           'optimizer_success':bool(result.success), 'optimizer_message':result.message,
           'terminal_height_constraint_residual_per_eye_distance':float(objective(x)[-1]),
           'candidate_all_target_rows_intersect_mesh': bool(normalized(x)[:,1].max() > levels.max()),
           'baseline_forward_vs_actual_bake_max_game_units':float(baseline_error.max()),
           'baseline_forward_vs_actual_bake_mean_game_units':float(baseline_error.mean()),
           'candidate_head_vertices':model.head_vertices(x).tolist(),
           'base_residual':objective(model.base).tolist(), 'candidate_residual':objective(x).tolist(),
           'target_levels_per_eye_distance':levels.tolist(), 'target_boundaries_per_eye_distance':target.tolist(),
           'target_terminal_x_per_eye_distance':terminal_x.tolist(),
           'target_terminal_y_per_eye_distance':terminal_y.tolist(),
           'residual_contract':'30 upper/mid left/right section errors, 41 full terminal bottom Y-at-X errors including true center, then actual bottom height; all in fixed eye-distance units. Missing required boundaries fail, no substitute contour.',
           'baseline_projected_vertices':normalized(model.base).tolist(),
           'candidate_projected_vertices':normalized(x).tolist(), 'patch':patch}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('x', encoding='utf-8') as f:
        json.dump(out,f,indent=2)
    print(json.dumps({k:out[k] for k in ['status','base','candidate','optimizer_success','patch']},indent=2))


if __name__ == '__main__':
    main()
