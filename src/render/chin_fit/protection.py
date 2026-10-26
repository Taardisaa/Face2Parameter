"""Feature guard measured on the candidate renderer's own skinned head streams.

Same rule and thresholds as ``chenger_reference_fit_20261012/feature_protection.py`` (handoff
4.2 item 3), evaluated on the bridge frame instead of the old CPU export:

* jaw-affected bone = one of the five chin bones or a descendant; feature bone = name contains
  Eye/eye/pupil/Nose/Ear/Mayu/Mouth;
* protected vertex = feature weight > 0.05, or cf_J_Head-local y > 0.25, or its front-view
  projection inside the baseline lip landmark box +-4 px; plus every vertex without jaw weight;
* o_head: protected vertices may move at most 2 px in every fitting view; o_namida (AIT/main
  namida full-face tear shell): protected vertices <= 1 px, its jaw part follows the native
  skinning; every other surface: all vertices <= 1 px;
* the minimum oriented-area ratio of o_head triangles, (n . n0) / |n0|^2 with n the 3-D
  triangle normal of the candidate and n0 of the baseline (guarded_predictor_fit.py rule),
  stays >= 0.2 (no folded or collapsed faces); evaluated in the front window.

Vertices come from the slot-0 skin stream the bridge draws (capture-start contents = frame N-2,
stride 40: position, normal, tangent, in the renderer's root-bone unscaled space) and go to world
space with that renderer's frozen root TRS from the bridge pre_render record; pixels use the
record's world-to-camera and non-jittered projection of bridge render 0.

One renderer per process (the EyeLook harness hosts Mono, which initialises once per process):
the masks are built from the front window (``build_masks``) and handed to every window's
``ViewGuard``; ``combine`` aggregates the per-window rows.
"""
from __future__ import annotations

import json

import numpy as np

from .controls import NAMES as JAW_NAMES

FEATURE_TOKENS = ('Eye', 'eye', 'pupil', 'Nose', 'Ear', 'Mayu', 'Mouth')
F = np.float32


def quat_matrix(q):
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def stream_positions(blob):
    a = np.frombuffer(blob, F).reshape(-1, 10)
    return a[:, 0:3].astype(float)


class ViewCamera:
    def __init__(self, record, roots):
        m = record['data']['camera_state']['matrices']
        self.w2c = np.asarray(m['worldToCameraMatrix'], float).reshape(4, 4)
        self.proj = np.asarray(m['nonJitteredProjectionMatrix'], float).reshape(4, 4)
        self.size = (1086, 1448)
        self.roots = roots          # renderer id -> (position, rotation xyzw)

    def world(self, renderer, local):
        p, q = self.roots[renderer]
        return local @ quat_matrix(q).T + np.asarray(p, float)

    def pixels(self, world):
        h = np.c_[world, np.ones(len(world))] @ (self.proj @ self.w2c).T
        ndc = h[:, :2] / h[:, 3:4]
        w, hgt = self.size
        return np.c_[(ndc[:, 0] + 1) * 0.5 * w, (1 - ndc[:, 1]) * 0.5 * hgt]


def bridge_camera(renderer):
    """ViewCamera of bridge render 0 of a BridgeCandidateRenderer window."""
    from src.render.hs2pipe import scene_v5 as sv
    cfg = renderer.cfg
    idx = renderer._window_index()
    recs = sv.resolve_records(json.loads((cfg.window / 'frames' / ('renders_%03d.json' % idx)).read_text(
        encoding='utf-8')), sv.prior_seed(cfg.window, idx))
    pr = sorted((r for r in recs if r['phase'] == 'pre_render' and r.get('bridge_render')),
                key=lambda r: r['bridge_render']['ordinal'])
    rec = pr[0]
    roots = {r['id']: (r['root_bone']['position'], r['root_bone']['rotation']) for r in rec['data']['renderers']
             if r['type'] == 'UnityEngine.SkinnedMeshRenderer' and r.get('root_bone')}
    return ViewCamera(rec, roots)


def slot0_chunks(renderer):
    """renderer id -> chunk of the skin stream the bridge draws (capture-start current = N-2)."""
    out = {}
    for p in renderer.points:
        if p['role'] == 'skinned_vertices_current' and p['capture_start']:
            out[p['renderer']] = p['chunk']
    return out


def renderer_names(r):
    names = {}
    for p in r.points:
        names[p['renderer']] = r.renderers[p['renderer']]['name']
    if len(set(names.values())) != len(names):
        raise RuntimeError('renderer names are not unique: %r' % sorted(names.values()))
    return names


def head_local_positions(r):
    """cf_J_Head-local positions of the baseline N-2 head/namida vertices (mask region only)."""
    from src.render.hs2pipe import frozen_pose as fpz
    fc = r.N - 2
    L = r.baseline['locals'][fc]
    parent = {}
    for t in r.tf:
        parent[t['id']] = r.tf[t['parent']]['id'] if t['parent'] >= 0 else r.by_id[t['id']]['parent_id']
    for tid in list(parent.values()):
        cur = tid
        while cur is not None and cur not in parent:
            parent[cur] = r.by_id[cur]['parent_id']
            cur = parent[cur]
    loc = {tid: (np.asarray(v['local_position'], F), np.asarray(v['local_rotation_xyzw'], F),
                 np.asarray(v['local_scale'], F)) for tid, v in L.items()}

    def world(tid):
        cols = fpz.global_matrix(r.emu, fpz.chain_of(loc, parent, tid))
        M = np.stack([np.asarray(c, float)[:4] for c in cols], axis=1)   # columns C0..C3 (affine)
        M[3] = (0.0, 0.0, 0.0, 1.0)
        return M

    head_id = next(t['id'] for t in r.tf if t['name'] == 'cf_J_Head')
    Hinv = np.linalg.inv(world(head_id))
    out = {}
    for p in r.points:
        if p['frame'] != fc or p['role'] != 'skinned_vertices_current':
            continue
        if r.renderers[p['renderer']]['name'] not in ('o_head', 'o_namida'):
            continue
        # unity_ObjectToWorld bytes are column-major: row-vector form v @ M
        O2W = np.frombuffer(r.baseline['object_matrices'][(p['renderer'], fc)][0], F).reshape(4, 4).astype(float)
        v = stream_positions(r.baseline['bytes'][p['chunk']])
        wpos = np.c_[v, np.ones(len(v))] @ O2W
        out[p['renderer']] = (wpos @ Hinv.T)[:, :3]
    if len(out) != 2:
        raise RuntimeError('head/namida N-2 skin streams not found')
    return out


def build_masks(r_front, front_lip_box):
    """Protected-vertex masks (by renderer name) from the front window's baseline streams."""
    names = renderer_names(r_front)
    tf = r_front.tf
    index_of = {t['id']: i for i, t in enumerate(tf)}
    jaw_index = {index_of[t['id']] for t in tf if t['name'] in JAW_NAMES}
    head_local = head_local_positions(r_front)
    cam = bridge_camera(r_front)
    chunks = slot0_chunks(r_front)
    lo, hi = np.asarray(front_lip_box[:2], float) - 4, np.asarray(front_lip_box[2:], float) + 4
    masks, roles = {}, {}
    for rid, name in names.items():
        rr = r_front.renderers[rid]
        bones = list(rr['bones'])
        jaw = np.zeros(len(bones), bool)
        feat = np.zeros(len(bones), bool)
        for k, b in enumerate(bones):
            j = b
            while j is not None and j >= 0:
                if j in jaw_index:
                    jaw[k] = True
                    break
                j = tf[j]['parent']
            feat[k] = any(tok in tf[b]['name'] for tok in FEATURE_TOKENS)
        mesh = r_front.win.meshes[rid]
        w = np.asarray(mesh['w'], float)
        bi = np.asarray(mesh['bi'], int)
        jw = np.sum(w * jaw[bi], axis=1)
        fw = np.sum(w * feat[bi], axis=1)
        if name in ('o_head', 'o_namida'):
            local = stream_positions(r_front.baseline['bytes'][chunks[rid]])
            px = cam.pixels(cam.world(rid, local))
            lips = np.all((px >= lo) & (px <= hi), axis=1)
            y = head_local[rid][:, 1]
            prot = (fw > 0.05) | (y > 0.25) | lips | (jw == 0)
            thr = 2.0 if name == 'o_head' else 1.0
            role = ('head: eyes/nose/ear/brow/upper face/lips protected' if name == 'o_head' else
                    'full-face tear shell: features/non-jaw protected, jaw part follows native skinning')
        else:
            prot = np.ones(len(w), bool)
            thr = 1.0
            role = 'other surface: unchanged'
        masks[name] = prot
        roles[name] = {'role': role, 'threshold_px': thr, 'vertices': int(len(prot)),
                       'protected': int(prot.sum()), 'jaw_following': int((~prot).sum()),
                       'jaw_weighted': int((jw > 0).sum())}
    return masks, roles


class ViewGuard:
    """Per-window part of the guard."""

    def __init__(self, r, masks, *, area_ratio=False):
        self.cam = bridge_camera(r)
        self.chunks = slot0_chunks(r)
        self.names = renderer_names(r)
        self.masks = {rid: np.asarray(masks[n], bool) for rid, n in self.names.items()}
        for rid, n in self.names.items():
            if len(self.masks[rid]) != len(stream_positions(r.baseline['bytes'][self.chunks[rid]])):
                raise RuntimeError('mask size differs from the stream of ' + n)
        self.head = next(rid for rid, n in self.names.items() if n == 'o_head')
        self.namida = next(rid for rid, n in self.names.items() if n == 'o_namida')
        self.base_px = {rid: self._pixels(r.baseline, rid) for rid in self.names}
        self.area_ratio = area_ratio
        if area_ratio:
            self.tri = np.asarray(r.win.meshes[self.head]['tri'], int).reshape(-1, 3)
            self.n0 = self._normals(r.baseline)
            self.area2 = np.sum(self.n0 * self.n0, axis=1)
            self.valid = self.area2 > 1e-18

    def _normals(self, streams):
        v = stream_positions(streams['bytes'][self.chunks[self.head]])
        t = v[self.tri]
        return np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])

    def _pixels(self, streams, rid):
        return self.cam.pixels(self.cam.world(rid, stream_positions(streams['bytes'][self.chunks[rid]])))

    def evaluate(self, streams):
        out = {'surfaces': {}, 'head_feature_max_px': 0.0, 'namida_feature_max_px': 0.0, 'other_max_px': 0.0,
               'namida_full_max_px': 0.0, 'head_full_max_px': 0.0}
        for rid, name in self.names.items():
            d = np.linalg.norm(self._pixels(streams, rid) - self.base_px[rid], axis=1)
            pm = float(d[self.masks[rid]].max()) if self.masks[rid].any() else 0.0
            out['surfaces'][name] = {'protected_max_px': pm, 'all_max_px': float(d.max())}
            if rid == self.head:
                out['head_feature_max_px'] = pm
                out['head_full_max_px'] = float(d.max())
            elif rid == self.namida:
                out['namida_feature_max_px'] = pm
                out['namida_full_max_px'] = float(d.max())
            else:
                out['other_max_px'] = max(out['other_max_px'], pm)
        if self.area_ratio:
            n = self._normals(streams)
            out['minimum_oriented_area_ratio'] = float(
                np.min(np.sum(n[self.valid] * self.n0[self.valid], axis=1) / self.area2[self.valid]))
        return out


def combine(rows):
    """Aggregate per-view guard rows; violation vector (head, namida, other, area ratio)."""
    keys = ('head_feature_max_px', 'namida_feature_max_px', 'other_max_px', 'namida_full_max_px', 'head_full_max_px')
    out = {k: max(r[k] for r in rows.values()) for k in keys}
    ratios = [r['minimum_oriented_area_ratio'] for r in rows.values() if 'minimum_oriented_area_ratio' in r]
    if len(ratios) != 1:
        raise RuntimeError('exactly one view evaluates the oriented-area ratio')
    out['minimum_oriented_area_ratio'] = ratios[0]
    out['violation'] = [max(0.0, out['head_feature_max_px'] - 2.0), max(0.0, out['namida_feature_max_px'] - 1.0),
                        max(0.0, out['other_max_px'] - 1.0), max(0.0, 0.2 - ratios[0])]
    out['feasible'] = not any(out['violation'])
    out['per_view'] = {v: {k: r[k] for k in r if k != 'surfaces'} for v, r in rows.items()}
    return out


def raycast_binding(r, px):
    """Bind an image point of window ``r`` to the o_head surface: the camera ray through pixel
    ``px`` (bridge render 0, non-jittered) intersected with every head triangle of the baseline
    slot-0 stream (Moeller-Trumbore in object space), nearest hit.  Returns triangle vertex ids
    and object-space barycentric weights (the step-4 binding form)."""
    cam, chunks, names = bridge_camera(r), slot0_chunks(r), renderer_names(r)
    head = next(rid for rid, n in names.items() if n == 'o_head')
    v = stream_positions(r.baseline['bytes'][chunks[head]])
    tri = np.asarray(r.win.meshes[head]['tri'], int).reshape(-1, 3)
    w, h = cam.size
    ndc = np.array([2 * px[0] / w - 1, 1 - 2 * px[1] / h])
    inv = np.linalg.inv(cam.proj @ cam.w2c)
    pts = [inv @ np.array([ndc[0], ndc[1], z, 1.0]) for z in (-1.0, 1.0)]
    a_w, b_w = (p[:3] / p[3] for p in pts)
    pos, q = cam.roots[head]
    R = quat_matrix(q)
    o = (a_w - np.asarray(pos, float)) @ R          # world -> root-bone object space (R orthonormal)
    d = (b_w - a_w) @ R
    d /= np.linalg.norm(d)
    p0, p1, p2 = v[tri[:, 0]], v[tri[:, 1]], v[tri[:, 2]]
    e1, e2 = p1 - p0, p2 - p0
    pv = np.cross(d, e2)
    det = np.sum(e1 * pv, axis=1)
    ok = np.abs(det) > 1e-14
    inv_det = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    tv = o - p0
    u = np.sum(tv * pv, axis=1) * inv_det
    qv = np.cross(tv, e1)
    vv = (qv @ d) * inv_det
    t = np.sum(e2 * qv, axis=1) * inv_det
    hit = ok & (u >= 0) & (vv >= 0) & (u + vv <= 1) & (t > 0)
    if not hit.any():
        raise ValueError('pixel %r does not hit the head surface' % (px,))
    k = int(np.argmin(np.where(hit, t, np.inf)))
    bary = [float(1 - u[k] - vv[k]), float(u[k]), float(vv[k])]
    return {'triangle_index': k, 'vertex_ids': [int(x) for x in tri[k]], 'barycentric': bary, 'ray_t': float(t[k])}
