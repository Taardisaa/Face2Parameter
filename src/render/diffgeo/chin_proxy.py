"""DIFFERENTIABLE PROXY of the chin objective -- not the chin objective.

The authoritative score is ``src/render/chin_fit/objective.py`` evaluated by the chin agent on
exact renders (BridgeCandidateRenderer + MediaPipe / image silhouettes).  This module only
provides a smooth surrogate with gradients for the hybrid strategy (optimise on the fast chain,
verify on the exact renderer).  It reuses the chin fit's frozen scoring frame and references
read-only (``domain.json`` and ``baseline_score.json`` of a chin fit directory: per view 2-D
similarity A, t; art chains and eye units; profile stations and normals) and replaces the two
image measurements by geometric stand-ins:

* front / oblique (MediaPipe jaw chains): each chain landmark id is bound once, on the exact
  baseline render, to a fixed point of the o_head surface -- the camera ray through the
  landmark's detected baseline position (``candidate_xy_render_px`` of the baseline score) hit
  on the window's exact baseline slot-0 stream (nearest hit), barycentric in object space; a
  ray that misses the head is bound to the nearest projected silhouette edge point (two
  vertices + parameter).  The proxy landmark of a candidate is the projection of that bound
  point.  A real detector re-locates contour landmarks as the jaw outline slides over the
  surface; the fixed binding does not (a stated proxy property).
* right / left profiles (image silhouette): the projected outer silhouette of o_head and the
  body (neck), selected each evaluation on the detached geometry (rasterised union of the
  projected triangles, mesh edges whose two sides fall inside / outside), then the signed
  distance along each fixed art station normal to the nearest silhouette edge crossing, as in
  ``objective.profile_offsets``; differentiable through the crossing edge's endpoints.  The
  image silhouette also contains the other drawn surfaces (eyelashes, tear shell, clothes) and
  anti-aliasing; the proxy does not.

Geometry per view: the view window's own exact baseline slot-0 stream (probe export skinned by
``paired_skin.unity_exact_vertices``) plus the torch chain's change from its baseline on the
geometry session's window (``DiffGeoSession``, normally the front window); the view window's
own frozen root of bridge render 0 (its PoseFreeze write set and pre_cull locals, the torch
N-1 locals elsewhere -- checked bit-exact against the view's render record at baseline); the
view's record camera (worldToCamera, non-jittered projection of render 0).  The window
difference of the root-relative stream enters only through the constant baseline term.

Baseline anchoring (default): a constant per landmark / station makes the proxy equal the exact
objective's own measurement at the fit's encoded baseline (q = 0); away from it the proxy adds
only the geometric change of the bound points / silhouette.

Units and cost follow the objective: residual = art px / eye unit / sqrt(n) per view, cost =
sum of squares (no guard penalty; the feature guard is not part of the proxy).
"""
from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import unity as U
from .fp import DT

F = np.float32
VIEWS = ('front', 'oblique', 'right', 'left')
SIZE = (1086, 1448)


def _read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


@dataclass
class ProxyView:
    view: str
    measure: str
    w2c: np.ndarray
    proj: np.ndarray
    A: np.ndarray
    t: np.ndarray
    eye_unit: float
    chain_ids: list = None
    chain_xy: np.ndarray = None
    stations: np.ndarray = None
    normals: np.ndarray = None
    bindings: list = None
    head_offset: torch.Tensor = None           # view exact baseline VB - session torch baseline VB
    frozen: dict = None
    body_world: np.ndarray = None
    body_tri: np.ndarray = None
    notes: dict = field(default_factory=dict)
    correction: torch.Tensor = None             # baseline anchoring (see ChinProxy._anchor)

    def pixels(self, world):
        """world (n, 3) torch -> render px (n, 2), the record camera of bridge render 0."""
        M = torch.as_tensor(self.proj @ self.w2c, dtype=DT)
        h = world @ M[:, :3].T + M[:, 3]
        ndc = h[:, :2] / h[:, 3:4]
        w, hgt = SIZE
        return torch.stack([(ndc[:, 0] + 1) * 0.5 * w, (1 - ndc[:, 1]) * 0.5 * hgt], 1)

    def to_art(self, px):
        return px @ torch.as_tensor(self.A, dtype=DT).T + torch.as_tensor(self.t, dtype=DT)


def _window_frozen(win_dir, session_dir, nonce, tf, N):
    """PoseFreeze write set and pre_cull locals of bridge render 0 of a window (probe v7)."""
    from src.render.hs2pipe import scene_v5 as sv
    from src.render.paired_skin import ProbeWindow
    idx = sorted(json.loads((win_dir / 'capture_receipt.json').read_text(encoding='utf-8'))['frame_counts_by_index']).index(N)
    recs = sv.resolve_records(_read(win_dir / 'frames' / ('renders_%03d.json' % idx)), sv.prior_seed(win_dir, idx))
    pc = {r['bridge_render']['ordinal']: r for r in recs if r['phase'] == 'pre_cull' and r.get('bridge_render')}
    pr = {r['bridge_render']['ordinal']: r for r in recs if r['phase'] == 'pre_render' and r.get('bridge_render')}
    arr = np.frombuffer(base64.b64decode(pc[0]['data']['character_transforms']['transforms_f32']), F).reshape(-1, 33)
    anim = _read(session_dir / ('pose_clip_bindings_%s.json' % nonce))['animators']
    written = {tf[i]['id'] for a in anim for i in a['transforms_written_by_current_state']}
    return {'record': pr[0], 'arrays': arr, 'written': written}


class ChinProxy:
    """Proxy cost and gradient on the chin views (see module doc)."""

    def __init__(self, session, fit_dir, windows_config, views=VIEWS, anchor_to_exact_baseline=True):
        from src.render import paired_skin as ps
        from src.render.hs2pipe.frame_skin import ancestor_chain
        from src.render.chin_fit import objective as obj
        self.s = session
        self.fit_dir = Path(fit_dir)
        dom = _read(self.fit_dir / 'domain.json')
        base = _read(self.fit_dir / 'baseline_score.json')
        cfg = _read(windows_config)
        self.views = [v for v in VIEWS if v in views]
        s = session
        names = {s.renderers[r]['name']: r for r in s.mesh_renderers}
        self.head = names['o_head']
        self.head_tri = np.asarray(s.win.meshes[self.head]['tri'], np.int64).reshape(-1, 3)
        self.edges = np.unique(np.sort(np.concatenate([self.head_tri[:, [0, 1]], self.head_tri[:, [1, 2]],
                                                         self.head_tri[:, [2, 0]]]), axis=1), axis=0)
        bl = s.torch_streams(s.window_patch, frames=(s.N - 2,), rids=[self.head])
        self.session_baseline_vb = bl['skin'][s.N - 2][self.head]['vb'].detach()
        self.pv = {}
        for v in self.views:
            d = dom['views'][v]
            vc = cfg['views'][v]
            win_dir = Path(cfg['bulk']) / ('window_' + vc['nonce'])
            wv = ps.ProbeWindow(win_dir) if vc['nonce'] != s.cfg.nonce else s.win
            if [t['id'] for t in wv.static['transforms']] != [t['id'] for t in s.tf]:
                raise ValueError('view window %s has another transform layout than the geometry session' % v)
            Nv = _read(win_dir / 'capture_receipt.json')['expected_capture_frame']
            fz = _window_frozen(win_dir, Path(cfg['bulk']), vc['nonce'], s.tf, Nv)
            rec = fz['record']['data']
            cam = rec['camera_state']['matrices']
            p = ProxyView(v, d['measure'], np.asarray(cam['worldToCameraMatrix'], float).reshape(4, 4),
                          np.asarray(cam['nonJitteredProjectionMatrix'], float).reshape(4, 4),
                          np.asarray(d['A'], float), np.asarray(d['t'], float),
                          float(dom['references'][v]['eye_unit_px']) if 'eye_unit_px' in dom['references'][v]
                          else float(obj.ReferenceView.load(v, verify_images=False).eye_unit_px))
            p.frozen = fz
            rid_rows = {r['id']: r for r in rec['renderers']}
            p.notes['record_root_head'] = rid_rows[self.head]['root_bone']
            # the view's exact baseline slot-0 head stream (its own probe export, N-2)
            anc = ancestor_chain(wv, Path(cfg['bulk']) / 'prepare_geometry_actor.json')
            emu = s.emu
            vb_exact = np.frombuffer(bytes(ps.unity_exact_vertices(emu, wv, wv.by_frame[Nv - 2], wv.static['renderers'][
                [r['id'] for r in wv.static['renderers']].index(self.head)], wv.meshes[self.head], anc)), F).reshape(-1, 10)
            p.head_offset = torch.as_tensor(vb_exact.astype(np.float64)) - self.session_baseline_vb
            p.notes['root_relative_stream_window_difference_max'] = float(p.head_offset.abs().max())
            sc = base['views'][v]
            if p.measure == 'landmarks':
                p.chain_ids = list(sc['point_ids'])
                p.chain_xy = np.asarray(sc['reference_xy_art_px'], float)
                p.bindings = [self._bind(p, vb_exact, xy) for xy in sc['candidate_xy_render_px']]
            else:
                st = dom['profile_stations'][v]
                p.stations = np.asarray(st['stations_art_px'], float)
                p.normals = np.asarray(st['normals'], float)
                bodies = [r['id'] for r in wv.static['renderers'] if r['name'] == 'o_body_cf']
                if len(bodies) != 1:
                    raise ValueError('one o_body_cf renderer required in view ' + v)
                body = bodies[0]
                vb_body = np.frombuffer(bytes(ps.unity_exact_vertices(emu, wv, wv.by_frame[Nv - 2], wv.static['renderers'][
                    [r['id'] for r in wv.static['renderers']].index(body)], wv.meshes[body], anc)), F).reshape(-1, 10)
                rb = rid_rows[body]['root_bone']          # body: parameter-independent, exact record root
                O = np.eye(4)
                O[:3, :3] = _quat_matrix(rb['rotation'])
                O[:3, 3] = rb['position']
                p.body_world = vb_body[:, :3].astype(float) @ O[:3, :3].T + O[:3, 3]
                p.body_tri = np.asarray(wv.meshes[body]['tri'], np.int64).reshape(-1, 3)
            self.pv[v] = p
            if wv is not s.win:
                del wv
        self.check_baseline()
        self.anchored = False
        if anchor_to_exact_baseline:
            self._anchor(base)

    def _anchor(self, base):
        """Baseline anchoring: a constant per landmark (art px, 2-D) / per station (px) so that the
        proxy equals the exact objective's measurement at the fit's encoded baseline (q = 0, the
        card); the proxy then contributes only the geometric change from that baseline."""
        rows = _read(self.fit_dir / 'encoding.json')['rows']
        with torch.no_grad():
            _, _, meas = self._measure({'bones': rows})
        for v, p in self.pv.items():
            sc = base['views'][v]
            if p.measure == 'landmarks':
                p.correction = torch.as_tensor(np.asarray(sc['candidate_xy_art_px'], float), dtype=DT) - meas[v]
            else:
                p.correction = torch.as_tensor(np.asarray(sc['offset_px'], float), dtype=DT) - meas[v]
            p.notes['baseline_anchor_max_px'] = float(p.correction.abs().max())
        self.anchored = True

    # ------------------------------------------------------------------ geometry
    def head_world(self, p, vb, L_prev):
        """World positions of the o_head slot-0 stream in view p (frozen root of render 0)."""
        s = self.s
        fz = p.frozen
        parent = s._parent_map()
        tid = s.tf[s.renderers[self.head]['root_bone']]['id']
        chain = []
        A = s.A
        while tid is not None:
            i = s.index_of.get(tid)
            if i is not None and tid in fz['written']:
                arr = fz['arrays']
                chain.append((A.f32(arr[i, 0:3]), A.f32(arr[i, 3:7]), A.f32(arr[i, 7:10])))
            else:
                r = L_prev[tid]
                chain.append((r['pos'], r['rot'], r['scl']))
            tid = parent[tid]
        pos, rot = U.global_position_rotation(A, chain)
        o2w = U.object_to_world(A, pos, rot)
        v = vb[:, 0:3] + p.head_offset[:, 0:3]
        return v @ o2w[:3, :3].T + o2w[:3, 3], (pos, rot)

    def _bind(self, p, vb_exact, xy):
        """Fixed o_head surface point under the baseline landmark pixel (exact baseline stream)."""
        rr = p.notes['record_root_head']
        R = _quat_matrix(rr['rotation'])
        pos = np.asarray(rr['position'], float)
        v = vb_exact[:, :3].astype(float)
        w, h = SIZE
        ndc = np.array([2 * xy[0] / w - 1, 1 - 2 * xy[1] / h])
        inv = np.linalg.inv(p.proj @ p.w2c)
        a_w, b_w = ((inv @ np.array([ndc[0], ndc[1], z, 1.0]))[:3] / (inv @ np.array([ndc[0], ndc[1], z, 1.0]))[3]
                    for z in (-1.0, 1.0))
        o = (a_w - pos) @ R
        d = (b_w - a_w) @ R
        d /= np.linalg.norm(d)
        tri = self.head_tri
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
        tt = np.sum(e2 * qv, axis=1) * inv_det
        hit = ok & (u >= 0) & (vv >= 0) & (u + vv <= 1) & (tt > 0)
        if hit.any():
            k = int(np.argmin(np.where(hit, tt, np.inf)))
            return {'kind': 'surface', 'vertex_ids': [int(x) for x in tri[k]],
                    'weights': [float(1 - u[k] - vv[k]), float(u[k]), float(vv[k])]}
        # nearest point of a projected silhouette edge
        world = v @ R.T + pos
        px = (torch.as_tensor(world) @ torch.as_tensor(p.proj @ p.w2c)[:, :3].T
              + torch.as_tensor(p.proj @ p.w2c)[:, 3]).numpy()
        px = np.stack([(px[:, 0] / px[:, 3] + 1) * 0.5 * w, (1 - px[:, 1] / px[:, 3]) * 0.5 * h], 1)
        sil = self._outer_edges(px, self.head_tri, self.edges)
        a, b = px[sil[:, 0]], px[sil[:, 1]]
        e = b - a
        tpar = np.clip(np.sum((np.asarray(xy) - a) * e, 1) / np.maximum(np.sum(e * e, 1), 1e-12), 0, 1)
        dist = np.linalg.norm(a + tpar[:, None] * e - np.asarray(xy), axis=1)
        k = int(np.argmin(dist))
        return {'kind': 'silhouette_edge', 'vertex_ids': [int(sil[k, 0]), int(sil[k, 1])],
                'weights': [float(1 - tpar[k]), float(tpar[k])], 'baseline_distance_px': float(dist[k])}

    @staticmethod
    def _outer_edges(px, tri, edges, probe=1.5, base_mask=None):
        """Mesh edges on the outer boundary of the projected union of triangles (detached);
        ``base_mask``: an already rasterised constant part of the union (uint8)."""
        import cv2
        w, h = SIZE
        img = np.zeros((h, w), np.uint8) if base_mask is None else base_mask.copy()
        poly = np.round(px[tri] * 16).astype(np.int32)          # 4 fractional bits (shift=4)
        # one call per triangle: the union (a multi-contour fillPoly would apply the even-odd rule)
        for t in poly:
            cv2.fillConvexPoly(img, t, 1, lineType=cv2.LINE_8, shift=4)
        mask = img.astype(bool)
        a, b = px[edges[:, 0]], px[edges[:, 1]]
        e = b - a
        n = np.stack([-e[:, 1], e[:, 0]], 1) / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-9)
        mid = 0.5 * (a + b)

        def inside(q):
            x = np.clip(np.round(q[:, 0]).astype(int), 0, w - 1)
            y = np.clip(np.round(q[:, 1]).astype(int), 0, h - 1)
            ok = (q[:, 0] >= 0) & (q[:, 0] < w) & (q[:, 1] >= 0) & (q[:, 1] < h)
            return ok & mask[y, x]
        s1, s2 = inside(mid + probe * n), inside(mid - probe * n)
        return edges[s1 != s2]

    def check_baseline(self):
        """At the window patch: torch frozen root == each view's render record (bit for bit)."""
        s = self.s
        tr = s.torch_streams(s.window_patch, frames=(s.N - 2, s.N - 1), rids=[self.head])
        L1 = tr['locals'][s.N - 1]
        for v, p in self.pv.items():
            _, (pos, rot) = self.head_world(p, tr['skin'][s.N - 2][self.head]['vb'], L1)
            rr = p.notes['record_root_head']
            ok = (np.asarray(pos.detach().numpy(), F).tobytes() == np.asarray(rr['position'], F).tobytes()
                  and np.asarray(rot.detach().numpy(), F).tobytes() == np.asarray(rr['rotation'], F).tobytes())
            p.notes['frozen_root_equals_record'] = bool(ok) if s.A.exact else 'not checked (float64 mode)'
            if s.A.exact and not ok:
                raise RuntimeError('frozen root of view %s differs from its render record' % v)

    # ------------------------------------------------------------------ residuals
    def _measure(self, patch, views=None):
        """Per view: landmark art px (n, 2) or station offsets (n,), before baseline anchoring."""
        s = self.s
        views = [v for v in (views or self.views)]
        tr = s.torch_streams(patch, frames=(s.N - 2, s.N - 1), rids=[self.head])
        vb = tr['skin'][s.N - 2][self.head]['vb']
        L1 = tr['locals'][s.N - 1]
        meas = {}
        for v in views:
            p = self.pv[v]
            world, _ = self.head_world(p, vb, L1)
            px = p.pixels(world)
            if p.measure == 'landmarks':
                pts = []
                for b in p.bindings:
                    ids = torch.as_tensor(b['vertex_ids'])
                    wts = torch.as_tensor(b['weights'], dtype=DT)
                    pts.append((px[ids] * wts[:, None]).sum(0))
                meas[v] = p.to_art(torch.stack(pts))
            else:
                meas[v] = self._silhouette(p, px)
        return views, tr, meas

    def residuals(self, patch, views=None):
        views, tr, meas = self._measure(patch, views)
        out, rep = {}, {}
        for v in views:
            p = self.pv[v]
            m = meas[v] + (p.correction if self.anchored else 0.0)
            if p.measure == 'landmarks':
                diff = m - torch.as_tensor(p.chain_xy, dtype=DT)
                n = len(p.chain_ids)
                out[v] = (diff / p.eye_unit / np.sqrt(n)).reshape(-1)
                rep[v] = {'rms_px': float(torch.sqrt((diff.detach() ** 2).sum(1).mean()))}
            else:
                n = len(p.stations)
                out[v] = m / p.eye_unit / np.sqrt(n)
                rep[v] = {'rms_px': float(torch.sqrt((m.detach() ** 2).mean()))}
        return out, rep, tr

    def _silhouette(self, p, px_head):
        px_body = p.pixels(torch.as_tensor(p.body_world, dtype=DT)).detach()
        nh = px_head.shape[0]
        allpx = torch.cat([px_head, px_body])
        if not hasattr(p, '_edges'):
            be = p.body_tri
            bed = np.unique(np.sort(np.concatenate([be[:, [0, 1]], be[:, [1, 2]], be[:, [2, 0]]]), axis=1), axis=0)
            p._edges = np.concatenate([self.edges, bed + nh])
        if not hasattr(p, '_body_mask'):
            import cv2
            w, h = SIZE
            p._body_mask = np.zeros((h, w), np.uint8)
            for t in np.round(px_body.numpy()[p.body_tri] * 16).astype(np.int32):
                cv2.fillConvexPoly(p._body_mask, t, 1, lineType=cv2.LINE_8, shift=4)
        sil = self._outer_edges(allpx.detach().numpy(), self.head_tri, p._edges, base_mask=p._body_mask)
        art = p.to_art(allpx)
        a, b = art[sil[:, 0]], art[sil[:, 1]]
        e = b - a
        st = torch.as_tensor(p.stations, dtype=DT)
        nv = torch.as_tensor(p.normals, dtype=DT)
        offs = []
        for k in range(len(st)):
            pk, nk = st[k], nv[k]
            den = nk[0] * (-e[:, 1]) - nk[1] * (-e[:, 0])
            rx, ry = a[:, 0] - pk[0], a[:, 1] - pk[1]
            safe = torch.where(den.abs() > 1e-12, den, torch.ones_like(den))
            u = (rx * (-e[:, 1]) - ry * (-e[:, 0])) / safe
            wpar = (nk[0] * ry - nk[1] * rx) / safe
            hit = (den.abs() > 1e-12) & (wpar >= 0) & (wpar <= 1) & (u.abs() <= 120.0)
            if not bool(hit.any()):
                raise ValueError('proxy silhouette: station %d of %s has no crossing' % (k, p.view))
            idx = torch.nonzero(hit).view(-1)
            j = idx[torch.argmin(u[idx].abs())]
            offs.append(u[j])
        return torch.stack(offs)

    def loss(self, patch, views=None):
        r, rep, tr = self.residuals(patch, views)
        total = sum((x ** 2).sum() for x in r.values())
        return total, rep, tr


def _quat_matrix(q):
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


# ---------------------------------------------------------------------- q parameterisation
def patch_from_q_torch(q, base_rows):
    """Torch twin of ``chin_fit.controls.patch_from_q`` (same linear map, differentiable in q)."""
    from src.render.chin_fit import controls as C
    if q.shape != (len(C.CONTROLS),):
        raise ValueError('complete control vector required')
    rows = {r['name']: {k: (torch.as_tensor(np.asarray(r[k], float), dtype=DT) if k in ('scale', 'position', 'rotation')
                            else r[k]) for k in r} for r in base_rows}
    for i, c in enumerate(C.CONTROLS):
        for j, name in enumerate(c['names']):
            ch, ax = c['channel'], c['axis']
            sign = -1 if j and ((ch == 'position' and ax == 0) or (ch == 'rotation' and ax in (1, 2))) else 1
            vec = rows[name][ch]
            onehot = torch.zeros(3, dtype=DT)
            onehot[ax] = 1.0
            if ch == 'scale':
                rows[name][ch] = vec * (1 + onehot * q[i] * c['range'])
            else:
                rows[name][ch] = vec + onehot * (sign * q[i] * c['range'])
    out = [rows[n] for n in C.NAMES]
    if any(r['length'] != 1.0 for r in out):
        raise RuntimeError('persistent length-1 branch lost')
    return {'bones': out}


def loss_and_grad(proxy, patch, views=None):
    """PROXY cost and its gradient w.r.t. every tensor value of ``patch`` that requires grad.

    patch: bridge patch dict ``{'bones': [...], 'native59': ...}``; values that are tensors with
    ``requires_grad`` receive gradients (``.grad``).  Returns (cost float, {path: grad}, report)."""
    leaves = {}
    for b in patch.get('bones') or ():
        for k, v in b.items():
            if isinstance(v, torch.Tensor) and v.requires_grad:
                leaves['%s.%s' % (b['name'], k)] = v
    nat = patch.get('native59')
    if isinstance(nat, torch.Tensor) and nat.requires_grad:
        leaves['native59'] = nat
    total, rep, _ = proxy.loss(patch, views)
    grads = torch.autograd.grad(total, list(leaves.values()), allow_unused=True) if leaves else []
    return float(total), {k: (None if g is None else g.detach()) for k, g in zip(leaves, grads)}, rep
