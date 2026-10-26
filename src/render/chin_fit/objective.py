"""Chin scoring domain: renderer output vs the approved Chenger art, view by view.

All measurements are taken on the candidate's rendered bridge target (``BridgeCandidateRenderer``
output, 1086x1448, top-down) of the window whose yaw matches the art view, and on the art image
with the same procedure.  Nothing of the candidate geometry is projected in place of an image
measurement; the mesh is used only for the frozen comparison frame and the feature guard.

Front (yaw 0) and three-quarter right (yaw +30): the user-approved step-2 master, i.e. the raw
MediaPipe jaw chains on the art (front both sides, 21 points; three-quarter near side, 11 points;
``raw_reference_chain_xy`` = art landmarks[``displayed_chain_ids``] of the step-4 receipts,
checked here) against the same model / options / ids on the render.  On these views the detector
is stable (8 seeds of +-1 level image noise move the chain by 0.5-0.7 px mean, 1.1-1.5 px max).

Profiles (yaw +90 / -90): MediaPipe is not a usable measurement there -- the same +-1 level noise
moves the chain by 13-16 px mean (up to 32 px) and the left profile loses the face in 5 of 8
seeds.  The profile is measured as the face silhouette against the plain background on both the
art and the render (``silhouette_contours``): fixed art stations every 6 px of arclength from the
lower lip past the chin and along the under-chin to the throat junction (todo item 7: lower lip
-> labiomental fold -> chin front -> chin bottom -> under-chin); residual = signed distance along
each art station normal to the render silhouette.  The internal jaw-angle line of a profile is
not on the silhouette and is not measured in these views (it is measured by the front and
three-quarter chains).

Comparison frame (frozen on the baseline, never refitted per candidate): 2-D similarity from the
step-4 mesh-bound upper-face anchors (eye corners, nose, forehead / temple surface bindings,
``*_anchor_bindings.json``) projected through the renderer's own skin stream and bridge camera,
to the art's detections of the same ids, the three groups weighted equally (the eye-fixed rule
of ``chenger_eye_head_frame_20261011`` on the step-4 anchor set).  Profiles keep that rotation
and scale and are translated to the fixed lower lip (todo item 7): the art and render lower-lip
silhouette points coincide.  Chin, jaw and cheek points never enter a frame.

Residual units: art px / the view's eye unit (``reference_eye_unit_pixels``) / sqrt(n), i.e. each
view contributes its RMS in eye units (the weighting of ``predictor_objective.py``).

Stated gaps: the art views are generated images without camera metadata.  The rendered yaws
(0, +30, +90, -90) are the bridge yaws nearest the step-4 observation estimates
(-2.37, 25.06, 83.58, -85.36 deg yaw; pitch -5.3 / -5.4 / -1.0 / +3.9 deg).  The bridge orbits
at pitch 0 and cannot reproduce those estimates; the remaining out-of-plane view difference is
left in the residual, not compensated.  The silhouette level (12 RGB units above the row
background) is a fixed contract on both images; anti-aliasing places it within a pixel of the
edge on both.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HS2MOD = Path('C:/Users/13666/Workspace/HS2Mod')
ALIGN = HS2MOD / 'artifacts/chenger_pose_alignment_step4_20261011'
ART_LANDMARKS = HS2MOD / 'artifacts/chenger_game_landmarks_step3_20261011'
EYE_FRAME = HS2MOD / 'artifacts/chenger_eye_head_frame_20261011'
MODEL = HS2MOD / 'tools/face_reference_landmarks/face_landmarker.task'
MODEL_SHA256 = '64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff'
OPTIONS = {'running_mode': 'IMAGE', 'num_faces': 1, 'output_facial_transformation_matrixes': True,
           'min_face_detection_confidence': 0.1, 'min_face_presence_confidence': 0.1}
VIEWS = ('front', 'oblique', 'right', 'left')
# bridge yaw of the render window used for each art view (camera orbit Quaternion.Euler(0, yaw, 0))
VIEW_YAW = {'front': 0.0, 'oblique': 30.0, 'right': 90.0, 'left': -90.0}
ANCHOR_GROUPS = {'eyes': (33, 133, 263, 362), 'nose': (168, 6, 197, 1, 2, 98, 327), 'upper': (10, 151, 127, 356)}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


class PredictionUnavailable(ValueError):
    """The fixed detector contract gives no single 478-point face (domain rejection)."""

    def __init__(self, view, faces):
        self.view, self.faces = view, faces
        super().__init__('%s: %d faces under the fixed detector contract' % (view, faces))


@dataclass
class ReferenceView:
    view: str
    art_path: str
    art_sha256: str
    chain_ids: list
    chain_xy: np.ndarray            # art pixels
    anchor_ids: list
    art_landmarks: np.ndarray       # 478 x 3 art pixels
    eye_unit_px: float
    step4_orientation: dict

    @classmethod
    def load(cls, view, verify_images=True):
        r = _read(ALIGN / (view + '_alignment_receipt.json'))
        art = _read(ART_LANDMARKS / (view + '_art_common_raw.json'))
        frame = _read(EYE_FRAME / (view + '_eye_frame.json'))
        if art['source']['sha256'] != r['art_source']['sha256']:
            raise ValueError('art landmark record and alignment receipt name different images: ' + view)
        if verify_images and sha256(r['art_source']['path']) != r['art_source']['sha256']:
            raise ValueError('approved art image changed: ' + r['art_source']['path'])
        L = np.asarray(art['landmarks'], float)
        ids = list(r['displayed_chain_ids'])
        chain = np.asarray(r['raw_reference_chain_xy'], float)
        if not np.array_equal(L[ids, :2], chain):
            raise ValueError('stored reference chain is not the art detection: ' + view)
        if (art['options']['min_face_detection_confidence'] != OPTIONS['min_face_detection_confidence']
                or art['options']['min_face_presence_confidence'] != OPTIONS['min_face_presence_confidence']):
            raise ValueError('art detection options differ from the scoring contract: ' + view)
        return cls(view, r['art_source']['path'], r['art_source']['sha256'], ids, chain, list(r['upper_anchor_ids']),
                   L, float(frame['reference_eye_unit_pixels']), r.get('observation_orientation'))


class FixedDetector:
    """MediaPipe FaceLandmarker, IMAGE mode (stateless), fixed model/options and a fixed
    unscaled detector input rectangle per view."""

    def __init__(self, model=MODEL):
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        if sha256(model) != MODEL_SHA256:
            raise ValueError('MediaPipe model changed')
        self.mp = mp
        opts = vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
            running_mode=vision.RunningMode.IMAGE, num_faces=OPTIONS['num_faces'],
            min_face_detection_confidence=OPTIONS['min_face_detection_confidence'],
            min_face_presence_confidence=OPTIONS['min_face_presence_confidence'],
            output_facial_transformation_matrixes=OPTIONS['output_facial_transformation_matrixes'])
        self.det = vision.FaceLandmarker.create_from_options(opts)

    def close(self):
        self.det.close()

    def detect(self, rgb, rect, view='?'):
        rgb = np.asarray(rgb)
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            raise ValueError('RGB(A) image required')
        x0, y0, x1, y1 = rect
        crop = np.ascontiguousarray(rgb[y0:y1, x0:x1, :3].astype(np.uint8))
        res = self.det.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=crop))
        if len(res.face_landmarks) != 1 or len(res.face_landmarks[0]) != 478:
            raise PredictionUnavailable(view, len(res.face_landmarks))
        w, h = x1 - x0, y1 - y0
        pts = np.asarray([[p.x * w + x0, p.y * h + y0, p.z * w] for p in res.face_landmarks[0]], float)
        if not np.isfinite(pts).all():
            raise ValueError('non-finite detector output')
        return pts


def select_rect(detector, rgb, view, step=50):
    """Detector input rectangle of a view, chosen once on the *baseline* render and then fixed
    for every candidate (step-4 ``detection_windows.json`` rule): the full frame when the
    contract finds one face there; otherwise the largest-area unscaled window on a 50-px grid
    (widths 1086..600, heights 1448..700) that does.  Every tried window is returned."""
    h, w = np.asarray(rgb).shape[:2]
    tried = []
    try:
        detector.detect(rgb, (0, 0, w, h), view)
        return [0, 0, w, h], [{'rect': [0, 0, w, h], 'faces': 1}]
    except PredictionUnavailable as e:
        tried.append({'rect': [0, 0, w, h], 'faces': e.faces})
    ok = []
    for ww in (1086, 1000, 900, 800, 700, 600):
        for hh in (1448, 1300, 1150, 1000, 900, 800, 700):
            if ww > w or hh > h:
                continue
            for x0 in range(0, w - ww + 1, step):
                for y0 in range(0, h - hh + 1, step):
                    r = (x0, y0, x0 + ww, y0 + hh)
                    try:
                        detector.detect(rgb, r, view)
                        ok.append((ww * hh, -x0, -y0, list(r)))
                        tried.append({'rect': list(r), 'faces': 1})
                    except PredictionUnavailable as e:
                        tried.append({'rect': list(r), 'faces': e.faces})
    if not ok:
        raise PredictionUnavailable(view, 0)
    ok.sort(reverse=True)
    return ok[0][3], tried


# ---------------------------------------------------------------------------- profile silhouette
SILHOUETTE_LEVEL = 12.0       # RGB distance from the row background (background noise <= 3, skin >= 30)
BG_STRIP = 24                 # columns at the image border on the face side (background in both images)
STATION_STEP = 6.0            # art px of arclength between stations
FACE_SIDE = {'right': 1, 'left': -1}


def silhouette_contours(rgb, side):
    """Foreground/background iso-contours of a profile image: D(x, y) = |rgb - bg(y)| with bg(y)
    the per-row median of the BG_STRIP border columns on the face side; marching squares at
    SILHOUETTE_LEVEL (skimage.measure.find_contours).  Returns a list of (n, 2) xy arrays."""
    from skimage import measure
    im = np.asarray(rgb)[..., :3].astype(float)
    strip = im[:, -BG_STRIP:] if side > 0 else im[:, :BG_STRIP]
    bg = np.median(strip, axis=1)
    D = np.linalg.norm(im - bg[:, None, :], axis=2)
    return [c[:, ::-1].copy() for c in measure.find_contours(D, SILHOUETTE_LEVEL)]


def _arclength(p):
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]


def profile_stations(art_rgb, art_landmarks, side, step=STATION_STEP):
    """Fixed art stations along the lower-face profile silhouette: from the contour point nearest
    the lower-lip landmark 17 down the chin front, past the chin (152), along the under-chin to
    the throat junction (the first contour point after 152 whose tangent is within 35 deg of
    straight down, i.e. the neck front).  Returns stations (n, 2) and unit normals pointing
    away from the face (into the background)."""
    L = np.asarray(art_landmarks, float)
    cs = silhouette_contours(art_rgb, side)
    lip, chin = L[17, :2], L[152, :2]
    best = min(cs, key=lambda c: np.min(np.linalg.norm(c - lip, axis=1)) + np.min(np.linalg.norm(c - chin, axis=1)))
    i0 = int(np.argmin(np.linalg.norm(best - lip, axis=1)))
    i1 = int(np.argmin(np.linalg.norm(best - chin, axis=1)))
    d = 1 if i1 > i0 else -1
    path = best[i0::d] if d > 0 else best[i0::-1]
    k152 = abs(i1 - i0)
    tang = np.gradient(path, axis=0)
    tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-12)
    # smooth tangent over +-5 samples to ignore pixel-scale wiggles of the iso-line
    ker = np.ones(11) / 11
    ts = np.stack([np.convolve(tang[:, 0], ker, 'same'), np.convolve(tang[:, 1], ker, 'same')], axis=1)
    ts /= np.maximum(np.linalg.norm(ts, axis=1, keepdims=True), 1e-12)
    down = np.nonzero((np.arange(len(path)) > k152) & (ts[:, 1] > np.cos(np.radians(35))))[0]
    if not len(down):
        raise ValueError('throat junction not found on the art silhouette')
    seg = path[:down[0] + 1]
    s = _arclength(seg)
    q = np.arange(0.0, s[-1] + 1e-9, step)
    st = np.stack([np.interp(q, s, seg[:, 0]), np.interp(q, s, seg[:, 1])], axis=1)
    t = np.gradient(st, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    n = np.stack([-t[:, 1], t[:, 0]], axis=1)
    # orient normals into the background: the face-side direction at the lip is +side in x
    if np.dot(n[0], [side, 0.0]) < 0:
        n = -n
    return st, n


def profile_offsets(contours_art, stations, normals, reach=120.0):
    """Signed distance along each station normal (positive = into the art's background side) to
    the nearest crossing of the candidate silhouette (already mapped into art pixels).  NaN where
    no crossing lies within +-reach."""
    out = np.full(len(stations), np.nan)
    segs = [np.stack([c[:-1], c[1:]], axis=1) for c in contours_art if len(c) > 1]
    if not segs:
        return out
    S = np.concatenate(segs)                      # (m, 2, 2)
    a, b = S[:, 0], S[:, 1]
    for k, (p, nv) in enumerate(zip(stations, normals)):
        # solve p + u nv = a + w (b - a), 0 <= w <= 1
        e = b - a
        den = nv[0] * (-e[:, 1]) - nv[1] * (-e[:, 0])
        ok = np.abs(den) > 1e-12
        rx, ry = a[:, 0] - p[0], a[:, 1] - p[1]
        u = np.where(ok, (rx * (-e[:, 1]) - ry * (-e[:, 0])) / np.where(ok, den, 1), np.nan)
        w = np.where(ok, (nv[0] * ry - nv[1] * rx) / np.where(ok, den, 1), np.nan)
        hit = ok & (w >= 0) & (w <= 1) & (np.abs(u) <= reach)
        if hit.any():
            out[k] = u[hit][np.argmin(np.abs(u[hit]))]
    return out


def anchor_weights(ids):
    """Equal weight per anchor group present, split evenly inside a group."""
    groups = {g: [i for i in ids if i in members] for g, members in ANCHOR_GROUPS.items()}
    present = [g for g in groups if groups[g]]
    w = {}
    for g in present:
        for i in groups[g]:
            w[i] = 1.0 / (len(present) * len(groups[g]))
    unknown = [i for i in ids if i not in w]
    if unknown:
        raise ValueError('anchor ids outside the step-4 groups: %r' % unknown)
    return np.asarray([w[i] for i in ids])


def fit_similarity(src, dst, weights):
    """Weighted least-squares 2-D similarity dst ~ A @ src + t (A = s R, det R = +1)."""
    src, dst, w = np.asarray(src, float), np.asarray(dst, float), np.asarray(weights, float)
    w = w / w.sum()
    ms, md = w @ src, w @ dst
    S, D = src - ms, dst - md
    C = (D * w[:, None]).T @ S
    U, sig, Vt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(U @ Vt))
    E = np.diag([1.0, d])
    R = U @ E @ Vt
    var = float(w @ np.sum(S * S, axis=1))
    s = float(np.trace(np.diag(sig) @ E) / var)
    A = s * R
    return A, md - A @ ms


def polyline_frames(xy):
    xy = np.asarray(xy, float)
    t = np.gradient(xy, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    return np.stack([-t[:, 1], t[:, 0]], axis=1), t


MEASURE = {'front': 'landmarks', 'oblique': 'landmarks', 'right': 'silhouette', 'left': 'silhouette'}


def load_bindings(view):
    """Step-4 mesh-surface bindings of the view's upper-face anchors (o_head triangle vertex ids
    and barycentric weights; ``used_ids`` only)."""
    d = _read(ALIGN / (view + '_anchor_bindings.json'))
    rows = {b['mediapipe_id']: b for b in d['bindings'] if b['status'] == 'surface_hit'}
    return [{'id': int(i), 'vertex_ids': [int(x) for x in rows[i]['vertex_ids']],
             'barycentric': [float(x) for x in rows[i]['barycentric']]} for i in d['used_ids']]


@dataclass
class ViewDomain:
    view: str
    yaw: float
    measure: str
    rect: list | None
    A: np.ndarray
    t: np.ndarray
    baseline_anchor_residual_px: list
    baseline_png_sha256: str
    anchor_render_px: list
    frame_notes: dict | None = None

    def to_json(self):
        return {'view': self.view, 'yaw': self.yaw, 'measure': self.measure, 'detector_input_rect': self.rect,
                'A': self.A.tolist(), 't': self.t.tolist(), 'scale': float(np.sqrt(abs(np.linalg.det(self.A)))),
                'rotation_deg': float(np.degrees(np.arctan2(self.A[1, 0], self.A[0, 0]))),
                'baseline_anchor_residual_px': self.baseline_anchor_residual_px,
                'baseline_png_sha256': self.baseline_png_sha256, 'anchor_render_px': self.anchor_render_px,
                'frame_notes': self.frame_notes}

    @classmethod
    def from_json(cls, d):
        return cls(d['view'], d['yaw'], d['measure'], d['detector_input_rect'], np.asarray(d['A'], float),
                   np.asarray(d['t'], float), d['baseline_anchor_residual_px'], d['baseline_png_sha256'],
                   d['anchor_render_px'], d.get('frame_notes'))


class ChinObjective:
    """Residual of a candidate's rendered views against the approved art (see module doc)."""

    def __init__(self, domains: dict, references: dict | None = None, detector: FixedDetector | None = None,
                 art_images: dict | None = None):
        self.domains = domains
        self.views = [v for v in VIEWS if v in domains]
        self.refs = references or {v: ReferenceView.load(v) for v in self.views}
        self.det = detector
        if self.det is None and any(domains[v].measure == 'landmarks' for v in self.views):
            self.det = FixedDetector()
        self.stations = {}
        for v in self.views:
            if domains[v].measure == 'silhouette':
                art = art_images[v] if art_images and v in art_images else _load_rgb(self.refs[v].art_path)
                self.stations[v] = profile_stations(art, self.refs[v].art_landmarks, FACE_SIDE[v])

    @staticmethod
    def build_domain(view, anchor_render_px, reference, rect, png_sha256):
        """Freeze the view's comparison frame: similarity from the mesh-bound upper-face anchors
        projected in the baseline render (``anchor_render_px``: id -> xy, from the renderer's own
        skin stream and bridge camera) to the art's detections of the same ids."""
        ids = [i for i in reference.anchor_ids if i in anchor_render_px]
        if ids != list(reference.anchor_ids):
            raise ValueError('every step-4 anchor of %s needs a mesh binding' % view)
        src = np.asarray([anchor_render_px[i] for i in ids], float)
        dst = reference.art_landmarks[ids, :2]
        A, t = fit_similarity(src, dst, anchor_weights(ids))
        res = (src @ A.T + t) - dst
        return ViewDomain(view, VIEW_YAW[view], MEASURE[view], None if rect is None else list(rect), A, t,
                          [[int(i), float(np.linalg.norm(r))] for i, r in zip(ids, res)], png_sha256,
                          src.tolist())

    def set_lip_translation(self, view, lip_render_px, baseline_rgb):
        """Profile views (todo item 7: the side path is positioned at the fixed lower lip): keep
        the anchor similarity's rotation and scale, translate so that the baseline render's
        lower-lip silhouette point (the silhouette point nearest the projection of the mesh-bound
        front lower-lip point, MediaPipe 17) lands on the art's (the first station, the art
        silhouette point nearest the art's landmark 17).  Frozen for every candidate; the lips
        are protected by the feature guard."""
        dom = self.domains[view]
        st, _ = self.stations[view]
        cs = silhouette_contours(baseline_rgb, FACE_SIDE[view])
        allp = np.concatenate(cs)
        q = allp[int(np.argmin(np.linalg.norm(allp - np.asarray(lip_render_px, float), axis=1)))]
        t_old = dom.t.copy()
        dom.t = st[0] - dom.A @ q
        dom.frame_notes = {'translation': 'lower lip (item 7)', 'lip_bound_point_render_px': list(map(float, lip_render_px)),
                           'lip_silhouette_point_render_px': q.tolist(), 'art_lip_silhouette_point': st[0].tolist(),
                           'snap_distance_render_px': float(np.linalg.norm(q - lip_render_px)),
                           'anchor_similarity_translation': t_old.tolist(),
                           'translation_change_art_px': float(np.linalg.norm(dom.t - t_old))}

    def n_residuals(self):
        return sum(2 * len(self.refs[v].chain_ids) if self.domains[v].measure == 'landmarks'
                   else len(self.stations[v][0]) for v in self.views)

    def score(self, images: dict):
        """images: view -> HxWx3/4 uint8 (top-down).  Returns (residual vector, report); raises
        PredictionUnavailable when a view's fixed measurement contract has no result."""
        residual, report = [], {}
        for v in self.views:
            dom, ref = self.domains[v], self.refs[v]
            if dom.measure == 'landmarks':
                pts = self.det.detect(images[v], dom.rect, v)
                r, rep = self._landmark_view(v, pts)
            else:
                r, rep = self._silhouette_view(v, images[v])
            residual.extend(r)
            report[v] = dict(rep, yaw=dom.yaw, measure=dom.measure)
        return np.asarray(residual), report

    def _landmark_view(self, v, pts):
        ref, dom = self.refs[v], self.domains[v]
        pts = np.asarray(pts, float)
        mapped = pts[ref.chain_ids, :2] @ dom.A.T + dom.t
        diff = mapped - ref.chain_xy
        n = len(ref.chain_ids)
        nrm, tan = polyline_frames(ref.chain_xy)
        anchors = pts[ref.anchor_ids, :2] @ dom.A.T + dom.t - ref.art_landmarks[ref.anchor_ids, :2]
        return (diff / ref.eye_unit_px / np.sqrt(n)).ravel(), {
            'point_ids': ref.chain_ids, 'reference_xy_art_px': ref.chain_xy.tolist(),
            'candidate_xy_art_px': mapped.tolist(), 'candidate_xy_render_px': pts[ref.chain_ids, :2].tolist(),
            'normal_px': np.sum(diff * nrm, axis=1).tolist(), 'tangent_px': np.sum(diff * tan, axis=1).tolist(),
            'rms_px': float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))),
            'max_px': float(np.linalg.norm(diff, axis=1).max()),
            'detected_anchor_offset_art_px_max': float(np.linalg.norm(anchors, axis=1).max())}

    def _silhouette_view(self, v, rgb):
        ref, dom = self.refs[v], self.domains[v]
        st, nv = self.stations[v]
        cs = [c @ dom.A.T + dom.t for c in silhouette_contours(rgb, FACE_SIDE[v])]
        off = profile_offsets(cs, st, nv)
        if np.isnan(off).any():
            raise PredictionUnavailable(v, -int(np.isnan(off).sum()))
        n = len(st)
        return off / ref.eye_unit_px / np.sqrt(n), {
            'stations_art_px': st.tolist(), 'normals': nv.tolist(), 'offset_px': off.tolist(),
            'candidate_xy_art_px': (st + off[:, None] * nv).tolist(),
            'reference_xy_art_px': st.tolist(),
            'rms_px': float(np.sqrt(np.mean(off ** 2))), 'max_px': float(np.abs(off).max()),
            'mean_signed_px': float(off.mean())}


def _load_rgb(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert('RGB'))
