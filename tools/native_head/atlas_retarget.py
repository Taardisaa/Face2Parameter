"""Retarget installed native skin by semantic UV anchors; never reshape vertices."""
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import LinearNDInterpolator

from tools.native_head.neck_geometry import ordered_loops
from tools.native_head.surface_regions import digest, inherited_faces, load, save


def lip_contour(image, profile):
    pixels = np.asarray(image.convert('RGBA'), int)
    h, w = pixels.shape[:2]
    box = profile['uv_box']; x0, x1 = int(box[0]*w), int(box[2]*w)
    y0, y1 = int((1-box[3])*h), int((1-box[1])*h)
    crop = pixels[y0:y1, x0:x1]
    mask = crop[:, :, 0]-crop[:, :, 1] > profile['red_minus_green_min']
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError('Declared native lip pigment region is absent')
    points = []
    for i, fraction in enumerate(profile['x_fractions']):
        x = int(round(xs.min()+fraction*(xs.max()-xs.min())))
        rows = np.flatnonzero(mask[:, x])
        if not len(rows):
            raise ValueError('Lip pigment contour has an unreviewed gap')
        y = (rows.min()+rows.max())/2 if i in (0, 6) else rows.min() if i < 6 else rows.max()
        points.append([(x0+x+.5)/w, 1-(y0+y+.5)/h])
    return np.array(points)


def eye_anchors(rig, vertices):
    # Weld only native UV duplicates for identifying actual open eye boundaries.
    unique = {}; aliases = {}
    for i, point in enumerate(rig.verts):
        unique.setdefault(point.tobytes(), i); aliases[i] = unique[point.tobytes()]
    loops = [np.array(x) for x in ordered_loops(rig.faces, aliases) if len(x) == 10]
    if len(loops) != 2:
        raise ValueError('Native head2 eye boundary topology changed')
    result = []
    # The installed head atlas reverses the mesh X axis. Match chart sides and
    # corner order in UV space, not an assumed world left/right convention.
    for loop in sorted(loops, key=lambda x: rig.uv[x, 0].mean()):
        # Native outer canthus UVs double back; copying their ordered contour
        # would fold a continuous FLAME chart. Author a convex six-point
        # envelope within the literal native opening's UV bounding rectangle.
        # This is an atlas-authoring choice, not an exact native loop transfer.
        a, b = rig.uv[loop].min(0), rig.uv[loop].max(0)
        shape = np.array([[0, .5], [.25, .95], [.75, .95],
                          [1, .5], [.75, .05], [.25, .05]])
        result.extend(a+shape*(b-a))
    return np.array(result)


def mapping(rig, native_vertices, source, image, profile):
    uv_template = Path(profile['uv_template'])
    corner_uv, landmarks, regions, report = load(source, uv_template)
    src = []; dst = []
    def append(a, b):
        src.extend(np.asarray(a).reshape(-1, 2)); dst.extend(np.asarray(b).reshape(-1, 2))
    keys = list(map(int, profile['landmark_uv']))
    append(landmarks[keys], [profile['landmark_uv'][str(k)] for k in keys])
    append(landmarks[36:48], eye_anchors(rig, native_vertices))
    donor_lips = lip_contour(image, profile['lip_pigment'])
    append(landmarks[48:60], donor_lips)
    # Mouth interior stays inside the lip outline; preserve an open UV area.
    bounds = np.array([donor_lips.min(0), donor_lips.max(0)])
    mid = bounds.mean(0)
    x = [0, .25, .5, .75, 1, .75, .5, .25]
    inner = [[mid[0]+(t-.5)*np.ptp(bounds[:, 0])*.85,
              mid[1]+(0 if i in (0, 4) else .003 if i < 4 else -.003)]
             for i, t in enumerate(x)]
    append(landmarks[60:68], inner)
    # Ear islands have authoritative UV domains in both models. Transfer their
    # chart extents with separate left/right correspondence, never nearest face.
    for name, side in [('right_ear', 'R'), ('left_ear', 'L')]:
        source_uv = corner_uv[regions[name] == 1].reshape(-1, 2)
        bones = [i for i, n in enumerate(rig.skin_bone_names) if 'Ear' in n and n.endswith(side)]
        weight = (rig.bone_w*np.isin(rig.bone_idx, bones)).sum(1)
        donor_uv = rig.uv[weight > .5]
        a, b = source_uv.min(0), source_uv.max(0)
        c, d = donor_uv.min(0), donor_uv.max(0)
        shape = np.array([[.5, .5], [.5, 0], [.5, 1], [0, .5], [1, .5]])
        append(a+shape*(b-a), c+shape*(d-c))
    edge = np.linspace(0, 1, 7)
    append(np.c_[edge, np.zeros(7)], np.c_[edge, np.zeros(7)])
    append(np.c_[edge, np.ones(7)], np.c_[edge, np.full(7, profile['scalp_max_v'])])
    append([[0, .25], [1, .25], [0, .6], [1, .6]], [[0, .16], [1, .16], [0, .6], [1, .6]])
    # FLAME's canonical UV U increases with game X; installed head2 U decreases.
    # Align this handedness before connecting to the literal native collar UVs.
    dst = np.asarray(dst)
    dst[:, 0] = 1-dst[:, 0]
    transform = LinearNDInterpolator(np.array(src), np.array(dst))
    cells = transform.tri.simplices
    def area(points):
        a, b = points[:, 1]-points[:, 0], points[:, 2]-points[:, 0]
        return a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0]
    if np.any(area(np.asarray(src)[cells])*area(np.asarray(dst)[cells]) >= 0):
        raise ValueError('Authored UV correspondences fold the reference chart')
    report['atlas_correspondence'] = dict(profile=profile, source_uv=np.asarray(src).tolist(),
        donor_uv=np.asarray(dst).tolist(), method='Piecewise affine authored UV correspondences',
        chart_orientation=-1, game_axis_alignment='FLAME canonical U/game X vs native head2 reversed U/game X',
        native_eye_boundaries='Authored convex envelopes within literal installed welded 10-vertex eye-loop UV bounds; native canthus contour doubles back',
        native_lip_boundary='Declared pigment contour on installed albedo; not a native anatomical segmentation claim')
    return transform, corner_uv, regions, report


def retarget(source, arrays, design, source_ids, face_keep, target_faces, attrs,
             rig, native_vertices, image, out):
    profile_path = Path(__file__).with_name('native_atlas_profile.json')
    profile = json.loads(profile_path.read_text())
    transform, corner_uv, regions, report = mapping(rig, native_vertices, source, image, profile)
    parents, labels = inherited_faces(arrays, face_keep, regions)
    # Faces refined by neck cutting retain their original triangle reference.
    reference = arrays['placed_original_vertices']
    original_faces = arrays['original_faces']
    points = arrays['before_vertices'][source_ids][target_faces]
    uv_corners = attrs['uv'][target_faces].copy()
    valid = parents >= 0
    triangle = reference[original_faces[parents[valid]]]
    a = triangle[:, 0]; ab = triangle[:, 1]-a; ac = triangle[:, 2]-a
    ap = points[valid]-a[:, None, :]
    dot = lambda x, y: np.sum(x*y, axis=-1)
    aa, cc, cross = dot(ab, ab), dot(ac, ac), dot(ab, ac)
    denom = aa*cc-cross*cross
    t = (cc[:, None]*dot(ap, ab[:, None])-cross[:, None]*dot(ap, ac[:, None]))/denom[:, None]
    u = (aa[:, None]*dot(ap, ac[:, None])-cross[:, None]*dot(ap, ab[:, None]))/denom[:, None]
    bary = np.stack([1-t-u, t, u], axis=-1)
    # Clamp only newly authored neck surface attributes; retained face corners
    # have literal decoder vertex/face lineage and remain exact.
    bary = np.maximum(bary, 0); bary /= bary.sum(-1, keepdims=True)
    original_uv = (corner_uv[parents[valid], None]*bary[..., None]).sum(2)
    mapped = transform(original_uv.reshape(-1, 2)).reshape(original_uv.shape)
    if not np.isfinite(mapped).all():
        raise ValueError('Native atlas anchors do not cover source chart')
    uv_corners[valid] = mapped
    # The retained native collar keeps exact original native UVs. Blend the new
    # bridge's attribute rows between its true source/native endpoint charts.
    # Source ring values are taken only from source-side incident corners.
    lookup = {int(v): i for i, v in enumerate(source_ids)}
    ring = [lookup[int(i)] for i in arrays['source_ring']]
    ring_set = set(ring); edge_uv = {}
    for face, uv in zip(target_faces[valid], mapped):
        for j, k in ((0, 1), (1, 2), (2, 0)):
            a, b = int(face[j]), int(face[k])
            if a in ring_set and b in ring_set:
                edge_uv[tuple(sorted((a,b)))] = {a: uv[j], b: uv[k]}
    width = len(ring)
    lower = attrs['uv'][[lookup[int(i)] for i in arrays['native_upper']]]
    bridge_start = len(arrays['vertices'])-5*width
    columns = {index: (i, 0.) for i, index in enumerate(ring)}
    interior = set()
    for row in range(5):
        ids = [lookup[int(i)] for i in range(bridge_start+row*width,bridge_start+(row+1)*width)]
        interior.update(ids)
        columns.update({index:(i, (row+1)/6) for i,index in enumerate(ids)})
    columns.update({lookup[int(index)]:(i,1.) for i,index in enumerate(arrays['native_upper'])})
    for j in np.flatnonzero(~valid):
        face = target_faces[j]
        if not any(int(index) in interior for index in face):
            continue  # Retained native collar keeps its literal native chart.
        pair = sorted({columns[int(index)][0] for index in face})
        if len(pair) != 2:
            raise ValueError('Bridge face no longer joins two corresponding ring columns')
        endpoints = edge_uv[tuple(sorted((ring[pair[0]],ring[pair[1]])))]
        # Endpoint attributes belong to the incident source-side triangle.
        # Never average U=0/U=1 seam copies into a false U=.5 neck stripe.
        for k,index in enumerate(face):
            column,blend = columns[int(index)]
            uv_corners[j,k] = (1-blend)*endpoints[ring[column]]+blend*lower[column]
    # All exterior charts must stay outside the donor's separate oral tile.
    box = profile['excluded_oral_uv_box']
    in_oral_tile = ((uv_corners[..., 0] >= box[0]) & (uv_corners[..., 0] <= box[2]) &
                    (uv_corners[..., 1] >= box[1]) & (uv_corners[..., 1] <= box[3]))
    if in_oral_tile.any():
        raise ValueError('Exterior head UV entered the donor oral-interior tile')
    # Split only UV discontinuities. All position/normal/skin attributes gather
    # their literal old vertex; triangle order and geometry stay unchanged.
    remap = {}; old_ids = []; values = []; new_faces = np.empty_like(target_faces)
    for j, (face, uv) in enumerate(zip(target_faces, uv_corners)):
        for k, (index, coord) in enumerate(zip(face, uv)):
            # Remove barycentric roundoff between shared source triangle corners;
            # real UV discontinuities retain separate vertices.
            coord = np.round(coord, 6)
            key = (int(index), np.asarray(coord, '<f4').tobytes())
            if key not in remap:
                remap[key] = len(old_ids); old_ids.append(int(index)); values.append(coord)
            new_faces[j, k] = remap[key]
    report['profile_sha256'] = digest(profile_path.read_bytes())
    report['positions_modified'] = False
    report['bridge_attribute_policy'] = 'Per-edge incident source UVs to exact native collar; no seam-copy averaging'
    report['uv_split_vertices'] = len(old_ids)
    save(report, parents, labels, out)
    np.savez(out/'atlas_uv.npz',uv=np.asarray(values, '<f4'),faces=new_faces,
             original_vertex_ids=np.asarray(old_ids),parent_face_ids=parents)
    return np.asarray(old_ids), new_faces, np.asarray(values, '<f4'), report
