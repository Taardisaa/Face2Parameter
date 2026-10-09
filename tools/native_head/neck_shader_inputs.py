"""Isolate authored neck UVs and restore source-derived neck mask inputs.

This authors texture coordinates and a transition field, not head geometry or a
replacement for the game's shader. Face texels and face UV corners stay literal.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
import UnityPy
import xatlas

from tools.native_head.audit_neck_shading import point_texels, source_frame, material


def sha(data):
    return hashlib.sha256(data).hexdigest()


def body_mask(capture, design, arrays):
    snapshot = json.loads(capture.read_text(encoding='utf-8-sig'))
    body = next(m for m in snapshot['meshes'] if m['mesh_name'] == 'o_body_cf')
    if body['source_geometry_sha256'] != design['body_source_geometry_sha256']:
        raise ValueError('Neck material capture does not contain the authored actual body')
    draw = next(m for m in body['material_state']['materials'] if m['name'] == 'cf_m_skin_body_00')
    binding = next(t for t in draw['textures'] if t['property'] == '_NailMask')
    if binding['name'] != 'cf_body_00_mask' or binding['is_null']:
        raise ValueError('Unaudited body skin region mask')
    path = Path('E:/HoneySelect2_ArcticFox/abdata/chara/mm_base.unity3d')
    env = UnityPy.load(str(path))
    draw = material(env, 'cf_m_skin_body_00').read()
    mask = dict(draw.m_SavedProperties.m_TexEnvs)['_NailMask'].m_Texture.read()
    points, _ = source_frame(body, 'cf_J_Head_s')
    seam = arrays['vertices'][arrays['native_outer_ring_authored_ids']]
    distance, ids = cKDTree(points).query(seam)
    if distance.max() > 1e-5:
        raise ValueError('Actual body and authored native rim no longer agree')
    pixels = point_texels(mask.image, np.asarray(body['source']['uv'])[ids])
    if not np.all(pixels == pixels[0]):
        raise ValueError('Nonuniform neck mask needs an explicit circumferential field')
    return pixels[0], dict(capture_sha256=sha(capture.read_bytes()),
        body_source_geometry_sha256=body['source_geometry_sha256'],
        bundle=str(path), bundle_sha256=sha(path.read_bytes()),
        texture=mask.m_Name, source_pixels_sha256=sha(np.asarray(mask.image).tobytes()),
        body_neck_rgba=pixels[0].tolist(), interface_match_max=float(distance.max()))


def raster(uv_triangles, values, size=2048):
    """Barycentric attribute bake at texel centers; no geometric fitting."""
    image = np.zeros((size, size, values.shape[-1]), float)
    covered = np.zeros((size, size), bool)
    for uv, value in zip(uv_triangles, values):
        xy = uv*[size, -size]+[0, size]
        lo = np.maximum(np.floor(xy.min(0)).astype(int), 0)
        hi = np.minimum(np.ceil(xy.max(0)).astype(int), size-1)
        yy, xx = np.mgrid[lo[1]:hi[1]+1, lo[0]:hi[0]+1]
        points = np.stack([xx+.5, yy+.5], -1)
        a, b = xy[1]-xy[0], xy[2]-xy[0]
        determinant = a[0]*b[1]-a[1]*b[0]
        if abs(determinant) < 1e-10:
            raise ValueError('Degenerate neck UV triangle')
        q = points-xy[0]
        t = (q[..., 0]*b[1]-q[..., 1]*b[0])/determinant
        u = (a[0]*q[..., 1]-a[1]*q[..., 0])/determinant
        inside = (t >= -1e-8) & (u >= -1e-8) & (t+u <= 1+1e-8)
        bary = np.stack([1-t-u, t, u], -1)
        image[yy[inside], xx[inside]] = bary[inside] @ value
        covered[yy[inside], xx[inside]] = True
    return image, covered


def author(vertices, faces, uv, integrated_ids, arrays, parents, donor, capture, design, out):
    selected = np.flatnonzero(parents < 0)
    old_corners = uv[faces].copy()
    mask_rgba, source = body_mask(capture, design, arrays)
    atlas = xatlas.Atlas()
    # xatlas has an absolute positional epsilon. Work in smaller length units
    # so it does not discard the connector's retained thin triangles. This
    # conversion is ONLY inside the UV solver; gathered game vertices stay exact.
    atlas.add_mesh(np.asarray(vertices*10000, np.float32), np.asarray(faces[selected], np.uint32))
    options = xatlas.PackOptions(); options.resolution = 512; options.padding = 16
    atlas.generate(pack_options=options)
    original_ids, indices, coords = atlas[0]
    if not np.array_equal(original_ids[indices], faces[selected]):
        raise ValueError('UV unwrapping changed ordered triangle corners')
    # The existing face authoring excludes this original oral tile. Reserve an
    # interior rectangle with a guard against face filtering footprints.
    box = np.array([.402, .902, .598, .988])
    source_face_uv = old_corners[parents >= 0]
    if np.any(source_face_uv[..., 1] >= .89):
        raise ValueError('Reserved oral tile is not free of retained face UVs')
    neck_uv = coords*(box[2:]-box[:2])+box[:2]
    np.savez(out/'neck_chart.npz',original_ids=original_ids,faces=indices,uv=neck_uv,
             original_face_ids=selected,vertices=vertices[original_ids])
    corners = old_corners.copy(); corners[selected] = neck_uv[indices]
    # Retained very thin triangles can collapse when packed UVs are quantized
    # to Unity float32. Give those faces their own small charts rather than
    # dropping faces or changing geometry. The main chart starts at V=.902;
    # this disjoint strip is also inside the unused oral tile.
    q = corners[selected]
    a, b = q[:,1]-q[:,0], q[:,2]-q[:,0]
    degenerate = np.flatnonzero(abs(a[:,0]*b[:,1]-a[:,1]*b[:,0]) < 1e-15)
    if len(degenerate):
        width = .196/len(degenerate)
        if width <= 4/2048:
            raise ValueError('Reserved micro-chart strip is too small')
        for i, face in enumerate(degenerate):
            low = .402+i*width+1/2048; high = .402+(i+1)*width-1/2048
            corners[selected[face]] = [[low,.893],[high,.893],[low,.899]]
    field = np.zeros(len(arrays['vertices']))
    width = len(arrays['source_ring'])
    bridge_start = len(arrays['vertices'])-5*width
    native_start = bridge_start-len(arrays['native_original_ids'])
    if not np.all(arrays['native_upper'] >= native_start):
        raise ValueError('Unaudited authored neck row layout')
    field[native_start:bridge_start] = 1
    for row in range(5):
        t = (row+1)/6
        field[bridge_start+row*width:bridge_start+(row+1)*width] = t*t*(3-2*t)
    blend = field[integrated_ids][faces[selected]]
    # Independent UV chart prevents a neck correction changing scalp/lip texels.
    baked, covered = raster(corners[selected], np.concatenate([
        old_corners[selected], blend[..., None]], axis=-1))
    size = len(covered)
    yy, xx = np.mgrid[:size, :size]
    tile = (xx/size >= .4) & (xx/size < .6) & (1-yy/size >= .89) & (1-yy/size < 1.)
    nearest = distance_transform_edt(~covered, return_distances=False, return_indices=True)
    baked[tile] = baked[nearest[0][tile], nearest[1][tile]]
    pixels = np.asarray(donor.convert('RGBA')).copy()
    if pixels.shape[:2] != (size, size):
        raise ValueError('Audited donor must be 2048 square')
    # Bilinear source sampling with explicit clamp; source atlas pixels outside
    # the unused oral tile are preserved byte for byte.
    sampled_uv = baked[tile, :2]
    xy = sampled_uv*[size, -size]+[-.5, size-.5]
    xy = np.clip(xy, 0, size-1)
    lo = np.floor(xy).astype(int); hi = np.minimum(lo+1, size-1); f = xy-lo
    color = ((1-f[:, :1])*(1-f[:, 1:])*pixels[lo[:, 1], lo[:, 0]]+
             f[:, :1]*(1-f[:, 1:])*pixels[lo[:, 1], hi[:, 0]]+
             (1-f[:, :1])*f[:, 1:]*pixels[hi[:, 1], lo[:, 0]]+
             f[:, :1]*f[:, 1:]*pixels[hi[:, 1], hi[:, 0]])
    pixels[tile] = np.rint(color).astype(np.uint8)
    mask = np.full((size, size, 4), [0, 255, 0, 255], np.uint8)
    t = baked[tile, 2:3]
    mask[tile] = np.rint((1-t)*[0, 255, 0, 255]+t*mask_rgba).astype(np.uint8)
    mask_path = out/'neck_region_mask.png'; Image.fromarray(mask).save(mask_path)
    albedo_path = out/'neck_albedo_atlas.png'; Image.fromarray(pixels).save(albedo_path)
    gather, values, remap = [], [], {}
    new_faces = np.empty_like(faces)
    for j, (face, chart) in enumerate(zip(faces, corners)):
        for k, (index, coord) in enumerate(zip(face, chart)):
            key = (int(index), np.asarray(coord, '<f4').tobytes())
            if key not in remap:
                remap[key] = len(gather); gather.append(int(index)); values.append(coord)
            new_faces[j, k] = remap[key]
    policy = dict(format='native_neck_shader_inputs_v1', source=source,
        neck_face_ids=selected.tolist(), reserved_uv_box=box.tolist(),
        atlas_library='xatlas==0.0.11', chart_count=atlas.get_mesh_chart_count(0),
        uv_solver_length_unit_scale=10000, game_vertices_scaled=False,
        float32_collapsed_uv_faces_recharted=selected[degenerate].tolist(),
        mask_path=str(mask_path.resolve()), mask_pixels_sha256=sha(mask.tobytes()),
        albedo_path=str(albedo_path.resolve()), albedo_pixels_sha256=sha(pixels.tobytes()),
        original_albedo_outside_unused_tile_unchanged=bool(np.array_equal(pixels[~tile],np.asarray(donor)[~tile])),
        face_uv_corners_unchanged=bool(np.array_equal(corners[parents >= 0],old_corners[parents >= 0])),
        geometry_modified=False, ao_and_geometric_normals_matched=False,
        mask_policy='Actual body neck mask at collar; topology-row smoothstep to unchanged face policy',
        limitations='Current authored five-row connector and compatible constant native neck mask; AO/normal continuity remains open')
    (out/'neck_shader_inputs.json').write_text(json.dumps(policy,indent=2)+'\n')
    return np.asarray(gather), new_faces, np.asarray(values,'<f4'), Image.fromarray(pixels), policy
