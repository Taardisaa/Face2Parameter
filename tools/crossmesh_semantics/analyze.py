"""Read-only source/pose-bound crossing diagnostics. Never certifies shader visibility."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools' / 'geometry_quality'))
sys.path.insert(0, str(ROOT))
from cross_mesh import load_certified_head, pair_key
from mesh_quality import plane_slice, triangle_intersection
from tools.surface_calibration.core import Camera, SurfaceMesh, all_intersections
from tools.surface_calibration.pixel_certificate import certify_pixel_contract


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bound(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}


def require(value, message):
    if not value:
        raise ValueError(message)


def barycentric(triangle, points):
    triangle, points = np.asarray(triangle), np.asarray(points)
    edge = (triangle[1:] - triangle[0]).T
    weights = np.linalg.lstsq(edge, (points - triangle[0]).T, rcond=None)[0].T
    bary = np.c_[1 - weights.sum(axis=1), weights]
    return {'weights': bary.tolist(), 'min_weight': float(bary.min()),
            'max_reconstruction_error': float(np.linalg.norm(bary @ triangle - points, axis=1).max())}


def segment(a, b, epsilon):
    """Use the existing classifier/slicing epsilon unchanged; recover its line interval."""
    a, b = np.asarray(a), np.asarray(b)
    classification = triangle_intersection(a, b, epsilon)
    na, nb = np.cross(a[1]-a[0], a[2]-a[0]), np.cross(b[1]-b[0], b[2]-b[0])
    na, nb = na / np.linalg.norm(na), nb / np.linalg.norm(nb)
    da, db = (a-b[0]) @ nb, (b-a[0]) @ na
    result = {'classification': classification,
              'signed_a_to_b_plane': da.tolist(), 'signed_b_to_a_plane': db.tolist()}
    if classification is None or classification['kind'] != 'proper_crossing':
        return result
    direction = np.cross(na, nb)
    direction /= np.linalg.norm(direction)
    origin = (a[0] + b[0]) / 2
    pa, pb = plane_slice(a, da, epsilon), plane_slice(b, db, epsilon)
    ia = (np.asarray(pa) - origin) @ direction
    ib = (np.asarray(pb) - origin) @ direction
    lo, hi = max(ia.min(), ib.min()), min(ia.max(), ib.max())
    line = origin + np.linalg.solve(np.array([na, nb, direction]),
                                    [na @ (a[0]-origin), nb @ (b[0]-origin), 0])
    points = np.array([line + lo*direction, line + hi*direction])
    result.update(endpoints_world=points.tolist(), length=float(np.linalg.norm(points[1]-points[0])),
                  left_barycentric=barycentric(a, points), right_barycentric=barycentric(b, points),
                  plane_residual_max=float(max(abs((points-a[0]) @ na).max(), abs((points-b[0]) @ nb).max())))
    return result


def submesh_for_triangle(mesh, ids):
    """Match the exact ordered index triplet. Ambiguous binding is not guessed."""
    matches = []
    for sub in mesh['source']['submeshes']:
        if sub['topology'] != 'Triangles' and sub['topology'] != 0:
            continue
        indices = np.asarray(sub['indices']).reshape(-1, 3)
        if not sub.get('indices_apply_base_vertex', False):
            indices = indices + sub.get('base_vertex', 0)
        rows = np.flatnonzero((indices == ids).all(axis=1))
        matches.extend({'submesh_index': sub['submesh_index'], 'local_triangle_id': int(row)} for row in rows)
    return matches


def surface_side(mesh, cm, tri_id, points, base_mesh, base_cm):
    ids = cm.faces[tri_id]
    require(np.array_equal(np.asarray(mesh['source']['triangles']).reshape(-1, 3), cm.faces), 'Baked/source triangle ordering mismatch')
    correspondence = all(mesh['source'][k] == base_mesh['source'][k] for k in mesh['source'])
    require(mesh['source_geometry_sha256'] == base_mesh['source_geometry_sha256'] and correspondence,
            'Baseline/current complete source correspondence changed')
    bary = barycentric(cm.vertices[ids], points)
    weights = np.asarray(bary['weights'])
    uv = np.asarray(mesh['source']['uv'])[ids]
    mapped_uv = weights @ uv
    uv2 = np.asarray(mesh['source']['uv2'])
    mapped_uv2 = weights @ uv2[ids] if len(uv2) else None
    colors=np.asarray(mesh['source_vertex_colors_rgba'])
    slots = submesh_for_triangle(mesh, ids)
    materials = []
    for match in slots:
        index = match['submesh_index']
        material = mesh['material_state']['materials'][index]
        materials.append({'submesh_index': index, 'material_slot': material['slot'],
                          'name': material['name'], 'shader_name': material['shader_name'],
                          'material_state': material,
                          'texture_uv_diagnostics': [dict(t, uv_after_recorded_scale_offset=(mapped_uv*np.asarray(t['scale'])+np.asarray(t['offset'])).tolist(),
                              texel_edge_coordinates_before_wrap=((mapped_uv*np.asarray(t['scale'])+np.asarray(t['offset']))*np.array([t['width'],t['height']])).tolist(),
                              texture_sample_value_certified=False,
                              actual_shader_uv_formula_certified=False) for t in material.get('textures',[]) if not t['is_null']]})
    return {'mesh_name': mesh['mesh_name'], 'renderer_path': cm.path, 'source_geometry_sha256': cm.source_hash,
            'triangle_id': tri_id, 'vertex_ids': ids.tolist(), 'world_candidate': cm.candidate,
            'actual_baked_to_world_matrix': mesh['baked']['world_candidates'][cm.candidate]['matrix'],
            'world_triangle': cm.vertices[ids].tolist(), 'source_triangle': np.asarray(mesh['source']['vertices'])[ids].tolist(),
            'uv_triangle': uv.tolist(), 'uv_at_segment_endpoints': mapped_uv.tolist(),
            'uv2_triangle': uv2[ids].tolist() if len(uv2) else None,
            'uv2_at_segment_endpoints': mapped_uv2.tolist() if mapped_uv2 is not None else None,
            'missing_uv2_is_not_assumed_equal_to_uv0': True,
            'source_vertex_colors_rgba_triangle':colors[ids].tolist() if len(colors) else None,
            'barycentric_vertex_color_candidate':(weights@colors[ids]).tolist() if len(colors) else None,
            'actual_shader_vertex_color_formula_certified':False,
            'barycentric': bary, 'baseline_same_barycentric_world': (weights @ base_cm.vertices[ids]).tolist(),
            'baseline_same_barycentric_uv': (weights @ np.asarray(base_mesh['source']['uv'])[ids]).tolist(),
            'baseline_current_complete_source_equal': correspondence, 'submesh_matches': slots,
            'engine_material_slot_binding_independently_validated': False, 'materials': materials}


def pairing(view, snapshot, path):
    pair = view['paired_geometry']
    require(Path(pair['path']).resolve() == Path(path).resolve(), 'Paired geometry file differs')
    signature = snapshot['pose_signature']
    require(view['paired_pose_unchanged'] is True and pair['pose_signature'] == signature
            and view['pose_signature_before_render'] == signature == view['pose_signature_after_render'], 'Pose pairing failed')
    require(pair['frame_count'] == snapshot['frame_count'] == snapshot['frame_count_end']
            == view['frame_count'] == view['frame_count_before_render'], 'Frame pairing failed')
    require(pair['capture_state'] == snapshot['capture_state'], 'Paired actual state differs')


def execute(live_path, quality_dir, output, marker_reports, asset_path):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    quality_dir = Path(quality_dir)
    live = read(live_path); case = live['cases'][0]
    baseline_path = Path(case['source_history']['geometry']['path'])
    baseline = read(baseline_path)
    base_cert_path = quality_dir/'baseline_lbs.json'
    base, _ = load_certified_head(baseline, sha(baseline_path), read(base_cert_path))
    baseline_report_path=quality_dir/'baseline_absolute.json'
    baseline_report=read(baseline_report_path)
    require(baseline_report['source_sha256']==sha(baseline_path),'Baseline report source mismatch')
    baseline_crossings={pair_key(p) for p in baseline_report['cross_mesh_report']['pairs'] if p['kind'] in ('proper_crossing','coplanar_overlap')}
    base_cm = {m.path:m for m in base}
    base_src = {m['renderer_path']:m for m in baseline['meshes']}
    inputs = [bound(live_path), bound(baseline_path), bound(base_cert_path), bound(baseline_report_path), bound(asset_path)]
    inputs.extend(bound(p) for p in marker_reports)
    assets = read(asset_path)
    asset_renderers = [r for b in assets['bundles'] for r in b['renderers'] if r['prefab'] == 'p_cf_head_02']
    windows = []
    for window in case['windows']:
        name = window['window']; path = Path(window['geometry']['path']); snapshot = read(path)
        quality_path = quality_dir/(name+'_quality.json'); quality = read(quality_path)
        require(sha(path) == quality['geometry_sha256'] and sha(baseline_path) == quality['baseline_geometry_sha256'], 'Quality/source SHA mismatch')
        cert_path = Path(quality['world_certificate']['certificate_path'])
        require(sha(cert_path) == quality['world_certificate']['certificate_sha256'], 'LBS certificate SHA mismatch')
        meshes, skipped = load_certified_head(snapshot, sha(path), read(cert_path))
        current = {m.path:m for m in meshes}; source = {m['renderer_path']:m for m in snapshot['meshes']}
        inputs.extend([bound(path), bound(quality_path), bound(cert_path)])
        raw_surfaces = [SurfaceMesh(m.path, m.vertices, m.faces, m.source_hash, world_policy_certified=True) for m in meshes]
        records = []
        for n, pair in enumerate(quality['cross_mesh_baseline_comparison']['new_crossing_pairs'],1):
            require(pair_key(pair) not in baseline_crossings,'A claimed new crossing already existed at baseline')
            sides = [pair[s] for s in ('left','right')]
            cm = [current[s['renderer_path']] for s in sides]
            triangles = [m.vertices[m.faces[s['triangle']]] for m,s in zip(cm,sides)]
            geometry = segment(*triangles, pair['length_epsilon'])
            require(geometry['classification']['kind'] == pair['kind'], 'Recomputed intersection classification differs')
            require(abs(geometry['length']-pair['intersection_length']) < 1e-12, 'Recomputed intersection length differs')
            points = np.asarray(geometry['endpoints_world'])
            bm = [base_cm[s['renderer_path']] for s in sides]
            baseline_triangles = [m.vertices[m.faces[s['triangle']]] for m,s in zip(bm,sides)]
            # Recompute original baseline tolerance, not candidate tolerance.
            all_base_vertices = np.concatenate([m.vertices for m in base])
            baseline_epsilon = 1e-7*np.linalg.norm(np.ptp(all_base_vertices,axis=0))+sum(m.residual_floor for m in bm)
            record = {'id': n, 'reported_pair': pair, 'intersection': geometry,
                      'baseline_same_pair': segment(*baseline_triangles,baseline_epsilon),
                      'baseline_length_epsilon': baseline_epsilon,
                      'shader_visible_defect_certified': False, 'anatomical_classification': 'unknown',
                      'pair_retained_by_existing_quality_gate': True,
                      'sides': [surface_side(source[s['renderer_path']],m,s['triangle'],points,base_src[s['renderer_path']],b)
                                for s,m,b in zip(sides,cm,bm)]}
            records.append(record)
        views = []
        for view in window['capture']['views']:
            pairing(view,snapshot,path)
            png = Path(view['path']); inputs.append(bound(png))
            failures = []; pixel = None
            for marker in marker_reports:
                try:
                    pixel = certify_pixel_contract(marker,view); break
                except (ValueError,KeyError,TypeError,OSError) as exc:
                    failures.append(str(exc))
            camera = Camera.from_capture(view,pixel_certificate=pixel,diagnostic=pixel is None)
            image = Image.open(png).convert('RGB'); draw = ImageDraw.Draw(image)
            draw.rectangle((0,0,image.width,33),fill='black')
            draw.text((5,3),name+f" yaw {view['yaw']} | "+('certified coordinates' if pixel else 'UNCERTIFIED COORDINATE DIAGNOSTIC'),fill='white')
            draw.text((5,17),'lines 2px / circles amplified; shader visibility UNKNOWN',fill='white')
            projected = []
            for record in records:
                points = np.asarray(record['intersection']['endpoints_world'])
                endpoints = [camera.project(p) for p in points]; midpoint = camera.project(points.mean(axis=0))
                xy = np.asarray(midpoint['xy']); ray = camera.ray(xy)
                hits = all_intersections(raw_surfaces,ray)
                distance = float((points.mean(axis=0)-ray.origin) @ ray.direction)
                occluders = [h for h in hits if h['distance_from_near'] < distance-record['reported_pair']['length_epsilon']]
                p = {'pair_id':record['id'], 'endpoints':endpoints,'midpoint':midpoint,
                     'projected_length_px':float(np.linalg.norm(np.asarray(endpoints[1]['xy'])-endpoints[0]['xy'])),
                     'raw_ray_midpoint_distance':distance,'raw_ray_hits':hits,
                     'raw_foreground_hits_above_existing_pair_epsilon':occluders,
                     'raw_geometry_depth_margin': distance-hits[0]['distance_from_near'] if hits else None,
                     'raw_occlusion_is_shader_visibility_certificate':False}
                projected.append(p)
                color = '#ffdd22' if not occluders else '#ff66cc'
                draw.line([tuple(x['xy']) for x in endpoints],fill=color,width=2)
                x,y=xy; draw.ellipse((x-5,y-5,x+5,y+5),outline=color,width=1)
                draw.text((x+7,y+((record['id']%3)-1)*12),str(record['id']),fill=color)
            overlay_path=output/f'{name}_yaw_{view["yaw"]:g}_diagnostic.png'; image.save(overlay_path)
            views.append({'yaw':view['yaw'],'png':bound(png),'paired_geometry':bound(path),
                          'view_payload_sha256':hashlib.sha256(json.dumps(view,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
                          'actual_capture_camera':view['capture_camera'],'actual_capture_controls':{k:view.get(k) for k in ('distance','ortho_size','target','width','height','force_skinning_recalculation')},
                          'pose_pairing_validated':camera.pose_pairing_validated,'pixel_coordinate_certified':pixel is not None,
                          'pixel_certificate':pixel.__dict__ if pixel is not None else None,
                          'pixel_certificate_failures':failures,'overlay':bound(overlay_path),'projections':projected})
        windows.append({'window':name,'snapshot':bound(path),'snapshot_frame':snapshot['frame_count'],
                        'baseline_crossing_count':baseline_report['cross_mesh_report']['crossing_count'],
                        'new_pair_count':len(records),'pairs':records,'views':views,'skipped_renderers':skipped})
    result={'schema_version':1,'scope':'exact-source and exact-pose-bound raw geometry intersection semantics; shader visibility unknown',
            'inputs':inputs,'windows':windows,'baseline_asset_context':{'evidence':bound(asset_path),'renderers':asset_renderers,
             'source_mesh_geometry_directly_compared_to_installed_asset':False,'fragment_clip_formula_validated':False,
             'prefab_name_match_is_not_source_geometry_equivalence':True},
            'thresholds_changed':False,'quality_gate_changed':False,'new_pairs_whitelisted':False,
            'shader_visibility_certified':False,'anatomy_certified':False,
            'heldout_images_or_annotations_read':False,'tools_source':bound(__file__)}
    (output/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    brief={'new_pairs':[w['new_pair_count'] for w in windows],
           'coordinate_certified_views':sum(v['pixel_coordinate_certified'] for w in windows for v in w['views']),
           'max_projected_segment_length_px':max(p['projected_length_px'] for w in windows for v in w['views'] for p in v['projections']),
           'raw_foreground_by_window':{}}
    brief['raw_foreground_by_window']={w['window']:[{'yaw':v['yaw'],'foreground_meshes':dict(Counter(p['raw_foreground_hits_above_existing_pair_epsilon'][0]['renderer_path'].split('/')[-1] if p['raw_foreground_hits_above_existing_pair_epsilon'] else 'none' for p in v['projections']))} for v in w['views']] for w in windows}
    (output/'summary.json').write_text(json.dumps(brief,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(brief))


if __name__ == '__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--live',required=True); ap.add_argument('--quality',required=True)
    ap.add_argument('--output',required=True); ap.add_argument('--marker',action='append',default=[]); ap.add_argument('--asset-evidence',required=True)
    args=ap.parse_args(); execute(args.live,args.quality,args.output,args.marker,args.asset_evidence)
