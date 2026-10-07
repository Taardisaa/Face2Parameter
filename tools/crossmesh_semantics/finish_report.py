"""Build a source-bound finding table and minimal follow-up acquisition contract."""
from pathlib import Path
import json
import subprocess
import sys
from analyze import ROOT, bound, read

OUT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else ROOT/'outputs/crossmesh_semantics_20261005/actual_lowered_v2'
r=read(OUT/'report.json')
asset=read(OUT/'asset_geometry_comparison.json')
rows=[]
for w in r['windows']:
    for p in w['pairs']:
        rows.append({'window':w['window'],'pair_id':p['id'],
                     'left':[p['sides'][0]['mesh_name'],p['sides'][0]['triangle_id']],
                     'right':[p['sides'][1]['mesh_name'],p['sides'][1]['triangle_id']],
                     'submeshes':[s['submesh_matches'] for s in p['sides']],
                     'world_length':p['intersection']['length'],
                     'baseline_classification':p['baseline_same_pair']['classification'],
                     'uv_endpoints':[s['uv_at_segment_endpoints'] for s in p['sides']],
                     'current_source_equals_baseline':all(s['baseline_current_complete_source_equal'] for s in p['sides']),
                     'per_view':[{'yaw':v['yaw'], 'projected_length_px':v['projections'][p['id']-1]['projected_length_px'],
                                  'pixel_coordinate_certified':v['pixel_coordinate_certified'],
                                  'foreground_raw':v['projections'][p['id']-1]['raw_foreground_hits_above_existing_pair_epsilon'][0]['renderer_path'],
                                  'depth_margin':v['projections'][p['id']-1]['raw_geometry_depth_margin']} for v in w['views']],
                     'visible_or_interior_certified':False})
camera=r['windows'][0]['views'][0]
options={'camera':{'yaws':[-30,0,30],'pitch':0,'roll':0,'distance':8.935848,
                   'orthographic':True,'ortho_size':1.12654757,'fov':24,
                   'target':[0.09552551,14.7151775,0.04619336]},
         'output':{'out_path':'<fresh absolute path>/full.png','geometry_out':'<fresh absolute path>/geometry.json',
                   'geometry_texture_dir':'<fresh absolute path>/textures','geometry_blendshape_frames':True,
                   'width':512,'height':512,'anti_aliasing':1,'contact_sheet':False},
         'scene':{'hide_hair':True,'hide_accessories':True,'freeze_pose':True,'force_skinning_recalculation':False,
                  'settle_frames':3,'visibility_method':'renderer_enabled'}}
requirements={'schema_version':1,'scope':'requirements only; no game acquisition executed by this agent',
              'actual_observation_report':bound(OUT/'report.json'),
              'coordinate':{'one_fresh_marker_scope_covers_all_nine_only_if_exact_match':True,
                'maker_capture_options':dict(options,camera=dict(options['camera'],yaws=[0]),
                   output=dict(options['output'],pixel_calibration=True,geometry_out=None,geometry_texture_dir=None)),
                'required_actual_camera':camera['actual_capture_camera'],
                'loaded_module_must_match':'ea8e1a9b-2dbd-45bb-83ef-824405408af5',
                'minimum_proof':'Fresh saved marker PNG+raw capture JSON SHA; independent evaluate+certify_pixel_contract against each actual target view. Any new MVID requires its own source/IL review. No reuse of prior green.'},
              'paired_original_shader_and_textures':{'maker_capture_options':options,
                'minimum_meshes':['o_head','o_eyebase_L','o_eyebase_R','o_eyeshadow','o_eyelashes','o_namida','o_tooth','o_tang'],
                'no_card_or_expression_edits_to_obtain_image':True,
                'same_frame_requirements':['actual renderer-transform matrices, baked vertices/normals, bone world/local transforms, bindposes and blendshape weights',
                  'capture frame before/render/paired geometry equal; exact pose and capture-state readbacks',
                  'material/shader IDs, queues, keywords, properties, submesh indices, UV0/UV1 and actual texture IDs/size/wrap/filter/scale/offset',
                  'all texture PNG SHA+source instance IDs and fresh linear/sRGB 2x2 readback calibration; unknown/unreadable outputs retained',
                  'original shader full RGB and repeated full RGB; full saved native/MCP request/response bindings'],
                'minimum_body_readbacks':['/maker/snapshot?regions=all','/maker/face/base','/maker/body/base','/maker/face/express','/maker/abmx'],
                'restoration':'Independent before/after full public state, actor/bridge identity, actual all-renderer and all-scene-light capture states; retain cleanup errors. Do not relabel previous history/prefix as current.',
                'new_pose_policy':'If all30 native cases alter pose/source, reacquire and recompute intersections from NEW source/pose. Do not carry these triangle segment coordinates into a different pose.',
                'texture_sampling_scope':'UV+recorded texture scale/offset is a diagnostic candidate; texture sampling cannot certify actual shader channel/UV/remap/parallax/mip/clip/dither without those formulas.'},
              'optional_conditional_rgb':{'protocol':'HS2Mod/docs/hs2_ocular_visibility.md',
                'meshes':['o_eyelashes','o_namida','o_eyeshadow'],
                'variants':'All eight subsets plus full repeat, isolated_layer to preserve enabled states; same frozen pose/camera and paired geometry+material textures.',
                'gates':'Existing exact source/state gates, world geometry 1e-6, repeat RGB <=2 byte, effect >4 byte. Do not change thresholds.',
                'additional_renderer_effects':'If needed, separate full vs one excluded o_eyebase_L/R/o_tooth/o_head measurements with full repeat; only conditional renderer RGB effects, never triangle-specific visibility.',
                'certifies_depth_or_anatomy':False},
              'depth_requirement':{'current_native_capability_certified':False,
                'not_required_for_next_minimal_readback':'Capture geometry/RGB/textures+marker first; no shader feature code requested now.',
                'future_if_needed':'Actual original-shader fragment survival/depth/renderer-triangle identity at same pixel sample, camera, pose and material inputs. Geometric depth, override shader depth and zero RGB effect cannot replace this.',
                'small_segments':'All reported segments <0.3 px at 512. A magnified ROI needs a new actual projection marker, not display enlargement; no fitted visibility threshold.'}}
(OUT/'capture_requirements.json').write_text(json.dumps(requirements,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'pair_table.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
test=subprocess.run([sys.executable,str(ROOT/'tools/crossmesh_semantics/test_analyze.py')],capture_output=True,text=True)
sources=[bound(p) for p in sorted((ROOT/'tools/crossmesh_semantics').glob('*.py'))]
finding={'report':bound(OUT/'report.json'),'asset_comparison':bound(OUT/'asset_geometry_comparison.json'),
         'requirements':bound(OUT/'capture_requirements.json'),'pair_table':bound(OUT/'pair_table.json'),
         'source_files':sources,'tests':{'command':[sys.executable,'tools/crossmesh_semantics/test_analyze.py'],
             'exit_code':test.returncode,'stdout':test.stdout,'stderr':test.stderr},
         'baseline_existing_crossings':r['windows'][0]['baseline_crossing_count'],
         'new_crossings':[w['new_pair_count'] for w in r['windows']],
         'coordinate_certified_views':sum(v['pixel_coordinate_certified'] for w in r['windows'] for v in w['views']),
         'shader_visible_certified_pairs':0,'anatomy_certified_pairs':0,
         'all84_raw_geometric_rays_have_foreground':all(p['raw_foreground_hits_above_existing_pair_epsilon'] for w in r['windows'] for v in w['views'] for p in v['projections']),
         'max_barycentric_reconstruction_error':max(s['barycentric']['max_reconstruction_error'] for w in r['windows'] for p in w['pairs'] for s in p['sides']),
         'max_projected_segment_px':max(p['projected_length_px'] for w in r['windows'] for v in w['views'] for p in v['projections']),
         'source_asset_relationship':'Installed p_cf_head_02 source ordered vertices/triangles/UV0/normals/tangents/weights/indices/bindposes match float32 or int64 for all8 live head meshes; UV1 present only head and namida, absent elsewhere. Exact full live source remains unchanged across baseline/candidate.',
         'port_semantics':'Face2 src/render/scene.py deliberately omits namida/eyeshadow (DEFERRED), and its own comments state their formulas are incomplete. This port cannot certify the live shaders.',
         'port_source_bindings':[bound(ROOT/'src/render/scene.py'),bound(ROOT/'src/render/shading.py')],
         'no_whitelist_or_quality_threshold_change':True,
         'limitations':(['9 pixel coordinate contracts reject projection mismatch'] if not all(v['pixel_coordinate_certified'] for w in r['windows'] for v in w['views']) else [])+['84 raw two-sided ray foreground hits do not prove fragment occlusion','Live texture contents not read back in these snapshots','No original-shader depth/fragment survival certificate','Asset geometry equality does not imply anatomical suitability']}
(OUT/'findings.json').write_text(json.dumps(finding,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:finding[k] for k in ['baseline_existing_crossings','new_crossings','all84_raw_geometric_rays_have_foreground','max_barycentric_reconstruction_error','max_projected_segment_px']}))
raise SystemExit(test.returncode)
