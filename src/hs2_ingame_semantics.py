"""Explicit consumer contracts; recorded state is not a runtime/likeness proof."""
from __future__ import annotations

import copy
import hashlib
import json
from numbers import Integral
from pathlib import Path

import numpy as np

N_SLIDERS = 59
MODES = {'native_only', 'installed_stateful_diagnostic', 'native_radial_target_v1'}
FLAGS = ('_hasBaseline', '_lenModForceUpdate', '_lenModNeedsPositionRestore',
         '_changedScale', '_changedRotation', '_changedPosition', '_forceApply')


class SemanticsError(ValueError):
    """Fail before mutation/scoring when a declared contract is unavailable."""


def require(condition, reason):
    if not condition:
        raise SemanticsError(reason)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_report_binding(descriptor, label):
    require(isinstance(descriptor, dict) and isinstance(descriptor.get('path'), str)
            and isinstance(descriptor.get('sha256'), str), label+': explicit report path/SHA required')
    path = Path(descriptor['path'])
    require(path.is_absolute() and path.suffix.lower() == '.json' and path.is_file()
            and file_sha(path) == descriptor['sha256'], label+': report bytes/path binding unavailable')
    return {'path': str(path.resolve()), 'sha256': descriptor['sha256'], 'hash_is_not_runtime_truth': True}


def finite_vector(value, size, label):
    raw = np.asarray(value)
    require(raw.shape == (size,) and raw.dtype.kind in 'fiu', f'{label}: exactly {size} numeric components required')
    result = raw.astype(np.float64)
    require(np.isfinite(result).all(), f'{label}: finite values required')
    return result


def shape_values(response):
    require(response.get('count') == N_SLIDERS and isinstance(response.get('shapes'), list)
            and len(response['shapes']) == N_SLIDERS, 'Complete actual native59 readback required')
    values, seen = np.empty(N_SLIDERS, dtype=np.float64), set()
    for row in response['shapes']:
        index = row.get('index')
        require(isinstance(index, Integral) and not isinstance(index, bool) and 0 <= index < N_SLIDERS
                and index not in seen, 'Readback indices must cover 0..58 exactly once')
        values[index] = finite_vector([row['value']], 1, 'actual shape')[0]
        seen.add(index)
    return values


def profile_and_bounds(capabilities, range_mode='native', requested_profile=None):
    require(range_mode in {'native', 'installed'}, 'range_mode must be native or installed')
    require(capabilities.get('face_shape_count') == N_SLIDERS and capabilities.get('native_range') == [0, 1],
            'Current morph capability/native59 contract unavailable')
    plugin = capabilities.get('plugin')
    require(isinstance(plugin, dict) and type(plugin.get('loaded')) is bool, 'Loaded sampler capability unavailable')
    if plugin['loaded']:
        require(plugin.get('guid') == 'com.bepis.bepinex.sliderunlocker'
                and plugin.get('version') in {'18.2', '18.2.0', '18.2.0.0'}
                and capabilities.get('sampler_profile') == 'slider_unlocker_18_2'
                and capabilities.get('sampler_patches_registered') is True,
                'Loaded sampler profile is unsupported; native input bounds do not disable the plugin')
        profile = 'slider_unlocker_18_2'
    else:
        profile = 'vanilla'
    require(requested_profile is None or requested_profile == profile, 'Declared sampler profile differs from the actual loaded sampler')
    bounds = np.asarray([0., 1.])
    if range_mode == 'installed':
        require(plugin['loaded'] and capabilities.get('installed_available') is True, 'Installed range opt-in is unavailable')
        bounds = finite_vector(capabilities.get('installed_range'), 2, 'installed bounds')
        require(bounds[0] <= 0 and bounds[1] >= 1 and bounds[0] < bounds[1], 'Unsupported installed bounds')
    return profile, bounds


def validate_write(values, indices, capabilities, range_mode, frozen=()):
    idx = list(range(N_SLIDERS)) if indices is None else list(indices)
    require(idx and all(isinstance(i, Integral) and not isinstance(i, bool) and 0 <= i < N_SLIDERS for i in idx)
            and len(set(idx)) == len(idx), 'Requested indices must be unique integers 0..58')
    vals = finite_vector(values, len(idx), 'requested shape values')
    _, bounds = profile_and_bounds(capabilities, range_mode)
    require(not (set(idx) & set(frozen)), 'Requested frozen indices cannot be silently skipped; use an explicit supported installed range')
    require(np.all(vals >= bounds[0]) and np.all(vals <= bounds[1]),
            f'Requested values exceed {range_mode} bounds {bounds.tolist()}; values are never clipped')
    return vals, [int(i) for i in idx]


def inspect_abmx(abmx, mode):
    require(mode in MODES, 'Unknown named semantics mode')
    require(isinstance(abmx, dict) and isinstance(abmx.get('bones'), list), 'Full inherited ABMX state unavailable')
    inherited, names, baselines = [], set(), {}
    for row in abmx['bones']:
        name = row.get('name')
        require(isinstance(name, str) and name and name not in names, 'Inherited ABMX names missing/duplicated')
        names.add(name)
        scale = finite_vector(row.get('scale'), 3, name+'/scale')
        position = finite_vector(row.get('position'), 3, name+'/position')
        rotation = finite_vector(row.get('rotation'), 3, name+'/rotation')
        length = finite_vector([row.get('length')], 1, name+'/length')[0]
        require(np.all(scale > 0), 'Unsupported nonpositive inherited ABMX scale: '+name)
        active = bool(np.any(scale != 1) or length != 1 or np.any(position != 0) or np.any(rotation != 0))
        mixed = bool(length != 1 and np.any(position != 0))
        wrapper = row.get('runtime_baseline')
        if row.get('exists') is True:
            require(isinstance(wrapper, dict) and wrapper.get('missing_fields') == []
                    and isinstance(wrapper.get('fields'), dict), 'Inherited private cache unavailable: '+name)
            fields = wrapper['fields']
            require(all(type(fields.get(flag)) is bool for flag in FLAGS), 'Inherited cache flags unavailable: '+name)
            for key, size in (('_sclBaseline', 3), ('_posBaseline', 3), ('_rotBaseline', 4), ('_positionBaseline', 3)):
                finite_vector(fields.get(key), size, name+'/'+key)
            finite_vector([fields.get('_lenBaseline')], 1, name+'/_lenBaseline')
            baselines[name] = {key: copy.deepcopy(value) for key, value in wrapper.items() if key != 'frame_count'}
            if mode == 'native_only':
                require(not any(fields[flag] for flag in FLAGS if flag != '_hasBaseline'),
                        'Inherited pending ABMX restore/force state requires an explicit diagnostic/provider mode: '+name)
        elif active:
            raise SemanticsError('Active inherited ABMX modifier has no actual instance: '+name)
        if active:
            inherited.append({'name': name, 'mixed_length_position': mixed})
    if mode == 'native_only':
        require(not inherited, 'native_only refuses inherited nonidentity ABMX, including unknown/stateful Length+Position; user ABMX is not reset')
    return {'inherited_active': inherited, 'baselines': baselines,
            'runtime_certified': False, 'unknown_additional_writers_not_certified': True}


def snapshot_context(boundary, mode, range_mode, requested_profile=None, *, include_actor_transforms=False):
    caps = boundary('/maker/morph/capabilities')
    profile, bounds = profile_and_bounds(caps, range_mode, requested_profile)
    native = shape_values(boundary('/maker/face/shapes'))
    require(type(caps.get('abmx_plugin_loaded')) is bool, 'ABMX loaded capability unavailable')
    abmx = boundary('/maker/abmx?runtime_baseline=true') if caps['abmx_plugin_loaded'] else {'bones': [], 'plugin_loaded': False}
    inspection = inspect_abmx(abmx, mode)
    geometry = boundary('/maker/geometry'+('?include_actor_transforms=true' if include_actor_transforms else ''), timeout=60)
    if include_actor_transforms:
        require(geometry.get('include_actor_transforms') is True
                and geometry.get('transform_scope') == 'actor_hierarchy_and_ancestors',
                'Loaded bridge did not confirm same-frame actor transform coverage')
    parameters = boundary('/maker/snapshot?region=all')
    require(isinstance(parameters, dict) and 'body_shapes' in parameters and 'face_base' in parameters,
            'Source/body public configuration snapshot unavailable')
    require(geometry.get('snapshot_kind') == 'maker_live_skinned_geometry' and geometry.get('schema_version') == 1,
            'Actual geometry context unavailable')
    character, game = geometry['character'], geometry['game']
    require(character.get('head_id') == caps.get('head_id') and character.get('native_count') == N_SLIDERS
            and np.array_equal(np.asarray(character['shape_value_face'], dtype=np.float32), native.astype(np.float32)),
            'Head/native changed while reading context')
    require(isinstance(character.get('transform_id'), int) and isinstance(game.get('game_assembly_sha256'), str),
            'Actual actor/game source identity unavailable')
    transforms = {row['id']: row for row in geometry.get('transforms', [])}
    root_id = character.get('head_root_transform_id')
    require(root_id in transforms, 'Complete actual head-root hierarchy unavailable')
    head_ids = set()
    for identity in transforms:
        cursor, visited = identity, set()
        while cursor in transforms and cursor not in visited:
            if cursor == root_id:
                head_ids.add(identity)
                break
            visited.add(cursor)
            cursor = transforms[cursor].get('parent_id')
    head_names = {transforms[identity]['name'] for identity in head_ids}
    empty_head_modifiers = [row['name'] for row in abmx['bones'] if row.get('exists') is True
                            and ((row.get('runtime_baseline') or {}).get('bone_transform_id') in head_ids or row['name'] in head_names)]
    if mode == 'native_only':
        require(not empty_head_modifiers, 'native_only refuses existing head-descendant modifiers even with identity/clean public state: native refresh/Reset can alter historical-radius pose; do not silently remove them')
    meshes = sorted(({'renderer_path': row['renderer_path'], 'source_geometry_sha256': row['source_geometry_sha256']}
                     for row in geometry['meshes'] if row.get('mesh_name') == 'o_head'), key=lambda row: row['renderer_path'])
    require(meshes, 'Actual head source identity unavailable')
    public_abmx = [{key: copy.deepcopy(row.get(key)) for key in ('name', 'exists', 'scale', 'length', 'position', 'rotation', 'coordinate_modifiers')}
                   for row in sorted(abmx['bones'], key=lambda row: row['name'])]
    binding = {'actor_transform_id': character['transform_id'], 'head_id': character['head_id'], 'head_sources': meshes,
               'game': game, 'head_root_transform_id': root_id, 'head_descendant_modifier_names': empty_head_modifiers,
               'profile': profile, 'range_mode': range_mode, 'capabilities': caps,
               'native59': native.tolist(), 'abmx_coordinate': abmx.get('coordinate'), 'abmx_version': abmx.get('plugin_version'),
               'public_abmx': public_abmx, 'private_cache': inspection['baselines'],
               'expression_configuration': copy.deepcopy(character.get('expression', {})),
               'native_driver_configuration': copy.deepcopy((geometry.get('native_face_drivers') or {}).get('configured', {})),
               'other_public_parameters': {key: value for key, value in parameters.items() if key not in ('face_shapes', 'outlier_summary')}}
    return {'binding': binding, 'binding_sha256': digest(binding), 'mode': mode, 'profile': profile, 'bounds': bounds.tolist(),
            'capabilities': caps, 'native59': native.tolist(), 'abmx': abmx, 'inspection': inspection,
            'geometry': geometry, 'geometry_read_json_sha256': digest(geometry), 'source_parameters': parameters,
            'bridge_mvid': game.get('bridge_mvid'), 'bridge_mvid_available': game.get('bridge_mvid') is not None,
            'runtime_certified': False}


def check_context(previous, current, *, allow_native_cache_refresh=False, expected_native=None):
    before, after = copy.deepcopy(previous['binding']), copy.deepcopy(current['binding'])
    if allow_native_cache_refresh:
        old_cache, new_cache = before.pop('private_cache'), after.pop('private_cache')
        require(set(old_cache) == set(new_cache), 'Native update changed inherited modifier/cache coverage')
        for name in old_cache:
            a, b = copy.deepcopy(old_cache[name]), copy.deepcopy(new_cache[name])
            af, bf = a.pop('fields'), b.pop('fields')
            require(a == b, 'Native update changed private cache source/bone identity: '+name)
            if previous['mode'] == 'native_only' and name not in before['head_descendant_modifier_names']:
                require(af == bf, 'Face native write/capture changed immutable non-head cache fields: '+name)
            # Native TRS baselines and flags may refresh through the installed
            # public UpdateShapeFace path; nonzero persistent history may not.
            for key, meaningful in (('_lenBaseline', float(af['_lenBaseline']) > 0),
                                    ('_positionBaseline', bool(np.any(np.asarray(af['_positionBaseline'], dtype=np.float32) != 0)))):
                require(not meaningful or np.array_equal(np.asarray(af[key], dtype=np.float32), np.asarray(bf[key], dtype=np.float32)),
                        'Native update changed measured persistent ABMX history: '+name+'/'+key)
    if expected_native is not None:
        require(np.array_equal(np.asarray(after['native59'], dtype=np.float32), np.asarray(expected_native, dtype=np.float32)),
                'Actual native readback differs from the requested values')
        before.pop('native59')
        after.pop('native59')
    require(before == after, 'Context is stale: actor/head/source/native/profile/ABMX/cache changed; create or explicitly refresh a context')


def validate_capture(response, expected_context, yaws, directory, resolution):
    require(response.get('capture_state_restored') is True, 'Capture state restoration not confirmed')
    views = response.get('views')
    require(isinstance(views, list) and len(views) == len(yaws), 'One complete native multi-yaw response required')
    frames, signatures, framing, images = set(), set(), [], []
    geometry_descriptor = response.get('paired_geometry')
    require(isinstance(geometry_descriptor, dict) and 'sha256' in geometry_descriptor, 'Paired geometry export unavailable')
    geometry_path = Path(geometry_descriptor['path']).resolve()
    require(geometry_path.parent == Path(directory).resolve() and file_sha(geometry_path) == geometry_descriptor['sha256'],
            'Actual paired geometry path/bytes mismatch')
    geometry = json.loads(geometry_path.read_text(encoding='utf-8'))
    require(geometry['frame_count'] == geometry['frame_count_end'] == geometry_descriptor['frame_count'], 'Geometry changed frame')
    require(geometry['character']['transform_id'] == expected_context['binding']['actor_transform_id']
            and geometry['character']['head_id'] == expected_context['binding']['head_id']
            and np.array_equal(np.asarray(geometry['character']['shape_value_face'], dtype=np.float32),
                               np.asarray(expected_context['native59'], dtype=np.float32)), 'Paired actual input readback differs')
    paired_sources = sorted(({'renderer_path': row['renderer_path'], 'source_geometry_sha256': row['source_geometry_sha256']}
                             for row in geometry['meshes'] if row.get('mesh_name') == 'o_head'), key=lambda row: row['renderer_path'])
    require(paired_sources == expected_context['binding']['head_sources'] and geometry['game'] == expected_context['binding']['game'],
            'Paired actual geometry/game source changed')
    require(geometry['character'].get('expression', {}) == expected_context['binding']['expression_configuration']
            and (geometry.get('native_face_drivers') or {}).get('configured', {})
                == expected_context['binding']['native_driver_configuration'],
            'Paired expression/native driver configuration changed')
    for view, yaw in zip(views, yaws):
        require(view.get('yaw') == yaw and view.get('paired_pose_unchanged') is True
                and view.get('paired_capture_state_unchanged') is True and view.get('paired_lights_unchanged') is True
                and view.get('paired_visibility_sampled_unchanged') is True, 'Multi-view frozen pose/state/lights/visibility evidence failed')
        pair = view['paired_geometry']
        require(pair['sha256'] == geometry_descriptor['sha256'] and pair['pose_signature'] == geometry['pose_signature']
                and view['pose_signature_before_render'] == view['pose_signature_after_render'] == geometry['pose_signature']
                and view['frame_count'] == view['frame_count_before_render'] == geometry['frame_count'], 'View/geometry pose-frame binding failed')
        require(view['visibility_sample_signature_before_render'] == view['visibility_sample_signature_after_render']
                == geometry['visibility_sample_signature'], 'Actual visibility signature values differ')
        path = Path(view['path']).resolve()
        require(path.parent == Path(directory).resolve() and path.is_file(), 'Rendered PNG missing or outside the evaluation output')
        data = path.read_bytes()
        require(data[:8] == b'\x89PNG\r\n\x1a\n' and len(data) >= 24
                and int.from_bytes(data[16:20], 'big') == resolution and int.from_bytes(data[20:24], 'big') == resolution,
                'Actual PNG dimensions/signature differ')
        images.append({'path': str(path), 'sha256': hashlib.sha256(data).hexdigest(), 'yaw': yaw})
        frames.add(view['frame_count'])
        signatures.add(view['pose_signature_before_render'])
        framing.append({key: view.get(key) for key in ('target', 'ortho_size', 'distance', 'fov', 'head_center', 'head_size')})
    require(len(frames) == len(signatures) == 1 and all(value == framing[0] for value in framing), 'Multi-view framing/state was not shared')
    return {'images': images, 'geometry': geometry_descriptor, 'actual_input_readback': geometry['character'],
            'frame': next(iter(frames)), 'pose_signature': next(iter(signatures)), 'fixed_framing': framing[0],
            'capture_response_json_sha256': digest(response), 'same_frame_pose_and_framing_verified': True,
            'source_model_or_anatomical_truth_certified': False, 'beauty_is_likeness': False}
