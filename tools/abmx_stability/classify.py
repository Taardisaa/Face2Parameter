"""Classify fixed-input Apply dynamics; simulated time is never runtime evidence."""
from __future__ import annotations
import copy
import numpy as np
from tools.abmx_replay.model import (FLAG_FIELDS, cache_state, modifier, replay_apply, serialized,
                                     transform, unity_zero)
from tools.abmx_replay.validate_trace import sha, trs_errors, cache_errors


def source_contract(contract):
    if contract.get('plugin_version') != '4.4.6.0' or contract.get('plugin_mvid') != '5442c72a-f463-4bf9-831a-247be87146c8':
        raise ValueError('Unsupported installed source branch')
    for path, digest in ((contract['assembly_path'], contract['assembly_sha256']),
                         (contract['unity_core_path'], contract['unity_core_sha256'])):
        if sha(path) != digest:
            raise ValueError('Installed source assembly hash changed')
    if any(sha(row['path']) != row['sha256'] for row in contract['sources']):
        raise ValueError('Installed source decompilation hash changed')
    return {'assembly_sha256': contract['assembly_sha256'], 'unity_core_sha256': contract['unity_core_sha256'],
            'apply_method_il_sha256': contract['apply_method_il_sha256'], 'sources': contract['sources']}


def regimes(value):
    m = modifier(value)
    length, position = m['length'] != np.float32(1), np.any(m['position'] != np.float32(0))
    if length and position:
        return 'combined'
    if length:
        return 'length_only'
    if position:
        return 'position_only'
    if np.any(m['scale'] != 1) or np.any(m['rotation'] != 0):
        return 'scale_rotation_only'
    return 'identity_no_active_channels'


def classify(event, *, long_count=512):
    """Initial actual before/cache + fixed resolved modifier; no observed after input."""
    if type(long_count) is not int or not 2 <= long_count <= 10000:
        raise ValueError('Explicit diagnostic count 2..10000 required')
    state = cache_state(event['before']['cache']['fields'])
    local = transform({key: event['before'][key] for key in ('local_position', 'local_rotation_xyzw', 'local_scale')})
    m = modifier(event['resolved_modifier'])
    regime = regimes(event['resolved_modifier'])
    reasons = []
    if event['additional_modifiers'] or event['is_during_h_scene'] or event['no_rotation_excluded']:
        reasons.append('additional_scene_or_rotation_exclusion_branch')
    if not state['_hasBaseline'] or any(state[key] for key in FLAG_FIELDS if key != '_hasBaseline'):
        reasons.append('nonclean_initial_flags_restore_or_pending_force_branch')
    if regime == 'identity_no_active_channels':
        reasons.append('identity_no_active_channels_outside_four_regimes')
    if float(m['length']) < .1:
        reasons.append('length_below_0_1_historical_direction_branch')
    if unity_zero(local['local_position']):
        reasons.append('unity_approximately_zero_current_position')
    if unity_zero(state['_positionBaseline']):
        reasons.append('unity_approximately_zero_historical_position_channel')
    if state['_lenBaseline'] <= 0:
        reasons.append('nonpositive_historical_length_baseline')
    result = {'regime': regime, 'supported_selected_branch': not reasons, 'unsupported_reasons': reasons,
              'initial_local': serialized(local), 'initial_private_fields': serialized(state), 'modifier': serialized(m),
              'input_source': 'complete actual first candidate before plus earlier-measured private state, never actual after',
              'assumptions': ['Resolved modifier and all baseline/history fields fixed between calls.',
                              'No animator, native Update, restore, other plugin or unobserved writer changes local state.',
                              'Clean initial flags and persistent nonzero position channels; non-H scene and nonexcluded rotation.',
                              'Current-direction path remains selected; no approximately-zero or special-length fallback.'],
              'actual_runtime_stability_proven': False, 'controlled_common_count_proven': False,
              'numeric_long_run_is_runtime_evidence': False}
    if reasons:
        result['mathematical_classification'] = 'unsupported_branch_no_generalization'
        return result
    R = float(state['_lenBaseline'])*float(m['length'])
    d = m['position'].astype(float)
    D = float(np.linalg.norm(d))
    if regime == 'combined':
        unit = d/D
        result.update(mathematical_classification='generally_nonidempotent_direction_iteration',
                      position_map='p[n+1] = R * normalize(p[n]) + d', R=R, D=D,
                      positive_fixed_point=((R+D)*unit).tolist(),
                      negative_fixed_point=(-(R-D)*unit).tolist() if R > D else None,
                      local_angular_contraction_at_positive_fixed_point=R/(R+D),
                      negative_fixed_point_angular_multiplier=R/(R-D) if R > D else None,
                      exact_arithmetic_limit='Non-collinear initial direction converges toward d direction while selected branch stays valid; anti-parallel fixed point exists when R>D and is angularly unstable.',
                      special_boundary='R=D and anti-parallel gives zero next position, outside this selected branch proof.')
    else:
        result.update(mathematical_classification='idempotent_after_one_apply_under_fixed_selected_branch',
                      position_map={'length_only': 'p[n+1] = R * normalize(p[n]); radius R after first call',
                                    'position_only': 'p[n+1] = _posBaseline + d',
                                    'scale_rotation_only': 'p[n+1] = p[n] under clean position flags'}[regime],
                      scale_rotation_map='Scale=baseline_scale*modifier_scale; q=baseline_q*Euler(delta), not cumulative composition.',
                      mathematical_idempotence_is_bitwise_float32_claim=False)
    first, previous, tail = None, None, []
    records, branch_counts = [], {}
    initial = copy.deepcopy(local)
    for i in range(1, long_count+1):
        prediction = replay_apply(before=serialized(local), cache=serialized(state), coordinate_modifiers=[serialized(m)],
            coordinate=event['coordinate'], additional_modifiers=[], bone_exists=True,
            rotation_excluded=False, is_during_h_scene=False)
        if 'length_use_historical_direction' in prediction['branches']:
            result['supported_selected_branch'] = False
            result['unsupported_reasons'].append('simulated_path_enters_historical_direction_fallback_at_call_'+str(i))
            break
        for branch in prediction['branches']:
            branch_counts[branch] = branch_counts.get(branch, 0)+1
        local, state = transform(prediction['after']), cache_state(prediction['cache_after'])
        step = float(np.linalg.norm(local['local_position'].astype(float)-previous)) if previous is not None else None
        if first is None:
            first = copy.deepcopy(prediction)
        if step is not None and i > long_count-16:
            tail.append(step)
        if i in {1, 2, 4, 8, 16, 32, 64, 128, 256, long_count}:
            records.append({'call': i, 'local_position': local['local_position'].tolist(), 'step_position_l2': step,
                            'position_drift_from_initial_l2': float(np.linalg.norm(local['local_position'].astype(float)-initial['local_position']))})
        previous = local['local_position'].astype(float)
    result['simulation'] = {'requested_count': long_count, 'completed_count': i if result['supported_selected_branch'] else i-1, 'sparse_steps': records,
                            'branch_counts': branch_counts, 'last_16_step_max_position_l2': max(tail) if tail else None,
                            'first_vs_final_trs': trs_errors(first['after'], serialized(local)) if first else None,
                            'first_vs_final_private_fields': cache_errors(first['cache_after'], serialized(state)) if first else None,
                            'runtime_verified': False}
    return result


def private_difference(a, b):
    """Compare fields separately from frame/other runtime wrapper metadata."""
    x, y = a['fields'], b['fields']
    if set(x) != set(y):
        raise ValueError('Private field scope changed')
    changes = {key: {'early': x[key], 'late': y[key]} for key in x if x[key] != y[key]}
    return {'changed_private_fields': changes,
            'changed_runtime_metadata_keys': [key for key in sorted(set(a) | set(b)) if key != 'fields' and a.get(key) != b.get(key)],
            'private_fields_unchanged': not changes,
            'metadata_change_implies_private_change': False}
