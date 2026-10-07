"""Read-only review of shared Torch Apply against independent actual-call replay.

No game calls. Observed before/cache anchors certify individual calls only.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from src.hs2_abmx_torch import FLAGS, apply_transition, tensor_cache
from tools.abmx_replay.model import IDENTITY, effective_modifier, replay_apply, require
from tools.abmx_replay.validate_trace import (cache_errors, local_only, read,
                                            replay_event, sha, trs_errors, validate)


def pack(value):
    return [*value['scale'], float(value['length']), *value['position'], *value['rotation']]


def execute(before, fields, effective, *, excluded=False, h_scene=False,
            device='cpu', mutate_cache=None):
    local = tuple(torch.tensor([before[key]], dtype=torch.float32, device=device,
                               requires_grad=True)
                  for key in ('local_position', 'local_rotation_xyzw', 'local_scale'))
    cache = tensor_cache(fields, device=device)
    cache = {key: value.detach().clone().requires_grad_(True) if key not in FLAGS
             else value.clone() for key, value in cache.items()}
    if mutate_cache:
        mutate_cache(cache)
    mod = torch.tensor([pack(effective)], dtype=torch.float32, device=device, requires_grad=True)
    result, after_cache = apply_transition(local, cache, mod, rotation_excluded=excluded,
                                          is_during_h_scene=h_scene)
    output = {key: value.detach().cpu().numpy()[0].tolist()
              for key, value in zip(('local_position', 'local_rotation_xyzw', 'local_scale'), result)}
    def recorded(value):
        array = value.detach().cpu().numpy()[0]
        if np.asarray(array).dtype.kind == 'f' and not np.isfinite(array).all():
            return {'invalid_nonfinite_cache': str(array)}
        return array.tolist()
    state = {key: recorded(value) for key, value in after_cache.items()}
    # Linear probe avoids loss overflow in the deliberate 1e20 input case.
    loss = sum(value.sum() for value in result)
    inputs = {'local_position': local[0], 'local_rotation_xyzw': local[1],
              'local_scale': local[2], 'modifier': mod,
              **{key: value for key, value in cache.items() if key not in FLAGS}}
    grads = torch.autograd.grad(loss, tuple(inputs.values()), allow_unused=True)
    bad = [key for key, grad in zip(inputs, grads) if grad is not None and not torch.isfinite(grad).all()]
    return {'after': output, 'cache_after': state, 'gradient_nonfinite_inputs': bad,
            'gradient_all_finite': not bad, 'gradient_loss': 'sum of output local TRS components',
            'unused_gradient_inputs': [key for key, grad in zip(inputs, grads) if grad is None]}


def baseline():
    before = {'local_position': [3., 4., 0.], 'local_rotation_xyzw': [0., 0., 0., 1.],
              'local_scale': [1., 1., 1.]}
    fields = {key: False for key in FLAGS}
    fields.update(_hasBaseline=True, _sclBaseline=[1., 1., 1.], _posBaseline=[3., 4., 0.],
                  _positionBaseline=[3., 4., 0.], _rotBaseline=[0., 0., 0., 1.], _lenBaseline=5.)
    return before, fields


def synthetic(device):
    cases = []
    for name in ('identity_skip', 'zero_length_baseline_inactive', 'current_zero_fallback',
                 'short_length_fallback', 'h_scene_fallback', 'position_restore_overwrites_length',
                 'pending_length_restore_no_history', 'finite_inactive_overflow'):
        before, fields = baseline()
        mod = copy.deepcopy(IDENTITY)
        h_scene = False
        if name == 'zero_length_baseline_inactive':
            before['local_position'] = [0., 0., 0.]
            fields['_positionBaseline'] = [0., 0., 0.]
            fields['_lenBaseline'] = 0.
        elif name == 'current_zero_fallback':
            before['local_position'] = [0., 0., 0.]
            mod['length'] = 1.2
        elif name == 'short_length_fallback':
            mod['length'] = .05
        elif name == 'h_scene_fallback':
            h_scene, mod['length'] = True, 1.2
        elif name == 'position_restore_overwrites_length':
            fields['_changedPosition'] = True
            mod['length'] = 1.2
        elif name == 'pending_length_restore_no_history':
            fields['_lenModForceUpdate'] = True
            fields['_positionBaseline'] = [0., 0., 0.]
        elif name == 'finite_inactive_overflow':
            before['local_position'] = [1e20, 0., 0.]
            fields['_positionBaseline'] = [0., 0., 0.]
            fields['_lenBaseline'] = 1e20
        expected = replay_apply(before=before, cache=fields, coordinate_modifiers=[mod], coordinate=0,
                                additional_modifiers=[], bone_exists=True, rotation_excluded=False,
                                is_during_h_scene=h_scene)
        row = {'name': name, 'before': before, 'cache': fields, 'modifier': mod,
               'scope': 'pathological finite float32 robustness probe, not observed runtime range'
                        if name == 'finite_inactive_overflow' else 'analytic branch probe'}
        try:
            result = execute(before, fields, mod, h_scene=h_scene, device=device)
            row.update(result, local_error=trs_errors(result['after'], expected['after']),
                       cache_error=cache_errors(result['cache_after'], expected['cache_after']))
        except Exception as exc:
            row['rejected'] = str(exc)
        cases.append(row)
    # tensor_cache rejects this, but direct apply_transition currently trusts its cache tensors.
    before, fields = baseline()
    def corrupt(cache):
        cache['_sclBaseline'] = torch.full((1, 3), float('nan'), device=device, requires_grad=True)
    row = {'name': 'direct_tensor_cache_nan_in_unused_scale_branch',
           'scope': 'invalid fabricated cache bypassing tensor_cache; guard diagnostic only'}
    try:
        result = execute(before, fields, IDENTITY, device=device, mutate_cache=corrupt)
        row.update(result)
    except Exception as exc:
        row['rejected'] = str(exc)
    cases.append(row)
    return cases


def analytic_length_gradient(device):
    before, fields = baseline()
    local = tuple(torch.tensor([before[key]], device=device, requires_grad=True)
                  for key in ('local_position', 'local_rotation_xyzw', 'local_scale'))
    cache = tensor_cache(fields, device=device)
    modifier = torch.tensor([[1., 1., 1., 1.4, .02, .03, .01, 0., 0., 0.]],
                            device=device, requires_grad=True)
    after, _ = apply_transition(local, cache, modifier)
    gp, gm = torch.autograd.grad(after[0].sum(), (local[0], modifier))
    # d(sum(5 L p/|p| + offset))/dp = 5L/|p| (1 - sum(p)p/|p|^2).
    expected_p = np.array([.224, -.168, 1.4])
    expected_m = np.array([0., 0., 0., 7., 1., 1., 1., 0., 0., 0.])
    p_error = float(np.max(np.abs(gp.detach().cpu().numpy()[0] - expected_p)))
    m_error = float(np.max(np.abs(gm.detach().cpu().numpy()[0] - expected_m)))
    return {'name': '3_4_5_active_length_analytic_gradient', 'position_gradient': gp.tolist(),
            'modifier_gradient': gm.tolist(), 'position_max_error': p_error,
            'modifier_max_error': m_error, 'tolerance': 2e-6, 'passed': max(p_error, m_error) < 2e-6}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    require(not args.out.exists(), 'New review output path required')
    summary, contract = read(args.summary), read(args.contract)
    require(sha(contract['assembly_path']) == contract['assembly_sha256'], 'Installed DLL changed')
    require(sha(contract['unity_core_path']) == contract['unity_core_sha256'], 'Unity core changed')
    require(all(sha(item['path']) == item['sha256'] for item in contract['sources']), 'Installed source changed')
    report = {'schema_version': 1, 'device': args.device, 'torch_version': torch.__version__,
              'scope': 'Single actual Apply call conditioned on observed before/cache; not future candidate prediction across external boundaries',
              'shared_core_modified': False, 'summary_sha256': sha(args.summary),
              'contract_sha256': sha(args.contract), 'cases': [], 'numeric_dtype': 'float32'}
    for case in summary['cases']:
        require(sha(case['trace_path']) == case['trace_sha256'], 'Trace changed')
        trace = read(case['trace_path'])
        independent = validate(trace, contract)
        require(independent['passed'], 'Independent NumPy replay failed')
        rows = []
        for event in trace['events']:
            before = local_only(event['before'])
            fields = event['before']['cache']['fields']
            expected = replay_event(event, before, fields)
            effective = effective_modifier([event['resolved_modifier']], event['coordinate'],
                                           event['additional_modifiers'], event['coordinate_specific'])
            require(effective is not None, 'Null modifier must be handled by caller')
            result = execute(before, fields, effective, excluded=event['no_rotation_excluded'],
                             h_scene=event['is_during_h_scene'], device=args.device)
            actual_error = trs_errors(result['after'], local_only(event['after']))
            numpy_error = trs_errors(result['after'], expected['after'])
            actual_cache = cache_errors(result['cache_after'], event['after']['cache']['fields'])
            numpy_cache = cache_errors(result['cache_after'], expected['cache_after'])
            rows.append({'sequence': event['sequence'], 'frame': event['frame'], 'branches': expected['branches'],
                         **result, 'actual_error': actual_error, 'numpy_error': numpy_error,
                         'actual_cache_error': actual_cache, 'numpy_cache_error': numpy_cache,
                         'forward_passed': all(item['passed'] for item in (actual_error, numpy_error, actual_cache, numpy_cache))})
        report['cases'].append({'name': case['name'], 'trace_path': case['trace_path'],
                                'trace_sha256': case['trace_sha256'], 'rows': rows,
                                'external_boundaries': independent['inter_call_boundaries']})
    all_rows = [row for case in report['cases'] for row in case['rows']]
    report.update(actual_call_count=len(all_rows), forward_passed=sum(row['forward_passed'] for row in all_rows),
                  finite_gradient_calls=sum(row['gradient_all_finite'] for row in all_rows),
                  max_actual_position_error=max(row['actual_error']['position_max_abs'] for row in all_rows),
                  max_actual_quaternion_error=max(row['actual_error']['quaternion_sign_equivalent_max_abs'] for row in all_rows),
                  max_numpy_position_error=max(row['numpy_error']['position_max_abs'] for row in all_rows),
                  max_numpy_quaternion_error=max(row['numpy_error']['quaternion_sign_equivalent_max_abs'] for row in all_rows))
    report['analytic_length_gradient'] = analytic_length_gradient(args.device)
    report['synthetic_diagnostics'] = synthetic(args.device)
    report['implementation_hashes'] = [{'path': str(path), 'sha256': sha(path)} for path in (
        Path(__file__), ROOT/'src/hs2_abmx_torch.py', ROOT/'src/hs2_deform_torch.py',
        Path(__file__).with_name('model.py'), Path(__file__).with_name('validate_trace.py'),
        Path(__file__).with_name('validate_geometry.py'))]
    report['package_import_compatibility'] = 'validate/verify_trace_header imported as tools.abmx_replay package'
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('actual_call_count', 'forward_passed', 'finite_gradient_calls',
                     'max_actual_position_error', 'max_actual_quaternion_error', 'max_numpy_position_error',
                     'max_numpy_quaternion_error', 'analytic_length_gradient')}, ensure_ascii=False))
    print('report='+str(args.out.resolve()))
    return 0 if all(row['forward_passed'] and row['gradient_all_finite'] for row in all_rows) else 2


if __name__ == '__main__':
    raise SystemExit(main())
