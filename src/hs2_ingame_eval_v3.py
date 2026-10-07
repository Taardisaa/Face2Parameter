"""V3 consumer adopts its predeclared actual capture after observer STOP."""
from __future__ import annotations

import copy
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.hs2_capture_gt import call
from src.hs2_ingame_semantics import (
    MODES,
    N_SLIDERS,
    check_context,
    digest,
    file_sha,
    finite_vector,
    require,
    shape_values,
    snapshot_context,
    validate_capture,
    validate_write,
    verify_report_binding,
)

__all__ = ['N_SLIDERS', 'InGameEvaluator', 'slider_names']

ROOT = str(Path(__file__).resolve().parents[1])
YAWS_A = [-20., -10., 0., 10., 20.]
YAWS_B = [-16., -6., 3., 13., 24.]


class InGameEvaluator:
    """Context → guarded writes → one paired multi-yaw capture → downstream score.

    native_radial_target_v1 requires an explicit lifecycle provider; this module
    supplies no automatic provider, no ABMX reset and no global runtime certificate.
    """

    def __init__(self, res=512, hide_hair=True, scorer=None, work_dir=None,
                 yaws_a=None, yaws_b=None, force_single=False, *,
                 semantics_mode='native_only', range_mode='native', sampling_profile=None,
                 boundary=None, radial_provider=None, logical_controls=None, capture_options=None):
        require(semantics_mode in MODES, 'Unknown named semantics mode')
        require(range_mode in {'native', 'installed'}, 'Explicit native/installed range mode required')
        require(type(res) is int and 16 <= res <= 4096, 'Resolution must be integer 16..4096')
        self.res, self.hide_hair = res, int(bool(hide_hair))
        self.yaws = {'a': self._yaws(YAWS_A if yaws_a is None else yaws_a),
                     'b': self._yaws(YAWS_B if yaws_b is None else yaws_b)}
        self.work_dir = str(Path(work_dir or Path(os.environ.get('TEMP', '.'))/'hs2_ingame_eval').resolve())
        Path(self.work_dir).mkdir(parents=True, exist_ok=True)
        self.mode, self.range_mode, self.sampling_profile = semantics_mode, range_mode, sampling_profile
        self.boundary = boundary or call
        self.radial_provider, self.logical_controls = radial_provider, copy.deepcopy(logical_controls)
        if self.mode == 'native_radial_target_v1':
            require(radial_provider is not None and all(callable(getattr(radial_provider, name, None))
                    for name in ('prepare', 'apply', 'validate_capture', 'restore')),
                    'native_radial_target_v1 requires an explicit controlled provider; none is supplied by this consumer')
            require(scorer is not None and callable(getattr(scorer, 'score', None)),
                    'Controlled radial evaluation requires an explicit task scorer')
        else:
            require(radial_provider is None, 'Provider requires explicitly named native_radial_target_v1 mode')
        self.capture_options = self._options(capture_options or {})
        self._scorer, self._batch = scorer, False if force_single else None
        self.p0, self.frozen_oob = None, []
        self._context = self._original_context = None
        self._mutated = False
        self._active_execution = None
        self._active_request = None
        self._source_card = None
        self.last_capture = self.last_restore = None
        self.n_render = self.n_eval = 0
        self.t_render = self.t_score = 0.
        require(self.boundary('/status').get('inside_maker') is True, 'Character Maker is not open')

    @staticmethod
    def _yaws(values):
        require(isinstance(values, (list, tuple, np.ndarray)) and len(values) > 0, 'Nonempty yaw set required')
        return finite_vector(values, len(values), 'yaw set').tolist()

    @staticmethod
    def _options(options):
        allowed = {'settle', 'ortho_size', 'target', 'dist', 'pitch', 'roll', 'fov', 'margin', 'aa',
                   'visibility_method', 'force_skinning_recalculation'}
        require(isinstance(options, dict) and set(options) <= allowed, 'Unknown/unsafe capture option')
        result = {'settle': 3, **copy.deepcopy(options)}
        require(type(result['settle']) is int and 0 <= result['settle'] <= 60, 'Native settle is 0..60; never clamped')
        for key in ('ortho_size', 'dist', 'pitch', 'roll', 'fov', 'margin'):
            if key in result:
                result[key] = float(finite_vector([result[key]], 1, key)[0])
        if 'target' in result:
            result['target'] = ','.join(map(str, finite_vector(result['target'], 3, 'target')))
        if 'visibility_method' in result:
            require(result['visibility_method'] in {'renderer_enabled', 'isolated_layer'}, 'Unknown visibility method')
        if 'force_skinning_recalculation' in result:
            require(type(result['force_skinning_recalculation']) is bool, 'Explicit recalculation bool required')
            result['force_skinning_recalculation'] = str(result['force_skinning_recalculation']).lower()
        if 'aa' in result:
            require(type(result['aa']) is int and result['aa'] in (1, 2, 4, 8), 'Unsupported AA')
        return result

    def _snapshot(self, mode=None):
        return snapshot_context(self.boundary, mode or self.mode, self.range_mode, self.sampling_profile,
                                include_actor_transforms=getattr(self.radial_provider, 'requires_actor_transform_coverage', False))

    def _anchor(self, context):
        self._context = context
        if self.p0 is None:
            self.p0 = np.asarray(context['native59'], dtype=np.float64)
        if self._original_context is None:
            self._original_context = copy.deepcopy(context)
        self.frozen_oob = [i for i, value in enumerate(self.p0) if not 0 <= value <= 1] if self.range_mode == 'native' else []

    def _preflight(self):
        context = self._snapshot()
        if self._context is None:
            self._anchor(context)
        else:
            check_context(self._context, context)
            self._context = context
        return context

    def refresh_context(self):
        """Explicitly accept a different context only before our mutations."""
        require(not self._mutated, 'Restore before replacing the original context')
        self._context = self._original_context = self.p0 = None
        self._anchor(self._snapshot())
        return self._context['binding_sha256']

    @property
    def scorer(self):
        if self._scorer is None:
            from beauty_score import BeautyScorer
            self._scorer = BeautyScorer()
        return self._scorer

    def load_card(self, path):
        """Explicit load → preserve p0 → inherited-state check → warm capture."""
        require(not self._mutated, 'Restore before loading another card')
        card = Path(path).resolve()
        response = self.boundary('/maker/card/load', 'POST', {'path': str(card)}, timeout=180)
        self._context = self._original_context = None
        self.p0 = self.get_shapes().copy()
        self._source_card = {'path': str(card), 'sha256': file_sha(card) if card.is_file() else None, 'load_response': response}
        self._anchor(self._snapshot())  # Unsafe inherited state rejects BEFORE warm rendering.
        if self.mode != 'native_radial_target_v1':
            self._render(('a',), '_warm')
        return response

    def get_shapes(self):
        return shape_values(self.boundary('/maker/face/shapes'))

    def _write(self, values, indices, range_mode):
        body = {'values': list(map(float, values)), 'indices': indices, 'range_mode': range_mode}
        if self._batch is not False:
            try:
                self.boundary('/maker/face/shapes/batch', 'POST', body, timeout=60)
                self._batch = True
                return
            except SystemExit as exc:
                if not str(exc).startswith('[404]'):
                    raise
                self._batch = False
        for index, value in zip(indices, values):
            self.boundary('/maker/face/shapes', 'POST', {'index': index, 'value': float(value), 'range_mode': range_mode}, timeout=30)

    def set_shapes(self, values, indices=None):
        require(self.mode != 'native_radial_target_v1', 'Direct native writes invalidate radial offsets; use controlled evaluate')
        context = self._preflight()
        vals, idx = validate_write(values, indices, context['capabilities'], self.range_mode, self.frozen_oob)
        expected = np.asarray(context['native59']).copy()
        expected[idx] = vals
        self._mutated = True
        try:
            self._write(vals, idx, self.range_mode)
            fresh = self._snapshot()
            check_context(context, fresh, allow_native_cache_refresh=True, expected_native=expected)
            self._context = fresh
        except (Exception, SystemExit):
            self.restore()
            raise

    def _directory(self, tag):
        require(isinstance(tag, str) and re.fullmatch(r'[A-Za-z0-9_.-]{1,100}', tag) is not None, 'Unsafe evaluation tag')
        directory = Path(self.work_dir)/(tag+'_'+uuid.uuid4().hex)
        directory.mkdir()
        return directory

    @staticmethod
    def _save(path, value):
        with Path(path).open('x', encoding='utf-8') as handle:
            json.dump(value, handle, indent=2, allow_nan=False)

    def _render(self, sets, tag):
        sets = tuple(sets)
        require(sets and len(set(sets)) == len(sets) and all(name in self.yaws for name in sets), 'Unknown/duplicate/empty view sets')
        context = self._preflight()
        require(self.mode != 'native_radial_target_v1' or self._active_execution is not None, 'Controlled execution required before capture')
        directory = self._directory(tag)
        self._save(directory/'input_context.json', {'context': context, 'source_card': self._source_card,
                                                    'semantics_mode': self.mode, 'range_mode': self.range_mode,
                                                    'provider_request': self._active_request,
                                                    'provider_execution': self._active_execution})
        yaws = [yaw for name in sets for yaw in self.yaws[name]]
        query = {'w': self.res, 'h': self.res, 'yaws': ','.join(map(str, yaws)), 'hide_hair': self.hide_hair,
                 'freeze_pose': 'true', 'out': str(directory/'capture.png'), 'geometry_out': str(directory/'geometry.json'),
                 'geometry_blendshape_frames': 'true', **self.capture_options}
        if self.mode == 'native_radial_target_v1' and 'fixed_framing' in self._active_execution:
            framing = self._active_execution['fixed_framing']
            require(isinstance(framing, dict) and set(framing) == {'ortho_size', 'distance', 'target'},
                    'Controlled provider framing unavailable')
            query.update(ortho_size=float(finite_vector([framing['ortho_size']], 1, 'provider ortho')[0]),
                         dist=float(finite_vector([framing['distance']], 1, 'provider distance')[0]),
                         target=','.join(map(str, finite_vector(framing['target'], 3, 'provider target'))))
        if self.mode == 'native_radial_target_v1' and self._active_execution.get('requires_actor_transform_coverage') is True:
            query['geometry_include_actor_transforms'] = 'true'
        start = time.monotonic()
        try:
            # Native POST config comes from QUERY; body is reserved for visibility variants.
            require(self.mode == 'native_radial_target_v1', 'V3 consumer requires explicit radial provider')
            execution = self._active_execution
            require(execution.get('observer_stopped_before_return') is True,
                    'V3 observer must stop before consumer serialization and snapshot work')
            from src.hs2_controlled_radial import bound_read
            response = bound_read(execution['predeclared_consumer_capture'])
            require(digest(execution['consumer_capture_plan']) == digest(self._active_request['plan']['consumer_capture_plan']),
                    'Predeclared consumer capture plan changed')
            require(execution['consumer_capture_plan']['schedule']['flat_yaws'] == yaws,
                    'Consumer view request differs from full predeclared capture')
            actual_directory = Path(response['paired_geometry']['path']).resolve().parent
            self._save(directory/'capture_adoption.json', {
                'revision': 'predeclared_actual_capture_adoption_v3',
                'original_native_response': execution['predeclared_consumer_capture'],
                'original_native_response_digest': digest(response), 'all_views_retained': True,
                'native_capture_already_completed_before_return': True, 'observer_stopped': True})

            self._save(directory/'render_response.json', response)
            evidence = validate_capture(response, context, yaws, actual_directory, self.res)
            fresh = self._snapshot()
            check_context(context, fresh, allow_native_cache_refresh=True, expected_native=context['native59'])
            self._context = fresh
            evidence.update(semantics_mode=self.mode, range_mode=self.range_mode, sampler_profile=context['profile'],
                            input_context_binding_sha256=context['binding_sha256'],
                            input_context_file_sha256=file_sha(directory/'input_context.json'),
                            render_response_file_sha256=file_sha(directory/'render_response.json'),
                            render_response={'path': str(directory/'render_response.json'),
                                             'sha256': file_sha(directory/'render_response.json')},
                            restored_context_binding_sha256=fresh['binding_sha256'], source_card=self._source_card,
                            runtime_certified=False, beauty_is_likeness=False)
            if self.mode == 'native_radial_target_v1':
                validation = self.radial_provider.validate_capture(self._active_execution, copy.deepcopy(evidence))
                require(isinstance(validation, dict) and validation.get('accepted') is True
                        and validation.get('source_bound') is True, 'Provider capture proof unavailable/rejected; no scoring')
                require(isinstance(validation.get('report_bindings'), list) and validation['report_bindings'],
                        'Provider actual report bindings unavailable; no scoring')
                validation['verified_report_bindings'] = [verify_report_binding(row, 'provider actual report')
                                                        for row in validation['report_bindings']]
                if 'restored_before_acceptance' in validation:
                    require(validation['restored_before_acceptance'] is True,
                            'Provider terminal restoration failed; no scoring')
                    restored_report = verify_report_binding(validation.get('restoration_report'),
                                                            'provider terminal restoration')
                    restored = self._snapshot(mode='installed_stateful_diagnostic')
                    check_context(self._original_context, restored, allow_native_cache_refresh=True,
                                  expected_native=self.p0)
                    self.last_restore = {'restored': True, 'provider': restored_report,
                                         'restored_before_acceptance': True,
                                         'full_actor_restoration_certified': False,
                                         'private_cache_exact_restoration_certified':
                                             restored['binding']['private_cache'] == self._original_context['binding']['private_cache']}
                    self._context = restored
                    self._mutated = False
                    self._active_execution = self._active_request = None
                    evidence['terminal_restored_context_binding_sha256'] = restored['binding_sha256']
                evidence['provider_validation'] = validation
            self._save(directory/'capture_evidence.json', evidence)  # BEFORE model/scorer load.
            self.last_capture = {'directory': str(directory), 'evidence': evidence,
                                 'evidence_sha256': file_sha(directory/'capture_evidence.json')}
            self.n_render += len(yaws)
            self.t_render += time.monotonic()-start
            output, cursor = {}, 0
            for name in sets:
                output[name] = [row['path'] for row in evidence['images'][cursor:cursor+len(self.yaws[name])]]
                cursor += len(self.yaws[name])
            return output
        except (Exception, SystemExit) as exc:
            self._save(directory/'capture_failure.json', {'error': str(exc), 'semantics_mode': self.mode, 'scoring_started': False})
            raise

    def evaluate(self, values=None, sets=('a',), tag='eval', *, logical_controls=None):
        sets = tuple(sets)
        require(sets and len(set(sets)) == len(sets) and all(name in self.yaws for name in sets), 'Unknown/duplicate/empty view sets')
        require(isinstance(tag, str) and re.fullmatch(r'[A-Za-z0-9_.-]{1,100}', tag) is not None, 'Unsafe evaluation tag')
        try:
            if self.mode == 'native_radial_target_v1':
                self.radial_provider.declare_consumer_capture(
                    [{'id': name, 'yaws': self.yaws[name]} for name in sets],
                    settle=self.capture_options['settle'], hide_hair=self.hide_hair, options=self.capture_options)
                context = self._preflight()
                candidate = context['native59'] if values is None else values
                vals, _ = validate_write(candidate, None, context['capabilities'], self.range_mode, self.frozen_oob)
                controls = self.logical_controls if logical_controls is None else logical_controls
                require(controls is not None, 'Explicit logical sidecar/controls required for radial provider')
                plan = self.radial_provider.prepare(copy.deepcopy(context), vals.tolist(), logical_controls=copy.deepcopy(controls))
                require(isinstance(plan, dict) and plan.get('mode') == self.mode
                        and plan.get('input_context_binding_sha256') == context['binding_sha256']
                        and plan.get('sampler_profile') == context['profile'], 'Stale/unsupported provider plan')
                plan_hash = digest(plan)
                journal = self._directory(tag+'_plan')
                self._save(journal/'prepared_request.json', {'plan': plan, 'plan_binding_sha256': plan_hash,
                                                            'logical_controls': controls, 'source_context': context})
                self._active_request = {'plan': plan, 'plan_binding_sha256': plan_hash,
                                        'journal': str(journal), 'logical_controls': copy.deepcopy(controls)}
                self._mutated = True
                execution = self.radial_provider.apply(copy.deepcopy(plan), self.boundary)
                self._save(journal/'execution.json', execution)
                require(isinstance(execution, dict) and execution.get('accepted') is True
                        and execution.get('plan_binding_sha256') == plan_hash, 'Provider execution rejected/lost request binding')
                execution['verified_model_source'] = verify_report_binding(execution.get('model_source'), 'provider model/compiler source')
                execution['verified_guard_report'] = verify_report_binding(execution.get('guard_report'), 'provider execution guard')
                fresh = self._snapshot()
                require(np.array_equal(np.asarray(fresh['native59'], dtype=np.float32), vals.astype(np.float32)), 'Provider actual native differs')
                for key in ('actor_transform_id', 'head_id', 'head_sources', 'game', 'profile', 'range_mode', 'capabilities', 'other_public_parameters'):
                    require(fresh['binding'][key] == context['binding'][key], 'Provider changed source/body/profile: '+key)
                if 'owned_expression_patch_fields' in execution:
                    owned = execution['owned_expression_patch_fields']
                    allowed = {'eyes_ptn', 'eyebrow_ptn', 'mouth_ptn', 'eyes_blink', 'eyes_open_max',
                               'mouth_open_max', 'mouth_open_min', 'mouth_fixed', 'mouth_adjust_width', 'eyes_look_pattern'}
                    require(isinstance(owned, list) and len(set(owned)) == len(owned) and set(owned) <= allowed,
                            'Provider expression ownership is unsupported')
                    declared = execution.get('owned_expression_values')
                    require(isinstance(declared, dict) and set(owned) <= set(declared),
                            'Provider owned expression actual values unavailable')
                    old_expression = context['binding']['expression_configuration']
                    new_expression = fresh['binding']['expression_configuration']
                    require(set(old_expression) == set(new_expression), 'Provider expression coverage changed')
                    for key, value in new_expression.items():
                        require(value == (declared[key] if key in owned else old_expression[key]),
                                'Provider changed undeclared/mismatched expression field: '+key)
                self._context, self._active_execution = fresh, execution
            elif values is not None:
                self.set_shapes(values)
            paths = self._render(sets, tag)
            start = time.monotonic()
            views = {name: [float(self.scorer.score(path, use_detector=True)) for path in images] for name, images in paths.items()}
            require(all(np.isfinite(scores).all() for scores in views.values()), 'Nonfinite scorer result')
            self.t_score += time.monotonic()-start
            self.n_eval += 1
            result = {'views': views, **{f'S_{name}': float(np.mean(scores)) for name, scores in views.items()},
                      'semantics_mode': self.mode, 'range_mode': self.range_mode, 'sampler_profile': self._context['profile'],
                      'runtime_certified': False, 'beauty_is_likeness': False, 'capture_evidence': self.last_capture,
                      'scorer': {'class': type(self.scorer).__module__+'.'+type(self.scorer).__qualname__,
                                 'model_source': getattr(self.scorer, 'model_source', None)}}
            self._save(Path(self.last_capture['directory'])/'scores.json', result)
            return result
        except (Exception, SystemExit):
            if self._mutated:
                self.restore()
            raise

    def noise_floor(self, n=8, sets=('a',), verbose=True):
        require(type(n) is int and n >= 2, 'Noise count must be >=2')
        scores = {name: [] for name in sets}
        for i in range(n):
            result = self.evaluate(sets=sets, tag=f'noise{i}')
            for name in sets:
                scores[name].append(result[f'S_{name}'])
        output = {name: {'mean': float(np.mean(values)), 'sd': float(np.std(values, ddof=1)),
                         'min': float(np.min(values)), 'max': float(np.max(values))} for name, values in scores.items()}
        if verbose:
            for name, value in output.items():
                print(f"  [noise] set {name}: mean {value['mean']:.4f} sd {value['sd']:.4f} (n={n})")
        return output

    def restore(self):
        """Restore original public values only; record failures without masking errors."""
        report = {'restored': False, 'private_cache_exact_restoration_certified': False}
        try:
            if not self._mutated:
                report.update(restored=True, no_owned_native_or_abmx_writes=True)
            else:
                require(self.p0 is not None and self._original_context is not None, 'Original restoration context unavailable')
                if self.mode == 'native_radial_target_v1':
                    provider = self.radial_provider.restore(copy.deepcopy(self._original_context), self.boundary)
                    require(isinstance(provider, dict) and provider.get('restored') is True, 'Provider public restore failed')
                    report['provider'] = provider
                else:
                    current = self._snapshot('installed_stateful_diagnostic')
                    check_context(self._original_context, current, allow_native_cache_refresh=True, expected_native=current['native59'])
                    indices = np.flatnonzero(np.asarray(current['native59'], dtype=np.float32) != self.p0.astype(np.float32)).tolist()
                    if indices:
                        vals, idx = validate_write(self.p0[indices], indices, current['capabilities'], self.range_mode)
                        self._write(vals, idx, self.range_mode)
                restored = self._snapshot('installed_stateful_diagnostic')
                check_context(self._original_context, restored, allow_native_cache_refresh=True, expected_native=self.p0)
                report.update(restored=True, actual_native59=restored['native59'], public_abmx_and_other_parameters_preserved=True,
                              persistent_and_private_cache_identical=(restored['binding']['private_cache'] == self._original_context['binding']['private_cache']))
                self._context = restored
                self._mutated = False
                self._active_execution = None
                self._active_request = None
        except (Exception, SystemExit) as exc:  # noqa: BLE001 -- Record restoration failure without masking the original failure.
            report['error'] = str(exc)
            print('  [restore] failed: '+str(exc))
        self.last_restore = report
        self._save(Path(self.work_dir)/('restore_'+uuid.uuid4().hex+'.json'), report)
        return report

    def timing(self):
        count = max(self.n_eval, 1)
        return (f'{self.n_eval} evals, {self.n_render} views | render {self.t_render/count*1000:.0f} ms/eval, '
                f'score {self.t_score/count*1000:.0f} ms/eval | {self.mode}/{self.range_mode} | '
                f"slider path: {'batch' if self._batch else 'per-slider'}")


def slider_names(boundary=None):
    response = (boundary or call)('/maker/face/shapes')
    shape_values(response)
    return [row['name'] for row in sorted(response['shapes'], key=lambda row: row['index'])]
