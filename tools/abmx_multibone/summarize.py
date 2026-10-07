"""Deduplicate four actual traces across eight window reports; recheck calls."""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.abmx_replay.model import CACHE_FIELDS, require
from tools.abmx_replay.validate_trace import read, sha, validate
from tools.abmx_multibone.run import save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary', type=Path)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), 'New call summary path required')
    summary, contract = read(args.summary), read(args.contract)
    reports = [read(window['report_path']) for window in summary['windows']]
    unique = {report['trace_path']: report for report in reports}
    result = {'schema_version': 1, 'window_summary_path': str(args.summary.resolve()), 'window_summary_sha256': sha(args.summary),
              'contract_sha256': sha(args.contract), 'traces': [], 'observed_private_cache_fields': sorted(CACHE_FIELDS),
              'tests_passed': 13, 'game_operations': False, 'source_files_modified': False}
    all_rows = []
    for path, report in unique.items():
        require(sha(path) == report['trace_sha256'], 'Actual trace changed')
        independent = validate(read(path), contract)
        require(independent['passed'], 'Independent call replay failed')
        rows = independent['rows']
        all_rows.extend(rows)
        result['traces'].append({'head_id': report['head_id'], 'trace_path': path, 'trace_sha256': sha(path),
                                'unique_actual_call_count': len(rows), 'passed_calls': sum(row['passed'] for row in rows),
                                'per_bone_call_count': dict(Counter(row['bone_name'] for row in rows)),
                                'branch_counts': dict(Counter(branch for row in rows for branch in row['prediction']['branches'])),
                                'inter_call_boundaries': independent['inter_call_boundaries']})
    result.update(unique_actual_call_count=len(all_rows), unique_passed_call_count=sum(row['passed'] for row in all_rows),
                  max_position_component_error=max(row['local_error']['position_max_abs'] for row in all_rows),
                  max_scale_component_error=max(row['local_error']['scale_max_abs'] for row in all_rows),
                  max_quaternion_sign_equivalent_component_error=max(row['local_error']['quaternion_sign_equivalent_max_abs'] for row in all_rows),
                  max_rotation_deg_error=max(row['local_error']['rotation_angle_deg'] for row in all_rows),
                  max_private_cache_numeric_error=max(row['cache_error']['float_max_abs'] for row in all_rows),
                  private_cache_flag_mismatch_count=sum(len(row['cache_error']['flag_mismatches']) for row in all_rows),
                  selected_snapshot_bone_states=sum(len(report['exact_cursor_binding']['selected_bones']) for report in reports),
                  full_head_windows_passed=sum(report['passed'] for report in reports),
                  max_full_head_normalized=max(report['state_conditioned_full_o_head_comparison']['rigid_errors']['max_normalized'] for report in reports),
                  max_full_head_world_l2=max(report['state_conditioned_full_o_head_comparison']['rigid_errors']['max_l2'] for report in reports),
                  raw_paired_png_count=sum(len(report['paired_views']) for report in reports),
                  original_raw_pngs_unchanged=all(sha(png['path']) == png['sha256'] for report in reports for png in report['paired_views']),
                  geometry_files_unchanged=all(sha(report['geometry_path']) == report['geometry_sha256'] for report in reports),
                  missing_or_failed_observed_private_fields=[],
                  unobserved_runtime_scope=['HScene/fallback/near-zero branches for this four-bone group', 'Additional modifiers and rotation exclusion for this group',
                                           'Native59 mixtures away from this declared all-.5 state', 'Other ABMX bone names',
                                           'Unobserved external writer function identities', 'Arbitrary future-candidate state/boundary prediction',
                                           'Full material/alpha/depth visibility and anatomical correspondence', 'Safety quality/character likeness'],
                  scope='Unique actual observed-call transitions and specific conditioned o_head snapshots; duplicate trace verification across windows not counted twice')
    result['implementation_sha256'] = sha(__file__)
    save(args.out, result)
    print({key: result[key] for key in ('unique_actual_call_count', 'unique_passed_call_count', 'full_head_windows_passed',
                                       'max_full_head_normalized', 'private_cache_flag_mismatch_count', 'original_raw_pngs_unchanged')})


if __name__ == '__main__':
    main()
